from torchtext import data
from torch.utils.data import DataLoader
from graph import LMBatcher, get_lm_dataset
from graph.lm import LMDataset
from modules import make_model
from optim import get_wrapper
from utils import unpack_params
import numpy as np
import torch as th
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
import yaml
import time


def run(proc_id, n_gpus, devices, config):
    th.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    th.cuda.manual_seed_all(config['seed'])

    dev_id = devices[proc_id]
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')

        world_size = n_gpus
        th.distributed.init_process_group(backend="nccl",
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=dev_id)

    TEXT = data.Field(lower=True, batch_first=True)
    train, dev, test = get_lm_dataset(config['dataset']).splits(TEXT, root='./data')
    TEXT.build_vocab(train)
    train = LMDataset(train, max_length=config['length'], part=(proc_id, n_gpus))
    dev = LMDataset(dev, max_length=config['length'])
    test = LMDataset(test, max_length=config['length'])

    batcher = LMBatcher(TEXT, fully=config['fully'])
    train_loader = DataLoader(dataset=train,
                              batch_size=config['batch_size'] // n_gpus,
                              collate_fn=batcher,
                              shuffle=True,
                              num_workers=6)

    dev_loader = DataLoader(dataset=dev,
                              batch_size=config['dev_batch_size'],
                              collate_fn=batcher,
                              shuffle=False,
                              num_workers=0)
    test_loader = DataLoader(dataset=test,
                              batch_size=config['batch_size'],
                              collate_fn=batcher,
                              shuffle=False,
                              num_workers=0)

    dim_embed = config['dim_embed'] 
    dim_model = config['dim_model'] 
    dim_ff = config['dim_ff']
    num_heads = config['num_heads']
    n_layers = config['n_layers']
    vocab_size = len(TEXT.vocab)

    model = make_model(vocab_size, dim_embed, dim_model, dim_ff, num_heads, vocab_size, n_layers,
        dropouti=config['dropouti'], dropouth=config['dropouth'], 
        dropouta=config.get('dropouta', 0.1), dropoutc=config['dropoutc'], 
        rel_pos=config['rel_pos'], ffn=config['ffn'])

    # tie weights
    if dim_embed == dim_model:
        model.generator.proj.weight = model.embed.lut.weight

    device = th.device(dev_id)
    model = model.to(device)

    embed_params, other_params, wd_params = unpack_params(model.named_parameters())

    optimizer = get_wrapper(config['opt_wrapper'])(
        optim.Adam([
            {'params': embed_params + other_params, 'lr': config.get('lr', 1e-3)},
            {'params': wd_params, 'lr': config.get('lr', 1e-3), 'weight_decay': 5e-5}]))

    best_val = 1e9
    best_test = 0

    for _ in range(config['n_epochs']):
        if proc_id == 0:
            print('training...')
        model.train()
        n_tokens = 0
        sum_loss = 0
        hit = 0
        tic = time.time()
        for i, batch in enumerate(train_loader):
            batch.y = batch.y.to(device)
            batch.g.edata['etype'] = batch.g.edata['etype'].to(device)
            batch.g.ndata['x'] = batch.g.ndata['x'].to(device)
            batch.g.ndata['pos'] = batch.g.ndata['pos'].to(device)

            out = model(batch)
            loss = F.nll_loss(out, batch.y)
            optimizer.zero_grad()
            loss.backward()
            if n_gpus > 1:
                for param in model.parameters():
                    if param.requires_grad and param.grad is not None:
                        th.distributed.all_reduce(param.grad.data, 
                                                  op=th.distributed.ReduceOp.SUM)
                        param.grad.data /= n_gpus

            nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            n = len(batch.y)
            n_tokens += n
            sum_loss += loss.item() * n
            hit += (out.max(dim=-1)[1] == batch.y).sum().item()

            if (i + 1) % config['log_interval'] == 0 and proc_id == 0:
                mem = th.cuda.max_memory_cached()
                print('ppl: ', np.exp(sum_loss / n_tokens), ' acc: ', hit * 1.0 / n_tokens,
                      ' #tokens/s: ', config['batch_size'] * config['log_interval'] * config['length'] / (time.time() - tic),
                      ' #mem: ', mem / 1024 / 1024 / 1024)
                tic = time.time()
                n_tokens, sum_loss, hit = 0, 0, 0

        if n_gpus > 1:
            th.distributed.barrier()
        if proc_id == 0:
            print('evaluating...')
        model.eval()
        n_tokens = 0
        sum_loss = 0
        hit = 0
        for batch in dev_loader:
            batch.y = batch.y.to(device)
            batch.g.edata['etype'] = batch.g.edata['etype'].to(device)
            batch.g.ndata['x'] = batch.g.ndata['x'].to(device)
            batch.g.ndata['pos'] = batch.g.ndata['pos'].to(device)

            with th.no_grad():
                out = model(batch)
                loss = F.nll_loss(out, batch.y, reduction='sum')
                n = len(batch.y)
                n_tokens += n
                sum_loss += loss.item()
                hit += (out.max(dim=-1)[1] == batch.y).sum().item()

        if proc_id == 0:
            if config['dataset'] == 'enwik8' or config['dataset'] == 'text8':
                print('bpc: ', (sum_loss / n_tokens) / np.log(2), ' acc: ', hit * 1.0 / n_tokens)
            else:
                print('ppl: ', np.exp(sum_loss / n_tokens), ' acc: ', hit * 1.0 / n_tokens)
        optimizer.adjust_lr(np.exp(sum_loss / n_tokens))
        val_ppl = np.exp(sum_loss / n_tokens)

        if proc_id == 0:
            print('testing...')
        model.eval()
        n_tokens = 0
        sum_loss = 0
        hit = 0
        for batch in test_loader:
            batch.y = batch.y.to(device)
            batch.g.edata['etype'] = batch.g.edata['etype'].to(device)
            batch.g.ndata['x'] = batch.g.ndata['x'].to(device)
            batch.g.ndata['pos'] = batch.g.ndata['pos'].to(device)

            with th.no_grad():
                out = model(batch)
                loss = F.nll_loss(out, batch.y, reduction='sum')
                n = len(batch.y)
                n_tokens += n
                sum_loss += loss.item()
                hit += (out.max(dim=-1)[1] == batch.y).sum().item()

        if proc_id == 0:
            if config['dataset'] == 'enwik8' or config['dataset'] == 'text8':
                print('bpc: ', (sum_loss / n_tokens) / np.log(2), ' acc: ', hit * 1.0 / n_tokens)
            else:
                print('ppl: ', np.exp(sum_loss / n_tokens), ' acc: ', hit * 1.0 / n_tokens)

        if val_ppl < best_val:
            best_val = val_ppl
            best_test = np.exp(sum_loss / n_tokens)

        if proc_id == 0:
            if config['dataset'] == 'enwik8' or config['dataset'] == 'text8':
                print('best val: %.2f ' % np.log2(best_val), 'best test: %.2f ' % np.log2(best_test))
            else:
                print('best val: %.2f ' % best_val, 'best test: %.2f ' % best_test)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("language modeling")
    argparser.add_argument('--config', type=str)
    argparser.add_argument('--gpu', type=str, default='0')
    args = argparser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    devices = list(map(int, args.gpu.split(',')))

    n_gpus = len(devices)
    if n_gpus == 1:
        run(0, n_gpus, devices, config)
    else:
        mp = th.multiprocessing
        mp.spawn(run, args=(n_gpus, devices, config), nprocs=n_gpus)
