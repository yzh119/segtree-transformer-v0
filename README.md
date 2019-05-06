# segtree-transformer-v0
This repo contains code and configs of [SegTree Transformer: Iterative Refinement of Hierarchical Features](https://rlgm.github.io/papers/67.pdf) (ICLR-RLGM 2019).

Transformer model can be viewed as a Graph Attention Network over complete graphs. Instead of complete graph, SegTree Transformer incorporates a latent Segment Tree structure with bottom-up and top-down edges, the time/space complexity per layer is O(d * n log n), where d refers to the hidden size and n refers to the sequence length. 

The model is implemented in *Deep Graph Library(DGL)* with PyTorch as backend.

# Requirements
- Python 3.6+
- PyTorch 1.0+
```
pip install torch torchvision
```
- torchtext 0.4+
```
pip install https://github.com/pytorch/text/archive/master.zip
```
- DGL (build from source code in master branch)
```
git clone https://github.com/dmlc/dgl.git --recurse
cd dgl
mkdir build && cd build
cmake ..
make -j4
cd ../python
python setup.py install
```
- yaml
```
pip install yaml
```
- nltk
```
pip install -U nltk
python
>>> import nltk
>>> nltk.download('punkt')
>>> exit()
```

# Submodules
Before we run experiments, the following submodules must be built manually.

## Graph Builder
The graph builder module is written in Cython to accelerate graph construction:
```
cd graphbuiler
python setup.py install
```

## Custom Op

The custom op module is written in CUDA, to accelerate graph attentions. *DGL 0.3 would provide much faster graph kernels, this submodule shall be deprecated after the release of DGL 0.3*.
```
cd customop
python setup.py install
```

# Experiments 
## Penn Tree Bank
```
python lm.py --config configs/ptb-*.yml
```
## WikiText-2
```
python lm.py --config configs/wiki-*.yml
```
## SST-1
```
python text_classification.py --config configs/sst1-super.yml
```
## IMDB
```
python text_classification.py --config configs/imdb-super.yml
```
