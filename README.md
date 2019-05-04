# segtree-transformer-v0
This repo contains code and configs of [SegTree Transformer: Iterative Refinement of Hierarchical Features](https://rlgm.github.io/papers/67.pdf) (ICLR-RLGM 2019).

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
- DGL 0.3rc (build from source)
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

# Install
The graph builder module is written in Cython, before we run experiments this module must be built:
```
cd graphbuiler
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
