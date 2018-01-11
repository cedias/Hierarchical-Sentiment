# Sentiment Analysis with Hierarchical Neural Networks - Pytorch flavor

This repository holds Pytorch (near)-implementations of the following papers:
- [Hierarchical Attention Networks for Document Classification paper](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)
- [Neural Sentiment Classification with User &amp; Product Attention paper](https://aclweb.org/anthology/D16-1171)


## Requirements:
    
- Pytorch 0.3
- TQDM
- Gensim (for `BuildW2VEmb.py`)
- Spacy 1.9 (for `prepare_data.py`)


## Quickstart:
    
`chmod +x minimal_ex_cuda.sh`
`./minimal_ex_cuda.sh`

## Main Scripts:

- `han.py` -> Hierarchical Attention Networks for Document Classification model training script.
- `nscupa.py.py` -> Neural Sentiment Classification with User &amp; Product Attention model training script.

## Helper scripts
- `prepare_data.py` transforms gzip files as found on [Julian McAuley Amazon product data page](http://jmcauley.ucsd.edu/data/amazon/) to a list of `(user,item,review,rating)` tuples.
- `minimal_ex(_cuda).sh` Does everything and start learning (just `chmod +x` them).
- `fmtl.py` holds data managing objects.
- `Nets.py` holds neural network models.
- `beer2json.py` is an helper script to convert ratebeer/beeradvocate datasets.
- `BuildW2VEmb.py` can help you build word embeddings from data.


## Note

This code has been written for python-3.6. If you are using python-2.7 you may want to reconsider your laziness, parenthesis on prints aren't that bad...
