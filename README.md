# Sentiment Analysis with Hierarchical Neural Networks - Pytorch flavor

This repository holds Pytorch (near)-implementations of the following papers:
- [Hierarchical Attention Networks for Document Classification paper](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)
- [Neural Sentiment Classification with User &amp; Product Attention paper](https://aclweb.org/anthology/D16-1171)

Implementations are single-gpu batched.

## Requirements:
    
- Pytorch 0.4
- TQDM

- Gensim (Only for `BuildW2VEmb.py` to build word embeddings with word2vec)
- Spacy 1.9 (Only for `prepare_data.py` to tokenize reviews and split in sentences)


## Quickstart:
    
1.  `chmod +x minimal_ex_cuda.sh`
2. `./minimal_ex_cuda.sh`

## Main Scripts:

- `han.py` -> Hierarchical Attention Networks for Document Classification model training script.
- `nscupa.py` -> Neural Sentiment Classification with User &amp; Product Attention model training script.

### Input formats:

#### Data input (filename argument)

`han.py` and `nscupa.py` expect a pickled dictionnary with the following keys as input:

```
input = {"data":data,"splits":splits,"rows":("user_id","item_id","review","rating")}
```
Where:
- The `"data"` field holds is a list of `("user_id","item_id","review","rating")` tuples where id's and ratings are `str` and review is tokenized and sentence-splitted text `list(list(str))`.
- The  `"splits"` key is a list integers `list(int)`. It's each data point split.
- The `"rows"` is used as helper for tuple indexing in the former scripts.

=> A helper script `prepare_data.py` is provided to create such input pickle file.

#### Pre-trained embeddings (emb optional argument)

`han.py` and `nscupa.py` can use pre-trained embedding. It expects a .txt where each line is a word followed by its vector. The first line of this file provides the number of words and the size of each vectors. `BuildW2VEmb.py` is provided to build word embeddings from data using the word2vec-skipgram algorithm.

If no pre-trained embeddings are provided `han.py` and `nscupa.py` build embedding dictionnaries and vectors on the fly (`---max-feat` arg).


## Helper scripts
- `prepare_data.py` transforms gzip files as found on [Julian McAuley Amazon product data page](http://jmcauley.ucsd.edu/data/amazon/) to a list of `(user,item,review,rating)` tuples.
- `minimal_ex(_cuda).sh` Does everything and start learning (just `chmod +x` them).
- `fmtl.py` holds data managing objects.
- `Nets.py` holds neural network models.
- `beer2json.py` is an helper script to convert ratebeer/beeradvocate datasets.
- `BuildW2VEmb.py` can help you build word embeddings from data.


## Note

This code has been written for python-3.6. If you are only using python-2.7, switch. 
