## A Study on Large-Scale Multi-Label Text Classification: from Label Treesto Neural Transfer Learning, including Few- and Zero-Shot Labels

## Requirements:

* \>= Python 3.6
* == TensorFlow 1.15
* == TensorFlow-Hub 0.7.0
* \>= Gensim 3.5.0
* == Keras 2.2.4
* \>= NLTK 3.4
* \>= Scikit-Learn 0.20.1
* \>= Spacy 2.1.0
* \>= TQDM 4.28.1

## Quick start:

### Install python requirements:

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python
>>> import nltk
>>> nltk.download('punkt')
```

### Get pre-trained word embeddings (GloVe + PubMED):

```
wget -P data/vectors/ http://nlp.stanford.edu/data/glove.6B.zip
unzip -j data/vectors/glove.6B.zip data/vectors/glove.6B.200d.txt
wget -P data/vectors/ https://archive.org/details/pubmed2018_w2v_200D.tar
tar xvzf data/vectors/pubmed2018_w2v_200D.tar
```

### Download datasets 

* EURLEX57K [download](http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/datasets.zip)
* AMAZON13K [download](https://drive.google.com/open?id=0B3lPMIHmG6vGYm9abnAzTU1XaTQ)
* MIMIC-III [download](https://mimic.physionet.org)

### Download in-domain BERT versions

* LegalBERT [download](legalbert.tar.gz)
* ClinicalBERT [download](https://www.dropbox.com/s/8armk04fu16algz/pretrained_bert_tf.tar.gz?dl=1)
* AmazonBERT [download](amazonbert.tar.gz)

### Select training options from the configuration JSON file:

E.g., run a Label-wise Attention Network with BIGRUs (BIGRU-LWAN) with the best reported hyper-parameters

```
nano ltmc_configuration.json

{
  "task": {
    "task_name": "eurovoc_classification",
    "dataset_type": "json",
    "dataset": "EURLEX57K",
    "decision_type": "multi_label",
    "cuDNN": true
  },
  "model": {
    "architecture": "LABEL_WISE_ATTENTION_NETWORK",
    "hierarchical": false,
    "document_encoder": "grus",
    "label_encoder": null,
    "attention_mechanism": null,
    "return_attention":  false,
    "token_encoding": "word2vec",
    "embeddings": "glove.6B.200d.txt",
    "bert": null,
    "revisions": null
  },
  "training": {
    "n_hidden_layers": 1,
    "hidden_units_size": 300,
    "dropout_rate": 0.4,
    "word_dropout_rate": 0.00,
    "lr": 0.001,
    "batch_size": 16,
    "epochs": 50,
    "patience": 5,
    "accumulation_steps": 0
  },
  "sampling": {
    "few_threshold": 50,
    "max_sequences_size": 16,
    "max_sequence_size": 2000,
    "max_label_size": 10,
    "sampling_ratio": null,
    "split_type": null,
    "validation_size": 0.2,
    "preprocessed": false,
    "load_from_disk": false,
    "dynamic_batching": true
  },
  "evaluation": {
    "evaluation@k": 10,
    "advanced_mode": true,
    "model_path": null
  }
}

```

**Supported models:** 
* LABEL_WISE_ATTENTION_NETWORK (referred as LWAN in paper)
* ZERO_LABEL_WISE_ATTENTION_NETWORK (referred as CLWAN in paper)
* GRAPH_LABEL_WISE_ATTENTION_NETWORK (referred as GCLWAN in paper)
* BERT

For all Label-wise Attention Networks, you have to define document encoder.
For zero-shot and graph-aware Label-wise Attention Networks, you also have to define label encoder.

**Supported document encoders:** grus, cnns

**Supported label encoders:** word2vec, word2vec+, node2vec, node2vec+, word2vec+node2vec

### Train a model:

```
python train_model.py
```

