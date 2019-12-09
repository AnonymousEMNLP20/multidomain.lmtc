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

### Get pre-trained word embeddings (GloVe + Law2Vec):

```
wget -P data/vectors/ http://nlp.stanford.edu/data/glove.6B.zip
unzip -j data/vectors/glove.6B.zip data/vectors/glove.6B.200d.txt
```

### Download datasets 

* EURLEX57K [download](http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/datasets.zip)
* AMAZON13K [download](https://drive.google.com/open?id=0B3lPMIHmG6vGYm9abnAzTU1XaTQ)
* MIMIC-III [download](https://mimic.physionet.org)

### Download in-domain BERT versions

* LegalBERT [download](legalbert.tar.gz)
* clinicalBERT [download](legalbert.tar.gz)
* AmazonBERT [download](legalbert.tar.gz)

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
    "architecture": "BERT",
    "hierarchical": false,
    "document_encoder": null,
    "label_encoder": null,
    "attention_mechanism": null,
    "return_attention":  false,
    "token_encoding": "word2vec",
    "embeddings": null,
    "bert": "bert-base",
    "revisions": null
  },
  "training": {
    "n_hidden_layers": 1,
    "hidden_units_size": 100,
    "dropout_rate": 0.1,
    "word_dropout_rate": 0.00,
    "lr": 0.00002,
    "batch_size": 8,
    "epochs": 20,
    "patience": 2,
    "accumulation_steps": 0
  },
  "sampling": {
    "few_threshold": 50,
    "max_sequences_size": 16,
    "max_sequence_size": 512,
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

**Supported models:** LABEL_WISE_ATTENTION_NETWORK, ZERO_LABEL_WISE_ATTENTION_NETWORK, GRAPH_LABEL_WISE_ATTENTION_NETWORK, BERT
**Supported token encodings:** word2vec, elmo 
**Supported document encoders:** grus, cnns

### Train a model:

```
python train_model.py
```

