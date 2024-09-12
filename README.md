# WikiOFGraph
Official Repository for paper [Ontology-Free General-Domain Knowledge Graph-to-Text Generation Dataset Synthesis using Large Language Model](https://arxiv.org/abs/2409.07088v1)

## Install

To use this repository, you must download the provided [QuestEval](./qeval/) directory. This directory contains modifications to the official QuestEval repository. These modifications are necessary for the specific analyses performed in this project, so please ensure you use the provided version.

```bash
git clone https://github.com/daehuikim/WikiOFGraph.git
cd WikiOFGraph
pip install transformers, vllm, openai, datasets
cd qeval
pip install -e .
```

## Data
The data used in this project is provided via the [Huggingface datasets](https://huggingface.co/datasets/andreaKIM/WikiOFGraph). 
You can download and prepare the dataset by running the following:

```
from datasets import load_dataset

dataset = load_dataset("andreaKIM/WikiOFGraph")
```
or

you can manually download files from this link.

[Download link](https://drive.google.com/drive/folders/1FaEdfgmcjHixVacdZLFCus6HO-k2yrR5?usp=sharing)

## Main Methods
This repository includes code to generate the WikiOFGraph, as described in the paper. 

The process involves several steps, such as data pre-processing, graph extraction, and Data-QuestEval Filtering. 

Detailed implementations are provided in the [process](./process/README.md).

## Qualitative Analysis

This directory contains the details of qualitative analysis described in the paper. 

Detailed analysis scripts and example outputs are provided in the [qualitativeAnalysis](./qualitativeAnalysis/README.md).

## Experiments

Detailed codes related to experiments are in the [experiments](./experiments/README.md).

Trainner and inference codes are implemented using [transformers](https://github.com/huggingface/transformers).