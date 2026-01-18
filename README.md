# PaCE: Parsimonious Concept Engineering for Large Language Models (NeurIPS 2024)

[Project Website](https://peterljq.github.io/project/pace) | [ArXiv](https://arxiv.org/abs/2406.04331) | [GitHub](https://github.com/peterljq/Parsimonious-Concept-Engineering)

> Jinqi Luo*, Tianjiao Ding*, Kwan Ho Ryan Chan, Darshan Thaker, Aditya Chattopadhyay, Chris Callison-Burch, René Vidal<br>
> *Equal contribution <br>
> https://arxiv.org/abs/2406.04331 <br>
>
>**Abstract:** Large Language Models (LLMs) are being used for a wide variety of tasks. While they are capable of generating human-like responses, they can also produce undesirable output including potentially harmful information, racist or sexist language, and hallucinations. Alignment methods are designed to reduce such undesirable outputs via techniques such as fine-tuning, prompt engineering, and representation engineering. However, existing methods face several challenges: some require costly fine-tuning for every alignment task; some do not adequately remove undesirable concepts, failing alignment; some remove benign concepts, lowering the linguistic capabilities of LLMs. To address these issues, we propose Parsimonious Concept Engineering (PaCE), a novel activation engineering framework for alignment. First, to sufficiently model the concepts, we construct a large-scale concept dictionary in the activation space, in which each atom corresponds to a semantic concept. Given any alignment task, we instruct a concept partitioner to efficiently annotate the concepts as benign or undesirable. Then, at inference time, we decompose the LLM activations along the concept dictionary via sparse coding, to accurately represent the activations as linear combinations of benign and undesirable components. By removing the latter ones from the activations, we reorient the behavior of the LLM towards the alignment goal. We conduct experiments on tasks such as response detoxification, faithfulness enhancement, and sentiment revising, and show that PaCE achieves state-of-the-art alignment performance while maintaining linguistic capabilities. 

## TLDR
Parsimonious Concept Engineering (PaCE) uses sparse coding on a large-scale concept dictionary to effectively improve the trustworthiness of Large Language Models by precisely controlling and modifying their neural activations. The repository contains our concept representation dataset PaCE-1M and the implementation of the framework in the paper.

## PaCE Framework

The framework of our approach is illustrated in the figure below. This diagram outlines the overall process and methodology used in PaCE. More details can be found in the [project page](https://peterljq.github.io/project/pace/index.html).

![Framework Figure](./image/framework_figure.png)

## PaCE-1M Dataset

To read the PaCE-1M concept representation dataset, we provide a Python script `pace1m_reader.py`. This script reads a concept index file and prints out the representations of each concept stored in individual files within a specified directory.

### Usage

To use the `pace1m_reader.py` script, follow the steps below:

1. Ensure you download the `concept_index.txt` and `concept.zip` files containing the frequency-ranked list of concepts and their contextual representations. Note that the concept list read from `concept_index.txt` is already in the ranked order.
2. Please unzip the `concept.zip` file to get the `./concept` folder. You can do this using the following command in a Unix-like operating system:

```
unzip concept.zip -d ./
```
### Reading the Dataset

Run the script using the following command to print out each of the concepts and their contextual representations:

```
python pace1m_reader.py --index_path ./concept_index.txt --representation_path ./concept/
```

### Sample Concepts and Their Stimuli

The image below provides examples of concepts and their stimuli in our PaCE-1M dataset. Our broad collection of concepts enables PaCE to accurately decompose a task input and modify the representation towards desired behaviors.

![Dataset Visualization](./image/dataset_visualization.png)


### Sampled Activation Space of LLaMA2-13B-Chat

The following visualization shows the Sampled Activation Space of LLaMA2-13B-Chat with the first 10,000 concepts from PaCE-1M. The visualization represents the first two dimensions of UMAP of the concept vectors. We observe that concepts of similar semantics are clustered together, indicating that the activation space has semantic structures.

![Cluster Figure](./image/cluster_figure.png)


### Representation Decomposition via Sparse Coding
We provide the code to sparsely decompose the representation on a collection of dictionary atoms. The `decompose()` function in `sparse_coding.py` enables zero-shot decomposition of target representations into sparse linear combinations of dictionary atoms using elastic net regularization. A basic use case is as follows:

```python
import torch
from sparse_coding import decompose

# Make target representation and the dictionary atoms
target = torch.randn(128)  # e.g., the representation be decomposed
dictionary = [torch.randn(128) for _ in range(10)]  # list of dictionary atoms

# Decompose the target representation using the dictionary atoms
coefficients = decompose(
    target=target,
    dl_dict=dictionary,
    tau=0.95,      
    alpha=0.05,    
    normalize=True 
)
```


## BibTeX
If you find our work helpful, please consider citing our paper:

```
@article{luo2024pace,
    title={PaCE: Parsimonious Concept Engineering for Large Language Models},
    author={Jinqi Luo and Tianjiao Ding and Kwan Ho Ryan Chan and Darshan Thaker and Aditya Chattopadhyay and Chris Callison-Burch and Ren{\'e} Vidal},
    journal={arXiv preprint arXiv:2406.04331},
    year={2024}
}
```
or use the bib entry in NeurIPS format:
```
@inproceedings{luo2024pace,
    title={PaCE: Parsimonious Concept Engineering for Large Language Models},
    author={Jinqi Luo and Tianjiao Ding and Kwan Ho Ryan Chan and Darshan Thaker and Aditya Chattopadhyay and Chris Callison-Burch and Ren{\'e} Vidal},
    booktitle={NeurIPS},
    year={2024}
}
```
