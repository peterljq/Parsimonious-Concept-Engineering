# Parsimonious-Concept-Engineering
Parsimonious Concept Engineering (PaCE) uses sparse coding on a large-scale concept dictionary to effectively improve the trustworthiness of Large Language Models by precisely controlling and modifying their neural activations. The repository contains our concept representation dataset PaCE-1M mentioned in the paper.

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

