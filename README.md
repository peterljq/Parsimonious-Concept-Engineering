# Parsimonious-Concept-Engineering
Parsimonious Concept Engineering (PaCE) uses sparse coding on a large-scale concept dictionary to effectively improve the trustworthiness of Large Language Models by precisely controlling and modifying their neural activations. The repository contains our concept representation dataset PaCE-1M mentioned in the paper.

## Read the PaCE-1M Dataset

To read the PaCE-1M concept representation dataset, we provide a Python script `pace1m_reader.py`. This script reads a concept index file and prints out the representations of each concept stored in individual files within a specified directory.

### Usage

To use the `pace1m_reader.py` script, follow the steps below:

1. Ensure you download the `concept_index.txt` and `concept.zip` files containing the frequency-ranked list of concepts and their contextual representations. Note that the concept list read from `concept_index.txt` is already in the ranked order.
2. Please unzip the `concept.zip` file to get the `./concept` folder. You can do this using the following command in a Unix-like operating system:

```
unzip concept.zip -d ./
```
### Print

Run the script using the following command to print out each of the concepts and their contextual representations:

```
python pace1m_reader.py --index_path ./concept_index.txt --representation_path ./concept/
```

