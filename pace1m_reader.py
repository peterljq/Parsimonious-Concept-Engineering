import argparse
import ast

def print_all_concepts(index_path, representation_path):

    with open(index_path, 'r') as concept_file:
        concept_list = ast.literal_eval(concept_file.read())

    for each_concept in concept_list:
        print_concept(each_concept, representation_path)

def print_concept(each_concept, representation_path):
    with open(f"{representation_path}{each_concept}.txt", 'r') as representation_file:
        each_concept_representation = ast.literal_eval(representation_file.read())

    print(f"Concept: {each_concept}")
    print(f"Concept Representation: {each_concept_representation}")
    print("\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print all concepts and their representations.")
    parser.add_argument('--index_path', type=str, help="Path to the concept index file.")
    parser.add_argument('--representation_path', type=str, help="Path to the concept representation directory.")

    args = parser.parse_args()

    print_all_concepts(args.index_path, args.representation_path)