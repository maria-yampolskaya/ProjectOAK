# ProjectOAK: Optimizing Architecture for Kind (of Pokemon)
Group project for PY 895 Machine Learning for Physicists (at Boston University)

Developers: Maria Yampolskaya & Sam Evans

Pokémon, a popular media franchise for all ages, features monsters organized by “types” such as fire, water, bug, and ghost. When first encountering a new creature, humans are often able to guess its type from appearance alone. (For example – one would be correct in guessing the caterpillar-like creature, Caterpie, is bug-typed.) Is this possible for all Pokémon? How strongly is the visual design of each Pokémon influenced by its type? Convolutional neural networks (CNNs) have found widespread use in classifying images according to provided labels. They are particularly successful when the categorized images correlate strongly to one another within each category. In this work, we train a CNN on images of Pokémon, using their types as labels, then try to predict types of unseen Pokémon based on their images.

# Code structure

## InitialProgress
A folder containing the two initial attempts at applying machine learning to the single-label problem.
## dataset
A folder containing the Pokémon images, as well as a "Pokédex" file containing information on all Pokémon. 

Data sources: [120x120 images of 809 Pokémon](https://www.kaggle.com/vishalsubbiah/pokemon-images-and-types), [256x256 images of 819 Pokémon](https://www.kaggle.com/kvpratama/pokemon-images-dataset), [variously-sized images of 150 Pokémon](https://www.kaggle.com/thedagger/pokemon-generation-one), [Pokédex csv](https://www.kaggle.com/takamasakato/pokemon-all-status-data).
## ClassifierChainExample
The code for a custom Keras neural net structure: classifier chains.
## DataProcessing
The code for the Full_Dataset class and other useful functions for processing the "Pokédex" file and splitting into training, test, and validation datasets.
## ImageProcessing
The code for processing the Pokémon images and standardizing the format.
## MetaTypes
The code for applying machine learning to classifing MetaTypes, which are combinations of the true types.
## ProjectOAK_base
The code containing the basic layout for processing the 3 datasets and applying a neural net.
## imageloadingNB
Some code showing how to load the Pokémon images.
## pokemonSorter
Some code for sorting the Pokémon images into folders according to their type (useful for some built-in Keras functions, like image_dataset_from_directory).
