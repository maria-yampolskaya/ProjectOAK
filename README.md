# ProjectOAK: Optimizing Architecture for Kind (of Pokemon)
Group project for PY 895 Machine Learning for Physicists (at Boston University)

Developers: Maria Yampolskaya & Sam Evans

Pokémon, a popular media franchise for all ages, features monsters organized by “types” such as fire, water, bug, and ghost. When first encountering a new creature, humans are often able to guess its type from appearance alone. (For example – one would be correct in guessing the caterpillar-like creature, Caterpie, is bug-typed.) Is this possible for all Pokémon? How strongly is the visual design of each Pokémon influenced by its type? Convolutional neural networks (CNNs) have found widespread use in classifying images according to provided labels. They are particularly successful when the categorized images correlate strongly to one another within each category. In this work, we train a CNN on images of Pokémon, using their types as labels, then try to predict types of unseen Pokémon based on their images.



# Datasets 

## Datasets used
You are encouraged to download the larger datasets using the links below!  
Check the top of the DataProcessing.py file to see what the folders should be named (or edit the file to use custom names).  

Main Data sources:
 - "Dataset 4": [(5000+) 200x200 images of ~898 Pokémon](https://www.kaggle.com/sevans7/pokemon-images-scraped-from-bulbapedia), created (scraped from Bulbapedia & cleaned) by me! -Sam
 - "CSV File" A.K.A "Pokédex File": [Pokédex csv](https://www.kaggle.com/takamasakato/pokemon-all-status-data). 
 
Initial Data sources / other Data sources:
 - "Dataset 1": [120x120 images of 809 Pokémon](https://www.kaggle.com/vishalsubbiah/pokemon-images-and-types)
 - "Dataset 2": [256x256 images of 819 Pokémon](https://www.kaggle.com/kvpratama/pokemon-images-dataset)
 - "Dataset 3": [variously-sized images of 150 Pokémon](https://www.kaggle.com/thedagger/pokemon-generation-one), 

## dataset (folder)
A folder containing the Pokémon images from the smallest dataset, as well as a "Pokédex" file containing information on all Pokémon.  
We exclude the larger datasets from this repository so that the repo does not get too large. 




# Code structure

## Notebooks (.ipynb)

### ProjectOAK_multi_label.ipynb
The code containing the more complicated code, for our Binary Relevance classifier implementation of multi_label classification. This file contains most of our most updated results.
### ProjectOAK_base.ipynb
The code containing the basic layout for processing the 3 datasets and applying a neural net.
### MetaTypes.ipynb
The code for applying machine learning to classifing MetaTypes, which are combinations of the true types.
### imageloadingNB.ipynb
Some code showing some of our work in developing the functions in `ImageProcessing.py`.
Also some code showing some work inspecting the scraped dataset
### ScrapingData.ipynb
Some code for scraping the data for Dataset 4, from Bulbapedia.

## Python Files (.py)

### DataProcessing.py
The code for the `Full_Dataset` class and other useful functions for processing the "Pokédex" file and splitting into training, test, and validation datasets.
### ImageProcessing.py
The code for processing the Pokémon images and standardizing the format.  
This file allows us to read all the images from each dataset in just a few lines.  
See the ImageLoading section of `ProjectOAK_base.ipynb` for an example of how this is done.
### ClassifierChains.py
The code for our Keras implementation of the neural net structure: classifier chains.  
Contains an example with a modified classifier chain with a single CNN preprocessing the images before the chain,
and an example with a full classifier chain with a different CNN in each classifier.  
### ScrapedLoading.py
File containing methods for loading in the data from the original full version of Dataset 4, which includes many more images than the linked dataset on kaggle, but many of those images are not clean images. The larger dataset is not available publicly.  
The methods in this file work on Dataset 4 as well. They are mainly useful for excluding images based on filenames. 
### AccuracyMetrics.py
The code for our custom implementation of precision and recall metrics.  
The code in this file is largely obsolete once you trust and understand the built-in keras implementation of these metrics, however we did not trust or understand the keras implementation at first. This code was a good was to check those.  
The main remaining use of this code is to calculate precision and recall for the quiz results from the Pokémon type quiz GUI.
### GUI.py
I made a GUI! -Sam  
This file runs a quiz which shows the player a random Pokémon image from Dataset 4, and asks them to identify its type (and whether they have seen it before). The quiz results are presented and analyzed a bit in our paper in the appendix. The quiz was a good way to check the Bayes Rate, i.e. it gives us an idea of how well people can perform at the task of identifying the types of Pokémon they have never seen before.
See screenshots in the `GUI_screenshots` folder.
### GUI_quiz_analysis.py
Some methods for analyzing the results from the GUI quiz.


## Miscellaneous

### GUI_screenshots
Screenshots from the Pokémon type identification quiz GUI.
### pokemonSorter
Some code for sorting the Pokémon images from the initial dataset into folders according to their type (useful for some built-in Keras functions, like image_dataset_from_directory).
### InitialProgress
A folder containing the two initial attempts at applying machine learning to the single-label problem.  
The code has matured greatly since then, but we keep these files to give an insight to how we got started.
### Filenames_200.txt
A txt file with the list of the names of all the image files in Dataset 4.
