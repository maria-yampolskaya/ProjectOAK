# ProjectOAK: Optimizing Architecture for Kind (of Pokemon)
Group project for PY 895 Machine Learning for Physicists (at Boston University)

Developers: Maria Yampolskaya & Sam Evans

Pokémon, a popular media franchise for all ages, features monsters organized by “types” such as fire, water, bug, and ghost. When first encountering a new creature, humans are often able to guess its type from appearance alone. (For example – one would be correct in guessing the caterpillar-like creature, Caterpie, is bug-typed.) Is this possible for all Pokémon? How strongly is the visual design of each Pokémon influenced by its type? Convolutional neural networks (CNNs) have found widespread use in classifying images according to provided labels. They are particularly successful when the categorized images correlate strongly to one another within each category. In this work, we train a CNN on images of Pokémon, using their types as labels, then try to predict types of unseen Pokémon based on their images.

Credit: [This kaggle dataset](https://www.kaggle.com/vishalsubbiah/pokemon-images-and-types)

Sorted1: sorts by primary type folder only.  
Sorted2: also place a copy of each pokemon image into the folder for its secondary type.  

[Here (~1GB, first 150 pokes, repeat images)](https://www.kaggle.com/thedagger/pokemon-generation-one) is another possible dataset to use for training!  
[Here (~70MB, 819 pokes, 256x256 images)](https://www.kaggle.com/kvpratama/pokemon-images-dataset) is another possible dataset.  
[Here](https://medium.com/@saswatraj/gotta-catch-them-all-building-your-pokédex-using-keras-e144a83e6040) is a page that seems to show a successful neural net for identifying individual pokemon, based on the dataset above (though I am skeptical...).  
