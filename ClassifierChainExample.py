#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FILE PURPOSE:
    In this file is code which implements a simple modified classifier chain in keras.
    Use it to get started with keras functional API, and/or to use classifier chains.

CLASSIFIER CHAINS:
    This method for multi-label classification allows for correlation between labels.
    The main idea is to do one binary classification at a time,
    then pass the result, along with the input data to a subsequent classifier.
    For example:
        Inputs <-- set of images, and labels
        C0(Inputs) --------------------> p(fire)   #probability(pokemon type == fire)
        C1(Inputs, p(fire)) -----------> p(water)
        C2(Inputs, p(fire), p(water)) -> p(grass) 
        Output = [p(fire), p(water), p(grass)]
    
    Because the results of previous classifiers are included in subsequent classifiers,
    correlations between labels can be captured by the neural network. This is
    desireable because the labels have some correlations. E.g. water+ice is more
    common than water+fire.
        

THIS EXAMPLE CLASSIFIER CHAIN:
    The classifier chain in this file has the structure:
    Inputs <-- set of images and labels
    CNN(Inputs) -------> A      (conv. neural net applied to input images)
    C0(A) -------------> p0     (prediction of classifier 0)
    C1(A, p0) ---------> p1     (prediction of classifier 1)
    ...
    CL(A, p0,...,pL) --> pL     (prediction of classifier L)
    Output = [p0, p1, ..., pL][perm]
    
    where perm is a pre-selected permutation.
    In this file each Ci is only two layers:
        - a dense layer with Ndense[i] neurons (5 by default).
        - a dense layer with 1 neuron and sigmoid activation.
    

Created on Tue Dec  8 16:27:00 2020

@author: Sevans
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

## set parameters by hand: ##
IMAGESHAPE=(256, 256, 3)
Ntypes = 8                               #18 for types; 8 for Sam's metatypes  
Ndense = [5 for i in range(Ntypes)]      #could choose custom Ndense here.

## define functions ##
def initial_layers(name='CNN'):
    '''initial layers; applies operations to images before applying classifier chain'''
    inputs = tf.keras.Input(shape=IMAGESHAPE)
    _x = layers.Conv2D(6, 3, 2, input_shape=IMAGESHAPE, activation='relu')(inputs)
    _x = layers.AveragePooling2D(pool_size = (2,2))(_x)
    _x = layers.Conv2D(12, 2, 1, input_shape=IMAGESHAPE, activation='relu')(_x)
    _x = layers.AveragePooling2D(pool_size = (2,2))(_x)
    _x = layers.Conv2D(24, 2, 1, input_shape=IMAGESHAPE, activation='relu')(_x)
    _x = layers.AveragePooling2D(pool_size = (2,2))(_x)
    _x = layers.Flatten()(_x)
    model = tf.keras.Model(inputs=inputs, outputs=_x, name=name)
    return model



def _lname(N, name): return str(N) + '-' + name    #function for naming layers nice things

def classifier_chain(Nlabels=Ntypes, classifier_label_order=None, name='cc'):
    '''creates classifier chain with Nlabels.
    returns model, classifier_label_order.
    
    Classifiers predict according to [c0, c1, ..., cNlabels][classifier_label_order].
    e.g. with classifier_label_order=[2,3,0,1], and classifiers==[c0,c1,c2,c3]:
        c2 predicts the label in index 0, c3 <--> label_idx=1, c0 <--> 2, and c1 <--> 3.
    By default, a random order will be selected via np.random.permutation(Ntypes).
    '''
    inputs = tf.keras.Input(shape=IMAGESHAPE, name='INPUT_IMAGES')
    cnn_out = initial_layers()(inputs)
    
    cc = [] #list of classifiers
    
    #classifier for class 0:
    _x  = layers.Dense(Ndense[0], activation='relu', name=_lname(0, 'Dense'))(cnn_out)
    c0  = layers.Dense(1, activation='sigmoid', name=_lname(0, 'Classifier'))(_x)
    cc += [c0]
    
    for i in range(1, Ntypes):
        #classifier for class i:
        _x = layers.concatenate([*cc, cnn_out], name=_lname(i, 'Inputs'))
        _x = layers.Dense(Ndense[i], activation='relu', name=_lname(i, 'Dense'))(_x)
        ci = layers.Dense(1, activation='sigmoid', name=_lname(i, 'Classifier'))(_x)
        cc += [ci]
        
    #outputs:
    clo = classifier_label_order
    clo = clo if clo is not None else np.random.permutation(Ntypes)
    outputs = layers.concatenate([cc[i] for i in clo], name='CLO')
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model, clo

## make model ##
mm, clo = classifier_chain()    
#example compilation of model; you can choose other options for compiling. 
mm.compile(optimizer=tf.keras.optimizers.SGD(),
           loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True),
           metrics=['accuracy'])


## outputs: ##
"""
We can check the classifier_label_order via print('clo=',clo):
>> array([2, 5, 0, 3, 4, 6, 7, 1])

We can check the model structure via mm.summary():
>>
Model: "cc"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
INPUT_IMAGES (InputLayer)       [(None, 256, 256, 3) 0                                            
__________________________________________________________________________________________________
CNN (Functional)                (None, 5400)         1644        INPUT_IMAGES[0][0]               
__________________________________________________________________________________________________
0-Dense (Dense)                 (None, 5)            27005       CNN[0][0]                        
__________________________________________________________________________________________________
0-Classifier (Dense)            (None, 1)            6           0-Dense[0][0]                    
__________________________________________________________________________________________________
1-Inputs (Concatenate)          (None, 5401)         0           0-Classifier[0][0]               
                                                                 CNN[0][0]                        
__________________________________________________________________________________________________
1-Dense (Dense)                 (None, 5)            27010       1-Inputs[0][0]                   
__________________________________________________________________________________________________
1-Classifier (Dense)            (None, 1)            6           1-Dense[0][0]                    
__________________________________________________________________________________________________
2-Inputs (Concatenate)          (None, 5402)         0           0-Classifier[0][0]               
                                                                 1-Classifier[0][0]               
                                                                 CNN[0][0]                        
__________________________________________________________________________________________________
2-Dense (Dense)                 (None, 5)            27015       2-Inputs[0][0]                   
__________________________________________________________________________________________________
2-Classifier (Dense)            (None, 1)            6           2-Dense[0][0]                    
__________________________________________________________________________________________________
3-Inputs (Concatenate)          (None, 5403)         0           0-Classifier[0][0]               
                                                                 1-Classifier[0][0]               
                                                                 2-Classifier[0][0]               
                                                                 CNN[0][0]                        
__________________________________________________________________________________________________
3-Dense (Dense)                 (None, 5)            27020       3-Inputs[0][0]                   
__________________________________________________________________________________________________
3-Classifier (Dense)            (None, 1)            6           3-Dense[0][0]                    
__________________________________________________________________________________________________
4-Inputs (Concatenate)          (None, 5404)         0           0-Classifier[0][0]               
                                                                 1-Classifier[0][0]               
                                                                 2-Classifier[0][0]               
                                                                 3-Classifier[0][0]               
                                                                 CNN[0][0]                        
__________________________________________________________________________________________________
4-Dense (Dense)                 (None, 5)            27025       4-Inputs[0][0]                   
__________________________________________________________________________________________________
4-Classifier (Dense)            (None, 1)            6           4-Dense[0][0]                    
__________________________________________________________________________________________________
5-Inputs (Concatenate)          (None, 5405)         0           0-Classifier[0][0]               
                                                                 1-Classifier[0][0]               
                                                                 2-Classifier[0][0]               
                                                                 3-Classifier[0][0]               
                                                                 4-Classifier[0][0]               
                                                                 CNN[0][0]                        
__________________________________________________________________________________________________
5-Dense (Dense)                 (None, 5)            27030       5-Inputs[0][0]                   
__________________________________________________________________________________________________
5-Classifier (Dense)            (None, 1)            6           5-Dense[0][0]                    
__________________________________________________________________________________________________
6-Inputs (Concatenate)          (None, 5406)         0           0-Classifier[0][0]               
                                                                 1-Classifier[0][0]               
                                                                 2-Classifier[0][0]               
                                                                 3-Classifier[0][0]               
                                                                 4-Classifier[0][0]               
                                                                 5-Classifier[0][0]               
                                                                 CNN[0][0]                        
__________________________________________________________________________________________________
6-Dense (Dense)                 (None, 5)            27035       6-Inputs[0][0]                   
__________________________________________________________________________________________________
6-Classifier (Dense)            (None, 1)            6           6-Dense[0][0]                    
__________________________________________________________________________________________________
7-Inputs (Concatenate)          (None, 5407)         0           0-Classifier[0][0]               
                                                                 1-Classifier[0][0]               
                                                                 2-Classifier[0][0]               
                                                                 3-Classifier[0][0]               
                                                                 4-Classifier[0][0]               
                                                                 5-Classifier[0][0]               
                                                                 6-Classifier[0][0]               
                                                                 CNN[0][0]                        
__________________________________________________________________________________________________
7-Dense (Dense)                 (None, 5)            27040       7-Inputs[0][0]                   
__________________________________________________________________________________________________
7-Classifier (Dense)            (None, 1)            6           7-Dense[0][0]                    
__________________________________________________________________________________________________
CLO (Concatenate)               (None, 8)            0           2-Classifier[0][0]               
                                                                 5-Classifier[0][0]               
                                                                 0-Classifier[0][0]               
                                                                 3-Classifier[0][0]               
                                                                 4-Classifier[0][0]               
                                                                 6-Classifier[0][0]               
                                                                 7-Classifier[0][0]               
                                                                 1-Classifier[0][0]               
==================================================================================================
Total params: 217,872
Trainable params: 217,872
Non-trainable params: 0
__________________________________________________________________________________________________


We can fit and make predictions just like any other model, via mm.fit() and mm.predict().

Enjoy :) 
"""
