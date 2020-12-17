#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FILE PURPOSE:
    In this file is code which implements a classifier chains in keras.
    Use it to get started with keras functional API, and/or to use classifier chains.
    The first example in this file is a simple modified classifier chain.
        It does not contain a separate CNN for each classifier.
    The second example in this file is more of a "classic" classifier chain..
        It contains a separate CNN for each classifier.
    
    The classifier chains code is kept here instead of in a notebook, because
    the classifier chains seemed to be performing poorly, overall.
    Because of this initial poor performance, combined with a time limit for the project,
    we did not continue looking into classifier chains for too long.
    We have not done enough tests with changing parameters to give up on classifier chains
    permanently, so we keep their code in this .py file.
    
HOW TO USE THE CODE:
    The easiest way to use this will be to replace the create_CNN() function
    in one of the ipynb notebook files and use the classifier_chain() function
    instead, to set the model. This step should be enough to test the classifier chains!

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


################ CLASSIFIER CHAIN EXAMPLE 1: A MODIFIED CHAIN ################
 
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
>> clo= array([2, 5, 0, 3, 4, 6, 7, 1])

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

################ CLASSIFIER CHAIN EXAMPLE 2: A FULL CHAIN OF CNNs ################

def _lname(N, name, N2=''): return str(N) + '_' + name + ('-' + str(N2) if N2 != '' else '')   #function for naming layers nice things

def input_layer(name='images_input'):
    return tf.keras.Input(shape=IMAGESHAPE, name=name)

def mini_cnn(inputs, parms=None, Ndense=1, name='unnamed_CNN'):
    _x = layers.Conv2D(6, 3, 1, input_shape=IMAGESHAPE, activation='relu')(inputs)
    _x = layers.AveragePooling2D(pool_size = (2,2))(_x)
    _x = layers.Flatten()(_x)
    outputs = layers.Dense(Ndense, activation='sigmoid')(_x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    
def classifier_chain(Ntypes=Ntypes, classifier_label_order=None, name='CC'):
    
    clo = classifier_label_order
    clo = clo if clo is not None else np.random.permutation(Ntypes)
    
    inputs = input_layer()
    cc = [] #list of classifiers
    
    #classifier for class 0:
    _x  = mini_cnn(inputs, name=_lname(0, 'CNN'))(inputs)
    c0  = layers.Dense(1, activation='sigmoid', name=_lname(0, 'Classify', clo[0]))(_x)
    cc += [c0]
    
    for i in range(1, Ntypes):
        #classifier for class i:
        _x = mini_cnn(inputs, name=_lname(i, 'CNN'))(inputs)
        _x = layers.concatenate([*cc, _x], name=_lname(i, 'Concat'))
        ci = layers.Dense(1, activation='sigmoid', name=_lname(i, 'Classify', clo[i]))(_x)
        cc += [ci]
        
    #outputs:
    outputs = layers.concatenate([cc[i] for i in clo], name='CLO')
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model, clo

## make model ##
mm, clo = classifier_chain()

## outputs: ##
"""
We can check the classifier_label_order via print('clo=',clo):
>> clo= array([2, 5, 0, 3, 4, 6, 7, 1])

We can check the model structure via mm.summary():
>>
Model: "CC"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
images_input (InputLayer)       [(None, 200, 200, 3) 0                                            
__________________________________________________________________________________________________
0_CNN (Functional)              (None, 1)            58975       images_input[0][0]               
__________________________________________________________________________________________________
0_Classify-5 (Dense)            (None, 1)            2           0_CNN[0][0]                      
__________________________________________________________________________________________________
1_CNN (Functional)              (None, 1)            58975       images_input[0][0]               
__________________________________________________________________________________________________
1_Concat (Concatenate)          (None, 2)            0           0_Classify-5[0][0]               
                                                                 1_CNN[0][0]                      
__________________________________________________________________________________________________
1_Classify-4 (Dense)            (None, 1)            3           1_Concat[0][0]                   
__________________________________________________________________________________________________
2_CNN (Functional)              (None, 1)            58975       images_input[0][0]               
__________________________________________________________________________________________________
2_Concat (Concatenate)          (None, 3)            0           0_Classify-5[0][0]               
                                                                 1_Classify-4[0][0]               
                                                                 2_CNN[0][0]                      
__________________________________________________________________________________________________
2_Classify-7 (Dense)            (None, 1)            4           2_Concat[0][0]                   
__________________________________________________________________________________________________
3_CNN (Functional)              (None, 1)            58975       images_input[0][0]               
__________________________________________________________________________________________________
3_Concat (Concatenate)          (None, 4)            0           0_Classify-5[0][0]               
                                                                 1_Classify-4[0][0]               
                                                                 2_Classify-7[0][0]               
                                                                 3_CNN[0][0]                      
__________________________________________________________________________________________________
3_Classify-2 (Dense)            (None, 1)            5           3_Concat[0][0]                   
__________________________________________________________________________________________________
4_CNN (Functional)              (None, 1)            58975       images_input[0][0]               
__________________________________________________________________________________________________
4_Concat (Concatenate)          (None, 5)            0           0_Classify-5[0][0]               
                                                                 1_Classify-4[0][0]               
                                                                 2_Classify-7[0][0]               
                                                                 3_Classify-2[0][0]               
                                                                 4_CNN[0][0]                      
__________________________________________________________________________________________________
4_Classify-1 (Dense)            (None, 1)            6           4_Concat[0][0]                   
__________________________________________________________________________________________________
5_CNN (Functional)              (None, 1)            58975       images_input[0][0]               
__________________________________________________________________________________________________
5_Concat (Concatenate)          (None, 6)            0           0_Classify-5[0][0]               
                                                                 1_Classify-4[0][0]               
                                                                 2_Classify-7[0][0]               
                                                                 3_Classify-2[0][0]               
                                                                 4_Classify-1[0][0]               
                                                                 5_CNN[0][0]                      
__________________________________________________________________________________________________
5_Classify-3 (Dense)            (None, 1)            7           5_Concat[0][0]                   
__________________________________________________________________________________________________
6_CNN (Functional)              (None, 1)            58975       images_input[0][0]               
__________________________________________________________________________________________________
6_Concat (Concatenate)          (None, 7)            0           0_Classify-5[0][0]               
                                                                 1_Classify-4[0][0]               
                                                                 2_Classify-7[0][0]               
                                                                 3_Classify-2[0][0]               
                                                                 4_Classify-1[0][0]               
                                                                 5_Classify-3[0][0]               
                                                                 6_CNN[0][0]                      
__________________________________________________________________________________________________
6_Classify-0 (Dense)            (None, 1)            8           6_Concat[0][0]                   
__________________________________________________________________________________________________
7_CNN (Functional)              (None, 1)            58975       images_input[0][0]               
__________________________________________________________________________________________________
7_Concat (Concatenate)          (None, 8)            0           0_Classify-5[0][0]               
                                                                 1_Classify-4[0][0]               
                                                                 2_Classify-7[0][0]               
                                                                 3_Classify-2[0][0]               
                                                                 4_Classify-1[0][0]               
                                                                 5_Classify-3[0][0]               
                                                                 6_Classify-0[0][0]               
                                                                 7_CNN[0][0]                      
__________________________________________________________________________________________________
7_Classify-6 (Dense)            (None, 1)            9           7_Concat[0][0]                   
__________________________________________________________________________________________________
CLO (Concatenate)               (None, 8)            0           5_Classify-3[0][0]               
                                                                 4_Classify-1[0][0]               
                                                                 7_Classify-6[0][0]               
                                                                 2_Classify-7[0][0]               
                                                                 1_Classify-4[0][0]               
                                                                 3_Classify-2[0][0]               
                                                                 0_Classify-5[0][0]               
                                                                 6_Classify-0[0][0]               
==================================================================================================
Total params: 471,844
Trainable params: 471,844
Non-trainable params: 0
__________________________________________________________________________________________________