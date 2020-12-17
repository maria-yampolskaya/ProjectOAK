#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 17:12:13 2020

@author: Sevans

File purpose:
    Methods to evaluate predictions.
    Especially concerned with: developing accuracy metrics.
    
    This file is mostly obsolete once you understand what the similar keras metrics do.
    However, those metrics are confusing and don't do exaclty what you might think sometimes.
    So this file is a good way to compare and test and check if they are doing what you think they are doing.
    
Example usage:
    import AccuracyMetrics as am
    y_true = ( true Nhot labels )
    y_pred = ( predictions from model, e.g. y_pred[0]=[0.5, 0.002, ..., 0.7, 0.1] )
    Nhot_pred  = am.prediction_to_Nhot(y_pred, thresh=2)  #takes the two most likely labels.
    classified = am.classify_Nhots(y_true, Nhot_pred)     #returns dict with number of TP, TN, FP, FN
    precision = am.d_to_precision(classified) #returns precisions of predictions
    recall    = am.d_to_recall(classified)    #returns recalls of predictions
    print('mean precision:', precision.mean(), 'mean recall:', recall.mean())
"""

import numpy as np
from tensorflow import one_hot

DEFAULT_THRESH = 0.5    #default minimum to cause "positive"
DEFAULT_MINPROB = 0.01  #default minimum below which everything will be negative no matter what.
eps = 1e-16 #eps to prevent division by 0 issues.

######### ACCURACY METRICS HERE #########

def precision(TP, FP):
    '''"What fraction of labeled-positive items are truly positive?" == TP / (TP + FP)'''
    return TP / (TP + FP + eps)

def recall(TP, FN):
    '''"What fraction of truly positive items are labeled-positive?" == TP / (TP + FN)'''
    return TP / (TP + FN + eps)

def d_to_precision(d): return precision(d['TP'], d['FP'])
def    d_to_recall(d): return    recall(d['TP'], d['FN'])

## code to convert guesses and true types to 

######### CLASSIFY {TP, TN, FP, FN} HERE #########

def _classify_multiple(y_true, y_pred, classify_fn, **kwargs):
    '''classify multiple predictions at once. **kwargs go to classify_fn.
    returns dict with keys TN, TP, FN, FP; vals = lists.
    e.g. r['TP'][i] = (TP for prediction i).
    '''
    cps = [classify_fn(y_true[i], y_pred[i], **kwargs) for i in range(len(y_true))]
    return {key:np.array([cps[i][key] for i in range(len(cps))]) for key in cps[0].keys()}

## for simple labels, e.g. guess=['Fire', 'Ground']
def classify_guess(truth, guess, N=None):
    '''tells number of true negatives, true positives, false negatives, and false positives.
    For a SINGLE guess. E.g. guess=['Ground', 'Fire'], truth=['Fire', ''].
    Allows for mismatched location. E.g. the example above has one true positive.
    N = total number of types. If N=None, TN will be None.
    returns dict with keys TN, TP, FN, FP.'''
    TP = int(guess[0] in truth) \
        + int(guess[1] in truth and guess[1] != '')
    FN = int(  (guess[1] == '') and (truth[1] != '')  )
    FP = int(guess[0] not in truth) \
        + int((guess[1] not in truth) and guess[1]!='')
    TN = None if N is None else N - (TP + FN + FP)
    return dict(TP=TP, TN=TN, FP=FP, FN=FN)

def classify_guesses(truths, guesses, N=None):
    '''classify_guess for multiple guesses. See _classify_multiple for more documentation.'''
    return _classify_multiple(truths, guesses, classify_guess, N=None)

## for Nhot vectors:
def classify_Nhot(Nhot_true, Nhot_pred):
    '''tells number of true negatives, true positives, false negatives, and false positives.
    For a SINGLE Nhot prediction.
    returns dict with keys TN, TP, FN, FP.
    '''
    t = np.array(Nhot_true, copy=False).astype(bool)
    p = np.array(Nhot_pred, copy=False).astype(bool)
    L = len(t)
    Npos = np.count_nonzero(t)
    Nneg = L - Npos
    TP = np.count_nonzero(p[t])
    TN = np.count_nonzero(~p[~t])
    FN = Npos - TP
    FP = Nneg - TN
    return dict(TP=TP, TN=TN, FP=FP, FN=FN)

def classify_Nhots(Nhots_true, Nhots_pred):
    '''classify_Nhot for multiple predictions. See _classify_multiple for more documentation.'''
    return _classify_multiple(Nhots_true, Nhots_pred, classify_Nhot)

## convert predictions to types (Nhot vectors)

def Nhot_accuracy(y_true, Nhot_pred, fn_d_to_acc, thresh=DEFAULT_THRESH):
    '''returns a measure of accuracy for Nhot labels.'''
    return fn_d_to_acc(classify_Nhots(y_true, Nhot_pred))

####### Let's make functions usable as metrics by keras. #########
# First, we need to convert predictions to Nhot vectors.
def prediction_to_Nhot(pred, thresh = 2, min_prob = DEFAULT_MINPROB):
    '''converts prediction to array of Nhot 'labels'.
    thresh can be:
        float < 1.0 -> Nhot==1 where prediction > thresh.
        int > 0     -> set the largest <thresh> values to 1 for each Nhot.
            refuse to set values to 1 if they are smaller than minprob.
    '''
    p = np.array(pred, copy=False)
    N = p.shape[-1]
    if type(thresh)==int:
        ags = np.argsort(-p)[...,:thresh]
        pred_vals = np.take_along_axis(p, ags, axis=-1) #this line takes the value of pred at the argmax locations.
        return np.array([np.sum(one_hot(ags[i][pred_vals[i]>min_prob], N,  on_value=1), axis=0)
                         for i in range(len(pred_vals))])
    else:
        return (p>thresh).astype(int)

# Next, we create a metric, i.e. a function of ONLY y_true, y_pred.
def Nhot_metric(fn_d_to_acc, name='unnamed_Nhot_metric', thresh=DEFAULT_THRESH):
    '''returns metric (usable by keras).'''
    def metric(y_true, y_pred):
        #.numpy() returns numpy array given tensor.
        return Nhot_accuracy(y_true.numpy(),
                             prediction_to_Nhot(y_pred.numpy(), thresh=thresh),
                             fn_d_to_acc)
    metric.__name__ = name
    return metric

def Nhot_precision_metric(name='precision', thresh=DEFAULT_THRESH):
    '''returns precision metric (a function) (useable by keras).'''
    return Nhot_metric(d_to_precision, name='precision', thresh=thresh)

def Nhot_recall_metric(name='recall', thresh=DEFAULT_THRESH):
    '''returns recall metric (a function) (useable by keras).'''
    return Nhot_metric(d_to_recall, name='recall', thresh=thresh)