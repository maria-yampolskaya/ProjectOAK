#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:28:12 2020

@author: Sevans
"""

import os
import numpy as np
from numpy import array as arr
import matplotlib.pyplot as plt
try:
    import QOL.plots as pqol #Sam's custom plotting stuff.
    pqol.fixfigsize((4,3))
    pqol.scale_fonts((3,3))
except:
    print('pqol not loaded, defaulting to matplotlib.pyplot.')
    pqol=plt

import ast #for save evaluation of string literal

import DataProcessing as dp
import AccuracyMetrics as am

def cleaned_guesses(guesses):
    '''cleans guesses. Nones are converted to empty strings.'''
    return arr([[g if g is not None else '' for g in guess] for guess in guesses])
    
def cummean(x):
    '''cumulative mean of x'''
    return np.cumsum(x)/(np.arange(len(x))+1)
    
def makeplot(y, include=None, plot_fn=plt.plot, **kwargs):
    '''makes plot of cumulative average of y. include is a boolean array.'''
    include = include if include is not None else np.ones(len(y), dtype=bool)
    R = np.arange(len(include))
    x = R[include] + 1
    plot_fn(x, cummean(y[include]), **kwargs)
    
def makeplots(recall, precision, seenery, qname):
    '''default plots for quiz result analysis.'''
    style = dict(alpha=1)
    mseen = dict(marker='o', fillstyle='none')
    mnew  = dict(marker='x')
    makeplot(recall, seenery[0], label='<recall( new )>', color='#2339C1', **mseen, **style)
    makeplot(recall, seenery[1], label='<recall( seen )>', color='#5A68BE', **mnew, **style)
    makeplot(precision, seenery[0], label='<precision( new )>', color='#F4A41A', **mseen, **style)
    makeplot(precision, seenery[1], label='<precision( seen )>', color='#E7B866', **mnew, **style)
    plt.xlabel('Total Number of Pokemon Shown')
    plt.ylabel('Value (See Legend)')
    plt.title(qname+' Quiz Results')
    plt.grid('on')
    plt.ylim([0,1])
    plt.xlim([1,None])
    plt.legend()


direc = 'GUI_results'

csvdata, cc = dp.read_csv(dp.CSVFILE2)
#pokerows    = csvdata[:,cc.CODE]=='1'
#POKENAMES   = csvdata[:,cc.NAME][pokerows]


files = os.listdir(direc)
quizs = sorted([os.path.join(direc, f) for f in files if os.path.splitext(f)[1]=='.txt'])
for quiz in quizs:
    qname = os.path.split(os.path.splitext(quiz)[0])[1]
    with open(quiz, 'r') as f:
        ss = f.read()
    ss = ss.split('\n')
    seens, pokes, guesses = [ast.literal_eval(s) for s in ss if s!=''] #s=='' for blank lines.
    if seens == []: continue #skips blank quizzes
    N = len(guesses) #number of pokes looked at.
    pokerows   = dp.pokes_to_rows(pokes, csvdata)
    tt = csvdata[pokerows][:,[cc.TYPE1, cc.TYPE2]]    #tt = true_types
    cg = cleaned_guesses(guesses)                     #cg = cleaned_gusses
    TeG = (tt==cg)
    
    perfect_guesses = np.all(TeG, axis=1)
    Pcorrect = arr([cg[i][0] in tt[i] for i in range(N)])
    Scorrect = arr([cg[i][1] in tt[i] for i in range(N)])
    single_typed = (tt[:,1]=='')
    
    
    seens = arr(seens)
    #-1 -> didn't answer, 0 -> not seen, 1-> seen, 2 or None -> maybe seen.
    seenery = [
        (seens == 0),
        (seens == 1),
        (seens == 2) | (seens == None),
        (seens == -1)
        ]
    
    #top2acc = (Pcorrect.astype(int) + Scorrect.astype(int))/2
    
    d = am.classify_guesses(tt, cg)
    recall    = am.d_to_recall(d)
    precision = am.d_to_precision(d)
    
    makeplots(recall, precision, seenery, qname)
    plt.savefig(os.path.join(direc, qname+'_Plot.png'), dpi=250, bbox_inches='tight')
    plt.close()




