#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 14:38:25 2020

@author: Sevans

Depends on: None
"""

#standard imports
import numpy as np
import matplotlib.pyplot as plt
import time

#imports for reading data from files
import os
import csv

#data processing imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#import for easy indexing of CSV data.
from collections import namedtuple 


## Pokemon constants:
NTYPES = 18

## enter file locations here! ##
CSVFILE1   = 'dataset/pokemon.csv'  #dropping support for this file because it is worse...
CSVFILE2   = 'dataset/Pokedex_Ver6.csv'
IMAGESFOLDER1  = 'dataset/images/images'
IMAGESFOLDER1J = 'dataset/images/image_jpgs' #folder where images from jpg-converted dataset are stored.
IMAGESFOLDER2  = 'dataset/images/archive/pokemon/pokemon'
IMAGESFOLDER2J = 'dataset/images/archive/pokemon_jpg/pokemon_jpg'

#edit default file locations here: default CSV; default images folder.
CSVFILE      = CSVFILE2  #default CSV file
IMAGESFOLDER = IMAGESFOLDER1 #default images file

## alltypes:
ALLTYPES = ['Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting', 'Fire', 'Flying', 'Ghost', 'Grass', 'Ground', 'Ice', 'Normal', 'Poison', 'Psychic', 'Rock', 'Steel', 'Water']
PADTYPE  = ''  #pad value for secondary type, for pokemon with no secondary type.
ALLTYPES = ALLTYPES + [PADTYPE]

IDXTYPE = {ALLTYPES[i]:i for i in range(len(ALLTYPES))} #dict with keys type, vals idx
TYPEIDX = {i:ALLTYPES[i] for i in range(len(ALLTYPES))} #dict with keys idx, vals type
def idx_to_type(indices): return np.vectorize(lambda i: TYPEIDX[i])(indices)   #convert idx to type
def type_to_idx(types):   return np.vectorize(lambda t: IDXTYPE[t])(types)     #convert type to idx
def prediction_to_idx(modelpredict):
    '''convert predictions (shape=(...,NTYPES+1)) to idx for types.'''
    return np.argmax(modelpredict, axis=-1)


## read csv ##
def read_csv(csvfilename=CSVFILE, return_indexer=True):
    '''return data from csv file. return [data_without_headings, indexer] if return_indexer, else all_data.'''
    now = time.time()
    with open(csvfilename) as csvfileobj:
        csvreader = csv.reader(csvfileobj)
        csvdata = [row for row in csvreader]
    result = [np.array(csvdata[1:]), _csv_indexer(csvdata)] if return_indexer else np.array(csvdata)
    print('Took {:5.2f} seconds to read data from {:s}'.format(time.time()-now, csvfilename))
    return result

def _read_csv_header(csvfilename):
    with open(csvfilename) as csvfileobj:
        csvreader = csv.reader(csvfileobj)
        csvdata = next(csvreader)
    return np.array([x.upper() for x in csvdata])

def _headers(csvdata):
    '''returns dict of headers:index numbers.
    e.g. row0=['Name','Type'] -> returns {'Name':0, 'Type':1}
    '''
    r = csvdata[0]
    return {r[i].upper():i for i in range(len(r))}

def _dict_to_struct(d, structname='indices', strfunc=lambda x: x.replace(' ', '_')):
    '''returns struct with attrs == keys of dict; vals == vals of dict.'''
    return namedtuple(structname, [strfunc(key) for key in d.keys()])(*d.values())

def _csv_indexer(csvdata, structname='indices', convert_spaces_to='_'):
    '''returns named tuple for easier indexing of csvdata.
    recommended: cc=csv_indexer(csvdata). then do: cc.TYPE1, or do cc.NAME, or some other key.
    example results, for CSVFILE2, produces:
    indices(NUMBER=0, CODE=1, SERIAL=2, NAME=3, TYPE1=4, TYPE2=5, COLOR=6, ABILITY1=7,
            ABILITY2=8, ABILITY_HIDDEN=9, GENERATION=10, LEGENDARY=11, MEGA_EVOLUTION=12,
            HEIGHT=13, WEIGHT=14, HP=15, ATK=16, DEF=17, SP_ATK=18, SP_DEF=19, SPD=20, TOTAL=21)
    '''
    strfunc = lambda x: x.replace(' ', convert_spaces_to)
    return _dict_to_struct(_headers(csvdata), structname=structname, strfunc=strfunc)

CC = _headers(_read_csv_header(CSVFILE)[np.newaxis,:])
_set_CC_keys = ['NAME', 'NUMBER', 'CODE', 'SERIAL'] #ensure these keys exist so that functions are well-defined.
for key in _set_CC_keys:
    CC[key] = CC[key] if (key in CC.keys()) else None

    
## convert between attributes: ##
def rows_where(val, csvdata, col_val=None):
    '''returns csvdata for the row(s) where csvdata[:,col_val]==val.'''
    return np.where(csvdata[:,col_val]==val)[0]

def poke_to_N(poke, csvdata, col_poke=CC['NAME'], col_N=CC['NUMBER']):
    '''converts (unique) pokemon to it's pokedex number.'''
    return csvdata[rows_where(poke,   csvdata, col_poke), col_N][0]

def N_to_poke(N, csvdata, col_poke=CC['NAME'], col_N=CC['NUMBER'], code=None, col_code=CC['CODE']):
    '''converts (non-unique) pokdex number N to the pokemon it represents.
    May return more than 1 pokemon, unless <code> is input.
    (e.g. 6 -> ['Charizard', 'Charizard Mega X', 'Charizard Mega Y']).
    use code=integer to specify code==integer must be true as well. code=1 gives base form.
    '''
    rows = rows_where(str(N),   csvdata, col_N)
    if code is not None:
        rows = [i for i in rows if csvdata[i,col_code]==str(code)]
        return '' if len(rows)==0 else csvdata[rows, col_poke][0]
    else:
        return '' if len(rows)==0 else csvdata[rows, col_poke]

def poke_to_S(poke, csvdata, col_poke=CC['NAME'], col_S=CC['SERIAL']):
    '''converts pokemon to it's (unique) serial number.'''
    return csvdata[rows_where(poke,   csvdata, col_poke), col_S][0]

def S_to_poke(S, csvdata, col_poke=CC['NAME'], col_S=CC['SERIAL']):
    '''converts serial number to the (unique) pokemon it represents.'''
    return csvdata[rows_where(str(S),   csvdata, col_S), col_poke][0]

def N_to_S(N, csvdata, col_S=CC['SERIAL'], col_N=CC['NUMBER'], code=None, col_code=CC['CODE']):
    '''converts (non-unique) pokdex number N to the serial number for the pokemon it represents.
    May return more than 1 pokemon, unless <code> is input.
    (e.g. 6 -> ['Charizard', 'Charizard Mega X', 'Charizard Mega Y']).
    use code=integer to specify code==integer must be true as well. code=1 gives base form.
    '''
    rows = rows_where(str(N),   csvdata, col_N)
    if code is not None:
        rows = [i for i in rows if csvdata[i,col_code]==str(code)]
        return '' if len(rows)==0 else csvdata[rows, col_S][0]
    else:
        return '' if len(rows)==0 else csvdata[rows, col_S]
    
def S_to_N(S, csvdata, col_N=CC['NUMBER'], col_S=CC['SERIAL']):
    '''converts (unique) serial number to the pokedex number of the pokemon it represents.'''
    return csvdata[rows_where(str(S),   csvdata, col_S), col_N][0]


## "Vectorize" conversion between attributes (inefficient, but it works, so... *shrug*): ##
def vectorized_rows_where(vals, csvdata, col_val=None):
    '''returns csvdata for the rows where csvdata[:,col_val]==val for val in vals.'''
    return [rows_where(val, csvdata, col_val) for val in vals]
def vectorized_row_where(vals, csvdata, col_val=None):
    '''returns csvdata for the first row where csvdata[:,col_val]==val for val in vals.'''
    return [rows_where(val, csvdata, col_val)[0] for val in vals]
def pokes_to_N(pokes, csvdata, col_poke=CC['NAME'], col_N=CC['NUMBER']):
    '''converts list of pokes to list of Ns.'''
    return [poke_to_N(poke, csvdata, col_poke, col_N) for poke in pokes]
def Ns_to_poke(Ns, csvdata, col_poke=CC['NAME'], col_N=CC['NUMBER'], code=None, col_code=CC['CODE']):
    '''converts list of Ns to list of pokes. code can be iterable, single value, or None.'''
    try: codes = iter(code)
    except TypeError: codes = iter([code]*len(Ns))
    return [N_to_poke(N, csvdata, col_poke, col_N, next(codes), col_code) for N in Ns]
def pokes_to_S(pokes, csvdata, col_poke=CC['NAME'], col_S=CC['SERIAL']):
    '''converts list of pokes to list of Ss.'''
    return [poke_to_S(poke, csvdata, col_poke, col_S) for poke in pokes]
def Ss_to_poke(S, csvdata, col_poke=CC['NAME'], col_S=CC['SERIAL']):
    '''converts list of Ss to list of pokes.'''
    return [S_to_poke(S, csvdata, col_poke, col_S) for S in Ss]
def Ns_to_S(Ns, csvdata, col_S=CC['SERIAL'], col_N=CC['NUMBER'], code=None, col_code=CC['CODE']):
    '''converts list of Ns to list of Ss.'''
    try:              codes = iter(code)
    except TypeError: codes = iter([code]*len(Ns))
    return [N_to_S(N, csvdata, col_S, col_N, next(codes), col_code) for N in Ns]
def Ss_to_N(Ss, csvdata, col_N=CC['NUMBER'], col_S=CC['SERIAL']):
    '''converts list of Ss to list of Ns.'''
    return [S_to_N(S, csvdata, col_N, col_S) for S in Ss]


## Exclude images without pokemon names.

## Data splitting ##
def argsplit(data, val_size=0.2, test_size=0.1):
    '''returns dict with indices for splitting data into train, val, test sets.
    test_size = portion of data for test. val_size = portion of data for validation.
    '''
    L = len(data)
    Nv = int(val_size * L) if val_size<1.0 else val_size    #number of validation points
    Nt = int(test_size * L) if test_size<1.0 else test_size #number of test points
    if Nv+Nt>0:
        X_train, Xt   = train_test_split(range(len(data)), test_size=Nv+Nt)
        if Nt>0:
            X_val, X_test = train_test_split(    Xt          , test_size=Nt)
            return {'train': X_train, 'val': X_val, 'test': X_test} #usually, function will end here.
        else:
            print('warning, test_size=0 --> test set will be empty.')
            return {'train': X_train, 'val': Xt, 'test': []}
    else:
        print('warning, val_size=0 and test_size=0 --> no splitting was performed.')
        return {'train': data, 'val': [], 'test': []}
    assert False #this line should never be reached - make error if it is reached.

    
## Data Scaling ##
def get_scaler(data):
    '''returns scaler for data. (remember to only input training data here.)'''
    s = StandardScaler()
    s.fit_transform(data.reshape(data.shape[0], -1))
    return s

def scale_images(data, scaler):
    '''scales images using scaler.'''
    shape = data.shape
    return scaler.transform(data.reshape(shape[0], -1)).reshape(shape)

"""
def scaled_datasets(dd):
    '''returns scaled splitdata. ( suggested use: scaled(splitdata(data, labels)) )'''
    s = get_scaler(dd['train']['data'])
    return {key:{'data':scale_images(dataset['data'], s), 'labels':dataset['labels']} for key, dataset in dd.items()}
#""";

## Full_Dataset class (does most of the preprocessing in one place.) ##
class Full_Dataset():
    '''class for storing data easily, and automatically doing a bunch of preprocessing.'''
    
    def __init__(self, data, labels, serials=None, val_size=0.2, test_size=0.1, do_scaling=True, verbose=True):
        '''initializes dataset: splits data & labels, and scales based on training data (if do_scaling=True).
        data:    images of pokemon.
        labels:  labels in machine-learning sense. For our work, these should be pokemon types.
        serials: identifiers; for human-readability. Decent naive use-case: serials = pokemon_names.
        '''
        self.data_input = data  #pointer to original data, in case self.data is scaled or altered.
        self.data       = data
        self.labels     = labels
        self.serials    = serials
        self.verbose    = verbose
        self.L          = len(data)
        self.val_size   = int(val_size  * self.L) if val_size <1.0 else val_size
        self.test_size  = int(test_size * self.L) if test_size<1.0 else test_size
        self.train_size = self.L - self.val_size - self.test_size
        self.argsplit   = argsplit(data, val_size=val_size, test_size=test_size)
        self._argsplit_attr('data')   #makes self.train_data, self.val_data, self.test_data.
        self._argsplit_attr('labels') #similarly but for labels.
        self._argsplit_attr('serials')#similarly but for serials.
        
        self.scaled       = 0   #Counts number of times do_scaling has been called.
        if do_scaling: self.do_scaling()
        
    def _argsplit_attr(self, attr_name, attr_values=None):
        attr_values = getattr(self,attr_name) if attr_values is None else attr_values
        if attr_values is None: return
        for key in self.argsplit.keys(): #(for key in ['train', 'val', 'test']):
            if self.verbose: print('| setting: {:15s}'.format(key+'_'+attr_name), end='')
            setattr(self, key+'_'+attr_name, attr_values[self.argsplit[key]])
        if self.verbose: print('')
                      
    def do_scaling(self):
        if self.verbose:
            print('scaling data')
            now = time.time()
        self.scaler     = get_scaler(self.train_data)
        self.scaled    += 1
        self.data       = scale_images(self.data, self.scaler)
        self.train_data = scale_images(self.train_data, self.scaler)
        self.val_data   = scale_images(self.val_data, self.scaler)
        self.test_data  = scale_images(self.test_data, self.scaler)
        print('Took {:5.2f} seconds to scale data'.format(time.time()-now))
