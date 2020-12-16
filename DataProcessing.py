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
IMAGESFOLDER3  = 'dataset/images/dataset150/' #folder containing [(folder with many images of poke) for poke in original 150 pokes]
IMAGESFOLDER4  = 'dataset/scraped' #folder containing [(folder with many images from bulbapedia of poke) for poke in pokes] 
IMAGESFOLDER4C = 'dataset/scraped_cleaned_200' #likst IMAGESFOLDER4 but with bad images culled and all images resized to 200x200.

#edit default file locations here: default CSV; default images folder.
CSVFILE      = CSVFILE2  #default CSV file
IMAGESFOLDER = IMAGESFOLDER1 #default images file

## alltypes:
ALLTYPES = ['Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting', 'Fire', 'Flying', 'Ghost', 'Grass', 'Ground', 'Ice', 'Normal', 'Poison', 'Psychic', 'Rock', 'Steel', 'Water']
PADTYPE  = ''  #pad value for secondary type, for pokemon with no secondary type.
ALLTYPES = ALLTYPES + [PADTYPE]

IDXTYPE = {ALLTYPES[i]:i for i in range(len(ALLTYPES))} #dict with keys type, vals idx
TYPEIDX = {i:ALLTYPES[i] for i in range(len(ALLTYPES))} #dict with keys idx, vals type
def idx_to_type(indices, TYPEIDX=TYPEIDX): return np.vectorize(lambda i: TYPEIDX[i])(indices)   #convert idx to type
def type_to_idx(types, IDXTYPE=IDXTYPE):   return np.vectorize(lambda t: IDXTYPE[t])(types)     #convert type to idx
def prediction_to_idx(modelpredict):
    '''convert predictions (shape=(...,NTYPES+1)) to idx for types.'''
    return np.argmax(modelpredict, axis=-1)


## read csv ##
def read_csv(csvfilename=CSVFILE, return_indexer=True):
    '''return data from csv file. return [data_without_headings, indexer] if return_indexer, else return all_data.
    suggested use:
        csvdata, cc = dp.read_csv(dp.CSVFILE2)
    '''
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

def pokerows(csvdata):
    '''returns boolean array for indexing all the pokemon'.
    pokerows    = csvdata[:,cc.CODE]=='1'
    POKENAMES   = csvdata[:,cc.NAME][pokerows]
    POKENUMBERS = csvdata[:,cc.NUMBER][pokerows]
    '''
    return csvdata[:,cc.CODE]=='1'

def poke_to_N(poke, csvdata, col_poke=CC['NAME'], col_N=CC['NUMBER']):
    '''converts (unique) pokemon to it's pokedex number.'''
    rows = rows_where(str(poke),   csvdata, col_poke)
    return '' if len(rows)==0 else csvdata[rows, col_N][0]

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
    rows = rows_where(str(poke),   csvdata, col_poke)
    return '' if len(rows)==0 else csvdata[rows, col_S][0]

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

def pokes_to_rows(pokes, csvdata):
    '''return [idx of row where NAME=poke, for poke in pokes]'''
    return vectorized_row_where(pokes, csvdata)


## Exclude images without pokemon names.

## Data splitting - split by index alone##
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
    
def split_from_argsplit(x, asplit, **argsplit_kwargs):
    '''splits x using asplit, a dict with idx for train, test, val. returns a dict with x[idx] for train, test, val.'''
    return {key:x[asplit[key]] for key in asplit.keys()}

def argsplit_watch_dups(x, val_size=0.2, test_size=0.1, shuffle=True):
    '''argsplit(x) while ensuring any duplicates in x stay together.
    
    Note: size of each category in result may fluctuate due to duplicates;
    val_size and test_size are used to calculate number of *unique* elements to include in each category.
    
    shuffle: whether to shuffle order of result. Without shuffle, duplicates will stay next to each other in result.
        e.g. without shuffle: x[result['train']]==[0,9,9,8,2,2,2,7,1,1].
        with shuffle: list order will be randomized so duplicates aren't necessarily adjacent.
        
    try this example to understand more:
        t = np.append([0,0, 1,1, 2,2,  3, 4,  5,6,7,8,9,  3, 4,  0,1])
        awd = dp.argsplit_watch_dups(t, 0.25, 0.25, shuffle=True) #or try shuffle=False
        print(dp.split_from_argsplit(t, awd))        #note that the duplicates always end up in the same set in the result.
    '''
    _, unidx, inv = np.unique(x, axis=0, return_index=True, return_inverse=True)
    invi_to_xi    = {i: [i] for i in unidx}
    xi_range      = np.arange(len(x))
    duplicate_xi  = xi_range[~np.isin(xi_range, unidx)]
    for i in duplicate_xi: invi_to_xi[unidx[inv[i]]] += [i]
    argsplit_unique_i = argsplit(unidx, val_size=val_size, test_size=test_size)
    split_unique_i    = split_from_argsplit(unidx, argsplit_unique_i)
    result = {TTV: [i for uniquei in split_unique_i[TTV] for i in invi_to_xi[uniquei]] for TTV in argsplit_unique_i.keys()}
    if shuffle: result = {key: np.random.permutation(val) for key,val in result.items()}
    return result
    
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
    
    def __init__(self, data, labels, serials=None, val_size=0.2, test_size=0.1, do_scaling=True, verbose=True,
                 watch_dups=True, shuffle=True):
        '''initializes dataset: splits data & labels, and scales based on training data (if do_scaling=True).
        <Please enter labels and data as numpy arrays.>
        data:    images of pokemon.
        labels:  labels in machine-learning sense. For our work, these should be pokemon types.
        serials: identifiers; for human-readability. Decent naive use-case: serials = pokemon_names.
        
        do_scaling: whether to do feature-wise scaling of data.
        verbose:    whether to print info during calculations.
        
        watch_dups: True, False, int, or array with length==len(data); default True
            whether to ensure "duplicates" are kept in the same category while argsplit happens.
            e.g. if we have two images of 'ditto', should we ensure they stay in the same category (train/val/test)?
            False ----------> do not care whether duplicates stay in the same category; just do a simple argsplit.
            True/int/array -> ensure duplicates stay in same category.
                "Duplicates" determined by:
                True --> serials.          (If serials is None in this case, watch_dups will be set to False.)
                int ---> serials[:, watch_dups ]. 
                array -> watch_dups.
        shuffle: True or False; default True; see shuffle kwarg in argsplit_watch_dups.
        '''
        self.data_input = np.array(data, copy=False)  #pointer to original data, in case self.data is scaled or altered.
        self.data       = np.array(data, copy=False)   #(could do copy=False to save memory.)
        self.labels     = np.array(labels, copy=False)
        self.serials    = np.array(serials, copy=False)
        self.verbose    = verbose
        self.L          = len(data)
        self.val_size   = int(val_size  * self.L) if val_size <1.0 else val_size
        self.test_size  = int(test_size * self.L) if test_size<1.0 else test_size
        self.train_size = self.L - self.val_size - self.test_size
        
        ## do argsplit: ##
        if (watch_dups is False) or (watch_dups is True and serials is None):
            if verbose: print('|> Splitting data with simple argsplit (not checking for duplicates).')
                
            self.argsplit   = argsplit(data, val_size=val_size, test_size=test_size)
            
        else:
            dupdstr = '|> "Duplicates" determined by whether '
            if verbose: print('|> Splitting data; ensuring duplicates stay together.')
            if watch_dups is True:
                self.dup_watcher = self.serials
                assert len(serials)==self.L
                if verbose: print(dupdstr + 'serials match. E.g. serial[0] =', self.serials[0])
            elif type(watch_dups==int):
                self.dup_watcher = self.serials[:,watch_dups]
                assert len(serials)==self.L
                if verbose: print(dupdstr + 'serials[:,{0:d}] match. E.g. serial[0,{0:d}] ='.format(watch_dups), self.serials[0])
            else:
                self.dup_watcher = watch_dups
                assert len(watch_dups)==len(data)
                if verbose: print(dupdstr + 'watch_dups elements match. E.g. watch_dups[0] =', watch_dups[0])
                    
            self.argsplit = argsplit_watch_dups(self.dup_watcher, val_size=val_size, test_size=test_size, shuffle=shuffle)
            
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


