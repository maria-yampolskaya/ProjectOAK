#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 14:38:25 2020

@author: Sevans

Depends on: None
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv

from collections import namedtuple #for easy indexing of CSV data.

## Pokemon constants:
NTYPES = 18

#enter file locations here!
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
def prediction_to_i(modelpredict):
    '''convert predictions (shape=(...,NTYPES+1)) to idx for types.'''
    return np.argmax(modelpredict, axis=-1)

## read csv ##
def read_csv(csvfilename=CSVFILE, return_indexer=True):
    '''return data from csv file. return [data_without_headings, indexer] if return_indexer, else all_data.'''
    with open(csvfilename) as csvfileobj:
        csvreader = csv.reader(csvfileobj)
        csvdata = [row for row in csvreader]
    return [np.array(csvdata[1:]), _csv_indexer(csvdata)] if return_indexer else np.array(csvdata)

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

## convert between attributes:
def rows_where(val, csvdata, col_idx_val=None):
    '''returns csvdata for the row(s) where csvdata[:,col_idx_val]==val.'''
    return np.where(csvdata[:,col_idx_val]==val)[0]

def poke_to_N(poke, csvdata, col_idx_poke=CC['NAME'], col_idx_N=CC['NUMBER']):
    '''converts (unique) pokemon to it's pokedex number.'''
    return csvdata[rows_where(poke,   csvdata, col_idx_poke), col_idx_N]

def N_to_poke(N, csvdata, col_idx_poke=CC['NAME'], col_idx_N=CC['NUMBER'], code=None, col_idx_code=CC['CODE']):
    '''converts (non-unique) pokdex number N to the pokemon it represents.
    May return more than 1 pokemon, unless <code> is input.
    (e.g. 6 -> ['Charizard', 'Charizard Mega X', 'Charizard Mega Y']).
    use code=integer to specify code==integer must be true as well. code=1 gives base form.
    '''
    rows = rows_where(str(N),   csvdata, col_idx_N)
    if code is not None: rows = [i for i in rows if csvdata[i,col_idx_code]==str(code)]
    return csvdata[rows, col_idx_poke]

def poke_to_S(poke, csvdata, col_idx_poke=CC['NAME'], col_idx_S=CC['SERIAL']):
    '''converts pokemon to it's (unique) serial number.'''
    return csvdata[rows_where(poke,   csvdata, col_idx_poke), col_idx_S]

def S_to_poke(S, csvdata, col_idx_poke=CC['NAME'], col_idx_S=CC['SERIAL']):
    '''converts serial number to the (unique) pokemon it represents.'''
    return csvdata[rows_where(str(S),   csvdata, col_idx_S), col_idx_poke]

def N_to_S(N, csvdata, col_idx_S=CC['SERIAL'], col_idx_N=CC['NUMBER'], code=None, col_idx_code=CC['CODE']):
    '''converts (non-unique) pokdex number N to the serial number for the pokemon it represents.
    May return more than 1 pokemon, unless <code> is input.
    (e.g. 6 -> ['Charizard', 'Charizard Mega X', 'Charizard Mega Y']).
    use code=integer to specify code==integer must be true as well. code=1 gives base form.
    '''
    rows = rows_where(str(N),   csvdata, col_idx_N)
    if code is not None: rows = [i for i in rows if csvdata[i,col_idx_code]==str(code)]
    return csvdata[rows, col_idx_S]

def S_to_N(S, csvdata, col_idx_N=CC['NUMBER'], col_idx_S=CC['SERIAL']):
    '''converts (unique) serial number to the pokedex number of the pokemon it represents.'''
    return csvdata[rows_where(str(S),   csvdata, col_idx_S), col_idx_N]