#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 18:00:11 2020

@author: Sevans

This file is to be used in conjunction with the scraped dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time

try:
    import QOL.plots as pqol #Sam's custom plotting stuff.
    pqol.fixfigsize((1,1))   #make default figsize small
    pqol.scale_fonts((2,2))  #make default fontsize small
except:
    print('pqol not loaded, defaulting to matplotlib.pyplot.')
    pqol=plt

direc = 'dataset/scraped/'   #default location for where images are.

def take_pre(s):
    '''takes prefix from s. e.g. s='078-test.png' -> return ['078', 'test.png']'''
    Ipre = 3
    return s[:Ipre], s[Ipre+1:]

def take_ext(s):
    '''takes extension from s. e.g. s='test.png' -> return ['test', 'png']'''
    try:
        Iext = -(s[::-1].index('.')+1)
        return s[:Iext], s[Iext+1:]
    except:
        print("didn't find '.' in s=",s)
        raise
        
class NamesList:
    '''class for manipulating list of names from folder.
    
    ---- main conveniences / reason to use this class ----
    Apply blacklists based on file names, so that images can be excluded according to certain rules.
    Get list of which images are shiny versions of pokemon.
    Get list of which images correspond to different forms (e.g. Mega evolution, Gigantamax, Alolan, Galar, Primal).
    
    ---- basic usage of this class; example ----
    x=NameList('pokemon', directory, number=N)
    x.get_files()   #list of which files haven't been blacklisted by default blacklists
    x.get_shinies() #list of which files are images of shiny versions of pokemon
    x.get_megas()   #list of which files are images of mega versions of pokemon
    
    ---- helpful hints ----
    You can use kwarg "csvdata" to have the number be calculated automatically, instead of passing it.
        This uses poke_to_N from file DataProcessing.py. If you don't have this file, just don't use kwarg csvdata.
        If number is passed, csvdata will be ignored.
        Example: x=NameList('Jirachi', directory, csvdata=csvdata)
    
    ---- advanced usage below ----
    For exploration of which images to use, you can use blacklist_defaults=False,
        then apply blacklists of your choosing.
    Each blacklist simply edits the variable self.idx, but does not change the data stored in this object.
    
    self.files = full names of files as stored in folders. (ignores 'hidden' files, i.e. files starting with '.')
    self.pre   = prefixes of files in folders == file image number on webpage when downloaded.
    self.ext   = extensions of images (e.g. 'png' or 'jpg').
    self.names = names of images without prefix and extension.
    self.comps = components of name (split by '_') (e.g. 'Spr_3e_063' -> ['Spr', '3e', '063'])
    self._idx  = np.arange(len(self.files))
    self.idx   = idx of files which have not been blacklisted by any applied blacklists.
    
    ---- alternate call signature ----
    You can initialize using a list of files, rather than a pokemon name and directory:
    x=NameList(['list','of','file','names'], False)
    You may optionally pass the kwarg 'name' to inform NameList of pokemon name, e.g.:
    x=NameList(['list','of','file','names'], False, name='Jirachi', number=385)
    (Some of the default blacklisting of NameList can only be done if pokemon name and number are known.)
    '''
    def __init__(self, pokemon, direc, number=None, **kwargs):
        if direc is not False:
            self.pokemon = pokemon
            self.files   = sorted([f for f in os.listdir(os.path.join(direc, pokemon)) if not f.startswith('.')])
        else:       #alternate call signature; order of inputs is messed up a bit but it's fineeee
            self.pokemon = kwargs.pop('name', None)
            self.files   = pokemon
        self._init_funcs(number=number, **kwargs)
                
    def _init_funcs(self, number=None, blacklist_defaults=True, blacklist_forms=True, csvdata=None):
        '''initial functions, once self.files is set.'''
        if number is not None: self.number=number
        else:
            if csvdata is not None:
                import DataProcessing as dp
                self.number = dp.poke_to_N(self.pokemon, csvdata)
            else:
                self.number=None
        if self.number is not None: self.number = str(self.number).zfill(3)
        
        took_pre   = [take_pre(s) for s in self.files]
        self.pre   = [x[0] for x in took_pre] 
        self.sites = [x[1] for x in took_pre]
        took_ext   = [take_ext(s) for s in self.sites]
        self.ext   = [x[1] for x in took_ext]
        self.names = [x[0] for x in took_ext]
        self.comps = [x.split('_') for x in self.names]
        self._idx  = np.arange(len(self.files))
        self.idx   = np.arange(len(self.files))
        if blacklist_defaults:
            self.blacklist_defaults()
        if blacklist_forms:
            self.blacklist_forms()
        
    ## "get" methods ##
    
    def get(self, attr='files'):
        '''returns self.attr, indexed by idx'''
        a = getattr(self, attr)
        return [a[i] for i in self.idx]
    
    def get_files(self): return self.get('files')
    def get_names(self): return self.get('names')
    
    
    ## some pre-defined blacklists ##
    
    def blacklist_sprites(self):    self._blacklist_comp('Spr')
    def blacklist_backs(self):      self._blacklist_i(self.get_backs()  )
    def blacklist_shinies(self):    self._blacklist_i(self.get_shinies())
    def blacklist_megas(self):      self._blacklist_i(self.get_megas()  )
    def blacklist_gigas(self):      self._blacklist_i(self.get_gigas()  )
    def blacklist_alola(self):      self._blacklist_i(self.get_forms()['alola'])
    def blacklist_longnames(self, long=30):  self._blacklist_names(lambda base: len(base)>long)
        
    def blacklist_defaults(self):
        self.blacklist_backs()
        self.blacklist_longnames(30)
        self.blacklist_misc()
        if not self.number is None: self.blacklist_orig_reqs()
    
    def blacklist_forms(self):
        self.blacklist_multiformed() #blacklist pokes with any differently-typed forms (I haven't done the code to handle these yet, so throw out this data.)
        self.blacklist_shinies()
        self.blacklist_megas()
        self.blacklist_gigas()
        forms = self.get_forms()
        for form in forms.keys(): self._blacklist_i(forms[form])
        
    def blacklist_misc(self):
        '''some miscellaneous blacklist things to remove some worse images.'''
        for comp in ['Menu', 'Box', 'C-Gear', 'XD', 'Serena', 'Clemont',]:
            self._blacklist_comp(comp)
        for name in ['CFS', 'DreamLeague', 'Dream_Eater', 'Battle_Royale', 'Poliwrath_OS_anime_2', '143._Snorlax',
                     '152Chikorita_158Totodile', 'Murkrow_200Misdreavus', '1997_GS_Pokemon', '234ODOSHI', '249Lugia_TCG_Model',
                     'Plusle_and_312Minun','484Palkia_alternate', 'Bad_Dreams', 'Dream_Fraxure_anime','Dream_Haxorus_anime',
                     'Hydreigon_BW_anime2', '649Genesect_TCG_Model', 'Genesect_BW_anime_', 'Pikachu_Froakie', 'DreamDreamCD',
                     '778Mimikyu_2', '785Tapu_Koko_2']:            
            self._blacklist_name(name)
        self._blacklist_names(lambda name: 'MS' in name)   #TODO: make this check that MS is followed by a number as well, e.g. MS3.
        self._blacklist_names(lambda name: name.endswith('OD'))   #remove overworld versions of pokes (usually smaller sprites).
    
    def blacklist_multipokes(self):
        '''blacklist name if multiple pokemon names appear in name. Requires csvdata to be passed.'''
        raise NameError('Not yet implemented')
    
    ## blacklist like original requests for images, but a bit smarter ##
        
    def blacklist_orig_reqs(self):
        '''blacklist anything that doesn't satisfy (a slightly more strict version of) the original requirements for which names to pull.
        original requirement: filename must:
            - start with number BUT not number and space
            - OR (contain name or number) AND a 'safeword' in SAFEWORDS = ['Dream', 'HOME', 'Spr', '20th Anniversary', ' s', 'OD']
            #note, 'Café Mix' was part of my requirements, but due to the é character, it didn't work out well.
        more strict requirement here:
            'Spr' -> 'Spr' in comps
            '_s' -> 's' in comps
        '''
        if self.number is None:
            print('Need pokemon number (use kwarg "number" at init or set self.number) to use blacklist_orig_reqs.')
            return
        if self.pokemon is None:
            print('Need pokemon name (use kwarg "name" at init or set self.pokemon) to use blacklist_orig_reqs.')
            return
        SAFEWORDS = ['Dream', 'HOME', '20th_Anniversary', 'OD']
        SAFECOMPS = ['Spr', 's']
        
        def some_a_in_b(a, b):
            for x in a:
                if x in b: return True
            return False
        
        iex = [] #idx to exclude
        for i in self.idx:
            if not self.names[i].startswith(self.number):
                if not (self.pokemon in self.names[i] or self.number in self.names[i]):
                    iex += [i]
                else:
                    if some_a_in_b(SAFEWORDS, self.names[i]) or some_a_in_b(SAFECOMPS, self.comps[i]):
                        pass
                    else:
                        iex += [i]
        self._blacklist_i(iex)


    ## helper functions for blacklists ##
        
    def _blacklist_names(self, rule): self.idx = [i for i in self.idx if not rule(self.names[i])]               
    def _blacklist_name(self, name): self._blacklist_names(lambda x: name in x)
        
    def _blacklist_comps(self, rule): self.idx = [i for i in self.idx if not rule(self.comps[i])]
    def _blacklist_comp(self, comp): self._blacklist_comps(lambda x: comp in x)
                              
    def _blacklist_i(self, blacklist): self.idx = [i for i in self.idx if not i in blacklist]
    
    def blacklist_multiformed(self):
        '''blacklists everything for pokes with differently-typed forms (I haven't implemented proper type association for these yet).'''
        if self.pokemon in ['Arceus', 'Darmanitan', 'Meloetta', 'Oricorio', 'Silvally', 'Necrozma', 'Calyrex', 'Castform']:
            self.idx = []
    
    def get_shinies(self):
        '''idx for images of shiny pokes. (roughly: images with "_s" in them.)'''
        if not hasattr(self, 'shinies'):
            self.shinies = []
            for i in self._idx:
                if 's' in self.comps[i]: self.shinies+=[i]
                elif '-Shiny' in self.names[i]: self.shinies+=[i]
                elif 'Shiny'  in self.comps[i]: self.shinies+=[i]
        return self.shinies
        
    def get_backs(self):
        '''idx for images of backs of pokes. (roughly: images with "_b" in them.)'''
        if not hasattr(self, 'backs'): self.backs = [i for i in self._idx if 'b' in self.comps[i]]
        return self.backs
    
    def get_megas(self):
        '''idx for mega evolutions of pokes'''
        if not hasattr(self, 'megas'):
            self.megas = []
            for i in self._idx:
                if  '-Mega' in self.names[i]: self.megas+=[i]
                elif 'Mega' in self.comps[i]: self.megas+=[i]
                elif self.number is not None:
                    for comp in self.comps[i]:
                        if comp.endswith(self.number+'M'): self.megas+=[i]
        return self.megas
    
    def get_gigas(self):
        '''idx for gigantamax forms of pokes'''
        if not hasattr(self, 'gigas'):
            self.gigas = []
            for i in self._idx:
                if  '-Gigantamax' in self.names[i]: self.gigas+=[i]
                elif 'Gigantamax' in self.comps[i]: self.gigas+=[i]
                elif self.number is not None:
                    for comp in self.comps[i]:
                        if comp.endswith(self.number+'Gi'): self.gigas+=[i]
        return self.gigas
    
    def get_forms(self):
        '''idx for alola, galar, primal, or other forms'''
        if not hasattr(self, 'forms'): self.forms = dict()
            
        FORM = 'alola'
        if FORM not in self.forms.keys():
            self.forms[FORM] = []
            for i in self._idx:
                if  '-Alola' in self.names[i]: self.forms[FORM]+=[i]
                elif 'Alola' in self.comps[i]: self.forms[FORM]+=[i]
                elif self.number is not None:
                    for comp in self.comps[i]:
                        if comp.endswith(self.number+'A'): self.forms[FORM]+=[i]
        
        FORM = 'galar'
        if FORM not in self.forms.keys():
            self.forms[FORM] = []
            for i in self._idx:
                if  '-Galar' in self.names[i]: self.forms[FORM]+=[i]
                elif 'Galar' in self.comps[i]: self.forms[FORM]+=[i]
                elif self.number is not None:
                    for comp in self.comps[i]:
                        if comp.endswith(self.number+'G'): self.forms[FORM]+=[i]
        #TODO: implement other forms
        return self.forms

    
    #FORMS STILL NEEDING WORK:
    #Castform, i=350, dex=351
    #Kyogre, Groudon, i=381,2 ; 'Primal' forms

def NamesLists(direc=direc, NOSPRITES=False, **kwargs):
    '''returns dict of NamesLists for each folder in direc.
    if NOSPRITES, blacklist all the sprites images.
    '''
    now = time.time()
    folders = [folder for folder in os.listdir(direc) if not folder.startswith('.')]
    result = dict()
    for folder in folders:
        result[folder] = NamesList(folder, direc, **kwargs)
        if NOSPRITES: result[folder].blacklist_sprites()
    print('Took {:5.2f} seconds to get filenames'.format(time.time()-now))
    return result

def get_ims_from_nameslists(direc=direc, **kwargs):
    '''returns ims from nameslists'''

def get_and_show_ims(poke, number=None, direc=direc, Ncol=6, verbose=3,
                     figsize=(2,2), NOSPRITES=False, csvdata=None, **kwargs):
    '''show a grid of ims of poke; labeled with their filenames (without the extension & prefix).
    kwargs go to ip.get_some_images_and_reshape.
    '''
    nl = NamesList(poke, direc, number=number, csvdata=csvdata)
    if NOSPRITES: nl.blacklist_sprites()
    gs = ip.get_some_images_and_reshape(os.path.join(direc,poke), imnames=nl.get_files(), verbose=verbose,
                                        some=np.inf, return_filesread=True, **kwargs)
    names = NamesList(gs[1], False, name=poke, number=nl.number).get_names()
    Nplot = len(names)
    if Nplot<1:
        print('no good images for ({:}) {:}'.format(nl.number, nl.pokemon))
        return gs
    Nrow = int(np.ceil(Nplot/Ncol))
    fig, axs = plt.subplots(Nrow, Ncol, figsize=(figsize[0]*Ncol, figsize[1]*Nrow))
    for i in range(Ncol):
        for j in range(Nrow):
            if Nrow==1: plt.sca(axs[i])
            else:       plt.sca(axs[j][i])
            plt.axis('off')
            idx = i + j * Ncol
            if idx < Nplot:
                im = gs[0][idx]
                if np.max(im) > 5: im = im/ 255
                plt.imshow(im)
                plt.title(names[idx])
    plt.tight_layout()
    plt.show()
    return gs
    

