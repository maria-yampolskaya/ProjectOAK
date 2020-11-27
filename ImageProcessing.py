#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 12:13:34 2020

@author: Sevans

depends on: DataProcessing
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time

import matplotlib.image as mpimg #for image reading
from matplotlib import colors    #for showing true color channels images

try:
    import QOL.plots as pqol #Sam's custom plotting stuff.
    pqol.fixfigsize((1,1))   #make default figsize small
    pqol.scale_fonts((2,2))  #make default fontsize small
except:
    print('pqol not loaded, defaulting to matplotlib.pyplot.')
    pqol=plt
    
import  DataProcessing as dp #ProjectOAK file with data  processing functions.
       
def get_image(pokename, folder=dp.IMAGESFOLDER, ext='.jpg'):
    '''returns image of pokemon named <pokename>.'''
    #should check if filename is abspath and ignore folder if it is.
    return mpimg.imread(os.path.join(folder, pokename+ext))

def show_rgba(imgdata, figsize=(10,2), pad=0.1):
    '''shows imgdata as true colors, then channels r, g, b, a.'''
    z = np.zeros(256)
    l = np.linspace(0.0,1.0,256)
    Rcmap = colors.ListedColormap(np.array([l, z, z, z+1]).T)
    Gcmap = colors.ListedColormap(np.array([z, l, z, z+1]).T)
    Bcmap = colors.ListedColormap(np.array([z, z, l, z+1]).T)
    Acmap = colors.ListedColormap(np.array([z, z, z, l  ]).T)
    fig, axs = plt.subplots(1,5, figsize=figsize)
    plt.sca(axs[0]); plt.imshow(imgdata)
    plt.sca(axs[1]); plt.imshow(imgdata[:,:,0], cmap=Rcmap); pqol.colorbar()
    plt.sca(axs[2]); plt.imshow(imgdata[:,:,1], cmap=Gcmap); pqol.colorbar()
    plt.sca(axs[3]); plt.imshow(imgdata[:,:,2], cmap=Bcmap); pqol.colorbar()
    plt.sca(axs[4]); plt.imshow(imgdata[:,:,3], cmap=Acmap); pqol.colorbar()
    plt.tight_layout(pad=pad)

## a few images were jpgs (only 3 channels rgb) so we convert 4-channels to 3-channels here.
def img_to_rgb(imgdata):
    '''converts imgdata to channels r, g, b.
    assumes imgdata is already rgb or has 4 channels (rgba) and has data values between 0. and 1. (inclusive).
    '''
    if imgdata.shape[-1]==4:
        imgrgb = imgdata[...,:-1]
        imga   = imgdata[...,-1:]
        background = colors.to_rgb('white')
        rgbdata = imgrgb * imga + (1-imga) * background
        rgbdata[rgbdata>1.0]=1.0    #max r,g,b value
    else:
        rgbdata = imgdata
    return rgbdata

def show_rgb(imgdata, figsize=(10,2), pad=0.1):
    '''shows imgdata as true colors, then channels r, g, b.
    converts to rgb if needed.
    '''
    rgbdata = img_to_rgb(imgdata)
    z = np.zeros(256)
    l = np.linspace(0.0,1.0,256)
    Rcmap = colors.ListedColormap(np.array([l, z, z, z+1]).T)
    Gcmap = colors.ListedColormap(np.array([z, l, z, z+1]).T)
    Bcmap = colors.ListedColormap(np.array([z, z, l, z+1]).T)
    fig, axs = plt.subplots(1,5, figsize=figsize)
    plt.sca(axs[0]); plt.imshow(rgbdata)
    plt.sca(axs[1]); plt.imshow(rgbdata[:,:,0], cmap=Rcmap); pqol.colorbar()
    plt.sca(axs[2]); plt.imshow(rgbdata[:,:,1], cmap=Gcmap); pqol.colorbar()
    plt.sca(axs[3]); plt.imshow(rgbdata[:,:,2], cmap=Bcmap); pqol.colorbar()
    axs[4].axis('off')
    plt.tight_layout(pad=pad)


### Put images into dict with pokemon names as keys! ###
def get_image_and_poke(filename, folder=dp.IMAGESFOLDER):
    '''returns [imagedata, pokemonname]. e.g. filename='squirtle.png' -> returns [image, 'squirtle'].'''
    name, ext = filename.split('.')
    ext = '.' + ext
    return [get_image(name, folder=folder, ext=ext), name]

def get_all_images(folder, imnames=None, verbose=True):
    '''returns dict of items pokemon:image. e.g. result['squirtle'] = image of squirtle.
    imnames: None or list of strings
        None -> default to all files in folder
        list of strings -> only read these images. (e.g. ['mew.png','pikachu.jpg'] -> just read these two.)
    '''
    now=time.time()
    allimagenames = imnames if imnames is not None else os.listdir(folder)
    #should include some line which skips non-images here. (not needed for this data though.)
    temp = [get_image_and_poke(name, folder=folder) for name in allimagenames]
    allimages = {t[1]: t[0] for t in temp}
    if verbose: print('got all {:3d} images in {:5.2} seconds (from folder = {:s})'.format(
                     len(allimagenames), time.time()-now, folder))
    return allimages

### Some images were jpgs, others were pngs. Convert all to jpgs. ###
def save_as_jpg(name, image, folder=dp.IMAGESFOLDER1J):
    '''saves image as jpg named name.jpg to folder.'''
    filename = name if name.endswith('.jpg') else name+'.jpg'
    plt.imsave(os.path.join(folder, filename), img_to_rgb(image))

def save_all_as_jpgs(folder_initial, folder_jpgs):
    '''saves every image in folder_initial as a jpg in folder_jpgs.'''
    allimages = get_all_images(folder_initial)
    for pokemon in allimages.keys():
        save_as_jpg(pokemon, allimages[pokemon], folder=folder_jpgs)
        
        
### Begin working with scond image dataset (70MB) ###
# these images tend to be named X.png with X a number (e.g. 1.png for bulbasaur)
# or named X-qualifiers.png with qualifiers a word or set of words separated by '-' (e.g. 'mega-y' or 'winter')

def numeric_name(name):
    '''returns whether name is like X.ext with X a number.
    e.g. True for 57.jpg, False for 6-mega-x.png.
    '''
    return name.split('.')[0].isnumeric()