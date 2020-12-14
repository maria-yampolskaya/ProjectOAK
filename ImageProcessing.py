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

import PIL
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

## Convenience functions ##
def image_info(img, verbose=True):
    '''prints then returns img shape, min, and max.'''
    s, mn, mx = img.shape, img.min(), img.max()
    if verbose: print('shape: {:}, min: {:6.2f}, max: {:6.2f}'.format(s, mn, mx))
    return s, mn, mx

## custom colormaps ##
z = np.zeros(256)
l = np.linspace(0.0,1.0,256)
Rcmap = colors.ListedColormap(np.array([l, z, z, z+1]).T)
Gcmap = colors.ListedColormap(np.array([z, l, z, z+1]).T)
Bcmap = colors.ListedColormap(np.array([z, z, l, z+1]).T)
Acmap = colors.ListedColormap(np.array([z, z, z, l  ]).T)

## get image ##       
def get_image(pokename, folder=dp.IMAGESFOLDER, ext='.jpg'):
    '''returns image of pokemon named <pokename>.'''
    #should check if filename is abspath and ignore folder if it is.
    return mpimg.imread(os.path.join(folder, pokename+ext))

# a few images were jpgs (only 3 channels rgb) so we convert 4-channels to 3-channels here.
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

## show image in true color channels ##
def show_rgba(imgdata, figsize=(10,2), pad=0.1):
    '''shows imgdata as true colors, then channels r, g, b, a.'''
    fig, axs = plt.subplots(1,5, figsize=figsize)
    plt.sca(axs[0]); plt.imshow(imgdata)
    plt.sca(axs[1]); plt.imshow(imgdata[:,:,0], cmap=Rcmap); pqol.colorbar()
    plt.sca(axs[2]); plt.imshow(imgdata[:,:,1], cmap=Gcmap); pqol.colorbar()
    plt.sca(axs[3]); plt.imshow(imgdata[:,:,2], cmap=Bcmap); pqol.colorbar()
    plt.sca(axs[4]); plt.imshow(imgdata[:,:,3], cmap=Acmap); pqol.colorbar()
    plt.tight_layout(pad=pad)

def show_rgb(imgdata, figsize=(10,2), pad=0.1):
    '''shows imgdata as true colors, then channels r, g, b.
    converts to rgb if needed.
    '''
    rgbdata = img_to_rgb(imgdata)
    fig, axs = plt.subplots(1,5, figsize=figsize)
    plt.sca(axs[0]); plt.imshow(rgbdata)
    plt.sca(axs[1]); plt.imshow(rgbdata[:,:,0], cmap=Rcmap); pqol.colorbar()
    plt.sca(axs[2]); plt.imshow(rgbdata[:,:,1], cmap=Gcmap); pqol.colorbar()
    plt.sca(axs[3]); plt.imshow(rgbdata[:,:,2], cmap=Bcmap); pqol.colorbar()
    axs[4].axis('off')
    plt.tight_layout(pad=pad)


### Put images into dict with pokemon names as keys! ###
def take_ext(s):
    '''takes extension from s. e.g. s='test.png' -> return ['test', 'png']'''
    try:
        Iext = -(s[::-1].index('.')+1)
        return s[:Iext], s[Iext+1:]
    except:
        print("didn't find '.' in s=",s)
        raise

def get_image_and_poke(filename, folder=dp.IMAGESFOLDER, verbose=True):
    '''returns [imagedata, pokemonname]. e.g. filename='squirtle.png' -> returns [image, 'squirtle'].'''
    name, ext = take_ext(filename)
    ext = '.' + ext
    try:
        return [get_image(name, folder=folder, ext=ext), name]
    except:
        if verbose: print('failed to get image:',filename)
        return None

def get_all_images(folder, imnames=None, verbose=True, to_rgb=True):
    '''returns dict of items pokemon:image. e.g. result['squirtle'] = image of squirtle.
    imnames: None or list of strings
        None -> default to all files in folder
        list of strings -> only read these images. (e.g. ['mew.png','pikachu.jpg'] -> just read these two.)
    '''
    now=time.time()
    allimagenames = imnames if imnames is not None else os.listdir(folder)
    #should include some line which skips non-images here. (not needed for this data though.)
    temp = []
    for name in allimagenames:
        gip = get_image_and_poke(name, folder=folder, verbose=verbose)
        if gip is not None: temp += [gip]
    to_rgb_fn =   img_to_rgb   if   to_rgb   else  lambda x: x
    allimages = {t[1]: to_rgb_fn(t[0]) for t in temp}
    print('got all {:3d} images in {:5.2f} seconds (from folder = {:s})'.format(
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
        
        
## rescale (possibly scaled) image to fit in [0,1] 
def rescaled(v, domain=[0,1]):
    '''linearly rescales v (a numpy array) so its values span domain.'''
    vm = v.min()
    vx = v.max()
    vW = vx - vm
    vc = vm + vW/2.
    W = domain[1] - domain[0]
    c = domain[0] + W/2.
    return (v - vc)/vW * W + c

def plt_rescaled(img, **imshow_kwargs):
    '''plots scaled img; rather than clipping values not in [0,1], makes all values fit in [0,1].'''
    plt.imshow(rescaled(img, [0,1]), **imshow_kwargs)

### Begin working with second image dataset (70MB) ###
# these images tend to be named X.png with X a number (e.g. 1.png for bulbasaur)
# or named X-qualifiers.png with qualifiers a word or set of words separated by '-' (e.g. 'mega-y' or 'winter')


def array_to_PIL(i):
    '''convert i to PIL Image.'''
    return PIL.Image.fromarray(np.uint8(rescaled(i, [0,255])))

def resize_image(i, shape, force_apply=False):
    '''resize image i so each channel has shape shape (e.g. shape=(120,120)). returns numpy array.'''
    return   i   if  (i.shape[:2]==shape or shape is None)  else   np.array(array_to_PIL(i).resize(shape))

def resize_images(ims, shape, verbose=True):
    '''resize ims (numpy array of images) so each channel of each image has shape shape. (e.g. shape=(120,120))
    returns numpy array with all the resized images; it's shape will be (Nimgs, shape[0], shape[1], Nchannels).
    '''
    if ims[0].shape[:2]==shape or shape is None:
        if verbose: print('(No resizing necessary; image shape is already the desired shape.)')
        return ims
    else:
        if verbose: now = time.time()
        result = np.array([resize_image(i, shape) for i in ims])  #the actual work of the function is here.
        if verbose: print('Took {:5.2f} seconds to resize images'.format(time.time()-now))
    return result

#PIL.Image.fromarray(np.uint8(ip.rescaled(im, [0,255]))).resize((256, 256))
        
        
## Begin working with third image dataset (1.21GB), images of first 150 pokes only, multiple images of each. ##
# images are in folders which are named by pokemon names. E.g. ./Abra/ has many pictures of abra.
        
def get_some_images_and_reshape(folder, some=50, shape=(240, 240), imnames=None,
                                exclude_smaller=False,
                                verbose=True, printfreq=5,                       #printfreq in seconds
                                random=False, to_rgb=True,
                                return_filesread=False, return_exclusion_info=False):
    '''returns np.array of some images from folder.
    some                  = max number of images to get from folder.
    shape                 = shape to make image channels.
    exclude_smaller       = whether to exclude images whose channels are initially smaller than shape.
    imnames               = image names to use (will os.listdir(folder) if imnames is None).
    verbose               = whether to print when there are errors with image reading.
        verbose goes up to 3, with 2 & 3 --> info about which images are being excluded & why.
    printfreq             = amount of time (in seconds) between progress updates.
    random                = whether to choose images randomly (if False, read them in order).
    to_rgb                = whether to output images in rgb. (to_rgb=False is not implemented yet.)
    return_filesread      = whether to also return the filenames of those images which were successfully read.
    return_exclusion_info = whether to return additionally a dict with info about which images were excluded.
    '''
    now=time.time()
    filenames = imnames if imnames is not None else os.listdir(folder)   #get image names.
    if random: filenames = np.random.permutation(filenames)          #permute names if random is True.
    to_rgb_fn  =  img_to_rgb   if   to_rgb   else  lambda x: x               #define to_rgb function if to_rgb is True.
    min_imsize =  0 if not exclude_smaller else shape[0]*shape[1]            #minimum size of image; 0 if not exclude_smaller.
    
    exclude_reasons = {0:'failed get_image_and_poke', 1:'not enough channels', 2:'too small'}
    excluded = {0:0, 1:0, 2:0, 'filenames':[], 'reasons':[]}                 #tracks reasons for images being excluded & which are excluded.
    def update_excluded(excluded, reason, filename):
        excluded[reason]     +=1
        excluded['filenames']+=[filename]
        excluded['reasons']  +=[reason]
    
    ## begin main loop for function ##
    printtime = now + printfreq
    imagelist = []
    filesread = []   #track which files were successfully added to imagelist, in case return_filenames is True.
    i = 0
    while len(imagelist)<some and i<len(filenames):
        if time.time() > printtime:
            print('.. getting image {:2d}.'.format(i), end='')
            printtime += printfreq
            
        gip = get_image_and_poke(filenames[i], folder=folder, verbose=verbose)
        if gip is not None:
            if len(gip[0].shape) == 3:     #check to ensure image has channels (for some reason some dont have channels).
                if (gip[0].shape[0] * gip[0].shape[1]) >= min_imsize:
                    imagelist += [resize_image(to_rgb_fn(gip[0]), shape)]
                    filesread += [filenames[i]]
                else: update_excluded(excluded, 2, filenames[i])
            else: update_excluded(excluded, 1, filenames[i])
        else: update_excluded(excluded, 0, filenames[i])
            
        i += 1
    allimages = np.array(imagelist)
    ## end main loop for function ##
    
    if verbose:
        print('Got {:3d} of the {:3d} images in {:5.2f} seconds (from folder = {:s}).'.format(
                len(imagelist), len(filenames), time.time()-now, folder))
        if len(imagelist) < len(filenames):
            print('(Some images were excluded. More info via verbose>1, return_filesread=True, or return_exclusion_info=True.)')
    
    if excluded[0]+excluded[1]+excluded[2] > 0:
        if verbose == 2: print(' >> Excluded images:',excluded['filenames'])
        if verbose  > 2:
            print(' >> List of excluded files and reasons for exclusion:')
            for i in range(len(excluded['filenames'])):
                print('{:>26s} | {:s}'.format(exclude_reasons[excluded['reasons'][i]], excluded['filenames'][i]))
    
    ## decide which things to return, based on optional parameters. (usually, just return allimages.) ##
    returns = [allimages]
    if return_filesread:      returns += [filesread]
    if return_exclusion_info: returns += [excluded]
    
    return returns if len(returns)>1 else returns[0]

def _print_clear(N=80):
    '''clears current printed line of up to N characters, and returns cursor to beginning of the line.
    debugging: make sure to use print(..., end=''), else your print statement will go to the next line.
    '''
    print('\r'+' '*N+'\r',end='')

def images_from_folders_in_dir(directory, Nf=None, nameslists=None, **kwargs):
    '''returns dict of images from folders in directory, with keys=folder names.
    Nf = max number of folders to get images from. Mainly useful when debugging.
    nameslists = optional argument for more control over which images are actually read.
        suggested use: in conjunction with ScrapedLoading.py & scraped data (dataset 4).
        instead of seeking all files in directory, will read files:
        [os.path.join(os.path.join(directory, folder), file) for folder in nameslists.keys() for file in folder.get_files()]
    **kwargs: get passed to get_some_images_and_reshape. see that function for more info.
    '''
    now=time.time()
    some = kwargs.pop('some', 10)
    verbose = kwargs.pop('verbose', False)
    if some < np.inf:
        print('Loading up to {:} images from each folder. Set kwarg "some" to another value to change this number.'.format(some))
    if nameslists is None:
        listdir = os.listdir(directory)
        folders = sorted([folder for folder in     listdir       if os.path.isdir(os.path.join(directory, folder))])
        imnames = [None for folder in listdir]
    else:
        folders = sorted([folder for folder in nameslists.keys() if os.path.isdir(os.path.join(directory, folder))])
        imnames = [nameslists[folder].get_files() for folder in folders]

    ## start loop of reading images.
    result = dict()
    for i in range(len(folders[:Nf])):
        folder, imns = folders[:Nf][i], imnames[i]
        print('Current folder:',folder, end='')
        result[folder] = get_some_images_and_reshape(os.path.join(directory, folder), some=some, verbose=verbose, imnames=imns, **kwargs)
        _print_clear(N=30)
    print('Finished in {:5.2f} seconds.'.format(time.time()-now))
    return result
    
def flatten_imlists(imlists):
    '''once-flatten list of lists of images to a single list of images.
    return (once-flattened list), (idx correponding to list for original list).
    e.g. [[A1,A2], [B1,B2,B3], [C1]] --> ([A1,A2,B1,B2,B3,C1], [0,0,1,1,1,2])
    '''
    idx  = [     i        for i in range(len(imlists)) for j in range(len(imlists[i]))]
    flat = [imlists[i][j] for i in range(len(imlists)) for j in range(len(imlists[i]))]
    return flat, idx