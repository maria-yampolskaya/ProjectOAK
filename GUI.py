#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 09:32:23 2020

@author: Sevans
"""



import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import os

import PySimpleGUI as sg
import DataProcessing as dp

direc = 'dataset/scraped_cleaned_200'

# Get list of images to read
with open('Filenames_200.txt', 'r') as f:
    s = f.read()
fnames = s.split('\n')

ALLTYPES = dp.ALLTYPES[:-1]  #18 types total. (the -1 ignores the None-type)
csvdata, cc = dp.read_csv(dp.CSVFILE2)
pokerows    = csvdata[:,cc.CODE]=='1'
POKENAMES   = csvdata[:,cc.NAME][pokerows]
POKENUMBERS = csvdata[:,cc.NUMBER][pokerows]
def poke_to_row(poke):
    return np.where(POKENAMES==poke)[0][0]

def save_results(results, name='untitled', direc='GUI_results'):
    n = 0
    def saveloc(n): return os.path.join(direc, str(n).zfill(3)+'-'+name)
    while os.path.exists(saveloc(n)):
        n+=1
    with open(saveloc(n), 'w') as f:
        for r in results:
            f.write(str(r)+'\n')

def types(poke):
    r = poke_to_row(poke)
    type1, type2 = csvdata[pokerows][r][[cc.TYPE1, cc.TYPE2]]
    if type1 == '': type1 = None #this should never happen
    if type2 == '': type2 = None
    return type1, type2

def strtypes(type1, type2):
    return type1 if type2 is None else type1+', '+type2

def correctness(poke, guess):
    type1, type2 = types(poke)
    g1, g2 = guess
    if (type1, type2) == (guess[0], guess[1]):
        return 3
    elif type1 in guess and type2 in guess:
        return 2
    elif type2 is None:
        if type1 in guess:
            return 1
        else:
            return 0
    elif type1 in guess or type2 in guess:
        return 1
    else:
        return 0
    
c2str = {3: 'You were 100% correct!!',
         2: 'You were basically correct, but mixed up the order.',
         1: 'You got one correct, but one incorrect.',
         0: 'You were 0% correct.'}
c2bar = ['0%', '1 right, 1 wrong', '2/2, wrong order', 'Perfect!']

xlim_max = [5]
def make_barh(corrs):
    bars = np.bincount(corrs, minlength=4)
    xlim_max[0] = xlim_max[0] if (xlim_max[0] >= max(bars)) else xlim_max[0] + 5
    plt.barh(c2bar, bars)
    plt.xlim([0,xlim_max[0]])
    plt.grid('on')
    plt.tight_layout()
    return plt.gcf()

def create_and_save_barh(corrs, saveloc='GUI_results/active_bar_plot.png'):
    figdpi=100
    plt.figure(figsize=(3.9, 2), dpi=figdpi)
    plt.xlabel('Number of Guesses')
    make_barh(corrs)
    plt.savefig(saveloc, dpi=figdpi)
    plt.close()
    return saveloc

def random_imgfile(i=None, fnames=fnames, direc=direc):
    '''return imgfile, pokename'''
    i = i if i is not None else np.random.randint(len(fnames))
    imgfile = os.path.join(direc, fnames[i])
    return [imgfile, pokename(imgfile)]

def pokename(imgfile):
    return os.path.split(os.path.split(imgfile)[0])[1]

def selection(x={'p':None, 's':None}):
    return 'Selected: ' + (', '.join([x[k] for k in ['p','s'] if x[k] is not None]))

bstyle = dict(font=('Helvetica', 12))
on  = dict(button_color=('white', '#23C18E'))
onP = on
onS = dict(button_color=('white', '#2E8569'))
off = dict(button_color=('white', '#252254'))

Lt = max([len(t) for t in ALLTYPES])
typesstyle = dict(size=(Lt, 1), **off, **bstyle)

tstyle = dict(font=('Comic Sans', 12)) #style for text

############################### begin GUI stuff ###############################

titletext   = sg.Text('Welcome to the Pokémon Universe!\n'+
                      'Each Pokémon has a Primary type from one of the 18 types below.\n'+
                      'Some Pokémon have Secondary types as well, but some do not.\n'+
                      'I will treat your first click as your guess for Primary type.\n'+
                      'You can hit Clear to clear your selection(s) at any time.',
                      justification='center', **tstyle)
                      
typecolumns = sg.Column([[sg.Button(ALLTYPES[i], **typesstyle), sg.Button(ALLTYPES[i+9], **typesstyle)]
                         for i in range(9)])
typebuttons = [button for row in typecolumns.Rows for button in row]

im0, poke0 = random_imgfile()
img         = sg.Image(im0)
selected    = sg.Text(selection(),
                       font=['Courier', 14], size=(len('Selected: ')+Lt*2+3, 1),
                       relief='ridge', border_width=2)
clear       = sg.Button('Clear', font=['Courier', 16], **off)
submit      = sg.Button('Submit', font=['Courier', 16], **off)
seen_text   = sg.Text('Have you ever seen this Pokemon before?', **tstyle)

seen_no     = sg.Button('No',  key='-SEEN_NO-', **off)
seen_yes    = sg.Button('Yes', key='-SEEN_YES-', **off)
seen_maybe  = sg.Button('Not sure', key='-SEEN_MAYBE-', **off)

hbartext    = sg.Text('–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––',
                        justification='center', **tstyle)
hbartextcopy= sg.Text('–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––',
                        justification='center', **tstyle)

correcttext = sg.Text('', key='-CORRECTNESS-',
                       font=['Courier', 12], size=(54, 3),
                       relief='ridge', border_width=2)
plotimage   = sg.Image(create_and_save_barh([]))

layout = [
          [titletext],
          [hbartext],
          [typecolumns, img],
          [clear, selected, submit],
          [seen_text, seen_no, seen_yes, seen_maybe],
          [hbartextcopy],
          [correcttext],
          [plotimage]
         ]

def update_typebuttons(typebuttons, x):
    '''makes x['p'] onP, x['s'] onS, and all other typebuttons off.'''
    texts = [button.ButtonText for button in typebuttons]
    for button in typebuttons: button.update(**off)
    if x['p'] is not None: typebuttons[texts.index(x['p'])].update(**onP)
    if x['s'] is not None: typebuttons[texts.index(x['s'])].update(**onS)
    
def update_tracker(x, event):
    if x['p'] is None:
        x['p'] = event
    elif x['p']==event:
        if x['s'] is None:
            x['p'] = None
        else:
            x['p'] = x['s']
            x['s'] = None
    else:
        if x['s'] is None:
            x['s'] = event
        elif x['s']==event:
            x['s'] = None
    return x

def seen_reset(buttons=[seen_yes, seen_no, seen_maybe]):
    for button in buttons: button.update(**off)
def update_seen(event):
    seen_reset()
    if   event == '-SEEN_YES-':
        seen_yes.update(**on)
        return 1
    elif event == '-SEEN_NO-':
        seen_no.update(**on)
        return 0
    elif event == '-SEEN_MAYBE-':
        seen_maybe.update(**on)
        seen = 2
          
# Create the window
window = sg.Window("Type Quiz!", layout, finalize=True) 

############################## Begin Event Loop ##############################
x = {'p':None, 's':None} #tracker for primary & secondary types selected.
seen = -1
im  = im0
poke = poke0

seens   = []
pokes   = []
guesses = []
corrs   = []
try:
    while True:
        event, values = window.read()
        
        if event in ALLTYPES:
            update_tracker(x, event)
            update_typebuttons(typebuttons, x)
        elif event == 'Clear':
            x = {'p':None, 's':None}
            update_typebuttons(typebuttons, x)
        elif event is not None and event.startswith('-SEEN'):
            seen = update_seen(event)
        elif event == 'Submit':
            if x['p'] is None and x['s'] is None:
                correcttext.update('Please select at least one type before submitting.')
            else:
                corr = correctness(poke, (x['p'], x['s']))
                correcttext.update('That was '+poke+', a '+strtypes(*types(poke))+' type!\n'+ 
                                   'You guessed '+strtypes(x['p'], x['s'])+'\n' +
                                   c2str[corr])
                corrs   += [corr]
                seens   += [seen]
                pokes   += [poke]
                guesses += [(x['p'], x['s'])]
                
                plotimage.update(create_and_save_barh(corrs))
                
                x = {'p':None, 's':None}
                update_typebuttons(typebuttons, x)
                seen = -1
                seen_reset()
                im, poke = random_imgfile()
        elif event == '-PLOT-':
            make_barh(corrs, xlim_max)
            plt.show(block=False)
        elif event == sg.WIN_CLOSED:
            break
        
        img.update(im)
        selected.update(selection(x))
except:
    print('There was an error raised! Saving data then raising again.')
    print(seens, pokes, guesses)
    save_results([seens, pokes, guesses])
    raise
    
print(seens, pokes, guesses)
save_results([seens, pokes, guesses])

    
#window.close()  #THIS LINE MAKES CRASH. DONT DO IT