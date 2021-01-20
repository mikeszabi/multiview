# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 12:41:43 2019

@author: szabo
"""
from threading import Event
from pynput import keyboard

quit_event=Event()
quit_event.clear()
measure_event=Event()
measure_event.clear()

def on_press(key):
    try:
        if key.char=='q':
            quit_event.set()
            print('quitting')
        if key.char=='n':
            if not measure_event.is_set():
                measure_event.set()
                print('measure mode on')
            else:
                measure_event.clear()
                print('measure mode off')
            quit_event.set()
    except AttributeError:
#        print('special key {0} pressed'.format(
#            key))
        pass

def on_release(key):
#    print('{0} released'.format(
#        key))
    if key == keyboard.Key.esc:
        # Stop listener
        quit_event.set()
        return False

listener = keyboard.Listener(
    on_press=on_press, 
    on_release=on_release)
listener.start()

