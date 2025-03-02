#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:57:54 2020

@author: Thomas Gaudelet
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'
from train_xgboost import train
import numpy as np

pathX = '../preprocesseddata/G_diseases.h5py'
pathY = '../preprocesseddata/G_drugs.h5py'
target = np.load('drugcentral.npy')
name='drug'
outpath = './'
train(pathX,pathY,target,name,outpath)
