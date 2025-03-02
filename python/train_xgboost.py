#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:32:21 2020

@author: Thomas Gaudelet
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
import h5py
import datetime
import itertools as it
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
import sys
import pandas as pd
import subprocess
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
import pickle

from typing import Tuple

def standardize(scores):
    avg = np.average(scores)
    std = np.std(scores)
    return (scores-avg)/(std+1e-6)
    
def train(path_X, path_Y, target,name,outpath,objective='binary:logitraw'):

    name = name
    
    fX = h5py.File(path_X,'r')['dataset'][:].astype(np.float32)
    fY = h5py.File(path_Y,'r')['dataset'][:].astype(np.float32)
    
    fX /= (np.linalg.norm(fX,axis=1,keepdims=True)+1e-8)
    fY /= (np.linalg.norm(fY,axis=1,keepdims=True)+1e-8)
    
    mX, mY = fX.shape[0], fY.shape[0]
    
    lbls = target.flatten()#np.load(path_target).flatten().astype(np.float32)
    
    ixs = np.array(list(it.product(np.arange(mX),np.arange(mY))))
    print(lbls.shape,ixs.shape)
    repeats = 10
    rows = ['avg','std']+[i for i in range(repeats)] 
    columns = ['AUROC','AUPRC']
    
    etas = [0.25,0.5,0.75]#,0.1,0.25]
    gammas = [0,10]#[0,1e-3,1e-2,1e-1]
    depths = [6,12]#[1e-3,1e-2]
    lambdas = [1,10,100]#,128,256]
    
    best = 0
    params = list(it.product(etas,gammas,depths,lambdas))
    xvals = np.zeros((len(params),1))
    num_rounds = 1000
    for i,(eta,gamma,depth,lambd) in enumerate(params):
    
        temp_predictions = np.zeros((mX,mY))
        temp_scores = np.zeros((repeats+2,2))
    
        param = {'eta':eta,'gamma':gamma,'max_depth':depth,'lambda':lambd, \
                  'objective':objective}
        param['eval_metric'] = 'auc'
        for repeat in range(repeats):
            train_ixs, test_ixs, train_lbls, test_lbls = train_test_split(ixs,lbls,test_size=0.1,stratify=lbls)
            print(train_lbls.shape,train_ixs.shape)
            train_ixs, val_ixs, train_lbls, val_lbls = train_test_split(train_ixs,train_lbls,test_size=0.1,stratify=train_lbls)
           
            dtrain = np.hstack((fX[train_ixs[:,0]],fY[train_ixs[:,1]]))
            dval = np.hstack((fX[val_ixs[:,0]],fY[val_ixs[:,1]]))
            dtest = np.hstack((fX[test_ixs[:,0]],fY[test_ixs[:,1]]))
            
            dtrain = xgb.DMatrix(dtrain,label=train_lbls)
            dval = xgb.DMatrix(dval,label=val_lbls)
            dtest = xgb.DMatrix(dtest,label=test_lbls)
            evallist = [(dtrain, 'train'),(dval, 'eval')]
            bst = xgb.train(param,dtrain,num_rounds,evallist,early_stopping_rounds=50)
            pickle.dump(bst,open('{}/{}_{}.xgb'.format(outpath,name,repeat),'wb'))
            
            out = bst.predict(dtest, iteration_range=(0, bst.best_iteration))


        
            fpr,tpr,th = roc_curve(test_lbls,out)
            temp_scores[repeat+2,0] = auc(fpr,tpr)
            pr, rec, th = precision_recall_curve(test_lbls,out)
            temp_scores[repeat+2,1] = auc(rec,pr)
            print(temp_scores[repeat+2])
        
            X, Y = fX[ixs[:,0]], fY[ixs[:,1]]
            dtest = xgb.DMatrix(np.hstack((X,Y)),label=lbls)
            out = bst.predict(dtest, iteration_range=(0, bst.best_iteration))


            out = standardize(out)
            temp_predictions += out.reshape(mX,mY)
        
        avg_auc = np.average(temp_scores[2:,0])
        temp_scores[0,0], temp_scores[1,0] = avg_auc,np.std(temp_scores[2:,0])
        temp_scores[0,1], temp_scores[1,1] = np.average(temp_scores[2:,1]),np.std(temp_scores[2:,1])
        xvals[i] = avg_auc
        if avg_auc > best:
            predictions = temp_predictions
            best = avg_auc
            scores = temp_scores
            for repeat in range(repeats):
                subprocess.call(['mv','{}/{}_{}.xgb'.format(outpath,name,repeat),'{}/{}_{}_best.xgb'.format(outpath,name,repeat)])
        else:
            for repeat in range(repeats):
                subprocess.call(['rm','{}/{}_{}.xgb'.format(outpath,name,repeat)])
            
    df = pd.DataFrame(data=xvals,index=params)
    df.to_csv('{}/xvalxgb_{}.tsv'.format(outpath,name),sep='\t')
    df = pd.DataFrame(data=scores,columns=columns, index=rows)
    df.to_csv('{}/xgb_{}_scores.tsv'.format(outpath,name),sep='\t')
    np.save('{}/xgb_{}_predictions'.format(outpath,name),predictions)


def train_(fX, fY, ixs,lbls, name, outpath, objective='binary:logitraw'):
    mX, mY = fX.shape[0], fY.shape[0]
    repeats = 10
    rows = ['avg','std']+[i for i in range(repeats)] 
    columns = ['AUROC','AUPRC']
    
    etas = [0.25,0.5,0.75]#,0.1,0.25]
    gammas = [0,10]#[0,1e-3,1e-2,1e-1]
    depths = [6,12]#[1e-3,1e-2]
    lambdas = [1,10,100]#,128,256]
    
    best = 0
    params = list(it.product(etas,gammas,depths,lambdas))
    xvals = np.zeros((len(params),1))
    num_rounds = 1000
    for i,(eta,gamma,depth,lambd) in enumerate(params):
    
        temp_predictions = 0
        temp_scores = np.zeros((repeats+2,2))
    
        param = {'eta':eta,'gamma':gamma,'max_depth':depth,'lambda':lambd, \
                  'objective':objective}
        param['eval_metric'] = 'auc'
        for repeat in range(repeats):
            train_ixs, test_ixs, train_lbls, test_lbls = train_test_split(ixs,lbls,test_size=0.2,stratify=lbls)
            print(train_lbls.shape,train_ixs.shape)
            train_ixs, val_ixs, train_lbls, val_lbls = train_test_split(train_ixs,train_lbls,test_size=0.11,stratify=train_lbls)
           
            dtrain = np.hstack((fX[train_ixs[:,0]],fY[train_ixs[:,1]]))
            dval = np.hstack((fX[val_ixs[:,0]],fY[val_ixs[:,1]]))
            dtest = np.hstack((fX[test_ixs[:,0]],fY[test_ixs[:,1]]))
            
            dtrain = xgb.DMatrix(dtrain,label=train_lbls)
            dval = xgb.DMatrix(dval,label=val_lbls)
            dtest = xgb.DMatrix(dtest,label=test_lbls)
            evallist = [(dtrain, 'train'),(dval, 'eval')]
            bst = xgb.train(param,dtrain,num_rounds,evallist,early_stopping_rounds=50)
            pickle.dump(bst,open('{}/{}_{}.xgb'.format(outpath,name,repeat),'wb'))
            
            out = bst.predict(dtest, iteration_range=(0, bst.best_iteration))


        
            fpr,tpr,th = roc_curve(test_lbls,out)
            temp_scores[repeat+2,0] = auc(fpr,tpr)
            pr, rec, th = precision_recall_curve(test_lbls,out)
            temp_scores[repeat+2,1] = auc(rec,pr)
            print(temp_scores[repeat+2])
        
            X, Y = fX[ixs[:,0]], fY[ixs[:,1]]
            dtest = xgb.DMatrix(np.hstack((X,Y)),label=lbls)
            out = bst.predict(dtest, iteration_range=(0, bst.best_iteration))


            out = standardize(out)
            temp_predictions += out
        
        avg_auc = np.average(temp_scores[2:,0])
        temp_scores[0,0], temp_scores[1,0] = avg_auc,np.std(temp_scores[2:,0])
        temp_scores[0,1], temp_scores[1,1] = np.average(temp_scores[2:,1]),np.std(temp_scores[2:,1])
        xvals[i] = avg_auc
        if avg_auc > best:
            predictions = temp_predictions
            best = avg_auc
            scores = temp_scores
            for repeat in range(repeats):
                subprocess.call(['mv','{}/{}_{}.xgb'.format(outpath,name,repeat),'{}/{}_{}_best.xgb'.format(outpath,name,repeat)])
        else:
            for repeat in range(repeats):
                subprocess.call(['rm','{}/{}_{}.xgb'.format(outpath,name,repeat)])
            
    df = pd.DataFrame(data=xvals,index=params)
    df.to_csv('{}/xvalxgb_{}.tsv'.format(outpath,name),sep='\t')
    df = pd.DataFrame(data=scores,columns=columns, index=rows)
    df.to_csv('{}/xgb_{}_scores.tsv'.format(outpath,name),sep='\t')
    np.save('{}/xgb_{}_predictions'.format(outpath,name),predictions)



