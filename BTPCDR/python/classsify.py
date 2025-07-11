import argparse
import random,os,sys
import numpy as np
import csv
from scipy import stats
import time
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
import pandas as pd
from keras.layers import Lambda
from keras import optimizers,utils
from keras.constraints import max_norm
from keras import regularizers
from keras.callbacks import ModelCheckpoint,Callback,EarlyStopping,History
from keras.utils import plot_model
import keras.backend as K
from keras.models import Model, Sequential
from keras.models import load_model
from keras.layers import Input,InputLayer,Multiply,ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense,Activation,Dropout,Flatten,Concatenate
from keras.layers import BatchNormalization


from tensorflow.keras.optimizers.legacy import Adam, SGD
from keras.models import model_from_json
import tensorflow as tf
from sklearn.metrics import average_precision_score
from scipy.stats import pearsonr
from model import KerasMultiSourceGCNModel
import hickle as hkl
import scipy.sparse as sp
import argparse

parser = argparse.ArgumentParser(description='Drug_response_pre')
parser.add_argument('-gpu_id', dest='gpu_id', type=str, default='0', help='GPU devices')
parser.add_argument('-use_mut', dest='use_mut', type=bool, default=True, help='use gene mutation or not')
parser.add_argument('-use_gexp', dest='use_gexp', type=bool, default=True, help='use gene expression or not')
parser.add_argument('-use_methy', dest='use_methy', type=bool, default=True, help='use methylation or not')

parser.add_argument('-israndom', dest='israndom', type=bool, default=False, help='randomlize X and A')
#hyparameters for GCN
parser.add_argument('-unit_list', dest='unit_list', nargs='+', type=int, default=[256,256,256],help='unit list for GCN')
parser.add_argument('-use_bn', dest='use_bn', type=bool, default=True, help='use batchnormalization for GCN')
parser.add_argument('-use_relu', dest='use_relu', type=bool, default=True, help='use relu for GCN')
parser.add_argument('-use_GMP', dest='use_GMP', type=bool, help='use GlobalMaxPooling for GCN')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
use_mut,use_gexp,use_methy = args.use_mut,args.use_gexp, args.use_methy
israndom=args.israndom
model_suffix = ('with_mut' if use_mut else 'without_mut')+'_'+('with_gexp' if use_gexp else 'without_gexp')+'_'+('with_methy' if use_methy else 'without_methy')

GCN_deploy = '_'.join(map(str,args.unit_list)) + '_'+('bn' if args.use_bn else 'no_bn')+'_'+('relu' if args.use_relu else 'tanh')+'_'+('GMP' if args.use_GMP else 'GAP')
model_suffix = model_suffix + '_' +GCN_deploy

TCGA_label_set = ["ALL","BLCA","BRCA","CESC","DLBC","LIHC","LUAD",
                  "ESCA","GBM","HNSC","KIRC","LAML","LCML","LGG",
                  "LUSC","MESO","MM","NB","OV","PAAD","SCLC","SKCM",
                  "STAD","THCA",'COAD/READ']
DPATH = '../data'
Drug_info_file = '%s/GDSC/1.Drug_listMon Jun 24 09_00_55 2019.csv'%DPATH
Cell_line_info_file = '%s/CCLE/Cell_lines_annotations_20181226.txt'%DPATH
Drug_feature_file = '%s/GDSC/drug_graph_feat'%DPATH
Genomic_mutation_file = '%s/CCLE/genomic_mutation_34673_demap_features.csv'%DPATH
Cancer_response_exp_file = '%s/CCLE/GDSC_IC50.csv'%DPATH
Gene_expression_file = '%s/CCLE/genomic_expression_561celllines_697genes_demap_features.csv'%DPATH
Methylation_file = '%s/CCLE/genomic_methylation_561celllines_808genes_demap_features.csv'%DPATH
IC50_thred_file = '%s/CCLE/IC50_thred.txt'%DPATH
Max_atoms = 100

def FeatureExtract(data_idx,drug_feature,mutation_feature,gexpr_feature,methylation_feature):
    cancer_type_list = []
    nb_instance = len(data_idx)
    nb_mutation_feature = mutation_feature.shape[1]
    nb_gexpr_features = gexpr_feature.shape[1]
    nb_methylation_features = methylation_feature.shape[1]
    drug_data = [[] for item in range(nb_instance)]
    mutation_data = np.zeros((nb_instance,1, nb_mutation_feature,1),dtype='float32')
    gexpr_data = np.zeros((nb_instance,nb_gexpr_features),dtype='float32') 
    methylation_data = np.zeros((nb_instance, nb_methylation_features),dtype='float32') 
    target = np.zeros(nb_instance,dtype='int16')
    for idx in range(nb_instance):
        cell_line_id,pubchem_id,binary_IC50,cancer_type = data_idx[idx]
        #modify
        feat_mat,adj_list,_ = drug_feature[str(pubchem_id)]
        #fill drug data,padding to the same size with zeros
        drug_data[idx] = CalculateGraphFeat(feat_mat,adj_list)
        #randomlize X A
        mutation_data[idx,0,:,0] = mutation_feature.loc[cell_line_id].values
        gexpr_data[idx,:] = gexpr_feature.loc[cell_line_id].values
        methylation_data[idx,:] = methylation_feature.loc[cell_line_id].values
        target[idx] = binary_IC50
        cancer_type_list.append([cancer_type,cell_line_id,pubchem_id])
    return drug_data,mutation_data,gexpr_data,methylation_data,target,cancer_type_list

#split into training and test set
def DataSplit(data_idx,ratio = 0.95):
    data_train_idx,data_test_idx = [], []
    for each_type in TCGA_label_set:
        data_subtype_idx = [item for item in data_idx if item[-1]==each_type]
        train_list = random.sample(data_subtype_idx,int(ratio*len(data_subtype_idx)))
        test_list = [item for item in data_subtype_idx if item not in train_list]
        data_train_idx += train_list
        data_test_idx += test_list
    return data_train_idx,data_test_idx

def MetadataGenerate(Drug_info_file,Cell_line_info_file,Genomic_mutation_file,Drug_feature_file,Gene_expression_file,Methylation_file,filtered,use_thred=True):
    #drug_id --> pubchem_id
    reader = csv.reader(open(Drug_info_file,'r'))
    rows = [item for item in reader]
    drugid2pubchemid = {item[0]:item[5] for item in rows if item[5].isdigit()}
    name2pubchemid = {item[1]:item[5] for item in rows if item[5].isdigit()}
    drug_names = [item.strip() for item in open(IC50_thred_file).readlines()[0].split('\t')]
    IC50_threds = [float(item.strip()) for item in open(IC50_thred_file).readlines()[1].split('\t')]
    drug2thred = {name2pubchemid[a]:b for a,b in zip(drug_names,IC50_threds) if a in name2pubchemid.keys()}

    #map cellline --> cancer type
    cellline2cancertype ={}
    for line in open(Cell_line_info_file).readlines()[1:]:
        cellline_id = line.split('\t')[1]
        TCGA_label = line.strip().split('\t')[-1]
        #if TCGA_label in TCGA_label_set:
        cellline2cancertype[cellline_id] = TCGA_label

    #load demap cell lines genomic mutation features
    mutation_feature = pd.read_csv(Genomic_mutation_file,sep=',',header=0,index_col=[0])
    cell_line_id_set = list(mutation_feature.index)

    # load drug features
    drug_pubchem_id_set = []
    drug_feature = {}
    for each in os.listdir(Drug_feature_file):
        drug_pubchem_id_set.append(each.split('.')[0])
        feat_mat,adj_list,degree_list = hkl.load('%s/%s'%(Drug_feature_file,each))
        drug_feature[each.split('.')[0]] = [feat_mat,adj_list,degree_list]
    assert len(drug_pubchem_id_set)==len(drug_feature.values())
    
    #load gene expression faetures
    gexpr_feature = pd.read_csv(Gene_expression_file,sep=',',header=0,index_col=[0])
    
    #only keep overlapped cell lines
    mutation_feature = mutation_feature.loc[list(gexpr_feature.index)]
    
    #load methylation 
    methylation_feature = pd.read_csv(Methylation_file,sep=',',header=0,index_col=[0])
    assert methylation_feature.shape[0]==gexpr_feature.shape[0]==mutation_feature.shape[0]        
    experiment_data = pd.read_csv(Cancer_response_exp_file,sep=',',header=0,index_col=[0])
    
    #filter experiment data
    drug_match_list=[item for item in experiment_data.index if item.split(':')[1] in drugid2pubchemid.keys()]
    experiment_data_filtered = experiment_data.loc[drug_match_list]
    
    data_idx = []
    for each_drug in experiment_data_filtered.index:
        for each_cellline in experiment_data_filtered.columns:
            pubchem_id = drugid2pubchemid[each_drug.split(':')[-1]]
            if str(pubchem_id) in drug_pubchem_id_set and each_cellline in mutation_feature.index:
                if not np.isnan(experiment_data_filtered.loc[each_drug,each_cellline]) and each_cellline in cellline2cancertype.keys():
                    ln_IC50 = float(experiment_data_filtered.loc[each_drug,each_cellline])
                    if use_thred:
                        if pubchem_id in drug2thred.keys():
                            binary_IC50 = 1 if ln_IC50 < drug2thred[pubchem_id] else 0
                            data_idx.append((each_cellline,pubchem_id,binary_IC50,cellline2cancertype[each_cellline])) 
                    else:
                        binary_IC50 = 1 if ln_IC50 > -2 else 0
                        data_idx.append((each_cellline,pubchem_id,binary_IC50,cellline2cancertype[each_cellline])) 

    nb_celllines = len(set([item[0] for item in data_idx]))
    nb_drugs = len(set([item[1] for item in data_idx]))
    print('%d instances across %d cell lines and %d drugs were generated.'%(len(data_idx),nb_celllines,nb_drugs))
    return mutation_feature, drug_feature,gexpr_feature,methylation_feature, data_idx


def NormalizeAdj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0).toarray()
    a_norm = adj.dot(d).transpose().dot(d)
    return a_norm
def random_adjacency_matrix(n):   
    matrix = [[random.randint(0, 1) for i in range(n)] for j in range(n)]
    # No vertex connects to itself
    for i in range(n):
        matrix[i][i] = 0
    # If i is connected to j, j is connected to i
    for i in range(n):
        for j in range(n):
            matrix[j][i] = matrix[i][j]
    return matrix
def CalculateGraphFeat(feat_mat,adj_list):
    assert feat_mat.shape[0] == len(adj_list)
    feat = np.zeros((Max_atoms,feat_mat.shape[-1]),dtype='float32')
    adj_mat = np.zeros((Max_atoms,Max_atoms),dtype='float32')
    if israndom:
        feat = np.random.rand(Max_atoms,feat_mat.shape[-1])
        adj_mat[feat_mat.shape[0]:,feat_mat.shape[0]:] = random_adjacency_matrix(Max_atoms-feat_mat.shape[0])        
    feat[:feat_mat.shape[0],:] = feat_mat
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i,int(each)] = 1
    assert np.allclose(adj_mat,adj_mat.T)
    adj_ = adj_mat[:len(adj_list),:len(adj_list)]
    adj_2 = adj_mat[len(adj_list):,len(adj_list):]
    norm_adj_ = NormalizeAdj(adj_)
    norm_adj_2 = NormalizeAdj(adj_2)
    adj_mat[:len(adj_list),:len(adj_list)] = norm_adj_
    adj_mat[len(adj_list):,len(adj_list):] = norm_adj_2    
    return [feat,adj_mat]  
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    recal = recall(y_true, y_pred)
    return 2.0*prec*recal/(prec+recal+K.epsilon())
def average_precision(y_true, y_pred):
    return tf.numpy_function(average_precision_score, (y_true, y_pred), tf.double) 


class MyCallback(Callback):
    def __init__(self,validation_data,patience):
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.best_weight = None
        self.patience = patience
    def on_train_begin(self,logs={}):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf
        return
    def on_train_end(self, logs={}):
        self.model.set_weights(self.best_weight)
        self.model.save('../checkpoint/MyBestDeepCDR_classify_%s.h5'%model_suffix)
        if self.stopped_epoch > 0 :
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        precision,recall,_, = metrics.precision_recall_curve(self.y_val,y_pred_val)
        pr_val = -np.trapz(precision,recall)
        print ('roc-val: %.4f, pr-val:%.4f' % (roc_val,pr_val))
        if roc_val > self.best:
            self.best = roc_val
            self.wait = 0
            self.best_weight = self.model.get_weights()
        else:
            self.wait+=1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
        return
def ModelTraining(model,X_drug_data_train,X_mutation_data_train,X_gexpr_data_train,X_methylation_data_train,Y_train,validation_data,nb_epoch=10):
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer = optimizer,loss='binary_crossentropy',metrics=['accuracy',precision,recall,f1_score,average_precision])
    #EarlyStopping(monitor='val_loss',patience=5)
    callbacks = [ModelCheckpoint('../checkpoint/best_DeepCDR_classify_%s.h5'%model_suffix,monitor='val_average_precision',save_best_only=True, save_weights_only=False),
                MyCallback(validation_data=validation_data,patience=10)]
    X_drug_feat_data_train = [item[0] for item in X_drug_data_train]
    X_drug_adj_data_train = [item[1] for item in X_drug_data_train]
    X_drug_feat_data_train = np.array(X_drug_feat_data_train)#nb_instance * Max_stom * feat_dim
    X_drug_adj_data_train = np.array(X_drug_adj_data_train)#nb_instance * Max_stom * Max_stom
    #validation data
    model.fit(x=[X_drug_feat_data_train,X_drug_adj_data_train,X_mutation_data_train,X_gexpr_data_train,X_methylation_data_train],y=Y_train,batch_size=16,epochs=nb_epoch,validation_split=0,callbacks=callbacks)
    return model


def ModelEvaluate(model,X_drug_data_test,X_mutation_data_test,X_gexpr_data_test,X_methylation_data_test,Y_test,cancer_type_test_list,file_path):
    X_drug_feat_data_test = [item[0] for item in X_drug_data_test]
    X_drug_adj_data_test = [item[1] for item in X_drug_data_test]
    X_drug_feat_data_test = np.array(X_drug_feat_data_test)#nb_instance * Max_stom * feat_dim
    X_drug_adj_data_test = np.array(X_drug_adj_data_test)#nb_instance * Max_stom * Max_stom    
    Y_pred = model.predict([X_drug_feat_data_test,X_drug_adj_data_test,X_mutation_data_test,X_gexpr_data_test,X_methylation_data_test])
    auROC_all = metrics.roc_auc_score(Y_test, Y_pred)
    fpr,tpr,_,= metrics.roc_curve(Y_test,Y_pred)
    precision,recall,_, = metrics.precision_recall_curve(Y_test,Y_pred)
    auPR_all = -np.trapz(precision,recall)
    print("The overall AUC and auPR is %.4f and %.4f."%(auROC_all,auPR_all))

    

def main():
    random.seed(0)
    mutation_feature, drug_feature,gexpr_feature,methylation_feature, data_idx = MetadataGenerate(Drug_info_file,Cell_line_info_file,Genomic_mutation_file,Drug_feature_file,Gene_expression_file,Methylation_file,False)
    data_train_idx,data_test_idx = DataSplit(data_idx)
    #Extract features for training and test 
    X_drug_data_train,X_mutation_data_train,X_gexpr_data_train,X_methylation_data_train,Y_train,cancer_type_train_list = FeatureExtract(data_train_idx,drug_feature,mutation_feature,gexpr_feature,methylation_feature)
    X_drug_data_test,X_mutation_data_test,X_gexpr_data_test,X_methylation_data_test,Y_test,cancer_type_test_list = FeatureExtract(data_test_idx,drug_feature,mutation_feature,gexpr_feature,methylation_feature)

    X_drug_feat_data_test = [item[0] for item in X_drug_data_test]
    X_drug_adj_data_test = [item[1] for item in X_drug_data_test]
    X_drug_feat_data_test = np.array(X_drug_feat_data_test)#nb_instance * Max_stom * feat_dim
    X_drug_adj_data_test = np.array(X_drug_adj_data_test)#nb_instance * Max_stom * Max_stom  
    
    validation_data = [[X_drug_feat_data_test,X_drug_adj_data_test,X_mutation_data_test,X_gexpr_data_test,X_methylation_data_test],Y_test]
    model = KerasMultiSourceGCNModel(use_mut,use_gexp,use_methy,regr=False).createMaster(X_drug_data_train[0][0].shape[-1],X_mutation_data_train.shape[-2],X_gexpr_data_train.shape[-1],X_methylation_data_train.shape[-1],args.unit_list,args.use_relu,args.use_bn,args.use_GMP)
    print('Begin training...')
    model = ModelTraining(model,X_drug_data_train,X_mutation_data_train,X_gexpr_data_train,X_methylation_data_train,Y_train,validation_data,nb_epoch=10)
    ModelEvaluate(model,X_drug_data_test,X_mutation_data_test,X_gexpr_data_test,X_methylation_data_test,Y_test,cancer_type_test_list,'%s/DeepCDR_%s.log'%(DPATH,model_suffix))

if __name__=='__main__':
    main()
    
