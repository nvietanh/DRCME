
# coding: utf-8

# # Load MNIST data

# In[1]:

import tensorflow as tf
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import csv
from scipy import sparse
import time
from matplotlib import pyplot as plt
import sys
#import mosek
#from mosek.fusion import *
#import cvxpy as cvx
from matplotlib import pyplot as plt
import ot
import pickle
from joblib import Parallel, delayed
import multiprocessing

from conditionalEstimates import *


# In[2]:

isDebugMode = False
runSequential = isDebugMode  


if not isDebugMode:
    expN= 100
    TrainSize_list = [20, 50, 100, 500]
    TestSize = 100
    Noise_list = [0]
    knns_list = np.arange(1,11)
    nada_N = 41
    mewn_alphas_list = [0.0001,0.001,0.01,0.1,1]
    mewn_gammak_list = [1,2,3,4,5,6]#[1, 3, 5]
    mewn_rhok_list = [0,1,2,3,4,5,6]
    mewn_N = 10
    mewn_paramsType = "k_and_ratio"
    bert_N = 40
    distType = "Euclidean"
#    distType = "Wasserstein"
    doClassify = False
    if not doClassify:
        result_filename = "LargeTest_results_"+distType+"7.dat"
    else:
        result_filename = "LargeTest_results_Classify_"+distType+"7.dat"
else:
    expN= 4
    TrainSize_list = [10,20]
    TestSize = 10
    Noise_list = [0]
    knns_list = [3]
    nada_N = 11
    mewn_alphas_list = [0.1]
    mewn_gammak_list = [3]
    mewn_rhok_list = [1]
    mewn_N = 2
    mewn_paramsType = "k_and_ratio"
    bert_N = 10
    distType = "Euclidean"
    doClassify = False    
    result_filename = "LargeTest_results_debug.dat"
    

    
print(result_filename)


# In[3]:

def saveData():
        file = open(result_filename,"wb")
        pickle.dump(fvals_test,file)
        pickle.dump(class_fvals_test,file)
        pickle.dump(knn_best_params,file)
        pickle.dump(mewn_best_params,file)
        pickle.dump(nada_best_params,file)        
        pickle.dump(nada_best_params2,file)        
        pickle.dump(bert_best_params,file)        
        pickle.dump(expN,file)
        pickle.dump(TrainSize_list,file)
        pickle.dump(TestSize,file)
        pickle.dump(Noise_list,file)    
        pickle.dump(knns_list,file)
        pickle.dump(mewn_alphas_list,file)
        pickle.dump(mewn_gammak_list,file)
        pickle.dump(mewn_rhok_list,file)
        pickle.dump(distType,file)
        pickle.dump(doClassify,file)
        pickle.dump(nada_N,file)
        pickle.dump(mewn_paramsType,file)
        pickle.dump(mewn_N,file)
        pickle.dump(bert_N,file)
        pickle.dump(all_results,file)        
        file.close()

def loadData():
        file = open(result_filename,"rb")
        fvals_test=pickle.load(file)
        class_fvals_test=pickle.load(file)
        knn_best_params=pickle.load(file)
        mewn_best_params=pickle.load(file)
        nada_best_params=pickle.load(file)        
        nada_best_params2=pickle.load(file)        
        bert_best_params=pickle.load(file)        
        expN=pickle.load(file)
        TrainSize_list=pickle.load(file)
        TestSize=pickle.load(file)
        Noise_list=pickle.load(file)           
        knns_list=pickle.load(file)
        mewn_alphas_list=pickle.load(file)
        mewn_gammak_list=pickle.load(file)
        mewn_rhok_list=pickle.load(file)
        distType=pickle.load(file)
        doClassify=pickle.load(file)       
        nada_N=pickle.load(file)
        mewn_paramsType=pickle.load(file)
        mewn_N=pickle.load(file)
        bert_N=pickle.load(file)
        all_results=pickle.load(file)        
        file.close()  
        return (fvals_test,class_fvals_test,knn_best_params,mewn_best_params,nada_best_params,nada_best_params2,bert_best_params,expN,TrainSize_list,TestSize,Noise_list,knns_list,mewn_alphas_list,mewn_gammak_list,mewn_rhok_list,distType,doClassify,nada_N,bert_N,mewn_paramsType,mewn_N,all_results)
        
def runTest(noiseId,TrainSizeId):
            #data
            CEprob = MNIST_ConditionalEstimateProblem(doClassify = doClassify)
            CEprob.set_noise(noiseType='Gaussian',noise_sigma=Noise_list[noiseId])
            (x_train2,y_train2,x_test2,y_test2) = CEprob.getDataSet(TrainSize_list[TrainSizeId],TestSize)
            
            #dist
            ImgDist = imageDistance(CEprob.ImgSize,distType = distType)
            (x_train2b,x_test2b) = ImgDist.load_hashTableFromData(x_train2,x_test2)
    
            #kNN
            knn = kNN_model(ImgDist.getDist,ks_list = knns_list)
            (NNk_best, NNk_best_fval, NNk_cv_fvals) = crossValidation(knn.getParamsList(),knn.predictor, x_train2b,y_train2,best_type='mean')
            knn.setParams(NNk_best); knn.setObs(x_train2b,y_train2)
            (kNN_perf,kNN_perf_CI,kNN_classPerf,kNN_fvals) = testPredictor(knn.predictor,x_test2b,y_test2,reportAll=True)
            
            #Wasserstein
            if mewn_paramsType == "k_params":
                mewn = wassersteinRobust_model(ImgDist.getDist,alphas_list = mewn_alphas_list,gammak_list = mewn_gammak_list,rhok_list = mewn_rhok_list)
            else:
                mewn = wassersteinRobust_model(ImgDist.getDist,paramsType=mewn_paramsType)
                mewn.setParamsListDD2(x_train2b,y_train2,mewn_N)
            (mewn_best, mewn_best_fval, mewn_cv_fvals) = crossValidation(mewn.getParamsList(),mewn.predictor, x_train2b,y_train2,best_type='mean')
            mewn.setParams(mewn_best); mewn.setObs(x_train2b,y_train2)
            (mewn_perf,mewn_perf_CI,mewn_classPerf,mewn_fvals) = testPredictor(mewn.predictor,x_test2b,y_test2,reportAll=True)

            #Nadaraja-Watson
            nada = nadarajaWatson_model(ImgDist.getDist)
            nada.setBandwidthList(x_train2b,nada_N)
            (nada_best, nada_best_fval, nada_cv_fvals) = crossValidation(nada.getParamsList(),nada.predictor, x_train2b,y_train2,best_type='mean')
            nada.setParams(nada_best); nada.setObs(x_train2b,y_train2)
            (nada_perf,nada_perf_CI,nada_classPerf,nada_fvals) = testPredictor(nada.predictor,x_test2b,y_test2,reportAll=True)

            #Nadaraja-Epanechnikov
            nada2 = nadarajaWatson_model(ImgDist.getDist,kernel="Epanechnikov")
            nada2.setBandwidthList(x_train2b,nada_N)
            (nada_best2, nada_best_fval2, nada_cv_fvals2) = crossValidation(nada2.getParamsList(),nada2.predictor, x_train2b,y_train2,best_type='mean')
            nada2.setParams(nada_best2); nada2.setObs(x_train2b,y_train2)
            (nada_perf2,nada_perf_CI2,nada_classPerf2,nada_fvals2) = testPredictor(nada2.predictor,x_test2b,y_test2,reportAll=True)
            
            #Bertsimas
            bert = wassersteinRobust_model(ImgDist.getDist)
            bert.setParamsList_Bertsimas(x_train2b,y_train2,bert_N)
            (bert_best, bert_best_fval, bert_cv_fvals) = crossValidation(bert.getParamsList(),bert.predictor, x_train2b,y_train2,best_type='mean')
            bert.setParams(bert_best); bert.setObs(x_train2b,y_train2)
            (bert_perf,bert_perf_CI,bert_classPerf,bert_fvals) = testPredictor(bert.predictor,x_test2b,y_test2,reportAll=True)            
            
            all_results = np.matrix((np.array(y_test2[0,:]).flatten(),kNN_fvals,mewn_fvals,nada_fvals,nada_fvals2,bert_fvals))
            return (kNN_perf,mewn_perf,nada_perf,nada_perf2,bert_perf,NNk_best,mewn_best,nada_best,nada_best2,bert_best,kNN_classPerf,mewn_classPerf,nada_classPerf,nada_classPerf2,bert_classPerf,all_results)


fvals_test = np.zeros((5,len(Noise_list),len(TrainSize_list),expN))
class_fvals_test = np.zeros((5,len(Noise_list),len(TrainSize_list),expN))
knn_best_params = np.zeros((1,len(Noise_list),len(TrainSize_list),expN))
mewn_best_params = np.zeros((3,len(Noise_list),len(TrainSize_list),expN))
nada_best_params = np.zeros((1,len(Noise_list),len(TrainSize_list),expN))
nada_best_params2 = np.zeros((1,len(Noise_list),len(TrainSize_list),expN))
bert_best_params = np.zeros((3,len(Noise_list),len(TrainSize_list),expN))

all_results = np.zeros((6,len(Noise_list),len(TrainSize_list),expN,TestSize))
for TrainSizeId in range(len(TrainSize_list)):
    for noiseId in range(len(Noise_list)):
        if runSequential:
            tmp_results = []
            for testid in range(expN):
                tmp_results.append(runTest(noiseId,TrainSizeId))
        else:
            def processInput(testid):
                return runTest(noiseId,TrainSizeId)
            tmp_results = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(processInput)(i) for i in range(expN))
        for testid in range(expN):
            fvals_test[:,noiseId,TrainSizeId,testid]=np.array(tmp_results[testid][0:5])
            knn_best_params[:,noiseId,TrainSizeId,testid]=tmp_results[testid][5]
            mewn_best_params[:,noiseId,TrainSizeId,testid]=tmp_results[testid][6]
            nada_best_params[:,noiseId,TrainSizeId,testid]=tmp_results[testid][7]
            nada_best_params2[:,noiseId,TrainSizeId,testid]=tmp_results[testid][8]
            bert_best_params[:,noiseId,TrainSizeId,testid]=tmp_results[testid][9]
            class_fvals_test[:,noiseId,TrainSizeId,testid] = np.array(tmp_results[testid][10:15])
            all_results[:,noiseId,TrainSizeId,testid,:] = tmp_results[testid][15]
        saveData()
        print("Done with Noise:"+str(Noise_list[noiseId])+", TrainSize:"+str(TrainSize_list[TrainSizeId]))

    


# In[ ]:



