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

class MNIST_ConditionalEstimateProblem:
    ImgSize = 0
    doClassify = False
    noiseType='None'
    noise_sigma=0.1
    protectTest = True
    doClassify = False
    classifyPair = []
    x_train_MNIST = []
    y_train_MNIST = []
    x_test_MNIST = []
    y_test_MNIST = []
    
    def __init__(self,noiseType='None',noise_sigma=0.1,protectTest = True, doClassify = False, classifyPair = []):    
        self.set_noise(noiseType,noise_sigma)
        self.noise_sigma = noise_sigma
        self.protectTest = protectTest
        (x_train_MNIST, y_train_MNIST), (x_test_MNIST, y_test_MNIST) = tf.keras.datasets.mnist.load_data()
        self.x_train_MNIST = x_train_MNIST; self.y_train_MNIST = y_train_MNIST; self.x_test_MNIST = x_test_MNIST; self.y_test_MNIST = y_test_MNIST; 
        self.ImgSize = [x_train_MNIST.shape[1],x_train_MNIST.shape[2]]
        self.doClassify = doClassify            
        if len(classifyPair)==2:
            self.classifyPair = classifyPair
        else:
            self.classifyPair = np.random.permutation(10)[1:3]
        
      
    def set_noise(self,noiseType='None',noise_sigma=0.1):
        self.noiseType = noiseType
        self.noise_sigma = noise_sigma    
        
    def noise_generator(self,ntrain, noise_type):
        if noise_type == "Bernoulli":
            return (2*np.random.binomial(1, 0.5, ntrain)-1)
        if noise_type == "Asym_Bernoulli":
            return (2*np.random.binomial(1, 0.8, ntrain)-1.6)/0.8
        if noise_type == "Gaussian":
            return np.random.randn(ntrain)
        if noise_type == "Uniform":
            return np.sqrt(3)*(2*rand.rand(ntrain) - 1)
        if noise_type == "Asym_Uniform":
            indicators = np.random.binomial(1,2/3,ntrain)
            return (indicators*rand.rand(ntrain)+(indicators-1)*2*rand.rand(ntrain))/np.sqrt(2/3)
        if noise_type == "ChiSquare":
            return ((2*np.random.binomial(1, 0.5, ntrain)-1) * rand.chisquare(1,ntrain)) / np.sqrt(3)
        if noise_type == "TrancateGaussian1":
            noise = np.max([-1*np.ones(ntrain), np.random.randn(ntrain)], axis = 0)
            noise = np.min([1*np.ones(ntrain), noise], axis = 0)
            return noise
        if noise_type == "TrancateGaussian2":
            noise = np.max([-2*np.ones(ntrain), np.random.randn(ntrain)], axis = 0)
            noise = np.min([2*np.ones(ntrain), noise], axis = 0)
            return noise
    
    def plotImage(self,imgVector):
        plt.imshow(imgVector.reshape(self.ImgSize)*255,cmap='gray', vmin=0, vmax=255)
        plt.show()

    def getDataSet(self,TrainSize,TestSize,doRandomize=True):
        x_train_MNIST = self.x_train_MNIST
        y_train_MNIST = self.y_train_MNIST
        x_test_MNIST = self.x_test_MNIST
        y_test_MNIST = self.y_test_MNIST
        
        if self.doClassify:
            indexes = np.array((y_train_MNIST==self.classifyPair[0]) | (y_train_MNIST==self.classifyPair[1])).flatten()
            x_train_MNIST = x_train_MNIST[indexes,:,:]; y_train_MNIST = 1*(y_train_MNIST[indexes]==self.classifyPair[1])
            indexes = np.array((y_test_MNIST==self.classifyPair[0]) | (y_test_MNIST==self.classifyPair[1])).flatten()
            x_test_MNIST = x_test_MNIST[indexes,:,:]; y_test_MNIST = 1*(y_test_MNIST[indexes]==self.classifyPair[1])

        if not self.protectTest:
            TrainSize = min(TrainSize,x_train_MNIST.shape[0])
            TestSize = min(TestSize,x_test_MNIST.shape[0])
            if doRandomize:
                indexes = np.random.permutation(x_train_MNIST.shape[0])
            else:
                indexes = np.arange(x_train_MNIST.shape[0])
            x_train1 = x_train_MNIST[indexes[:TrainSize],:,:]
            y_train1 = y_train_MNIST[indexes[:TrainSize]]
            if doRandomize:
                indexes = np.random.permutation(x_test_MNIST.shape[0])
            else:
                indexes = np.arange(x_test_MNIST.shape[0])
            x_test1 = x_test_MNIST[indexes[:TestSize],:,:]
            y_test1 = y_test_MNIST[indexes[:TestSize]]
        else:
            TestSize = int(min(TestSize,np.round(x_train_MNIST.shape[0]/2)))
            TrainSize = int(min(TrainSize,x_train_MNIST.shape[0]-TestSize))
            if doRandomize:
                indexes = np.random.permutation(x_train_MNIST.shape[0])
            else:
                indexes = np.arange(x_train_MNIST.shape[0])
            x_test1 = x_train_MNIST[indexes[TrainSize:TrainSize+TestSize],:,:]
            y_test1 = y_train_MNIST[indexes[TrainSize:TrainSize+TestSize]]
            x_train1 = x_train_MNIST[indexes[:TrainSize],:,:]
            y_train1 = y_train_MNIST[indexes[:TrainSize]]

        #format the data in matrix form
        x_train2 = np.divide(np.matrix(np.double(x_train1).reshape((x_train1.shape[0], np.prod(x_train1.shape[1:]))).transpose()),255)
        y_train2 = np.matrix(np.double(y_train1))
        x_test2 = np.divide(np.matrix(np.double(x_test1).reshape((x_test1.shape[0], np.prod(x_train1.shape[1:]))).transpose()),255)
        y_test2 = np.matrix(np.double(y_test1))

        #add noise
        if not self.noiseType=='None':
            y_train2 = y_train2+self.noise_sigma*self.noise_generator(y_train2.shape[1], self.noiseType)
            y_test2 = y_test2+self.noise_sigma*self.noise_generator(y_test2.shape[1], self.noiseType)

        return (x_train2,y_train2,x_test2,y_test2)
    
    

    
# Function to do cross validation

def crossValidation(params_list,predictor, x_train2, y_train2, CVExpN=np.inf, randomCV=False,cv_type='loo',best_type='mean'):
    CVExpN = min(x_train2.shape[1],CVExpN)
    fvals = np.zeros((len(params_list),0))
    for expid in range(CVExpN):
#    if cv_type == "loo":   
        if randomCV:
            tmp = np.random.permutation(x_train2.shape[1])
            cv_train_idx = tmp[:-1]
            cv_test_idx = tmp[-1:]
        else:
            tmp = np.arange(x_train2.shape[1])
            cv_train_idx = np.concatenate((tmp[:expid],tmp[expid+1:]))
            cv_test_idx = tmp[expid:expid+1]
        for cv_testk in range(len(cv_test_idx)):
            fvals = np.concatenate((fvals, np.zeros((len(params_list),1))),1)
            for nnk in range(len(params_list)):
                y0 = predictor(x_train2[:,cv_test_idx[cv_testk]], x_train2[:,cv_train_idx], y_train2[0,cv_train_idx], params_list[nnk])
                fvals[nnk,-1]+=np.power(y0-y_train2[0,cv_test_idx[cv_testk]],2)
    if best_type=='mean':
        tmp = np.mean(fvals,1)
    elif best_type=='conf90':
        tmp = np.mean(fvals,1)+stats.norm.ppf(0.95)*np.sqrt(np.var(fvals,1))
    elif best_type=='mean_conf90':
        tmp = np.mean(fvals,1)+np.divide(stats.norm.ppf(0.95),np.sqrt(fvals.shape[1]))*np.sqrt(np.var(fvals,1))
    return (params_list[np.argsort(tmp.squeeze())[0]],np.min(tmp.squeeze()),fvals)

def testPredictor(predictor,x_test2,y_test2,TestExpN = np.inf, doClassifyError = False, reportAll=False):
    TestExpN = min(TestExpN,x_test2.shape[1])
    fval_tests = np.zeros(TestExpN)
    class_tests = np.zeros(TestExpN)
    for expid in range(TestExpN):
        y0 = predictor(x_test2[:,expid])
        fval_tests[expid]=np.absolute(y0-y_test2[0,expid])
        class_tests[expid]=1*(int(np.round(y0))==int(y_test2[0,expid]))
    perf_mean = np.sqrt(np.mean(np.power(fval_tests,2)))
    perf_meanCI = np.sqrt(np.mean(fval_tests)+np.divide(stats.norm.ppf(0.95),np.sqrt(len(fval_tests)))*np.sqrt(np.var(fval_tests)))
    perf_meanL1 = np.mean(np.sqrt(fval_tests))
    if reportAll:
        return (perf_mean,perf_meanCI,np.mean(class_tests),fval_tests)
    elif doClassifyError:
        return (perf_mean,perf_meanCI,np.mean(class_tests))
    else:
        return (perf_mean,perf_meanCI)

    

        
# Distance object

class imageDistance:
    TransportCostImg = []
    distType = "Wasserstein"
    distHashTable = True
    distDict = {}
    ImgSize = []
    loaded_xs = []
    loaded_dists = []
    loaded_en = False
    
    def __init__(self,ImgSize,distType = "Wasserstein",distHashTable = True):
        self.ImgSize = ImgSize
        TransportCostImg =np.zeros((self.ImgSize[0],self.ImgSize[1],self.ImgSize[0],self.ImgSize[1]))
        for k in range(self.ImgSize[0]):
            for kk in range(self.ImgSize[1]):
                for kkk in range(self.ImgSize[0]):
                    for kkkk in range(self.ImgSize[1]):
                        TransportCostImg[k,kk,kkk,kkkk] = np.linalg.norm(np.divide(np.array([k, kk]),self.ImgSize)-np.divide(np.array([kkk, kkkk]),self.ImgSize),None,0).item()
        self.TransportCostImg = TransportCostImg.reshape((np.prod(self.ImgSize),np.prod(self.ImgSize)))        
        self.distType = distType
        self.distDict = {}
        
    def distFun_raw(self,x1,x2):
        if self.distType == "Euclidean":
            #normalize data CHECK IF IT WORKS BETTER
            x1 = np.divide(x1,np.linalg.norm(x1,1,0))
            x2 = np.divide(x2,np.linalg.norm(x2,1,0))
            
            tmp = np.zeros((x1.shape[1],x2.shape[1]))
            for k in range(x1.shape[1]):
                tmp[k,:] = np.linalg.norm(x1[:,k]-x2,None,0)
            if x1.shape[1]==1 and x2.shape[1]==1:
                tmp = tmp.item()
            return tmp
        elif self.distType == "Wasserstein":
            if x1.shape[1]==1 and x2.shape[1]==1:
                x1 = np.array(np.divide(x1,np.sum(x1))).flatten()
                x2 = np.array(np.divide(x2,np.sum(x2))).flatten()
#                print((x1.shape,x2.shape,self.TransportCostImg.shape))
                return ot.emd2(x1,x2,self.TransportCostImg)
            else:
                tmp = np.zeros((x1.shape[1],x2.shape[1]))
                for k in range(x1.shape[1]):
                    for kk in range(x2.shape[1]):
                        if kk>=k:
                            tmp[k,kk] = self.distFun_raw(x1[:,k],x2[:,kk])
                        else:
                            tmp[k,kk] = tmp[kk,k]
                return tmp

    def getKey(self,imVector):
        return tuple(map(tuple, np.round(np.array(imVector)*255)))        
        
    def xIsLoaded(self,x1):
        return x1[1,0]==-1 #loaded x's are 2xn with a -1 in second row
    
    def getDist(self,x1,x2):
            if len(x1.shape)==1:
                x1 = x1.reshape(len(x1),1)
            if len(x2.shape)==1:
                x2 = x2.reshape(len(x2),1)
            if self.loaded_en and self.xIsLoaded(x1) and self.xIsLoaded(x2):
                return (self.loaded_dists[x1[0,:].astype(int),:][:,x2[0,:].astype(int)]).reshape(x1.shape[1],x2.shape[1])
            elif self.loaded_en and self.xIsLoaded(x1):
                return self.getDist(self.loaded_xs[:,x1[0,:].astype(int)],x2)
            elif self.loaded_en and self.xIsLoaded(x2):
                return self.getDist(x1,self.loaded_xs[:,x2[0,:].astype(int)])
            elif x1.shape[1]==1 and x2.shape[1]==1:
                if not self.distHashTable:
                    return self.distFun_raw(x1[:,k],x2[:,kk])       
                else:
                    if self.getKey(x1) in self.distDict.keys():
                        sub_dict = self.distDict[self.getKey(x1)]
                        if not self.getKey(x2) in sub_dict.keys():
                            sub_dict[self.getKey(x2)] = self.distFun_raw(x1,x2)
                        return sub_dict[self.getKey(x2)]
                    elif self.getKey(x2) in self.distDict.keys():
                        sub_dict = self.distDict[self.getKey(x2)]
                        if not self.getKey(x1) in sub_dict.keys():
                            sub_dict[self.getKey(x1)] = self.distFun_raw(x1,x2)
                        return sub_dict[self.getKey(x1)]
                    else:
                        self.distDict[self.getKey(x1)] = {self.getKey(x2):self.distFun_raw(x1,x2)}
                        return self.distDict[self.getKey(x1)][self.getKey(x2)]
            else:
                if self.distType == "Euclidean":
                    tmp = self.distFun_raw(x1,x2)
                else:
                    tmp = np.zeros((x1.shape[1],x2.shape[1]))
                    for k in range(x1.shape[1]):
                        for kk in range(x2.shape[1]):
                            if kk>=k:
                                tmp[k,kk] = self.getDist(x1[:,k],x2[:,kk])
                            else:
                                tmp[k,kk] = tmp[kk,k]
                return tmp
            
    def load_hashTableFromData_old(self,x_train,x_test):
        self.getDist(x_train,x_train)
        self.getDist(x_train,x_test)
        return (x_train,x_test)

    def load_hashTableFromData(self,x_train,x_test):
        self.loaded_en = True
        self.loaded_xs = np.concatenate((x_train,x_test),1)
        self.loaded_dists = self.distFun_raw(self.loaded_xs,self.loaded_xs)
        x_trainb = -1*np.ones((2,x_train.shape[1]))
        x_trainb[0,:]=np.arange(x_train.shape[1])
        x_testb = -1*np.ones((2,x_test.shape[1]))
        x_testb[0,:]=np.arange(x_train.shape[1],x_train.shape[1]+x_test.shape[1])
        return (x_trainb,x_testb)

    def save_hashTableToFile(self,filename = "hashTable_Backup.dat"):
        file = open(filename,"wb")
        pickle.dump(self.distDict,file)
        pickle.dump(self.distType,file)
        pickle.dump(self.TransportCostImg,file)
        file.close()

    def load_hashTableFromFile(self,filename = "hashTable_Backup.dat"):
        file = open(filename,"rb")
        self.distDict = pickle.load(file)
        distType=pickle.load(file)
        TransportCostImg=pickle.load(file)
        #we should check that file has same type and transport cost
        file.close()
        
        
class kNN_model:
    getDist = []
    k = 0
    ks_list = [1,3,5,10,25,50,100,200]
    xobs = []
    yobs = []
    
    def __init__(self,getDist,k=0,ks_list = []):
        if not len(ks_list)==0:
            self.ks_list = ks_list
        self.getDist = getDist
        self.k = k
        
    def getParamsList(self):
        tmp = []
        for k in range(len(self.ks_list)):
            tmp.append([self.ks_list[k]])
        return tmp
        
    def set_ks_list(self,ks):
        self.ks = ks
    
    def setParams(self,params):
        self.k = np.ceil(params[0]).astype(np.int)
    
    def getParams(self):
        return [self.k]
    
    def setObs(self,xobs,yobs):
        self.xobs = xobs
        self.yobs = yobs
    
    def predictor(self,x0,xobs=[],yobs=[],params=[]):
        if len(xobs)==0:
            xobs = self.xobs; yobs = self.yobs
        if len(params)==0:
            params = self.getParams()
        k = params[0]
        tmp = self.getDist(x0,xobs)
        nn_idx = np.argsort(tmp)[0,:k]
        return np.mean(yobs[0,nn_idx]) 

        
class wassersteinRobust_model:
    getDist = []
    rho = 0
    gamma = 0
    alpha = 1
    rhok = 0
    gammak = 0
    paramsType = "k_params"
    rhos_list = []
    gammas_list = []
    rhok_list = [0,1,3,5]
    gammak_list = [1, 3, 5]
    alphas_list = [0.01, 0.1, 1]    
    rhoRatio = 0
    rhoRatio_list = []
    xobs = []
    yobs = []
    
    def __init__(self,getDist,rho = 0,gamma = 0,alpha = 1,rhok = 0,gammak = 0,paramsType = "k_params",rhos_list = [],alphas_list = [],gammas_list = [],rhok_list = [],gammak_list = []):
        self.getDist = getDist
        self.rho = rho
        self.gamma = gamma
        self.alpha = alpha
        self.rhok = rhok
        self.gammak = gammak
        self.paramsType = paramsType
        self.set_params_list(rhos_list = rhos_list,alphas_list = alphas_list,gammas_list = gammas_list,rhok_list = rhok_list,gammak_list = gammak_list)
        
    def getParamsList(self):
        if self.paramsType == "k_params":
            p1=self.rhok_list;p2=self.gammak_list;p3=self.alphas_list;
        elif self.paramsType == "k_and_ratio":
            p1=self.rhoRatio_list;p2=self.gammak_list;p3=self.alphas_list;            
        elif self.paramsType == "k_and_straight":
            p1=self.rhos_list;p2=self.gammak_list;p3=self.alphas_list;            
        else:
            p1=self.rhos_list;p2=self.gammas_list;p3=self.alphas_list;
        tmp = []
        for k in range(len(p1)):
            for kk in range(len(p2)):
                for kkk in range(len(p3)):
                    tmp.append([p1[k],p2[kk],p3[kkk]])
        return tmp
        
    def set_params_list(self,rhos_list = [],alphas_list = [],gammas_list = [],rhok_list = [],gammak_list = []):
        if not len(alphas_list)==0:
            self.alphas_list = alphas_list
        if not len(rhos_list)==0:
            self.use_k_params = False
            self.rhos_list = rhos_list
        if not len(gammas_list)==0:
            self.use_k_params = False
            self.gammas_list = gammas_list
        if not len(rhok_list)==0:
            self.use_k_params = True
            self.rhok_list = rhok_list
        if not len(gammak_list)==0:
            self.use_k_params = True
            self.gammak_list = gammak_list
    
    def setParams(self,params):
        if self.paramsType == "k_params":
            self.rhok = params[0]
            self.gammak = params[1]
            self.alpha = params[2]
        elif self.paramsType == "k_and_ratio":
            self.rhoRatio = params[0]
            self.gammak = params[1]
            self.alpha = params[2]            
        elif self.paramsType == "k_and_straight":
            self.rho = params[0]
            self.gammak = params[1]
            self.alpha = params[2]            
        else:
            self.gamma = params[1]
            self.alpha = params[2]
    
    def getParams(self):
        if self.paramsType == "k_params":
            return [self.rhok, self.gammak, self.alpha];
        elif self.paramsType == "k_and_ratio":
            return [self.rhoRatio, self.gammak, self.alpha];
        elif self.paramsType == "k_and_straight":
            return [self.rho, self.gammak, self.alpha];
        else:
            return [self.rho, self.gamma, self.alpha];
    
    def setObs(self,xobs,yobs):
        self.xobs = xobs
        self.yobs = yobs

        
    def setParamsListDD_straight(self,x_train,y_train,N):
        tmp = self.getDist(x_train,x_train).reshape(1,np.power(x_train.shape[1],2))
        b = max(tmp[tmp>0])
        a = np.divide(min(tmp[tmp>0]),1000)
        tmp2 = a*np.power(np.power(np.divide(b,a),np.divide(1,N)),np.arange(N+1))
        self.rhos_list = np.concatenate((tmp2,[0]))
        b = max(tmp[tmp>0])
        a = np.divide(min(tmp[tmp>0]),10)
        self.gammas_list = a*np.power(np.power(np.divide(b,a),np.divide(1,N)),np.arange(N+1))
        b = np.divide(np.mean(tmp),np.mean(np.absolute(y_train)))*100
        a = np.divide(b,100*1000)
        self.alphas_list = a*np.power(np.power(np.divide(b,a),np.divide(1,N)),np.arange(N+1))

    def setParamsListDD2(self,x_train,y_train,N):
        if self.paramsType == "straight":
            self.setParamsListDD_straight(x_train,y_train,N)
        elif self.paramsType == "k_and_ratio":
            tmp = self.getDist(x_train,x_train).reshape(1,np.power(x_train.shape[1],2))
            b = 10; a = 1
            self.gammak_list = a*np.power(np.power(np.divide(b,a),np.divide(1,N)),np.arange(N+1))
            b = 1; a = 0.001
            self.rhoRatio_list = np.concatenate(([0],a*np.power(np.power(np.divide(b,a),np.divide(1,N)),np.arange(N+1))))
            b = np.divide(np.mean(tmp),np.mean(np.absolute(y_train)))*100
            a = np.divide(b,100*1000)
            self.alphas_list = a*np.power(np.power(np.divide(b,a),np.divide(1,N)),np.arange(N+1))

    def setParamsList_Bertsimas(self,x_train,y_train,N):
        self.paramsType = "k_and_straight"
        tmp = self.getDist(x_train,x_train).reshape(1,np.power(x_train.shape[1],2))
        b = 10; a = 1; gammaN = min(N,b-a+1)
        self.gammak_list = np.round(a+(b-a)*np.divide(np.arange(gammaN),gammaN-1))
        alpha = 0
        self.alphas_list = [alpha];
        b = 20; a = 0.001
        self.rhos_list = np.concatenate(([0],a*np.power(np.power(np.divide(b,a),np.divide(1,N)),np.arange(N+1))))
            
            
        
    def predictor(self,x0,xobs=[],yobs=[],params=[]):
        if len(xobs)==0:
            xobs = self.xobs; yobs = self.yobs
        if len(params)==0:
            params = self.getParams()
        return self.meWN2_1d_kNN2(xobs, yobs, x0, params)


    def meWN_1d_cost(self, xobs, yobs, x0, beta, rho, gamma,idx_I1,idx_I2,xDists, alpha = 1):
        y_I1, y_I2 = yobs[idx_I1], yobs[idx_I2]
        D_minus_gamma = xDists[idx_I2] - gamma    
        D_minus_gamma[D_minus_gamma<0] = 0
        rho_minus_D_I1 = rho * np.ones_like(y_I1)
        rho_minus_D_I2 = rho - D_minus_gamma

        v_I1 = np.power(np.divide(rho_minus_D_I1,alpha)+np.abs(y_I1-beta),2)
        v_I2 = np.power(np.divide(rho_minus_D_I2,alpha)+np.abs(y_I2-beta),2)
        zmask_I1 = np.ones_like(y_I1)
        zmask_I2 = np.zeros_like(y_I2)
        if len(zmask_I1)==0:
            indexes = np.argsort(-v_I2)
            fval = v_I2[indexes[0]]
            zmask_I2[indexes[0]] = 1
        else:
            fval = np.sum(v_I1); count = np.sum(zmask_I1)
            indexes = np.argsort(-v_I2)
            for k in range(len(indexes)):
                if v_I2[indexes[k]]>np.divide(fval,count):
                    zmask_I2[indexes[k]]=1; count+=1
                    fval += v_I2[indexes[k]]
                else:
                    break
            fval = np.divide(fval,count)

        tmp_I1 = 2*np.multiply(np.sqrt(v_I1),(np.double(beta*np.ones_like(y_I1)>y_I1)-np.double(beta*np.ones_like(y_I1)<y_I1)))
        tmp_I2 = 2*np.multiply(np.sqrt(v_I2),(np.double(beta*np.ones_like(y_I2)>y_I2)-np.double(beta*np.ones_like(y_I2)<y_I2)))
        gradV = np.divide(zmask_I1.dot(tmp_I1)+zmask_I2.dot(tmp_I2),np.sum(zmask_I1)+np.sum(zmask_I2))
        return (fval,gradV)

    def mean_estimation_Winfty_Neighbor_1d(self, xobs, yobs, x0, rho = 1e-2, gamma = 1e-2, alpha = 1, tol = 1e-8, verbose = True):
        #This function uses a bisection algorithm to speed up in the case of 1d y
        yobs = np.array(yobs).flatten()
        xDists = self.getDist(x0,xobs)
        xDists = xDists.flatten()  
        if not alpha==0:
            idx_I = (xDists <= gamma + rho)
            idx_I1 = (xDists + rho <= gamma)
        else: #the idea is to treat rho as a bound on Y but not movement in X
            idx_I = (xDists <= gamma)
            idx_I1 = idx_I
            alpha = 1
        idx_I2 = idx_I & (~idx_I1)
        y_I = yobs[idx_I]
        if len(y_I)==0:
            return yobs[np.argsort(xDists)[0]]
        beta_range = np.array([y_I.min(0)-np.divide(rho,alpha), y_I.max(0)+np.divide(rho,alpha)]).flatten()
        best_beta = []; best_fval = np.inf
        while (beta_range[1]-beta_range[0]>tol) or (best_fval==np.inf):
            beta = np.mean(beta_range)
            (fval,gradV)=self.meWN_1d_cost(xobs, yobs, x0, beta, rho, gamma,idx_I1,idx_I2,xDists,alpha)
            if fval<best_fval:
                best_beta=beta; best_fval = fval;
            if gradV>0:
                beta_range[1]=beta
            else:
                beta_range[0]=beta
        return best_beta #(best_beta,best_fval)

    def getFracRank(self,sorted_list,rank):
        if rank < 1:
            return rank*sorted_list[0]
        elif rank == int(rank):
            return sorted_list[int(rank)-1]
        else:
            return (rank-np.floor(rank))*(sorted_list[int(np.ceil(rank)-1)]-sorted_list[int(np.ceil(rank-1)-1)])+sorted_list[int(np.ceil(rank-1)-1)]    
        
    def meWN2_1d_kNN(self, xobs, yobs, x0, rhok = 1, gammak = 5, alpha = 1, verbose = True):
        #choose rho and gamma based on k neirest neighbour
        xDists = self.getDist(x0,xobs)
        xDists = xDists.flatten()    
        xDists = np.sort(xDists)
        gamma = self.getFracRank(xDists,min(len(xDists),gammak))
        if rhok==0:
            rho = 0
        else:
            rho = self.getFracRank(xDists,min(len(xDists),rhok))
        return self.mean_estimation_Winfty_Neighbor_1d(xobs, yobs, x0, rho, gamma, alpha, verbose = verbose)

    def meWN2_1d_kNN2(self, xobs, yobs, x0, params, verbose = True):
        #choose rho and gamma based on k neirest neighbour
        xDists = self.getDist(x0,xobs)
        xDists = xDists.flatten()    
        xDists = np.sort(xDists)
        if self.paramsType == "k_params":
            rhok=params[0];gammak=params[1];alpha=params[2]
            gamma = self.getFracRank(xDists,min(len(xDists),gammak))
            if rhok==0:
                rho = 0
            else:
                rho = self.getFracRank(xDists,min(len(xDists),rhok))
        elif self.paramsType == "k_and_ratio": 
            rhoRatio=params[0];gammak=params[1];alpha=params[2]
            gamma = self.getFracRank(xDists,min(len(xDists),gammak))
            rho = rhoRatio*gamma
        elif self.paramsType == "k_and_straight": 
            rho=params[0];gammak=params[1];alpha=params[2]
            gamma = self.getFracRank(xDists,min(len(xDists),gammak))
        else:
            rho=params[0];gamma=params[1];alpha=params[2]
        return self.mean_estimation_Winfty_Neighbor_1d(xobs, yobs, x0, rho, gamma, alpha, verbose = verbose)

class nadarajaWatson_model:
    getDist = []
    bandwidth = 0
    bandwidth_list = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    kernel = "Gaussian"
    xobs = []
    yobs = []
    
    def __init__(self,getDist,kernel="Gaussian",bandwidth=0,bandwidth_list = []):
        if not len(bandwidth_list)==0:
            self.bandwidth_list = bandwidth_list
        self.getDist = getDist
        self.bandwidth = bandwidth
        self.kernel = kernel
        
    def getParamsList(self):
        tmp = []
        for k in range(len(self.bandwidth_list)):
            tmp.append([self.bandwidth_list[k]])
        return tmp
        
    def set_bandwidth_list(self,bandwidth_list):
        self.bandwidth_list = bandwidth_list
    
    def setParams(self,params):
        self.bandwidth = params[0]
    
    def getParams(self):
        return [self.bandwidth]
    
    def setObs(self,xobs,yobs):
        self.xobs = xobs
        self.yobs = yobs

    def setBandwidthList(self,x_train,N):
        tmp = self.getDist(x_train,x_train).reshape(1,np.power(x_train.shape[1],2))
        b = np.mean(np.array(tmp).flatten())
        b = max(tmp[tmp>0])
        a = np.divide(min(tmp[tmp>0]),1000)
        self.bandwidth_list = a*np.power(np.power(np.divide(b,a),np.divide(1,N)),np.arange(N+1))
        self.bandwidth_list = np.concatenate((self.bandwidth_list,[0]))
        
        
        
    def predictor(self,x0,xobs=[],yobs=[],params=[]):
        if len(xobs)==0:
            xobs = self.xobs; yobs = self.yobs
        if len(params)==0:
            params = self.getParams()
        bandwidth = params[0]
        tmp = self.getDist(x0,xobs)
        if self.kernel=="Gaussian":
            if np.square(bandwidth)==0:
                density = np.ones_like(tmp)
            else:
                density = np.exp(-np.divide(np.square(tmp),(2 * np.square(bandwidth))))
            if np.sum(density)==0:
                return np.mean(yobs)
            else:
                return np.divide(yobs.dot(density.transpose()),np.sum(density)).item()    
        elif self.kernel=="Epanechnikov":
            if np.square(bandwidth)==0:
                density = np.ones_like(tmp)
            else:
                density = np.divide(3,4)*np.maximum(np.zeros_like(tmp),1 - np.square(np.divide(tmp,bandwidth)))
            if np.sum(density)==0:
                return np.mean(yobs)
            else:
                return np.divide(yobs.dot(density.transpose()),np.sum(density)).item()