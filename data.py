# -*- coding: utf-8 -*-
import matplotlib.pylab as plt
import numpy as np
import os
import pandas as pd
import re
import gzip
import pdb
from sklearn import datasets

####################
# 分類デモデータ用のクラス
class classification:
    #-------------------
    # パス、ラベルの設定
    def __init__(self,negLabel=-1,posLabel=1):
        self.path = "data"
        self.negLabel = negLabel
        self.posLabel = posLabel
    #-------------------
    
    #-------------------
    # データの作成
    # dataType: データの種類（整数スカラー）
    def makeData(self,dataType=1):
        self.dataType = dataType
    
        # 建物等級）説明変数:GrLivArea
        if dataType == 1:
            data = pd.read_csv(os.path.join(self.path,"house-prices-advanced-regression-techniques/train.csv"))
            self.X = data[(data['MSSubClass']==30) |(data['MSSubClass']==60)][['GrLivArea']].values
            self.Y = data[(data['MSSubClass']==30) |(data['MSSubClass']==60)][['MSSubClass']].values
            self.Y[self.Y==30] = self.negLabel
            self.Y[self.Y==60] = self.posLabel
            self.xLabel = "x[ft^2]"
            self.yLabel = "y"

        # 建物等級）説明変数:GrLivArea,GarageArea
        elif dataType == 2:
            data = pd.read_csv(os.path.join(self.path,"house-prices-advanced-regression-techniques/train.csv"))
            self.X = data[(data['MSSubClass']==30) |(data['MSSubClass']==60)][['GrLivArea','GarageArea']].values
            self.Y = data[(data['MSSubClass']==30) |(data['MSSubClass']==60)][['MSSubClass']].values
            self.Y[self.Y==30] = self.negLabel
            self.Y[self.Y==60] = self.posLabel
            self.xLabel = "housing area x[ft^2]"
            self.yLabel = "garage area x[ft^2]"

        # トイデータ） 線形分離可能な2つのガウス分布に従場合
        elif dataType == 3:
            dNum = 120
            np.random.seed(1)
            
            cov = [[1,-0.6],[-0.6,1]]
            X = np.random.multivariate_normal([1,2],cov,int(dNum/2))
            X = np.concatenate([X,np.random.multivariate_normal([-2,-1],cov,int(dNum/2))],axis=0)
            Y = np.concatenate([self.negLabel*np.ones([int(dNum/2),1]),self.posLabel*np.ones([int(dNum/2),1])],axis=0)
            randInds = np.random.permutation(dNum)
            self.X = X[randInds]
            self.Y = Y[randInds]
            self.xLabel = "$x_1$"
            self.yLabel = "$x_2$"

        # トイデータ）分類境界がアルファベッドのCの形をしている場合
        elif dataType == 4:
            dNum = 120
            np.random.seed(1)
            
            cov1 = [[1,-0.8],[-0.8,1]]
            cov2 = [[1,0.8],[0.8,1]]
                
            X = np.random.multivariate_normal([0.5,1],cov1,int(dNum/2))
            X = np.concatenate([X,np.random.multivariate_normal([-1,-1],cov1,int(dNum/4))],axis=0)
            X = np.concatenate([X,np.random.multivariate_normal([-1,4],cov2,int(dNum/4))],axis=0)
            Y = np.concatenate([self.negLabel*np.ones([int(dNum/2),1]),self.posLabel*np.ones([int(dNum/2),1])],axis=0)
            randInds = np.random.permutation(dNum)
            self.X = X[randInds]
            self.Y = Y[randInds]
                
            self.xLabel = "$x_1$"
            self.yLabel = "$x_2$"

        # トイデータ）複数の島がある場合
        elif dataType == 5:
            dNum = 120
            np.random.seed(1)
            
            cov = [[1,-0.8],[-0.8,1]]
            X = np.random.multivariate_normal([0.5,1],cov,int(dNum/2))
            X = np.concatenate([X,np.random.multivariate_normal([-1,-1],cov,int(dNum/4))],axis=0)
            X = np.concatenate([X,np.random.multivariate_normal([2,2],cov,int(dNum/4))],axis=0)
            Y = np.concatenate([self.negLabel*np.ones([int(dNum/2),1]),self.posLabel*np.ones([int(dNum/2),1])],axis=0)
            
            # データのインデックスをシャッフル
            randInds = np.random.permutation(dNum)
            self.X = X[randInds]
            self.Y = Y[randInds]
            self.xLabel = "$x_1$"
            self.yLabel = "$x_2$"
            
        # トイデータ）分類境界がアルファベッドのCの形をしている場合（ノイズあり）
        elif dataType == 6:
            dNum = 120
            np.random.seed(1)
                    
            cov1 = [[1,-0.8],[-0.8,1]]
            cov2 = [[1,0.8],[0.8,1]]

            X = np.random.multivariate_normal([0.5,1],cov1,int(dNum/2))
            X = np.concatenate([X,np.random.multivariate_normal([-1,-1],cov1,int(dNum/4))],axis=0)
            X = np.concatenate([X,np.random.multivariate_normal([-1,4],cov2,int(dNum/4))],axis=0)
            Y = np.concatenate([self.negLabel*np.ones([int(dNum/2),1]),self.posLabel*np.ones([int(dNum/2),1])],axis=0)
            
            # ノイズ
            X = np.concatenate([X,np.array([[-1.5,-1.5],[-1,-1]])],axis=0)
            Y = np.concatenate([Y,self.negLabel*np.ones([2,1])],axis=0)
            dNum += 2
            
            randInds = np.random.permutation(dNum)
            self.X = X[randInds]
            self.Y = Y[randInds]
                
            self.xLabel = "$x_1$"
            self.yLabel = "$x_2$"