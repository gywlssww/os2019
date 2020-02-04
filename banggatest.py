#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 01:41:51 2019

@author: hyunjeong
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression


#import geopandas as gpd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

def scale(X):
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    return X_scaled

class jeonse:
    def __init__(self,p1,p2,p3):
        self.money=0
        self.tlist=[]
        df=pd.read_csv('realdata.csv')
        #print(len(df.values))
        #df=df[df['계약년도']==2019]

        df['구동']=df['자치구명']+df['법정동명']


        rent1=df[df['전월세구분']=='전세']
        
        #print(rent1)
        #a=rent1[['임대면적','건축년도']]
        a=rent1[['임대면적']]
        
        b=pd.get_dummies(rent1['구동'])
        self.simple_rent1=a.join(b)
        c=pd.get_dummies(rent1['임대건물명'])
        self.simple_rent1=self.simple_rent1.join(c)



        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.simple_rent1, rent1['보증금'],random_state=42)

        self.knn=KNeighborsRegressor(n_neighbors=3)
        self.lr=LinearRegression()
            
        self.train()
        
        
        self.getinput2(p1,p2,p3)
        self.getMoney()
        self.predict()
        '''
        while True:
            
            print("input >> ")
            tstr=input()
            if tstr=="-1":
                break
            self.getinput(tstr)
            self.predict()
        '''
            
    def train(self):
        
        self.knn.fit(self.X_train, self.Y_train)

        self.lr.fit(self.X_train,self.Y_train)

        print(self.lr.score(self.X_test,self.Y_test))
        print(self.knn.score(self.X_test,self.Y_test))
        #print(lr.score(X_test,Y_test))
        #svr = SVR(gamma='scale', C=1.0, epsilon=0.2)
        #svr.fit(X_train,Y_train) 
    
    def getinput(self,tstr):
        self.tlist=[]
        zero=np.zeros(len(self.simple_rent1.columns))

        arr=tstr.split(" ")
        t=pd.DataFrame([zero],columns=self.simple_rent1.columns)
        #t['임대면적']=float(arr[0])/3.305785
        t['임대면적']=float(arr[0])
        
        t[arr[1]]=1.0

        if arr[2]=='아파트':
            t['아파트']=1.0
        elif arr[2]=='오피스텔':
            t['오피스텔']=1.0
        else:
            t['다세대/연립']=1.0 
        #t['건축년도']=2017
        #t['보증금']=6
        self.tlist.append(t)
    
    def getMoney(self):
        return self.money
    
    def predict(self):

        for i in self.tlist:

            pred=self.lr.predict(i)
            pred=pred[0]
            if pred>=10000:
                p=str(pred)
                b=p[0]
                m=pred-10000*int(b)
                print(b+"억"+str(int(m))+"만원")
                self.money=b+"억"+str(int(m))+"만원"
            else:
                print(str(int(pred))+"만원")

                return str(str(int(pred))+"만원")
            print('\n')
            
    def getinput2(self,a,b,c):
        #a 동네
        #b 면적
        #c 유형
        
        self.tlist=[]
        zero=np.zeros(len(self.simple_rent1.columns))

        #arr=tstr.split(" ")
        t=pd.DataFrame([zero],columns=self.simple_rent1.columns)
        t['임대면적']=float(b)*3.305785
        #t['임대면적']=float(b)
        
        t[a]=1.0

        if c=='아파트':
            t['아파트']=1.0
        elif c=='오피스텔':
            t['오피스텔']=1.0
        else:
            t['다세대/연립']=1.0 

        self.tlist.append(t)
            
class wolse:
    def __init__(self,p1,p2,p3):
        
        
        df=pd.read_csv('realdata.csv')
        #print(len(df.values))
        #df=df[df['계약년도']==2019]
        

        df['구동']=df['자치구명']+df['법정동명']

        '''
        temp=df['구동']
        temp=temp.unique()
        temp=pd.DataFrame(temp)
        temp.to_excel('구동.xlsx')
        '''

        rent1=df[df['전월세구분']=='준월세']
        
        self.origin=rent1

        a=rent1[['임대면적','건축년도']]
        a=rent1[['임대면적']]

        #tempindex=a.index        
        #a=scale(a)
        #a=pd.DataFrame(a,columns=['임대면적'],index=tempindex)

        b=pd.get_dummies(rent1['구동'])
        self.simple_rent1=a.join(b)
        c=pd.get_dummies(rent1['임대건물명'])
        self.simple_rent1=self.simple_rent1.join(c)
   
        floor=[]
        for f in rent1['층'].values:
            
            if f==-1:
                floor.append(-1)
            else:
                floor.append(0)
        
        self.simple_rent1['floor']=floor


        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.simple_rent1, rent1['보증금'],random_state=42)
        self.X_train2, self.X_test2, self.Y_train2, self.Y_test2 = train_test_split(self.simple_rent1, rent1['임대료'],random_state=42)
        
        
        self.knn=KNeighborsRegressor(n_neighbors=3)
        self.knn2=KNeighborsRegressor(n_neighbors=3)
        self.lr=LinearRegression()

        self.tlist=[]
        
        self.train()
        
        self.getinput2(p1,p2,p3)
        self.predict()
        '''
        while True:
            
            print("input >> ")
            tstr=input()
            if tstr=="-1":
                break
            self.getinput(tstr)
            self.predict()
        '''
            
    def train(self):
        
        self.knn.fit(self.X_train, self.Y_train)
        self.knn2.fit(self.X_train2, self.Y_train2)

        self.lr.fit(self.X_train,self.Y_train)

    
    def getinput(self,tstr):
        self.tlist=[]
        zero=np.zeros(len(self.simple_rent1.columns))

        arr=tstr.split(" ")
        t=pd.DataFrame([zero],columns=self.simple_rent1.columns)
        #t['임대면적']=float(arr[0])/3.305785
        t['임대면적']=float(arr[0])

        t[arr[1]]=1.0

        if arr[2]=='아파트':
            t['아파트']=1.0
        elif arr[2]=='오피스텔':
            t['오피스텔']=1.0
        else:
            t['다세대/연립']=1.0 
        #t['건축년도']=2017
        #t['보증금']=6
        
            
        self.tlist.append(t)
    
                
    def getinput2(self,a,b,c):
        #a 동네
        #b 면적
        #c 유형
        
        self.tlist=[]
        zero=np.zeros(len(self.simple_rent1.columns))

        #arr=tstr.split(" ")
        t=pd.DataFrame([zero],columns=self.simple_rent1.columns)
        t['임대면적']=float(b)*3.305785
        #t['임대면적']=float(b)
        
        t[a]=1.0

        if c=='아파트':
            t['아파트']=1.0
        elif c=='오피스텔':
            t['오피스텔']=1.0
        else:
            t['다세대/연립']=1.0 

        self.tlist.append(t)
        
    def predict(self):

        for i in self.tlist:

            ind=self.knn.kneighbors(i,n_neighbors=3,return_distance=False)
            print(ind[0])
            ind2=self.knn2.kneighbors(i,n_neighbors=3,return_distance=False)
            print(ind2[0])
            
            indices=list(ind[0])
            indices2=list(ind2[0])
            
            y_tr=pd.DataFrame(self.Y_train)
            
            zlist=list(map(lambda i: y_tr.iloc[i,:], indices))  
           
            z=pd.DataFrame(zlist[1])
            
            rz1=str(int(z.loc['보증금',:]))
            
            
            y_tr2=pd.DataFrame(self.Y_train2)
            
            z2list=list(map(lambda i: y_tr2.iloc[i,:], indices2))  
           
            z2=pd.DataFrame(z2list[1])
            
            rz2=str(int(z2.loc['임대료',:]))
            
            r1=rz1+"/"+rz2
            
            z=pd.DataFrame(zlist[2])
            z2=pd.DataFrame(z2list[2])
            
            rz1=str(int(z.loc['보증금',:]))
            rz2=str(int(z2.loc['임대료',:]))
            
            r2=rz1+"/"+rz2
            
            z=pd.DataFrame(zlist[0])
            z2=pd.DataFrame(z2list[0])
            
            rz1=str(int(z.loc['보증금',:]))
            rz2=str(int(z2.loc['임대료',:]))
            
            r3=rz1+"/"+rz2
            
            self.resultstr=r1+'\n'+r2+'\n'+r3
            return str(self.resultstr)
            print('\n')

            print('\n')
            
            


