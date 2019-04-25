# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 17:54:24 2018
@author: Krishna Vamsy (tbrk)
"""

#%% Importing required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#%%
Train_set = pd.read_csv("train.csv")
Test_set = pd.read_csv("test.csv")
#Ori_Train = Train_set
#Ori_Test = Test_set
pd.set_option('display.max_columns', None)
Train_set.head(8)
Train_set = Train_set.drop(['Id'], axis = 1)
Test_set = Test_set.drop(['Id'], axis = 1)
#Train_set.describe()
Train_set.isnull().sum() #check for missing values

Train_set['Cover_Type'].value_counts()
sns.countplot(data=Train_set,x=Train_set['Cover_Type'])

for i in range(10,54):
    print (Train_set.iloc[:,i].value_counts())

Train_set = Train_set.drop(['Soil_Type7', 'Soil_Type15'], axis = 1)
Test_set = Test_set.drop(['Soil_Type7', 'Soil_Type15'], axis = 1)

#%% Visualizations
colnames = Train_set.columns

for i in colnames[0:10]:
    plt.figure()
    sns.violinplot(data=Train_set,x=Train_set['Cover_Type'],y=Train_set[i])
    plt.show()

#%% Correlation
corrdata = Train_set.iloc[:,:10]
data_corr = corrdata.corr()
level = 0.6
#print(data_corr) # Are the highly correlated ones

corrmat = Train_set.iloc[:,:10].corr()
sns.heatmap(corrmat,vmax=0.8,square=True) #for better visualization

high_corr = [] #to get highly correlated ones
for i in range(0, 10):
    for j in range(i+1, 10):
        if data_corr.iloc[i,j]>= level and data_corr.iloc[i,j]<1\
        or data_corr.iloc[i,j] <0 and data_corr.iloc[i,j]<=-level:
            high_corr.append([i,j,data_corr.iloc[i,j]])
sorted_high_corr = sorted(high_corr,key= lambda x: abs(x[2]),reverse = True)

cols = corrdata.columns
for i,j,corr in sorted_high_corr:
    print("%s and %s = %.2f" % (cols[i], cols[j], corr))
    
#visualization
for i,j,corr in sorted_high_corr:
    sns.pairplot(data = Train_set, hue='Cover_Type', height = 5, x_vars=cols[i], y_vars=cols[j])
    plt.show()

#%%
y=Train_set['Cover_Type']
Train_set=Train_set.drop(['Cover_Type'],1)

#Train-Test split for Cross-validation

x_train, x_test, y_train, y_test = train_test_split(Train_set, y, test_size=0.3, random_state=42)

from sklearn.feature_selection import SelectFromModel
sel = SelectFromModel(RandomForestClassifier(n_estimators=150,class_weight='balanced',n_jobs=2,random_state=42))
sel.fit(x_train, y_train)
selected_feat= x_train.columns[(sel.get_support())]

new_train = Train_set[selected_feat]
new_test = Test_set[selected_feat]

x_train, x_test, y_train, y_test = train_test_split(new_train , y, test_size=0.3, random_state=42)
rf=RandomForestClassifier(n_estimators=150,class_weight='balanced',n_jobs=2,random_state=42)
rf.fit(x_train,y_train)
pred=rf.predict(x_test)
confusion_matrix(pred,y_test)
acc=rf.score(x_test,y_test)
print(acc)

#%%
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(rf, random_state=42).fit(x_train, y_train)
eli5.show_weights(perm, feature_names = x_train.columns.tolist())
