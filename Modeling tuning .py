
# coding: utf-8

# In[8]:


from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, roc_curve
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import sklearn.model_selection as model_selection
from statistics import mean
from sklearn.ensemble import RandomForestClassifier, VotingClassifier


# In[2]:


# Since the dataset is so large, I just randomly select part of data to train model

train_smote=pd.read_csv('Cleaned_train_smote.csv')

test_smote=pd.read_csv('Cleaned_test_smote.csv')

df_nosampling=pd.read_csv('Cleaned_train_no_sampling.csv')


train_undersampling=pd.read_csv('Cleaned_train_undersampling.csv')

test_undersampling=pd.read_csv('Cleaned_test_undersampling.csv')
#train_oversample=pd.read_csv('Cleaned_train_oversample.csv')

#test_oversample=pd.read_csv('Cleaned_test_oversample.csv')

print(train_smote['TARGET'].value_counts())
#-------------Select part of data from the total dataset------------
random_train_list=np.random.randint(low=0, high=len(train_smote), size=15000)
random_test_list=np.random.randint(low=0, high=len(test_smote), size=5000)

train_smote=train_smote.iloc[list(random_train_list),:]
test_smote=test_smote.iloc[list(random_test_list),:]

random_list=np.random.randint(low=0, high=len(df_nosampling), size=20000)
df_nosampling=df_nosampling.iloc[list(random_list),:]

random_test_list=np.random.randint(low=0, high=len(test_undersampling), size=5000)
test_undersampling=test_undersampling.iloc[list(random_test_list),:]
print(test_undersampling.shape)
#-------Dataset with smote oversampling
train_data_smote=train_smote.drop(columns=['TARGET','Unnamed: 0'])

train_label_smote=train_smote['TARGET']

test_data_smote=test_smote.drop(columns=['TARGET','Unnamed: 0'])
test_label_smote=test_smote['TARGET']

##--------Dataset without sampling-------------
data_nosampling=df_nosampling.drop(columns=['TARGET','Unnamed: 0'])

label_nosampling=df_nosampling['TARGET']

##------Dataset with undersampling-------------
train_data_under=train_undersampling.drop(columns=['TARGET','Unnamed: 0'])

train_label_under=train_undersampling['TARGET']

test_data_under=test_undersampling.drop(columns=['TARGET','Unnamed: 0'])
test_label_under=test_undersampling['TARGET']

#---------Get column name----------------------
features_names_smote  =list(train_data_smote.columns)
features_names_no = list(data_nosampling.columns)
features_names_under = list(train_data_under.columns)



#-------------Normalization----------------
def normalize(X):
    X = StandardScaler().fit_transform(X)
    return X

train_data_smote=normalize(train_data_smote)

test_data_smote=normalize(test_data_smote)
data_nosampling=normalize(data_nosampling)
train_data_under=normalize(train_data_under)

test_data_under=normalize(test_data_under)


#-----Split the dataset without oversampling
train_data_no, test_data_no, train_label_no, test_label_no = train_test_split(data_nosampling, label_nosampling, test_size=0.3, random_state=42)


# ## SVM

# In[ ]:



def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

print(svc_param_selection(train_data_under, train_label_under, 10))


# In[7]:


gammas = [0.001, 0.01, 0.1, 1]
recalls=[]
precs=[]
accuracys=[]
f1s=[]
for i in gammas:

    
    kfold = model_selection.KFold(n_splits=10, random_state=42)
    model=SVC(kernel='rbf',gamma=i)


    recall = cross_val_score(model, train_data_under, train_label_under, cv=10, scoring='recall')
    accuracy = cross_val_score(model,train_data_under, train_label_under, cv=10)
    f1=cross_val_score(model,train_data_under, train_label_under, cv=10, scoring='f1_macro')
    prec=cross_val_score(model, train_data_under, train_label_under, cv=10, scoring='precision')
    
    recalls.append(recall)
    accuracys.append(accuracy)
    f1s.append(f1)
    precs.append(prec)

df=pd.DataFrame({'Gamma value':gammas,'Accuracy':np.mean(accuracys,axis=1),'Recall':np.mean(recalls,axis=1),'Precision':np.mean(precs,axis=1),'F1':np.mean(f1s,axis=1)})
print(df)


# In[14]:


Cs = [0.001, 0.01, 0.1, 1, 10]
recalls=[]
precs=[]
accuracys=[]
f1s=[]
for i in Cs:

    
    kfold = model_selection.KFold(n_splits=10, random_state=42)
    model=SVC(kernel='rbf',gamma=0.001,C=i)


    recall = cross_val_score(model, train_data_under, train_label_under, cv=10, scoring='recall')
    accuracy = cross_val_score(model,train_data_under, train_label_under, cv=10)
    f1=cross_val_score(model,train_data_under, train_label_under, cv=10, scoring='f1_macro')
    prec=cross_val_score(model, train_data_under, train_label_under, cv=10, scoring='precision')
    
    recalls.append(recall)
    accuracys.append(accuracy)
    f1s.append(f1)
    precs.append(prec)

df=pd.DataFrame({'C value':Cs,'Accuracy':np.mean(accuracys,axis=1),'Recall':np.mean(recalls,axis=1),'Precision':np.mean(precs,axis=1),'F1':np.mean(f1s,axis=1)})
print(df)


# ## Random Forest

# In[4]:


Num_trees = [50,100, 500,1000]
recalls=[]
precs=[]
accuracys=[]
f1s=[]
for i in Num_trees:

    start_time = time.time()
    kfold = model_selection.KFold(n_splits=10, random_state=42)
    model= RandomForestClassifier(n_estimators=i)


    recall = cross_val_score(model, train_data_under, train_label_under, cv=10, scoring='recall')
    accuracy = cross_val_score(model,train_data_under, train_label_under, cv=10)
    f1=cross_val_score(model,train_data_under, train_label_under, cv=10, scoring='f1_macro')
    prec=cross_val_score(model, train_data_under, train_label_under, cv=10, scoring='precision')
    
    recalls.append(recall)
    accuracys.append(accuracy)
    f1s.append(f1)
    precs.append(prec)
    print(recall,accuracy,f1,prec)
    print("--- %s seconds ---" % (time.time() - start_time))
df=pd.DataFrame({'Num_trees':Num_trees ,'Accuracy':np.mean(accuracys,axis=1),'Recall':np.mean(recalls,axis=1),'Precision':np.mean(precs,axis=1),'F1':np.mean(f1s,axis=1)})
print(df)


# In[6]:


max_depth = [5,10,20,30]
recalls=[]
precs=[]
accuracys=[]
f1s=[]
for i in max_depth:

    start_time = time.time()
    kfold = model_selection.KFold(n_splits=10, random_state=42)
    model= RandomForestClassifier(n_estimators=1000,max_depth=i)


    recall = cross_val_score(model, train_data_under, train_label_under, cv=10, scoring='recall')
    accuracy = cross_val_score(model,train_data_under, train_label_under, cv=10)
    f1=cross_val_score(model,train_data_under, train_label_under, cv=10, scoring='f1_macro')
    prec=cross_val_score(model, train_data_under, train_label_under, cv=10, scoring='precision')
    
    recalls.append(recall)
    accuracys.append(accuracy)
    f1s.append(f1)
    precs.append(prec)
    print(recall,accuracy,f1,prec)
    print("--- %s seconds ---" % (time.time() - start_time))
df=pd.DataFrame({'Max_depth':max_depth,'Accuracy':np.mean(accuracys,axis=1),'Recall':np.mean(recalls,axis=1),'Precision':np.mean(precs,axis=1),'F1':np.mean(f1s,axis=1)})
print(df)


# In[7]:



max_depth = [5,10,20,30]
recalls=[]
precs=[]
accuracys=[]
f1s=[]
for i in max_depth:

    start_time = time.time()
    kfold = model_selection.KFold(n_splits=10, random_state=42)
    model= RandomForestClassifier(n_estimators=1000,max_depth=i)


    recall = cross_val_score(model, train_data_under, train_label_under, cv=10, scoring='recall')
    accuracy = cross_val_score(model,train_data_under, train_label_under, cv=10)
    f1=cross_val_score(model,train_data_under, train_label_under, cv=10, scoring='f1_macro')
    prec=cross_val_score(model, train_data_under, train_label_under, cv=10, scoring='precision')
    
    recalls.append(recall)
    accuracys.append(accuracy)
    f1s.append(f1)
    precs.append(prec)
    print(recall,accuracy,f1,prec)
    print("--- %s seconds ---" % (time.time() - start_time))
df=pd.DataFrame({'Max_depth':max_depth,'Accuracy':np.mean(accuracys,axis=1),'Recall':np.mean(recalls,axis=1),'Precision':np.mean(precs,axis=1),'F1':np.mean(f1s,axis=1)})
print(df)


# ## LR

# In[11]:


c = [0.01,0.1,1,10]
recalls=[]
precs=[]
accuracys=[]
f1s=[]
for i in c:

    start_time = time.time()
    kfold = model_selection.KFold(n_splits=10, random_state=42)
    model= LogisticRegression(solver='lbfgs',C=i)


    recall = cross_val_score(model, train_data_under, train_label_under, cv=10, scoring='recall')
    accuracy = cross_val_score(model,train_data_under, train_label_under, cv=10)
    f1=cross_val_score(model,train_data_under, train_label_under, cv=10, scoring='f1_macro')
    prec=cross_val_score(model, train_data_under, train_label_under, cv=10, scoring='precision')
    
    recalls.append(recall)
    accuracys.append(accuracy)
    f1s.append(f1)
    precs.append(prec)
   
    print("--- %s seconds ---" % (time.time() - start_time))
df=pd.DataFrame({'C value':c ,'Accuracy':np.mean(accuracys,axis=1),'Recall':np.mean(recalls,axis=1),'Precision':np.mean(precs,axis=1),'F1':np.mean(f1s,axis=1)})
print(df)


# ## Majority vote

# In[13]:


#Accuracy 0.6763870967741935
#Recall 0.6668387096774193
#Precision 0.6804633858046878
#F1 0.6770432334755305
estimators = []
model1 = LogisticRegression(solver='lbfgs',C=1)
estimators.append(('logistic', model1))
model2 = RandomForestClassifier(n_estimators=1000,max_depth=10)
estimators.append(('rf', model2))
model3 = SVC(kernel='rbf',gamma=0.001,C=10)
estimators.append(('svm', model3))
ensemble = VotingClassifier(estimators)
acc = cross_val_score(ensemble, train_data_under, train_label_under, cv=10)
recall=cross_val_score(ensemble, train_data_under, train_label_under, cv=10, scoring='recall')
f1=cross_val_score(ensemble,train_data_under, train_label_under, cv=10, scoring='f1_macro')
prec=cross_val_score(ensemble, train_data_under, train_label_under, cv=10, scoring='precision')
print('Accuracy',acc.mean())
print('Recall',recall.mean())
print('Precision',prec.mean())
print('F1',f1.mean())

