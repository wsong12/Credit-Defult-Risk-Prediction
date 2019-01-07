
# coding: utf-8

# In[2]:


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


# In[4]:


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


# In[16]:


print(test_label_under.value_counts())


# In[5]:


def principleCA(train_data,test_data,num_c):
    pca = PCA(n_components=num_c)
    pca.fit(train_data)
    train_data = pca.transform(train_data)
    test_data = pca.transform(test_data)
    pcatrain = pd.DataFrame(data = train_data)
    pcatest= pd.DataFrame(data = test_data)
    
    return pcatrain,pcatest


# In[6]:


# SVM feature selection
def f_importances(imp, names):
    
    
    imp=abs(imp)
    df=pd.DataFrame({'Name':names,'Coef':imp})
    df=df.sort_values(by=['Coef'],ascending=False)
    plt.figure(figsize=(10,5))
    plt.barh(range(len(names[0:10])),df['Coef'][0:10], align='center')
    plt.yticks(range(len(names[0:10])), df['Name'][0:10])
    
    plt.show()
    svm_feature=df['Name']
    return svm_feature


# In[5]:


def auc(model,testX,testy):
# predict probabilities
    probs = model.predict_proba(testX)
# keep probabilities for the positive outcome only
    probs = probs[:, 1]
# calculate AUC
    auc = roc_auc_score(testy, probs) 
    return auc


# In[7]:


from sklearn.svm import SVC

svm = SVC(kernel='linear')

svm.fit(train_data_under, train_label_under)
coef=svm.coef_ 
svm_feature=f_importances(coef[0], features_names_under)


# In[8]:




def accuracy(y_test,y_pred):
   
    

    #clf=svm.SVC(kernel='rbf',gamma='scale')
    #clf=KNeighborsClassifier(n_neighbors=10)
    #clf = LogisticRegression(C=1)
    #clf=MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    #clf.fit(X_train,y_train)
    
    #y_pred=clf.predict(X_test)
    acc=accuracy_score(y_test, y_pred)
    #scores = cross_val_score(clf, X_train, y_train, cv=10)
    #avg_score=np.mean(scores)
    #report=metrics.classification_report(y_test, y_pred)
    return acc

def cross_validate(clf, X_train, y_train,cv=10):
    X_train=pd.DataFrame(X_train)
    #y_train=pd.DataFrame(y_train)
    scores = cross_val_score(clf, X_train, y_train, cv=10)
    avg_score=np.mean(scores)
    return avg_score



# ## SVM

# In[36]:





def svmacc(train_data,train_label,test_data,test_label):
    start_time = time.time()
    
    clf=SVC(kernel='rbf')
    clf.fit(train_data,train_label)
    y_pred=clf.predict(test_data)
    acc=accuracy(test_label,y_pred)
    cv=cross_validate(clf, train_data, train_label,cv=10)
    prec=precision_score(test_label,y_pred)
    recall=recall_score(test_label,y_pred)
    f1=2 * (prec * recall) / (prec + recall)
    #aucc=auc(clf,test_data)
    #acc,cv,prec,recall,f1,aucc=model_performance(train_data,train_label,test_data,test_label,clf,y_pred)
    print("--- %s seconds ---" % (time.time() - start_time))
    return acc,cv,prec,recall,f1
acc,cv,prec,recall,f1=svmacc(train_data_smote,train_label_smote,test_data_smote,test_label_smote)

acc1,cv1,prec1,recall1,f11=svmacc(train_data_no,train_label_no,test_data_no,test_label_no)
acc2,cv2,prec2,recall2,f12=svmacc(train_data_under,train_label_under,test_data_under,test_label_under)

accs=[acc,acc1,acc2]
cvs=[cv,cv1,cv2]
precs=[prec,prec1,prec2]
f1s=[f1,f11,f12]
recalls=[recall,recall1,recall2]
#auccs=[aucc,aucc1,aucc2]
Name=['Dataset using smote oversampling','Dataset without oversampling','Dataset with undersampling']
df=pd.DataFrame({'Name':Name,'Test Accuracy':accs,'Cross-Validation Accuracy':cvs,'Precision':precs,'F1 score':f1s,'Recall':recalls})
print(df)

 


# In[9]:


def model_performance(train_data,train_label,test_data,test_label,y_pred,clf):
    acc=accuracy(test_label,y_pred)
    cv=cross_validate(clf, train_data, train_label,cv=10)
    prec=precision_score(test_label,y_pred)
    recall=recall_score(test_label,y_pred)
    #f1=f1_score(test_label, y_pred)
    f1 = 2 * (prec * recall) / (prec + recall)
    #aucc=auc(clf,test_data)
    return acc,cv,prec,recall,f1


# In[37]:


## PCA
import matplotlib.pyplot as plt
accs=[]
cvs=[]
precs=[]
recalls=[]
f1s=[]
aucs=[]
component_num=[10,20,30,40,60,80]
for i in component_num:
    start_time = time.time()
    #Xnew=principleCA(X,i)
    newXtrain,newXtest=principleCA(train_data_under,test_data_under,i)
    clf=svm.SVC(kernel='rbf')
    clf.fit(newXtrain,train_label_under)
    y_pred=clf.predict(newXtest)
    acc=accuracy(test_label_under,y_pred)
    cv=cross_validate(clf, newXtrain, train_label_under,cv=10)
    prec=precision_score(test_label_under,y_pred)
    recall=recall_score(test_label_under,y_pred)
    f1=2 * (prec * recall) / (prec + recall)
    #aucc=auc(clf,newXtest,test_label_under)
    accs.append(acc)
    cvs.append(cv)
    precs.append(prec)
    recalls.append(recall)
    f1s.append(f1)

    print("--- %s seconds ---" % (time.time() - start_time))
#print('The accuracy of the SVM with PCA feature selection is:', acc[0])
print('The performace of SVM with PCA feature selection is:')
df=pd.DataFrame({'PCs':component_num,'Test Accuracy':accs,'Cross-Validation Accuracy':cvs,'Precision':precs,'F1 score':f1s,'Recall':recalls})
print(df)
plt.plot(component_num,accs)
plt.title('The accuray of SVM using different number of components ')
plt.show()
plt.plot(component_num,recalls)
plt.title('The Recall of SVM using different number of components ')
plt.show()


#------Plot explained variance------
c_list=list(range(1,81))
clf=PCA(n_components=80)
clf.fit_transform(train_data_under)
variance=clf.explained_variance_ratio_.cumsum()
plt.plot(c_list,variance)
plt.show()


# In[11]:


## Analyze its importance
#Select ten top important features
from sklearn import svm

train_data_under=pd.DataFrame(train_data_under)
test_data_under=pd.DataFrame(test_data_under)

accs=[]
cvs=[]
precs=[]
recalls=[]
f1s=[]

idx=[10,20,30,40,60]
for i in idx:
    Xnew=train_data_under.iloc[:,svm_feature.index[0:i]]
    clf=svm.SVC(kernel='rbf')
    clf.fit(Xnew,train_label_under)
    X_test_new=test_data_under.iloc[:,svm_feature.index[0:i]]
    y_pred=clf.predict(X_test_new)
    acc=accuracy(test_label_under,y_pred)
    cv=cross_validate(clf, Xnew, train_label_under,cv=10)
    prec=precision_score(test_label_under,y_pred)
    recall=recall_score(test_label_under,y_pred)
    f1=2 * (prec * recall) / (prec + recall)
    
    accs.append(acc)
    cvs.append(cv)
    precs.append(prec)
    recalls.append(recall)
    f1s.append(f1)
    #aucs.append(aucc)


#print('The performance of SVM with the most important '+str(i)+' feature is:',acc)
df=pd.DataFrame({'Feature num ':idx,'Test Accuracy':accs,'Cross-Validation Accuracy':cvs,'Precision':precs,'F1 score':f1s,'Recall':recalls})
print(df)
    

plt.plot(idx,accs)
plt.title('The accuray of SVM using different number of components ')
plt.show()

plt.plot(idx,recalls)
plt.title('The F1 of SVM using different number of components ')
plt.show()

#print('Cross validation is:',acc[1])


# ## KNN

# In[31]:



def knnacc(train_data,train_label,test_data,test_label):


    clf=KNeighborsClassifier(n_neighbors=10)
    clf.fit(train_data,train_label)
    y_pred=clf.predict(test_data)
   # acc=accuracy(test_label,y_pred)
   # cv=cross_validate(clf, train_data, train_label,cv=10)
    acc,cv,prec,recall,f1=model_performance(train_data,train_label,test_data,test_label,y_pred)
    return acc,cv,prec,recall,f1

acc,cv,prec,recall,f1=knnacc(train_data_smote,train_label_smote,test_data_smote,test_label_smote)
acc1,cv1,prec1,recall1,f11=knnacc(train_data_no,train_label_no,test_data_no,test_label_no)
acc2,cv2,prec2,recall2,f12=knnacc(train_data_under,train_label_under,test_data_under,test_label_under)
accs=[acc,acc1,acc2]
cvs=[cv,cv1,cv2]
precs=[prec,prec1,prec2]
f1_score=[f1,f11,f12]
recalls=[recall,recall1,recall2]
#auccs=[aucc,aucc1,aucc2]
Name=['Dataset using smote oversampling','Dataset without oversampling','Dataset with undersampling']
df=pd.DataFrame({'Name':Name,'Test Accuracy':accs,'Cross-Validation Accuracy':cvs,'Precision':precs,'F1 score':f1_score,'Recall':recalls})
print(df)


#print('Cross validation is:',acc[1])

#-----Feature selection--------


# In[19]:


import matplotlib.pyplot as plt

## PCA

accs=[]
cvs=[]
precs=[]
recalls=[]
f1s=[]

component_num=[10,20,30,40,60,80,100,130]
for i in component_num:
    start_time = time.time()
    newXtrain,newXtest=principleCA(train_data_smote,test_data_smote,i)
    clf=KNeighborsClassifier(n_neighbors=10)
    clf.fit(newXtrain,train_label_smote)
    y_pred=clf.predict(newXtest)
    #acc=accuracy(test_label_smote,y_pred)
    acc,cv,prec,recall,f1=model_performance(newXtrain,train_label_smote,newXtest,test_label_smote,y_pred,clf)
    accs.append(acc)
    cvs.append(cv)
    precs.append(prec)
    recalls.append(recall)
    f1s.append(f1)

    print("--- %s seconds ---" % (time.time() - start_time))

print('The performace of KNN with PCA feature selection is:')
df=pd.DataFrame({'PCs':component_num,'Test Accuracy':accs,'Cross-Validation Accuracy':cvs,'Precision':precs,'F1 score':f1s,'Recall':recalls})
print(df)
plt.plot(component_num,accs)
plt.title('The accuray of KNN using different number of components ')
plt.show()
plt.plot(component_num,recalls)
plt.title('The Recall of KNN using different number of components ')
plt.show()



# In[81]:


#select important feature for each model
#one hot encoding


# ## Logistic Regression

# In[40]:



from sklearn.linear_model import LogisticRegression


def logisacc(train_data,train_label,test_data,test_label):
    start_time = time.time()
    
    clf = LogisticRegression(solver='lbfgs')
    clf.fit(train_data,train_label)
    y_pred=clf.predict(test_data)
    acc,cv,prec,recall,f1=model_performance(train_data,train_label,test_data,test_label,y_pred,clf)
    #aucc=auc(clf,test_data)
    #acc,cv,prec,recall,f1,aucc=model_performance(train_data,train_label,test_data,test_label,clf,y_pred)
    print("--- %s seconds ---" % (time.time() - start_time))
    return acc,cv,prec,recall,f1
acc,cv,prec,recall,f1=logisacc(train_data_smote,train_label_smote,test_data_smote,test_label_smote)

acc1,cv1,prec1,recall1,f11=logisacc(train_data_no,train_label_no,test_data_no,test_label_no)
acc2,cv2,prec2,recall2,f12=logisacc(train_data_under,train_label_under,test_data_under,test_label_under)

accs=[acc,acc1,acc2]
cvs=[cv,cv1,cv2]
precs=[prec,prec1,prec2]
f1_score=[f1,f11,f12]
recalls=[recall,recall1,recall2]
#auccs=[aucc,aucc1,aucc2]
Name=['Dataset using smote oversampling','Dataset without oversampling','Dataset with undersampling']
df=pd.DataFrame({'Name':Name,'Test Accuracy':accs,'Cross-Validation Accuracy':cvs,'Precision':precs,'F1 score':f1_score,'Recall':recalls})
print(df)


# In[18]:


#---------PCA------------

## PCA

accs=[]
cvs=[]
precs=[]
recalls=[]
f1s=[]

component_num=[10,20,30,40,60,80,100,130]
for i in component_num:
    start_time = time.time()
    newXtrain,newXtest=principleCA(train_data_smote,test_data_smote,i)
    clf=LogisticRegression(solver='lbfgs')
    clf.fit(newXtrain,train_label_smote)
    y_pred=clf.predict(newXtest)
    #acc=accuracy(test_label_smote,y_pred)
    acc,cv,prec,recall,f1=model_performance(newXtrain,train_label_smote,newXtest,test_label_smote,y_pred,clf)
    accs.append(acc)
    cvs.append(cv)
    precs.append(prec)
    recalls.append(recall)
    f1s.append(f1)

    print("--- %s seconds ---" % (time.time() - start_time))

print('The performace of KNN with PCA feature selection is:')
df=pd.DataFrame({'PCs':component_num,'Test Accuracy':accs,'Cross-Validation Accuracy':cvs,'Precision':precs,'F1 score':f1s,'Recall':recalls})
print(df)
plt.plot(component_num,accs)
plt.title('The accuray of Logistic Regression using different number of components ')
plt.show()
plt.plot(component_num,recalls)
plt.title('The Recall of Logistic Regression using different number of components ')
plt.show()




# In[46]:


#------Plot explained variance------
c_list=list(range(1,len(train_data_smote[1,:])+1))
clf=PCA(n_components=len(train_data_smote[1,:]))
clf.fit_transform(train_data_smote)
variance=clf.explained_variance_ratio_.cumsum()
plt.plot(c_list,variance)
plt.show()


# # Random Forest

# In[48]:


#-------Random Forest--------------------
from sklearn.ensemble import RandomForestClassifier

def randomacc(train_data,train_label,test_data,test_label):
    start_time = time.time()
    
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(train_data,train_label)
    y_pred=clf.predict(test_data)
    acc,cv,prec,recall,f1=model_performance(train_data,train_label,test_data,test_label,y_pred,clf)
    #aucc=auc(clf,test_data)
    #acc,cv,prec,recall,f1,aucc=model_performance(train_data,train_label,test_data,test_label,clf,y_pred)
    print("--- %s seconds ---" % (time.time() - start_time))
    return acc,cv,prec,recall,f1
acc,cv,prec,recall,f1=logisacc(train_data_smote,train_label_smote,test_data_smote,test_label_smote)

acc1,cv1,prec1,recall1,f11=randomacc(train_data_no,train_label_no,test_data_no,test_label_no)
acc2,cv2,prec2,recall2,f12=randomacc(train_data_under,train_label_under,test_data_under,test_label_under)

accs=[acc,acc1,acc2]
cvs=[cv,cv1,cv2]
precs=[prec,prec1,prec2]
f1_score=[f1,f11,f12]
recalls=[recall,recall1,recall2]
#auccs=[aucc,aucc1,aucc2]
Name=['Dataset using smote oversampling','Dataset without oversampling','Dataset with undersampling']
df=pd.DataFrame({'Name':Name,'Test Accuracy':accs,'Cross-Validation Accuracy':cvs,'Precision':precs,'F1 score':f1_score,'Recall':recalls})
print(df)


# In[14]:


# Feature selection-random forest

rf = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)


#df.rename(columns=list(column_name))
#print(df)

#feature=list(X.columns)
rf.fit(train_data_smote,train_label_smote)
featimportance=rf.feature_importances_
#print(len(featur),len(featimportance))


#feat=pd.DataFrame({'feature':features_names[0:10],'importance':featimportance[0:10]})
rf_feat=f_importances(featimportance, features_names_smote)


# In[15]:


# Feature selection using feature importance
train_data_smote=pd.DataFrame(train_data_smote)
test_data_smote=pd.DataFrame(test_data_smote)

accs=[]
cvs=[]
precs=[]
recalls=[]
f1s=[]

idx=[10,20,30,40,60,80,100]
for i in idx:
    Xnew=train_data_smote.iloc[:,rf_feat.index[0:i]]
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(Xnew,train_label_smote)
    X_test_new=test_data_smote.iloc[:,rf_feat.index[0:i]]
    
    y_pred=clf.predict(X_test_new)
    
    acc,cv,prec,recall,f1=model_performance(Xnew,train_label_smote,X_test_new,test_label_smote,y_pred,clf)
    accs.append(acc)
    cvs.append(cv)
    precs.append(prec)
    recalls.append(recall)
    f1s.append(f1)
    #aucs.append(aucc)


#print('The performance of SVM with the most important '+str(i)+' feature is:',acc)
df=pd.DataFrame({'Feature num ':idx,'Test Accuracy':accs,'Cross-Validation Accuracy':cvs,'Precision':precs,'F1 score':f1s,'Recall':recalls})
print(df)
    

plt.plot(idx,accs)
plt.title('The accuray of RF using different most important features')
plt.show()

plt.plot(idx,recalls)
plt.title('The Recall of RF using different most important features')
plt.show()



# In[22]:


#Nerual Network


def nn(train_data,train_label,test_data,test_label):


    clf=MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
    clf.fit(train_data,train_label)
    y_pred=clf.predict(test_data)
   # acc=accuracy(test_label,y_pred)
   # cv=cross_validate(clf, train_data, train_label,cv=10)
    acc,cv,prec,recall,f1=model_performance(train_data,train_label,test_data,test_label,y_pred,clf)
    return acc,cv,prec,recall,f1

acc,cv,prec,recall,f1=nn(train_data_smote,train_label_smote,test_data_smote,test_label_smote)
acc1,cv1,prec1,recall1,f11=nn(train_data_no,train_label_no,test_data_no,test_label_no)
acc2,cv2,prec2,recall2,f12=nn(train_data_under,train_label_under,test_data_under,test_label_under)
accs=[acc,acc1,acc2]
cvs=[cv,cv1,cv2]
precs=[prec,prec1,prec2]
f1_score=[f1,f11,f12]
recalls=[recall,recall1,recall2]
#auccs=[aucc,aucc1,aucc2]
Name=['Dataset using smote oversampling','Dataset without oversampling','Dataset with undersampling']
df=pd.DataFrame({'Name':Name,'Test Accuracy':accs,'Cross-Validation Accuracy':cvs,'Precision':precs,'F1 score':f1_score,'Recall':recalls})
print(df)


# In[17]:


#------Lightgbm---------
# Implement lightgbm
import lightgbm as lgb
from sklearn.model_selection import train_test_split
feature_importances=np.zeros(len(features_names_under))
# Feature selection - light GBM
model=lgb.LGBMClassifier(objective='binary', boosting_type = 'goss', n_estimators = 10000, class_weight = 'balanced')
for i in range(2):
    
    # Train using early stopping
    model.fit(train_data_under, train_label_under, early_stopping_rounds=100, eval_set = [(test_data_under, test_data_under)], 
              eval_metric = 'auc', verbose = 200)
    
    # Record the feature importances
    feature_importances += model.feature_importances_
# Make sure to average feature importances! 
feature_importances = feature_importances / 2
feature_importances = pd.DataFrame({'feature': features_names, 'importance': feature_importances}).sort_values('importance', ascending = False)

print(feature_importances)


# In[13]:


# Compare the accuracy performance

model_type=['SVM','KNN','LR','RF','Neural Network']
acc_over=[0.598,0.43,0.56,0.56,0.53]
acc_under=[0.566,0.6258,0.57,0.68,0.538]
recall_over=[0.6487,0.8325,0.795,0.795,0.62]
recall_under=[0.7968,0.528,0.75,0.624,0.75]
cv_over=[0.876,0.6939,0.7,0.69,0.76]
cv_under=[0.668,0.58,0.67,0.67,0.63]

f1_over=[0.19,0.156,0.206,0.20,0.15]
f1_under=[0.21,0.174,0.20,0.22,0.192]
x=np.array([0,1,2,3,4])
plt.xticks(x, model_type)
plt.plot(x, acc_over,label='Smote oversampled')
plt.plot(x,acc_under,label='Undersampled')
plt.legend()
plt.title('Acuracy comparison among different modeling methods')
plt.show()

#----------------------
x=np.array([0,1,2,3,4])
plt.xticks(x, model_type)
plt.plot(x, cv_over,label='Smote oversampled')
plt.plot(x,cv_under,label='Undersampled')
plt.legend()
plt.title('Cross-validation comparison among different modeling methods')
plt.show()
#-----------------------
x=np.array([0,1,2,3,4])
plt.xticks(x, model_type)
plt.plot(x, recall_over,label='Smote oversampled')
plt.plot(x,recall_under,label='Undersampled')
plt.legend()
plt.title('Recall comparison among different modeling methods')
plt.show()

#-----------------------
x=np.array([0,1,2,3,4])
plt.xticks(x, model_type)
plt.plot(x, f1_over,label='Smote oversampled')
plt.plot(x,f1_under,label='Undersampled')
plt.legend()
plt.title('F1 score comparison among different modeling methods')
plt.show()


