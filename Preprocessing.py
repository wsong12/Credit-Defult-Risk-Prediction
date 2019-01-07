
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import pandas_profiling as pp
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
import numpy as np


# In[13]:


df=pd.read_csv('application_train.csv')


# ## Find the distribution of TARGET column

# In[4]:



print(df['TARGET'].value_counts())
df['TARGET'].astype(int).plot.hist()

print('The shape of our original dataset is:',df.shape)


# ## Find the missing values

# In[14]:



def missing_value(df):
    mis=df.isnull().sum()
    mis_percen=df.isnull().sum()/len(df)
    table=pd.DataFrame({'Missing value counts':mis,'Missing value percentage':mis_percen})
    table=table.sort_values('Missing value counts',ascending=False)
    return table


# ## find the rows with missing values belong to which class

# In[10]:



inds = pd.isnull(df).any(1).nonzero()[0]
print('The number of the data with missing values is:', len(inds))
oneclass=[]
zeroclass=[]
for i in inds:
    if df['TARGET'][i]==1:
        oneclass.append(i)
    else:
        zeroclass.append(i)
print('The number of the data with missing values that belong to class one is:',len(oneclass))
print('The number of the data with missing values that belong to class zero is:',len(zeroclass))


# ## Drop the feature with missing value greater than 0.6

# In[15]:



miss=missing_value(df)
miss_column=miss[miss['Missing value percentage']>0.6].index
print('The number of the feature with the missing value percentage greater than 0.6 is:',len(miss_column))
df=df.drop(columns=miss_column)
missing_value(df)


# ## Delete rows which have high missing data percentage

# In[16]:



import matplotlib.pyplot as plt


row_zero=df.loc[df['TARGET'] == 0]
row_one=df.loc[df['TARGET'] == 1]

miss_row_zero=row_zero.isnull().sum(axis=1)

miss_row_one=row_one.isnull().sum(axis=1)

print('This is the histogram of missing data percentage for label zero:')
plt.hist(miss_row_zero)
plt.show()
print('This is the histogram of missing data percentage for label one:')
plt.hist(miss_row_one)
plt.show()

miss_one_selected=miss_row_one[miss_row_one.values<30]

miss_zero_selected=miss_row_zero[miss_row_zero.values<30]
print('This is the histogram of missing data percentage for label one we selected:')
plt.hist(miss_one_selected)
plt.show()
print('The number of data is:',len(miss_one_selected))
print('This is the histogram of missing data percentage for label zero we selected:')
plt.hist(miss_zero_selected)
plt.show()
print('The number of data is:',len(miss_zero_selected))
row_selected=[]

#row_selected=[]
#row_selected.append(miss_zero_selected.index)
#row_selected.append(miss_one_selected.index)
row_selected=miss_zero_selected.index.union(miss_one_selected.index)
print(row_selected)
#row_selected
df=df.iloc[row_selected,:]
print(len(df))


# In[12]:


print(df.shape)


# In[60]:


missing_value(df)


# ## Data encoding

# In[12]:



#If we only have two classes, label coding will be fine
#If we have more than two classes, one hot encoding will be safe
onehot=[]
encode=[]
def encoding(df):
    a=0
    one_hot=0
    LE=LabelEncoder()

    for col in df:
    #print(df[col].dtype)
        if df[col].dtype=='object':
        #a+=1
    # label encoder
            if len(list(df[col].unique()))<=2:
       
                LE.fit(df[col])
                df[col]=LE.transform(df[col])
                a+=1
                encode.append(col)
            else:
                one_hot+=1 
                onehot.append(col)
       # elif df[col].dtype=='int64': 
        #b+=1
#print('The number of features that needs labelEncoder is:',le)
#print('The number of features that needs one_hot_coding is:',one_hot)

    # one-hot encoding
    df=pd.get_dummies(df)
    print('There are '+str(a)+' features using label encoding')
    print('There are '+str(one_hot)+' features using one hot encoding')
    return df

#column_name=df.columns
df=encoding(df)

#print(columns_af_enconding)
print('Training data size:',df.shape)
print(encode)
print(onehot)


# In[62]:


print(df)


# ## Impute missing value

# In[24]:




#from sklearn.impute import SimpleImputer
imputer = Imputer(strategy = 'median')
num_col=[]

for col in df:
    if df[col].dtype!='object':
        num_col.append(col)
   
          
imputer.fit(df[num_col])

df[num_col]=imputer.transform(df[num_col])

#df[str_col]=df[str_col].fillna(df[str_col].mode().iloc[0].index[0])

print(missing_value(df))


# In[25]:


df1=df.drop(columns=['TARGET'])
feats=df1.columns
print(feats)


# ## Undersampling

# In[32]:




from sklearn.utils import resample

X=df.drop(columns=['TARGET'])

y=df['TARGET']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(len(X_train))
print(len(X_test))
#df=df.iloc[row_selected,:]
X_train=pd.DataFrame(X_train,columns=X.columns)
y_train=pd.DataFrame(y_train,columns=['TARGET'])
train= pd.concat((X_train, y_train), axis = 1)

# Separate majority and minority classes
df_majority = train[train['TARGET']==0]
df_minority = train[train['TARGET']==1]
print(len(df_majority))
print(len(df_minority))
# Downsample majority class
df_majority_underampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=len(df_minority),     # to match minority class
                                 random_state=123) # reproducible results
 
# Combine minority class with downsampled majority class
train = pd.concat([df_majority_underampled, df_minority])
X_train_new=train.drop(columns=['TARGET'])
y_train_new=train['TARGET']
 
# Display new class counts
#train['TARGET'].value_counts()
unique, counts = np.unique(y_train_new, return_counts=True)

y_pos = np.arange(len(unique))

 
plt.bar(y_pos, counts, align='center', alpha=0.5)
plt.xticks(y_pos, unique)
plt.ylabel('Amount of data')
plt.title('The data label distribution after undersampling')
 
plt.show()


# In[31]:


unique, counts = np.unique(y, return_counts=True)

y_pos = np.arange(len(unique))

 
plt.bar(y_pos, counts, align='center', alpha=0.5)
plt.xticks(y_pos, unique)
plt.ylabel('Amount of data')
plt.title('The data label distribution before sampling')
 
plt.show()


# ## SMOTE sampling

# In[19]:




X=df.drop(columns=['TARGET'])
y=df['TARGET']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
sm = SMOTE(random_state=12, ratio = 1.0)
X_train_new, y_train_new = sm.fit_sample(X_train, y_train)

unique, counts = np.unique(y_train_new, return_counts=True)

y_pos = np.arange(len(unique))

 
plt.bar(y_pos, counts, align='center', alpha=0.5)
plt.xticks(y_pos, unique)
plt.ylabel('Amount of data')
plt.title('The data label distribution after SMOTE oversampling')
 
plt.show()


# In[20]:


X_train_new=pd.DataFrame(X_train_new,columns=feats)
X_test_new=pd.DataFrame(X_test,columns=feats)

y_train_new=pd.DataFrame(y_train_new,columns=['TARGET'])

y_test_new=pd.DataFrame(y_test,columns=['TARGET'])


# In[223]:


print(y_train_new)
print(X_train_new.shape)


# ## Removed collinear features

# In[36]:


#Removed collinear features as measured by the correlation coefficient greater than 0.9
#Removed any columns with greater than 80% missing values in the train or test set
#Removed all features with non-zero feature importances
import numpy as np
import pandas as p
# step 1:remove collinear features
label=y_train_new
data=X_train_new
threshold=0.9
corr_matrix=data.corr().abs()
corr_matrix.head()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))


to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
other_cols=[]
for i in to_drop:
    
    other_col=upper[upper[i]>threshold]
    other_col=list(other_col.index)
    other_cols.append(other_col)

print('There are %d columns to remove.' % (len(to_drop)))

data=data.drop(columns=to_drop)
print('The columns we dropped are:',to_drop)
print('The columns we reserved are:',other_cols)
print(data.shape)


# In[5]:


print(data.shape)


# ## Implement lighgbm to emilating no importance features

# In[37]:



import lightgbm as lgb
from sklearn.model_selection import train_test_split
feature_importances=np.zeros(len(data.columns))
# Feature selection - light GBM
model=lgb.LGBMClassifier(objective='binary', boosting_type = 'goss', n_estimators = 10000, class_weight = 'balanced')
for i in range(2):
    train_features, valid_features, train_y, valid_y = train_test_split(data,label, test_size = 0.25, random_state = i)
    # Train using early stopping
    model.fit(train_features, train_y, early_stopping_rounds=100, eval_set = [(valid_features, valid_y)], 
              eval_metric = 'auc', verbose = 200)
    
    # Record the feature importances
    feature_importances += model.feature_importances_
# Make sure to average feature importances! 
feature_importances = feature_importances / 2
feature_importances = pd.DataFrame({'feature': list(data.columns), 'importance': feature_importances}).sort_values('importance', ascending = False)

feature_importances.head()

zero_importance=[]
for i in range(len(feature_importances)):
    if feature_importances['importance'][i]==0:
        zero_importance.append(feature_importances['feature'][i])
print(zero_importance)
data=data.drop(columns=zero_importance)  
print(data.shape)


# In[38]:


print(len(zero_importance))


# ## Add domain knowledge features

# In[226]:




data['CREDIT_INCOME_PERCENT']=data['AMT_CREDIT']/data['AMT_INCOME_TOTAL']
X_test_new['CREDIT_INCOME_PERCENT']=X_test_new['AMT_CREDIT']/X_test_new['AMT_INCOME_TOTAL']
data['ANNUITY_INCOME_PERCENT']=data['AMT_ANNUITY']/data['AMT_INCOME_TOTAL']
X_test_new['ANNUITY_INCOME_PERCENT']=X_test_new['AMT_ANNUITY']/X_test_new['AMT_INCOME_TOTAL']
data['CREDIT_TERM']=data['AMT_ANNUITY']/data['AMT_CREDIT']

X_test_new['CREDIT_TERM']=X_test_new['AMT_ANNUITY']/X_test_new['AMT_CREDIT']
data['DAYS_EMPLOYED_PERCENT']=data['DAYS_EMPLOYED']/data['DAYS_BIRTH']
X_test_new['DAYS_EMPLOYED_PERCENT']=X_test_new['DAYS_EMPLOYED']/X_test_new['DAYS_BIRTH']
print(data.shape)


# In[201]:


print(label.value_counts())


# In[227]:



selected_col=list(data.columns)

X_test_select=X_test_new[selected_col]


# In[229]:


#print(df.shape)


finaltest = pd.concat((X_test_select, y_test_new), axis = 1)
#print(finaltest.shape)
finaltrain.to_csv('Cleaned_train_undersampling.csv')
finaltest.to_csv('Cleaned_test_undersampling.csv')

