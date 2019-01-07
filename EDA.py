
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import pandas_profiling as pp
df=pd.read_csv('application_train.csv')


data=df.drop(columns=['TARGET'])
label=df['TARGET']


# In[2]:


print(df.shape)


# In[14]:


print(df.shape)
import numpy as np
unique, counts = np.unique(df['TARGET'], return_counts=True)

y_pos = np.arange(len(unique))
plt.bar(y_pos, counts, align='center', alpha=0.5)
plt.xticks(y_pos, unique)
plt.ylabel('Amount of data')
plt.title('The data distribution of different labels')
 
plt.show()


# In[20]:


# Find the distribution of TARGET column
print(df['TARGET'].value_counts())
df['TARGET'].astype(int).plot.hist()

# We can find that it is a imbalanced dataset


# In[3]:


# Find the missing values
def missing_value(df):
    mis=df.isnull().sum()
    mis_percen=df.isnull().sum()/len(df)
    table=pd.DataFrame({'Missing value counts':mis,'Missing value percentage':mis_percen})
    table=table.sort_values('Missing value counts',ascending=False)
    return table

miss=missing_value(df)
miss_column=miss[miss['Missing value percentage']>0].index
print(len(miss_column))
#df=df.drop(columns=miss_column)
missing_value(df)


# In[10]:



#row_zero=df.loc[df['TARGET'] == 0]
#row_one=df.loc[df['TARGET'] == 1]

miss_row=df.isnull().sum(axis=1)

#miss_row_one=row_one.isnull().sum(axis=1)

print('This is the histogram of missing data percentage for the whole dataset:')
plt.hist(miss_row)
plt.show()


# In[2]:


# find the rows with missing values belong to which class
inds = pd.isnull(df).any(1).nonzero()[0]
print(len(inds))
oneclass=[]
zeroclass=[]
for i in inds:
    if df['TARGET'][i]==1:
        oneclass.append(i)
    else:
        zeroclass.append(i)
print(len(oneclass))
print(len(zeroclass))
df=oneclass.append(zerocalss)


# In[16]:


# Overveiw
pp.ProfileReport(df)


# In[6]:


# Number of each data type in dataset
df.dtypes.value_counts()


# In[12]:


#find correlation
correlations = df.corr()['TARGET'].sort_values(ascending=False)
print('Most negative correlations:',correlations.tail(15))
print('Most positive correlations:',correlations.head(15))

 


# In[9]:


df['DAYS_EMPLOYED'].describe()
#plt.hist(df['DAYS_EMPLOYED'],bins='auto')

#plt.show()


# In[ ]:


# Detect outlier
# Classify the feature into catergories(common sense)
# missing value maybe drop the feature has nothing to with classify the 10% label
# different model may have different feature selection


# In[6]:


print(df.shape)


# In[18]:


import seaborn as sns
import pandas as pd
import numpy as np
df['DAYS_BIRTH']=abs(df['DAYS_BIRTH'])
plt.figure
sns.kdeplot(df.loc[df['TARGET']==0,'DAYS_BIRTH']/365,label='target==0')
sns.kdeplot(df.loc[df['TARGET']==1,'DAYS_BIRTH']/365,label='target==1')
plt.xlabel('Age(years)')
plt.ylabel('Density')
plt.title('Distribution of Ages')
plt.show()

df_age=df[['TARGET','DAYS_BIRTH']]
df_age['YEAR_BIRTH']=df_age['DAYS_BIRTH']/365

df_age['YEARS_BINNED']=pd.cut(df_age['YEAR_BIRTH'],bins=np.linspace(20,70,num=11))
df_age.head(10)


# In[19]:


age_groups=df_age.groupby('YEARS_BINNED').mean()
print(age_groups)


# In[20]:


plt.plot(age_groups['YEAR_BIRTH'],age_groups['TARGET'])
plt.xlabel('Age')
plt.ylabel('Average_Repay')
plt.show()


# In[28]:


# Implement lightgbm
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

a=feature_importances.head()
x=np.array(range(len(a)))
my_xticks = a['feature']
plt.xticks(x, my_xticks)
plt.plot(x, a['importance'])
plt.show()


# In[22]:


zero_importance=[]
for i in range(len(feature_importances)):
    if feature_importances['importance'][i]==0:
        zero_importance.append(feature_importances['feature'][i])
data=data.drop(columns=zero_importance)  
print(data.shape)


# In[67]:


missing_value(data)

