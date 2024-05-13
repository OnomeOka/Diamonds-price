#!/usr/bin/env python
# coding: utf-8

# In[70]:


pip install lightgbm


# In[71]:


pip install optuna


# In[72]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import optuna
import warnings
warnings.filterwarnings('ignore')


# In[73]:


df = pd.read_csv('diamond.csv')


# In[74]:


df.head()


# In[75]:


# checking for missing value
df.isnull().sum()


# In[76]:


df.describe()


# In[77]:


# checking for any duplicate
df.duplicated()


# In[78]:


# to see the culumns names
df.columns


# In[79]:


print(df.columns)


# In[80]:


df.columns = df.columns.str.replace('_', '-')


# In[81]:


print(df.columns)


# In[82]:


df.columns = df.columns.str.replace(r'\W', '')


# In[83]:


df.loc[df['x']==0, 'x'] = df.loc[df['x']==0, 'carat']*(df['x']/df['carat']).mean()
df.loc[df['y']==0, 'y'] = df.loc[df['y']==0, 'carat']*(df['y']/df['carat']).mean()
df.loc[df['z']==0, 'z'] = df.loc[df['z']==0, 'carat']*(df['z']/df['carat']).mean()


# In[84]:


df.hist(figsize=[10,10])
plt.tight_layout()


# In[85]:


selected_df = df[df['price']<15000]


# In[86]:


# it divide the data into two part categorical and numerical
cat_columns = selected_df.select_dtypes(include='object').columns.tolist()
num_columns = selected_df.select_dtypes(include='number').columns.tolist()
num_columns.remove('price')


# In[87]:


# creat a boxplot and violinplot for categorical columns
for col in cat_columns:
    plt.figure(figsize=[10,5])
    
    plt.subplot(1,2,1)
    sns.boxplot(x=col, y='price', data = selected_df)
    plt.title(f'violinplot for price v {col}')
    
    plt.subplot(1,2,2)
    sns.violinplot(x=col, y='price', data = selected_df)
    plt.title(f'violinplot for price v {col}')
    
    plt.tight_layout()
    plt.show()


# In[88]:


#using scatter plot to see relationship between price and other numerical features
for col in num_columns:
    plt.figure(figsize=[8,8])
    sns.scatterplot(x=col, y='price', data = selected_df)
    plt.show()


# In[89]:


#correlation between each numerical column and price
df[num_columns].corrwith(df['price'])


# In[90]:


# convert the categorical viariable to binary 
df = pd.concat([df, pd.get_dummies(df[['cut', 'color', 'clarity']])], axis=1)
df.drop(['cut', 'color', 'clarity'], axis=1, inplace=True)
X =df.copy()
Y =X.pop('price')


# In[95]:


import matplotlib.pyplot as plt
import seaborn as sns

def model_evaluator(model):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=42)
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(Y_test, pred)
    rmse = mean_squared_error(Y_test, pred, squared=False)
    
    print(f"Mean absolute error is {mae} and Root mean squared error is {rmse}.")
    
    # Plotting scatter plot of predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=pred, y=Y_test)
    plt.xlabel('Predictions')  # Corrected xlabel method
    plt.ylabel('Actual')       # Corrected ylabel method
    plt.title('Scatter of Actual vs Predictions')
    
    plt.subplot(1,2,2)
    ax = sns.histplot(pred, color='r', kde=True, label='Predictions')
    sns.histplot(Y_test, color='b', kde=True, label='Actual', ax = ax)
    plt.title('Distribution plot')

# Assuming X, Y are defined elsewhere
lgb_model = lgb.LGBMRegressor(verbose=0)
model_evaluator(lgb_model)


# In[99]:


# Convert 'z' column to numeric
X['z'] = pd.to_numeric(X['z'], errors='coerce')


# In[102]:


X['volume'] = X['x']*X['y']*X['z']


# In[104]:


model_evaluator(lgb_model)


# In[106]:


X['x_by_carat'] = X['x']/X['carat']
X['y_by_carat'] = X['y']/X['carat']

model_evaluator(lgb_model)


# In[114]:


best_params = {'num_leaves' : 113, 'learning_rate': 0.09392128552646012, 'feature_fraction': 0.8576073392999497, 
               'bagging_fraction': 0.9438230713459743, 'bagging_freq': 8, 'min_child_samples': 5}


# In[117]:


new_model = lgb.LGBMRegressor(**best_params, verbosity = -1)
model_evaluator(new_model)


# In[ ]:




