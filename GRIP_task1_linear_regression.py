#!/usr/bin/env python
# coding: utf-8

# # Maria David

# ### Task-1 Prediction Using supervised ML

# ### Simple Linear Regression

# -----

# ##### Introduction

# Simple Linear Regression is a linear model which depicts the relationship between the independent variable (x) and the dependent variable (y).
# 
# The simple linear regression model is y = B0 + B1*x 
# 
# where, y is the dependent variable
# 
# x is the independent variable
# 
# B0 and B1 are the coefficients

# ---

# ##### Objective

# The below analysis aims to predict the percentage of an student based on the no.of study hours using python.

# ----

# ##### About the dataset

# The dataset contains information about the study hours and the corresponding scores in percentage. It contains two variables "Hours" and "Scores".

# ---

# ##### Step 1:Importing the libraries

# In[22]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
from sklearn import metrics  


# ##### Step 2:Loading the dataset

# In[2]:


url = "http://bit.ly/w-data"
scores_data = pd.read_csv(url)


# ##### Step 3:Understanding the data

# In[3]:


scores_data.head()


# In[4]:


scores_data.shape


# There are 25 observations and 2 columns in the dataset.

# In[5]:


scores_data.describe()


# The average study hours of the 25 students is approximately 5 hours and their average scores is around 51%.

# In[6]:


scores_data.info()


# ##### Step 4:Data Cleaning

# In[7]:


#a.Checking for null values
scores_data.isnull().sum()


# There are no null values in the dataset.

# In[8]:


#b.Checking for outliers


# In[9]:


scores_data.plot.box(title="Boxplot of all columns",figsize=(10,8))


# There are no outliers in the dataset

# ##### Step 5:Exploratory data anlaysis

# In[10]:


#Understanding the distribution

plt.rcParams['figure.figsize'] = (16, 10)
plt.subplot(2, 2, 1)
sns.set(style = 'whitegrid')
sns.distplot(scores_data['Hours'])
plt.title('Distribution of Study Hours', fontsize = 20)
plt.xlabel('Range of study hours')
plt.ylabel('Count')

plt.subplot(2, 2, 2)
sns.set(style = 'whitegrid')
sns.distplot(scores_data['Scores'], color = 'red')
plt.title('Distribution of Scores', fontsize = 20)
plt.xlabel('Percentage of Scores')
plt.ylabel('Count')

plt.show()


# Both study hours and scores follows normal distribution. Study hours of the students fall between the range 1 to 9.And scores are in the range 20% to 95%. Most students study for 2 to 4 hours and most students scores ranges from 20% to 40%.

# In[11]:


# Plotting the distribution of scores
scores_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# There is a very strong positive relationship between percentage of Scores and Study Hours.

# ##### Step 6:Data preprocessing

# In[12]:


#Dividing the dependent and independent variables into y and X .y(score percentage) is the dependent variable and X (study hours) is the independent variable
X = scores_data.iloc[:, :-1].values  
y = scores_data.iloc[:, 1].values  


# In[13]:


#Dividing the dataset to training data and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# ##### Step 7:Training the algorithm

# In[14]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train) 


# In[15]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.plot(X, line);
plt.show()


# ##### Step 8:Making predictions

# In[16]:


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# In[17]:


# Comparing Actual vs Predicted and storing in df
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# ##### What will be predicted score if a student studies for 9.25 hrs/day?

# In[21]:


hours = [[9.25]]
own_pred = regressor.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# When a student studies for approximately 9 hours, the score percentage is approximately 93%.

# ##### Step 9:Evaluating the model using mean absolute error

# Mean Absolute Error is a model evaluation metric used with regression models. 

# In[23]:


print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# The error is just 4.18%.It is the difference between the predicted value and the measured value.

# ----

# ##### Conclusion

# This analysis summarizes the use of simple linear regression model to predict the score percentages according to the Study hours in 9 steps.

# In[ ]:




