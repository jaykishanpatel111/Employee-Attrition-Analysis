#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


cd F:\Dataset\Done projects\Employee Attrition Analysis


# In[3]:


pwd


# # Read Dataset

# In[4]:


pd.set_option('display.max_columns', None)
df = pd.read_csv("R_ML_Project_2_Emplo.csv")
df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


col = df.columns
col


# In[8]:


df.describe()


# # Checking null values in the dataset

# In[9]:


df.isna().sum()


# # Checking Duplicate in data

# In[10]:


df.duplicated().sum()


# # Plot Histograam mapes

# In[11]:


fig , s= plt.subplots(6,2, figsize = (20,20))
s[0][0].set_title("Histogram of Age column")
s[1][0].set_title("Histogram of BusinessTravel column")
s[2][0].set_title("Histogram of DailyRate column")
s[3][0].set_title("Histogram of Department column")
s[4][0].set_title("Histogram of EducationField column")
s[5][0].set_title("Histogram of Gender column")

s[0][1].set_title("Histogram of HourlyRate column")
s[1][1].set_title("Histogram of JobRole column")
s[2][1].set_title("Histogram of MaritalStatus column")
s[3][1].set_title("Histogram of NumCompaniesWorked column")
s[4][1].set_title("Histogram of OverTime column")
s[5][1].set_title("Histogram of TotalWorkingYears column")


s[0][0].hist(df['Age'])
s[1][0].hist(df['BusinessTravel'])
s[2][0].hist(df['DailyRate'])
s[3][0].hist(df['Department'])
s[4][0].hist(df['EducationField'])
s[5][0].hist(df['Gender'])


s[0][1].hist(df['HourlyRate'] )
s[1][1].hist(df['JobRole'])
s[2][1].hist(df['MaritalStatus'])
s[3][1].hist(df['NumCompaniesWorked'])
s[4][1].hist(df['OverTime'])
s[5][1].hist(df['TotalWorkingYears'])

plt.show()


# # Plot Scatterplot 

# In[12]:


df.head()


# In[13]:


plt.figure(figsize=(15,5))
sns.scatterplot(x= "BusinessTravel",y= 'Department', hue = 'Attrition', data = df)
plt.show()


# In[14]:


plt.figure(figsize=(15,5))
sns.scatterplot(x= "EducationField",y= 'Department', hue = 'Attrition', data = df)
plt.show()


# In[15]:


plt.figure(figsize=(15,5))
sns.scatterplot(x= "JobRole",y= 'HourlyRate', hue = 'Attrition', data = df)
plt.show()


# In[16]:


plt.figure(figsize=(15,5))
sns.scatterplot(x= "JobRole",y= 'NumCompaniesWorked', hue = 'Attrition', data = df)
plt.show()


# In[18]:


plt.figure(figsize=(15,5))
sns.scatterplot(x= "PercentSalaryHike",y= 'MonthlyIncome', hue = 'Attrition', data = df)
plt.show()


# In[19]:


plt.figure(figsize=(15,5))
sns.scatterplot(x= "OverTime",y= 'JobRole', hue = 'Attrition', data = df)
plt.show()


# # Label Encoding

# In[20]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Attrition'] = le.fit_transform(df['Attrition'])
df['BusinessTravel'] = le.fit_transform(df['BusinessTravel'])
df['Department'] = le.fit_transform(df['Department'])
df['EducationField'] = le.fit_transform(df['EducationField'])
df['Gender'] = le.fit_transform(df['Gender'])
df['JobRole'] = le.fit_transform(df['JobRole'])
df['MaritalStatus'] = le.fit_transform(df['MaritalStatus'])
df['OverTime'] = le.fit_transform(df['OverTime'])


# # Correlation between each features
# 

# In[21]:


corr_data  =df.corr()


# In[22]:


corr_data


# # Ploting heat map of the correlated data

# In[23]:


plt.figure(figsize = (20,10))
sns.heatmap(corr_data, annot = True, cmap = "RdYlGn")


# # Withot Scalling data

# # Model Building

# # Create features and target data

# In[24]:


X = df.drop(['Attrition'], axis= 1)
Y = df.Attrition


# In[25]:


X.shape, Y.shape


# # Spliting training and testing dataset

# In[28]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.30,random_state=41)


# In[29]:


X_train.shape, Y_train.shape  , X_test.shape, Y_test.shape


# # Creating Random Forest model

# In[30]:


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators = 15, random_state = 11)


# In[31]:


# train the model
model.fit( X_train , Y_train.ravel())


# In[32]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,precision_score,recall_score


# In[33]:


# Predicting values from the model
Y_pred = model.predict(X_test)
Y_pred = np.array([0 if i < 0.5 else 1 for i in Y_pred])
Y_pred


# # Checking accuracy score of our model

# In[34]:


def run_model(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train,Y_train.ravel())
    accuracy = accuracy_score(Y_test, Y_pred)
    print("pricison_score: ",precision_score(Y_test, Y_pred))
    print("recall_score: ",recall_score(Y_test, Y_pred))
    print("Accuracy = {}".format(accuracy))
    print(classification_report(Y_test,Y_pred,digits=5))
    print(confusion_matrix(Y_test,Y_pred))


# In[35]:


run_model(model, X_train, Y_train, X_test, Y_test)


# In[36]:


cm = confusion_matrix(Y_test, Y_pred)
cm


# In[37]:


# Heatmap of Confusion matrix
sns.heatmap(pd.DataFrame(cm), annot=True)


# In[ ]:





# # Classification Report

# In[38]:


from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))


# In[ ]:





# # With Scaling data

# In[39]:


from sklearn.preprocessing import RobustScaler
scaling = RobustScaler()
df = scaling.fit_transform(df)


# In[40]:


df = pd.DataFrame(df,columns=col)
df.head()


# # Model Building

# In[41]:


# Create features and target data


# In[42]:


X = df.drop(['Attrition'], axis= 1)
Y = df.Attrition


# In[43]:


X.shape, Y.shape


# # Spliting training and testing dataset

# In[44]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.30,random_state=41)


# In[45]:


X_train.shape, Y_train.shape  , X_test.shape, Y_test.shape


# # Creating Random Forest model

# In[46]:


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators = 15, random_state = 11)


# In[47]:


# train the model
model.fit( X_train , Y_train.ravel())


# In[48]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,precision_score,recall_score


# In[49]:


# Predicting values from the model
Y_pred = model.predict(X_test)
Y_pred = np.array([0 if i < 0.5 else 1 for i in Y_pred])
Y_pred


# # Checking accuracy score of our model

# In[50]:


def run_model(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train,Y_train.ravel())
    accuracy = accuracy_score(Y_test, Y_pred)
    print("pricison_score: ",precision_score(Y_test, Y_pred))
    print("recall_score: ",recall_score(Y_test, Y_pred))
    print("Accuracy = {}".format(accuracy))
    print(classification_report(Y_test,Y_pred,digits=5))
    print(confusion_matrix(Y_test,Y_pred))


# In[51]:


run_model(model, X_train, Y_train, X_test, Y_test)


# In[52]:


cm = confusion_matrix(Y_test, Y_pred)
cm


# In[53]:


# Heatmap of Confusion matrix
sns.heatmap(pd.DataFrame(cm), annot=True)


# # Classification Report

# In[54]:


from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))


# In[ ]:





# In[ ]:




