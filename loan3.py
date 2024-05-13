#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import datetime

import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder
import copy
from pandas.api.types import CategoricalDtype
import statsmodels.api as sm
import statsmodels.formula.api as smf7

from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# In[11]:


df = pd.read_csv('train_u6lujuX_CVtuZ9i (1) (1).csv')
df.head()


# In[12]:


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

def check_df(dataframe, head = 5):
  print("##################### HEAD ######################")
  print(dataframe.head())
  print("#################### INFO #######################")
  print(dataframe.info())
  print("################## QUANTILES ####################")
  print(dataframe.describe().T)
  print("#################### NA #########################")
  print(dataframe.isnull().sum())
  
check_df(df)


# In[13]:


for col in df.columns:
    print(col, ":", df[col].unique())
    print()


# # Exploratory Data Analysis

# In[14]:


df["Loan_ID"].value_counts()


# In[15]:


df = df.drop_duplicates(subset = ["Loan_ID"], keep = "last").reset_index(drop = True)
print(df.shape) 


# In[16]:


df["Loan_ID"].duplicated().sum()


# In[17]:


df.drop("Loan_ID", axis=1, inplace=True)


# In[18]:


df.head()


# # Gender

# In[19]:


df["Gender"].unique()


# In[20]:


df["Gender"].unique()


# In[21]:


df["Gender"].isna().sum() 


# In[22]:


df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])


# In[23]:


ax = sns.countplot(df['Gender']);

for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), 
       ha = 'center', va = 'center', xytext = (0, 4), textcoords = 'offset points')
    
ax.set_title("Gender");


# In[24]:


df["Married"].unique()


# In[25]:


df["Married"].value_counts()


# In[26]:


df["Married"].isna().sum() 


# In[27]:


df["Married"] = df["Married"].fillna(df["Married"].mode()[0])


# In[28]:


color = ["blue","green"]
sns.set_palette(sns.color_palette(color))
ax = sns.countplot(df['Married']);

for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), 
       ha = 'center', va = 'center', xytext = (0, 4), textcoords = 'offset points')
    
ax.set_title("Married");


# In[29]:


df["Dependents"].unique()


# In[30]:


df["Dependents"].value_counts()


# In[31]:


df["Dependents"].isna().sum() 


# In[32]:


df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])


# In[33]:


ax = sns.countplot(df['Dependents']);

for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), 
       ha = 'center', va = 'center', xytext = (0, 4), textcoords = 'offset points')
    
ax.set_title("Dependents");


# # Education

# In[34]:


df["Education"].unique()


# In[35]:


df["Education"].isna().sum() 


# In[36]:


color = ["orange","yellow"]
sns.set_palette(sns.color_palette(color))
print(df["Education"].value_counts())
print()

label =  "Graduate", "Not Graduate"
explode = (0.1, 0.1)
sizes = df["Education"].value_counts(normalize = True)*100

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels = label, autopct="%1.1f%%", shadow=True, startangle=70)
ax1.axis("equal") 
plt.title("Distribution by Education")
plt.rcParams['figure.figsize'] = [5, 5]
plt.show();


# # Self_Employed

# In[37]:


df["Self_Employed"].unique()


# In[38]:


print(df["Self_Employed"].value_counts())


# In[39]:


df["Self_Employed"].isna().sum() 


# In[40]:


df["Self_Employed"] = df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])


# In[41]:


df["Self_Employed"].isna().sum() 


# In[43]:


ax = sns.countplot(df["Self_Employed"]);

for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), 
       ha = 'center', va = 'center', xytext = (0, 4), textcoords = 'offset points')
    
ax.set_title("Self_Employed");


# # ApplicantIncome

# In[45]:


df["ApplicantIncome"].isna().sum() 


# In[46]:


fig, ax = plt.subplots(1,2, figsize = (13,5));

sns.histplot(df["ApplicantIncome"], kde = True, ax = ax[0]);
sns.boxplot(x = "ApplicantIncome", data = df, ax = ax[1]);

plt.suptitle("Distribution of ApplicantIncome");

print("Minimum", df["ApplicantIncome"].min())
print("Maximum", df["ApplicantIncome"].max())


# # CoapplicantIncome

# In[47]:


df["CoapplicantIncome"].isna().sum() 


# In[48]:


fig, ax = plt.subplots(1,2, figsize = (13,5));

sns.histplot(df["CoapplicantIncome"], kde = True, ax = ax[0],color='b');
sns.boxplot(x = "CoapplicantIncome", data = df, ax = ax[1]);

plt.suptitle("Distribution of CoapplicantIncome");

print("Minimum", df["CoapplicantIncome"].min())
print("Maximum", df["CoapplicantIncome"].max())


# # LoanAmount

# In[49]:


df["LoanAmount"].isna().sum() 


# In[50]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(df[['LoanAmount']])
df['LoanAmount'] = imputer.transform(df[['LoanAmount']])


# In[51]:


df["LoanAmount"].isna().sum() 


# In[52]:


fig, ax = plt.subplots(1,2, figsize = (13,5));

sns.histplot(df["LoanAmount"], kde = True, ax = ax[0], color='b');
sns.boxplot(x = "LoanAmount", data = df, ax = ax[1]);

plt.suptitle("Distribution of LoanAmount");

print("Minimum", df["LoanAmount"].min())
print("Maximum", df["LoanAmount"].max())


# # Loan_Amount_Term

# In[53]:


df["Loan_Amount_Term"].isna().sum() 


# In[54]:


df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0])


# In[55]:


df["Loan_Amount_Term"] = df["Loan_Amount_Term"].astype(int)
df["Loan_Amount_Term"].dtypes


# In[56]:


sns.set_palette(sns.color_palette("Paired"))

ax = sns.countplot(df["Loan_Amount_Term"]);

for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), 
       ha = 'center', va = 'center', xytext = (0, 4), textcoords = 'offset points')
    
ax.set_title("Loan_Amount_Term");


# # Credit_History

# In[57]:


df["Credit_History"].unique()


# In[58]:


df["Credit_History"].value_counts()


# In[59]:


df["Credit_History"].isna().sum() 


# In[60]:


imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(df[['Credit_History']])
df['Credit_History'] = imputer.transform(df[['Credit_History']])


# In[61]:


ax = sns.countplot(df["Credit_History"]);

for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), 
       ha = 'center', va = 'center', xytext = (0, 4), textcoords = 'offset points')
    
ax.set_title("Credit_History");


# # Property_Area

# In[62]:


df["Property_Area"].unique()


# In[63]:


df["Property_Area"].isna().sum()


# In[64]:


ax = sns.countplot(df["Property_Area"]);

for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), 
       ha = 'center', va = 'center', xytext = (0, 4), textcoords = 'offset points')
    
ax.set_title("Property_Area");


# # Loan_Status

# In[65]:


df["Loan_Status"] = df["Loan_Status"].map({"Y" : 1, "N" : 0})
df["Loan_Status"].value_counts()


# In[66]:


color = ["red","green"]
sns.set_palette(sns.color_palette(color))
ax = sns.countplot(df["Loan_Status"]);

for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), 
       ha = 'center', va = 'center', xytext = (0, 4), textcoords = 'offset points')
    
ax.set_title("Loan_Status");


# In[67]:


df.isna().sum()


# In[68]:


plt.figure(figsize = (14,7))
sns.heatmap(df.corr(), annot = True);


# In[69]:


sns.pairplot(df, size=2);


# In[70]:


sns.set_palette(sns.color_palette("Paired"))
pd.crosstab(df.Gender,df.Property_Area).plot(kind="bar", stacked=True, figsize=(5,5))
plt.title('Gender vs Property_Area')
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()


# In[71]:


pd.crosstab(df.Gender ,df.Married).plot(kind="bar", stacked=True, figsize=(5,5))
plt.title('Gender vs Married')
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()


# In[72]:


pd.crosstab(df.Gender ,df. Dependents).plot(kind="bar", stacked=True, figsize=(5,5))
plt.title('Gender vs Dependents')
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()


# In[73]:


pd.crosstab(df.Married ,df. Dependents).plot(kind="bar", stacked=True, figsize=(5,5))
plt.title('Married vs Dependents')
plt.xlabel('Married')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()


# In[74]:


df = pd.get_dummies(df)

# Drop columns
df = df.drop(['Gender_Female', 'Married_No', 'Education_Not Graduate', 
              'Self_Employed_No'], axis = 1)

# Rename columns name
new = {'Gender_Male': 'Gender', 'Married_Yes': 'Married', 
       'Education_Graduate': 'Education', 'Self_Employed_Yes': 'Self_Employed',
       'Loan_Status_Y': 'Loan_Status'}
       
df.rename(columns=new, inplace=True)


# In[75]:


X = df.drop(["Loan_Status"], axis=1)
y = df["Loan_Status"]
X, y = SMOTE().fit_resample(X, y)


# In[82]:


pip install imblearn


# In[83]:


from imblearn.over_sampling import SMOTE

X = df.drop(["Loan_Status"], axis=1)
y = df["Loan_Status"]
smote = SMOTE()
X, y = smote.fit_resample(X, y)


# In[84]:


sns.countplot(y=y, data=df)
plt.ylabel('Loan Status')
plt.xlabel('Total')
plt.show()


# In[85]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_auc_score,roc_curve


# In[86]:


X = MinMaxScaler().fit_transform(X)


# In[87]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[88]:


log_reg = LogisticRegression(solver="liblinear", max_iter=30, random_state=32)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

log_reg_acc = accuracy_score(y_pred,y_test)
print('Logistic Regression Accuracy: {:.2f}%'.format(log_reg_acc*100))

roc_auc = roc_auc_score(y_test, y_pred)
print('Auc Score: {:.2f}%'.format(roc_auc*100))


# In[89]:


knn = []
for i in range(1,21):
    KNclassifier = KNeighborsClassifier(n_neighbors = i)
    KNclassifier.fit(X_train, y_train)
    knn.append(KNclassifier.score(X_test, y_test))
    
plt.plot(range(1,21), knn)
plt.xticks(np.arange(1,21,1))
plt.xlabel("K value")
plt.ylabel("Score")
plt.show()
KNAcc = max(knn)
print("KNN best accuracy: {:.2f}%".format(KNAcc*100))


# In[90]:


SVCclassifier = SVC(kernel='rbf', max_iter=500)
SVCclassifier.fit(X_train, y_train)

y_pred = SVCclassifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score
SVCAcc = accuracy_score(y_pred,y_test)
print('SVC accuracy: {:.2f}%'.format(SVCAcc*100))

roc_auc = roc_auc_score(y_test, y_pred)
print('Auc Score: {:.2f}%'.format(roc_auc*100))


# In[91]:


NBclassifier1 = CategoricalNB()
NBclassifier1.fit(X_train, y_train)

y_pred = NBclassifier1.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score
NBAcc1 = accuracy_score(y_pred,y_test)
print('Categorical Naive Bayes accuracy: {:.2f}%'.format(NBAcc1*100))

roc_auc = roc_auc_score(y_test, y_pred)
print('Auc Score: {:.2f}%'.format(roc_auc*100))


# In[92]:


NBclassifier2 = GaussianNB()
NBclassifier2.fit(X_train, y_train)

y_pred = NBclassifier2.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score
NBAcc2 = accuracy_score(y_pred,y_test)
print('Gaussian Naive Bayes accuracy: {:.2f}%'.format(NBAcc2*100))

roc_auc = roc_auc_score(y_test, y_pred)
print('Auc Score: {:.2f}%'.format(roc_auc*100))


# In[93]:


scoreListDT = []
for i in range(2,21):
    DTclassifier = DecisionTreeClassifier(max_leaf_nodes=i)
    DTclassifier.fit(X_train, y_train)
    scoreListDT.append(DTclassifier.score(X_test, y_test))
    
plt.plot(range(2,21), scoreListDT)
plt.xticks(np.arange(2,21,1))
plt.xlabel("Leaf")
plt.ylabel("Score")
plt.show()
DTAcc = max(scoreListDT)
print("Decision Tree Accuracy: {:.2f}%".format(DTAcc*100))


# In[94]:


scoreListRF = []
for i in range(2,25):
    RFclassifier = RandomForestClassifier(n_estimators = 1000, random_state = 1, max_leaf_nodes=i)
    RFclassifier.fit(X_train, y_train)
    scoreListRF.append(RFclassifier.score(X_test, y_test))
    
plt.plot(range(2,25), scoreListRF)
plt.xticks(np.arange(2,25,1))
plt.xlabel("RF Value")
plt.ylabel("Score")
plt.show()
RFAcc = max(scoreListRF)
print("Random Forest Accuracy:  {:.2f}%".format(RFAcc*100))


# In[95]:


compare = pd.DataFrame({'Model': ['Logistic Regression',
                                  'SVM', 'Categorical NB', 
                                  'Gaussian NB'], 
                        'Accuracy': [log_reg_acc*100, SVCAcc*100, NBAcc1*100,
                                     NBAcc2*100,]})
compare.sort_values(by='Accuracy', ascending=False)


# In[ ]:




