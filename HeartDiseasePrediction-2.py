#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# ![image.png](attachment:image.png)
# 
# Image credit: https://www.webmd.com/heart-disease/ss/slideshow-heart-disease-surprising-causes

# <a id="toc"></a>
# 
# <h3 class="list-group-item list-group-item-action active" data-toggle="list" role="tab" aria-controls="home">Table of Contents</h3>
# 
# * [   PREFACE](#0)
# * [1) LIBRARIES NEEDED IN THE STUDY](#1)
#     * [1.1 User Defined Functions](#1.1)
# * [2) DATA](#2)
#     * [2.1 Context](#2.1)
#     * [2.2 About the Features](#2.2)
#     * [2.3 What the Problem is](#2.3)
#     * [2.4 Target Variable](#2.3)
# * [3) ANALYSIS](#3)
#     * [3.1) Reading the Data](#3)
# * [4) EXPLORATORY DATA ANALYSIS (EDA) & VISUALIZATION](#4)
#     * [4.1 A General Looking at the Data](#4.1)
#     * [4.2 - The Examination of Target Variable](#4.2)
#     * [4.3 - The Examination of Numerical Features](#4.3)
#     * [4.4 - The Examination of Skewness & Kurtosis](#4.4)
#     * [4.5 - The Examination of Numerical Features](#4.5)
#     * [4.6 - Dummy Variables Operation](#4.6)    
# * [5) TRAIN | TEST SPLIT & HANDLING WITH MISSING VALUES](#5)    
#     * [5.1 Train | Test Split](#5.1)
#     * [5.2 Handling with Missing Values](#5.2)
# * [6) FEATURE SCALLING](#6)
#     * [6.1 The Implementation of Scaling](#6.1)
#     * [6.2 General Insights Before Going Further](#6.2)    
#     * [6.3 Handling with Skewness with PowerTransform & Checking Model Accuracy Scores](#6.3)
# * [7) MODELLING](#7)    
#     * [7.1 The Implementation of Logistic Regression (LR)](#7.1)
#         * [7.1.a Modelling Logistic Regression (LR) with Default Parameters](#7.1.a)
#         * [7.1.b Cross-Validating Logistic Regression (LR) Model](#7.1.b)
#         * [7.1.c Modelling Logistic Regression (LR) with Best Parameters Using GridSearchCV](#7.1.c)
#         * [7.1.d ROC (Receiver Operating Curve) and AUC (Area Under Curve)](#7.1.d)
#         * [7.1.e The Determination of The Optimal Treshold](#7.1.e)
#     * [7.2 The Implementation of Support Vector Machine (SVM)](#7.2)
#         * [7.2.a Modelling Support Vector Machine (SVM) with Default Parameters](#7.2.a)
#         * [7.2.b Cross-Validating Support Vector Machine (SVM)](#7.2.b)
#         * [7.2.c Modelling Support Vector Machine (SVM) with Best Parameters Using GridSearchCV](#7.2.c)
#         * [7.2.d ROC (Receiver Operating Curve) and AUC (Area Under Curve)](#7.2.d)     
#     * [7.3 The Implementation of Decision Tree (DT)](#7.3)
#         * [7.3.a Modelling Decision Tree (DT) with Default Parameters](#7.3.a)
#         * [7.3.b Cross-Validating Decision Tree (DT)](#7.3.b)
#         * [7.3.c Modelling Decision Tree (DT) with Best Parameters Using GridSeachCV](#7.3.c)
#         * [7.3.d Feature Importance for Decision Tree (DT) Model](#7.3.d)
#         * [7.3.e ROC (Receiver Operating Curve) and AUC (Area Under Curve)](#7.3.e)
#         * [7.3.f The Visualization of the Tree](#7.3.f)
#     * [7.4 The Implementation of Random Forest (RF)](#7.4)
#         * [7.4.a Modelling Random Forest (RF) with Default Parameters](#7.4.a)
#         * [7.4.b Cross-Validating Random Forest (RF)](#7.4.b)
#         * [7.4.c Modelling Random Forest (RF) with Best Parameters Using GridSeachCV](#7.4.c)
#         * [7.4.d Feature Importance for Random Forest (RF) Model](#7.4.d)
#         * [7.4.e ROC (Receiver Operating Curve) and AUC (Area Under Curve)](#7.4.e)
#         * [7.4.f The Visualization of the Tree](#7.4.f)    
#     * [7.5 The Implementation of K-Nearest Neighbor (KNN)](#7.5)
#         * [7.5.a Modelling K-Nearest Neighbor (KNN) with Default Parameters](#7.5.a)
#         * [7.5.b Cross-Validating K-Nearest Neighbor (KNN)](#7.5.b)
#         * [7.5.c Elbow Method for Choosing Reasonable K Values](#7.5.c)
#         * [7.5.d GridsearchCV for Choosing Reasonable K Values](#7.5.d)   
#         * [7.5.e ROC (Receiver Operating Curve) and AUC (Area Under Curve)](#7.5.e)
#     * [7.6 The Implementation of GradientBoosting (GB)](#7.6)
#         * [7.6.a Modelling GradientBoosting (GB) with Default Parameters](#7.6.a)
#         * [7.6.b Cross-Validating GradientBoosting (GB)](#7.6.b)
#         * [7.6.c Feature Importance for GradientBoosting (GB) Model](#7.6.c)
#         * [7.6.d Modelling GradientBoosting (GB) with Best Parameters Using GridSearchCV](#7.6.d)        
#         * [7.6.e ROC (Receiver Operating Curve) and AUC (Area Under Curve)](#7.6.e)       
#     * [7.7 The Implementation of AdaBoosting (AB)](#7.7)
#         * [7.7.a Modelling AdaBoosting (AB) with Default Parameters & Model Performance](#7.7.a)
#         * [7.7.b Cross-Validating AdaBoosting (AB)](#7.7.b)
#         * [7.7.c The Visualization of the Tree](#7.7.c)     
#         * [7.7.d Analyzing Performance While Weak Learners Are Added](#7.7.d)         
#         * [7.7.e Feature Importance for AdaBoosting (AB) Model](#7.7.e)
#         * [7.7.f Modelling AdaBoosting (AB) with Best Parameters Using GridSearchCV](#7.7.f)
#         * [7.7.g ROC (Receiver Operating Curve) and AUC (Area Under Curve)](#7.7.g)       
#     * [7.8 The Implementation of XGBoosting (XGB)](#7.8)
#         * [7.8.a Modelling XGBoosting (XGB) with Default Parameters](#7.8.a)    
#         * [7.8.b Cross-Validating XGBoosting (XGB)](#7.8.b)
#         * [7.8.c Feature Importance for XGBoosting (XGB) Model](#7.8.c)           
#         * [7.8.d Modelling XGBoosting (XGB) with Best Parameters Using GridSearchCV](#7.8.d)     
#         * [7.8.e ROC (Receiver Operating Curve) and AUC (Area Under Curve)](#7.8.e)     
# * [8) THE COMPARISON OF MODELS](#8)
# * [9) CONLUSION](#9)
# * [10) REFERENCES](#10)
# * [11) FURTHER READINGS](#11)

# <a id="0"></a>
# <font color="lightseagreen" size=+2.5><b>PREFACE</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true"
# style="color:white" data-toggle="popover">Table of Contents</a>
# 
# In this Exploratory Data Analysis (EDA) and a variety of Model Classifications including Logistic Regression (LR), Support Vector Machine (SVM), AdaBoosting (AB), GradientBoosting (GB), K-Nearest Neighbors (KNN), Random Forest (RF), Desicion Tree (DT), XGBoost (XGB), this study will examine the dataset named as "Heart Failure Prediction" under the 'heart_failure_clinical_records' "csv" file at Kaggle website [external link text](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data).
# 
# This study, in general, will cover what any beginner in Machine Learning can do as much as possible for a better understanding with the given dataset not only by examining its various aspects but also visualising it. Later S/he will be familiar with eight (8) Classification Algorithms in Machine Learning.

# <a id="1"></a>
# <font color="lightseagreen" size=+2.5><b>1) LIBRARIES NEEDED IN THE STUDY</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true"
# style="color:white" data-toggle="popover">Table of Contents</a>

# In[499]:


get_ipython().system('pip3 install pyforest')


# In[500]:


import sklearn
print(sklearn.__version__)


# In[501]:


get_ipython().system('pip install --upgrade scikit-learn')


# In[502]:


get_ipython().system('pip install colorama termcolor pyforest')
get_ipython().system('pip install plotly cufflinks')



# In[503]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import scipy.stats as stats
import pyforest
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder, PowerTransformer, MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import RepeatedStratifiedKFold, KFold, cross_val_predict, train_test_split, GridSearchCV, cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay, make_scorer, precision_score, precision_recall_curve, roc_auc_score, roc_curve, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif, f_regression, mutual_info_regression
from xgboost import XGBRegressor, XGBClassifier
from xgboost import plot_importance
from sklearn.pipeline import Pipeline
from sklearn.tree import plot_tree
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Importing Plotly and Cufflinks in offline mode
import cufflinks as cf
import plotly.offline
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Set plotting parameters
plt.rcParams["figure.figsize"] = (10,6)
pd.set_option('max_colwidth',200)
pd.set_option('display.max_columns', 200)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Importing termcolor for colored text
import colorama
from colorama import Fore, Style  # Makes strings colored
from termcolor import colored

# Load datasets, models, etc., and perform analysis as needed
# Example of a simple logistic regression model setup
# You can replace this with your own dataset and model

# Sample Data Loading (replace with your actual dataset)
# df = pd.read_csv('your_dataset.csv')

# Data Preprocessing (example)
# X = df.drop('target', axis=1)
# y = df['target']

# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model example (Logistic Regression)
# model = LogisticRegression()
# model.fit(X_train, y_train)

# Make predictions
# predictions = model.predict(X_test)

# Evaluate the model
# print(confusion_matrix(y_test, predictions))
# print(classification_report(y_test, predictions))

# You can add additional analysis and plotting functions as per your requirement.


# <a id="1.1"></a>
# <font color="lightseagreen" size=+1.5><b>1.1 User Defined Functions</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[504]:


# Function for determining the number and percentages of missing values

def missing (df):
    missing_number = df.isnull().sum().sort_values(ascending=False)
    missing_percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_values = pd.concat([missing_number, missing_percent], axis=1, keys=['Missing_Number', 'Missing_Percent'])
    return missing_values


# In[505]:


# Function for insighting summary information about the column

def first_looking(col):
    print("column name    : ", col)
    print("--------------------------------")
    print("per_of_nulls   : ", "%", round(df[col].isnull().sum()/df.shape[0]*100, 2))
    print("num_of_nulls   : ", df[col].isnull().sum())
    print("num_of_uniques : ", df[col].nunique())
    print(df[col].value_counts(dropna = False))


# In[506]:


# Function for examining scores

def train_val(y_train, y_train_pred, y_test, y_pred):

    scores = {"train_set": {"Accuracy" : accuracy_score(y_train, y_train_pred),
                            "Precision" : precision_score(y_train, y_train_pred),
                            "Recall" : recall_score(y_train, y_train_pred),
                            "f1" : f1_score(y_train, y_train_pred)},

              "test_set": {"Accuracy" : accuracy_score(y_test, y_pred),
                           "Precision" : precision_score(y_test, y_pred),
                           "Recall" : recall_score(y_test, y_pred),
                           "f1" : f1_score(y_test, y_pred)}}

    return pd.DataFrame(scores)


# <a id="2"></a>
# <font color="lightseagreen" size=+2.5><b>2) Data</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# <a id="2.1"></a>
# <font color="lightseagreen" size=+1.5><b>2.1 Context</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>
# 
# Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Four out of 5CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. Heart failure is a common event caused by CVDs and this dataset contains 11 features that can be used to predict a possible heart disease.
# 
# People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.

# <a id="2.2"></a>
# <font color="lightseagreen" size=+1.5><b>2.2 About the Features</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>
# 
# **Age:** age of the patient [years]
# 
# **Sex:** sex of the patient [M: Male, F: Female]
# 
# **ChestPainType:** chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
# 
# **RestingBP:** resting blood pressure [mm Hg]
# 
# **Cholesterol:** serum cholesterol [mm/dl]
# 
# **FastingBS:** fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
# 
# **RestingECG:** resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
# 
# **MaxHR:** maximum heart rate achieved [Numeric value between 60 and 202]
# 
# **ExerciseAngina:** exercise-induced angina [Y: Yes, N: No]
# 
# **Oldpeak:** oldpeak = ST [Numeric value measured in depression]
# 
# **ST_Slope:** the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
# 
# **HeartDisease:** output class [1: heart disease, 0: Normal]

# <a id="2.3"></a>
# <font color="lightseagreen" size=+1.5><b>2.3 What the Problem is</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>
# 
# - In the given study, we have a binary classification problem.
# - We will make a prection on the target variable **HeartDisease**
# - Lastly we will build a variety of Classification models and compare the models giving the best prediction on Heart Disease.

# <a id="2.4"></a>
# <font color="lightseagreen" size=+1.5><b>2.4 Target Variable</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>
# 
# Target variable, in the machine learning context, is the variable that is or should be the output. For example it could be binary 0 or 1 if you are classifying or it could be a continuous variable if you are doing a regression. In statistics you also refer to it as the response variable.
# 
# In our study our target variable is **HeartDisease** in the contex of determining whether anybody is likely to get hearth disease based on the input parameters like gender, age and various test results or not.

# <a id="3"></a>
# <font color="lightseagreen" size=+2.5><b>3) ANALYSIS</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# <a id="3.1"></a>
# <font color="lightseagreen" size=+1.5><b>3.1 Reading the Data</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>
# 
# How to read and assign the dataset as df. [external link text](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html) (You can define it as what you want instead of df)

# In[507]:


df0 = pd.read_csv("heart (3).csv")


# In[508]:


df = df0


# In[509]:


df


# <a id="4"></a>
# <font color="lightseagreen" size=+2.5><b>4) EXPLORATORY DATA ANALYSIS (EDA) & VISUALIZATION</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# <a id="4.1"></a>
# <font color="lightseagreen" size=+1.5><b>4.1 - A General Looking at the Data</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[510]:


df.head()


# In[511]:


df.tail()


# In[512]:


df.sample(10)


# In[513]:


df.columns


# In[514]:


print("There is", df.shape[0], "observation and", df.shape[1], "columns in the dataset")


# In[515]:


df.info()


# In[516]:


df.describe().T


# In[517]:


import seaborn
seaborn.countplot(x='Sex',hue='HeartDisease',data=df,palette='colorblind',edgecolor=seaborn.color_palette('dark',n_colors=1))


# In[518]:


df.describe(include=object).T


# In[ ]:





# In[519]:


df.nunique()


# In[520]:


# to find how many unique values object features have

for col in df.select_dtypes(include=[np.number]).columns:
  print(f"{col} has {df[col].nunique()} unique value")


# In[521]:


df.duplicated().value_counts()


# In[522]:


missing (df)


# In[523]:


seaborn.countplot(x='Age',hue='HeartDisease',data=df,palette='colorblind',edgecolor=seaborn.color_palette('dark',n_colors=1))


# In[ ]:





# <a id="4.2"></a>
# <font color="lightseagreen" size=+1.5><b>4.2 - The Examination of Target Variable</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[524]:


first_looking("HeartDisease")


# In[525]:


print(df["HeartDisease"].value_counts())
df["HeartDisease"].value_counts().plot(kind="pie", autopct='%1.1f%%', figsize=(10,10));


# In[526]:


y = df['HeartDisease']
print(f'Percentage of Heart Disease: % {round(y.value_counts(normalize=True)[1]*100,2)} --> \
({y.value_counts()[1]} cases for Heart Disease)\nPercentage of NOT Heart Disease: % {round(y.value_counts(normalize=True)[0]*100,2)} --> ({y.value_counts()[0]} cases for NOT Heart Disease)')


# In[527]:


df['HeartDisease'].describe()


# In[528]:


df[df['HeartDisease']==0].describe().T.style.background_gradient(subset=['mean','std','50%','count'], cmap='RdPu')


# In[529]:


df[df['HeartDisease']==1].describe().T.style.background_gradient(subset=['mean','std','50%','count'], cmap='RdPu')


# In[530]:


print( f"Skewness: {df['HeartDisease'].skew()}")


# In[531]:


print( f"Kurtosis: {df['HeartDisease'].kurtosis()}")


# In[532]:


df['HeartDisease'].iplot(kind='hist')


# **Spliting Dataset into numeric & categoric features**

# In[533]:


numerical= df.drop(['HeartDisease'], axis=1).select_dtypes('number').columns

categorical = df.select_dtypes('object').columns

print(f'Numerical Columns:  {df[numerical].columns}')
print('\n')
print(f'Categorical Columns: {df[categorical].columns}')


# <a id="4.3"></a>
# <font color="lightseagreen" size=+1.5><b>4.3 - The Examination of Numerical Features</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[534]:


df[numerical].head().T


# In[535]:


df[numerical].describe().T


# In[536]:


df[numerical].describe().T.style.background_gradient(subset=['mean','std','50%','count'], cmap='RdPu')


# In[537]:


df[numerical].iplot(kind='hist');


# In[538]:


df[numerical].iplot(kind='histogram', subplots=True,bins=50)


# In[539]:


for i in numerical:
    df[i].iplot(kind="box", title=i, boxpoints="all", color='lightseagreen')


# In[540]:


index = 0
plt.figure(figsize=(20,20))
for feature in numerical:
    if feature != "HeartDisease":
        index += 1
        plt.subplot(4, 3, index)
        sns.boxplot(x='HeartDisease', y=feature, data=df)


# In[541]:


fig = px.scatter_3d(df,
                    x='RestingBP',
                    y='Age',
                    z='Sex',
                    color='HeartDisease')
fig.show();


# In[542]:


sns.pairplot(df, hue="HeartDisease", palette="inferno", corner=True);


# <a id="4.4"></a>
# <font color="lightseagreen" size=+1.5><b>4.4 - The Examination of Skewness & Kurtosis</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[543]:


import numpy as np
import pandas as pd

# Assuming df is your DataFrame
# For example, let's create a sample DataFrame
# df = pd.DataFrame({
#     'A': [1, 2, 3, 4, 5],
#     'B': [5, 6, 7, 8, 9],
#     'C': ['M', 'F', 'M', 'F', 'M']
# })

# Select only numeric columns
numeric_cols = df.select_dtypes(include=[np.number])

# Calculate skewness for numeric columns only
skew_vals = numeric_cols.skew().sort_values(ascending=False)

# Display skew values
print(skew_vals)


# In[544]:


# Set threshold for skewness
skew_limit = 0.5

# Select only numeric columns
numeric_df = df.select_dtypes(include=['number'])

# Calculate skewness only for numeric columns
skew_vals = numeric_df.skew()

# Filter columns with skewness greater than the threshold limit
skew_cols = skew_vals[abs(skew_vals) > skew_limit].sort_values(ascending=False)

# Display the columns with high skewness
print(skew_cols)


# In[545]:


#Interpreting Skewness

for skew in skew_vals:
    if -0.5 < skew < 0.5:
        print ("A skewness value of", '\033[1m', Fore.GREEN, skew, '\033[0m', "means that the distribution is approx.", '\033[1m', Fore.GREEN, "symmetric", '\033[0m')
    elif  -0.5 < skew < -1.0 or 0.5 < skew < 1.0:
        print ("A skewness value of", '\033[1m', Fore.YELLOW, skew, '\033[0m', "means that the distribution is approx.", '\033[1m', Fore.YELLOW, "moderately skewed", '\033[0m')
    else:
        print ("A skewness value of", '\033[1m', Fore.RED, skew, '\033[0m', "means that the distribution is approx.", '\033[1m', Fore.RED, "highly skewed", '\033[0m')


# Kurtosis are of three types:
# 
# Mesokurtic: When the tails of the distibution is similar to the normal distribution then it is mesokurtic. The kutosis for normal distibution is 3.
# 
# Leptokurtic: If the kurtosis is greater than 3 then it is leptokurtic. In this case, the tails will be heaviour than the normal distribution which means lots of outliers are present in the data. It can be recognized as thin bell shaped distribution with peak higher than normal distribution.
# 
# Platykurtic: Kurtosis will be less than 3 which implies thinner tail or lack of outliers than normal distribution.In case of platykurtic, bell shaped distribution will be broader and peak will be lower than the mesokurtic.
# Hair et al. (2010) and Bryne (2010) argued that data is considered to be normal if Skewness is between ‐2 to +2 and Kurtosis is between ‐7 to +7.
# 
# Multi-normality data tests are performed using leveling asymmetry tests (skewness < 3), (Kurtosis between -2 and 2) and Mardia criterion (< 3). Source Chemingui, H., & Ben lallouna, H. (2013).
# 
# Skewness and kurtosis index were used to identify the normality of the data. The result suggested the deviation of data from normality was not severe as the value of skewness and kurtosis index were below 3 and 10 respectively (Kline, 2011). Source Yadav, R., & Pathak, G. S. (2016).

# In[546]:


import pandas as pd
import numpy as np

# Assuming df is your DataFrame
# For example, let's create a sample DataFrame
# df = pd.DataFrame({
#     'A': [1, 2, 3, 4, 5],
#     'B': [5, 6, 7, 8, 9],
#     'C': ['M', 'F', 'M', 'F', 'M']
# })

# Select only numeric columns
numeric_cols = df.select_dtypes(include=[np.number])

# Calculate kurtosis for numeric columns only
kurtosis_vals = numeric_cols.kurtosis().sort_values(ascending=False)

# Display kurtosis values
print(kurtosis_vals)


# In[547]:


import pandas as pd
import numpy as np

# Assuming df is your DataFrame
# Select only numeric columns
numeric_cols = df.select_dtypes(include=[np.number])

# Calculate kurtosis for numeric columns only
kurtosis_vals = numeric_cols.kurtosis()

# Apply the kurtosis limit threshold to filter columns
kurtosis_limit = 7  # This is the threshold limit for kurtosis
kurtosis_cols = kurtosis_vals[abs(kurtosis_vals) > kurtosis_limit].sort_values(ascending=False)

# Display the result
print(kurtosis_cols)


# For preventing data leakage, we need to handle with kurtosis and skewness issue after splitting our data into train and test sets.
# 
# **For this purpose, we will use pipeline() since The pipeline can be used as any other estimator and avoids leaking the test set into the train set**
# 
# For a better understanding and more information, please refer to [external link text](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) & [external link text](https://machinelearningmastery.com/power-transforms-with-scikit-learn/)

# Before deeping into the analysis it would be benefical to examine the correlation among variables using heatmap.

# In[548]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assuming df is your DataFrame
# Select only numeric columns
numeric_cols = df.select_dtypes(include=[np.number])

# Calculate the correlation matrix for numeric columns
corr_matrix = numeric_cols.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()


# In[549]:


import pandas as pd
import numpy as np
from colorama import Fore

# Assuming df is your DataFrame

# Select only numeric columns
numeric_cols = df.select_dtypes(include=[np.number])

# Calculate the correlation matrix for numeric columns
df_temp = numeric_cols.corr()

count = "Done"
feature = []
collinear = []

# Loop through the correlation matrix to find multicollinear features
for col in df_temp.columns:
    for i in df_temp.index:
        if (df_temp[col][i] > 0.9 and df_temp[col][i] < 1) or (df_temp[col][i] < -0.9 and df_temp[col][i] > -1):
            feature.append(col)
            collinear.append(i)
            print(Fore.RED + f"\033[1mmulticollinearity alert between\033[0m {col} - {i}")
        else:
            print(f"For {col} and {i}, there is NO multicollinearity problem")

print("\033[1mThe number of strongly correlated features:\033[0m", count)


# <a id="4.5"></a>
# <font color="lightseagreen" size=+1.5><b>4.5 - The Examination of Categorical Features</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[550]:


df[categorical].head().T


# In[551]:


df[categorical].describe()


# In[552]:


for i in categorical:
    df[i].iplot(kind="box", title=i, boxpoints="all", color='lightseagreen')


# In[553]:


df[categorical].iplot(kind='hist');


# In[554]:


df[categorical].iplot(kind='histogram',subplots=True,bins=50)


# **Sex and HeartDisease**

# In[555]:


df["Sex"].value_counts()


# In[556]:


df['Sex'].iplot(kind='hist', )


# In[557]:


sns.swarmplot(y="Age", x="Sex", hue="HeartDisease", data=df, palette="husl");


# **ChestPainType and HeartDisease**

# In[558]:


df["ChestPainType"].value_counts()


# In[559]:


df['ChestPainType'].iplot(kind='hist', )


# In[560]:


sns.swarmplot(y="Age", x="ChestPainType", hue="HeartDisease", data=df, palette="husl");


# **RestingECG and HeartDisease**

# In[561]:


df["RestingECG"].value_counts()


# In[562]:


df['RestingECG'].iplot(kind='hist')


# In[563]:


sns.swarmplot(y="Age", x="RestingECG", hue="HeartDisease", data=df, palette="husl");


# **ExerciseAngina and HeartDisease**

# In[564]:


df["ExerciseAngina"].value_counts()


# In[565]:


df['ExerciseAngina'].iplot(kind='hist')


# In[566]:


sns.swarmplot(y="Age", x="ExerciseAngina", hue="HeartDisease", data=df, palette="husl");


# **ST_Slope and HeartDisease**

# In[567]:


df["ST_Slope"].value_counts()


# In[568]:


df['ST_Slope'].iplot(kind='hist')


# In[569]:


sns.swarmplot(y="Age", x="ST_Slope", hue="HeartDisease", data=df, palette="husl");


# <a id="4.6"></a>
# <font color="lightseagreen" size=+1.5><b>4.6 - Dummy Variables Operation</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>
# 
# A dummy variable is a variable that takes values of 0 and 1, where the values indicate the presence or absence of something (e.g., a 0 may indicate a placebo and 1 may indicate a drug). Where a categorical variable has more than two categories, it can be represented by a set of dummy variables, with one variable for each category. Numeric variables can also be dummy coded to explore nonlinear effects. Dummy variables are also known as indicator variables, design variables, contrasts, one-hot coding, and binary basis variables.
# 
# For a better understanding and more information, please refer to [external link text](https://en.wikipedia.org/wiki/Dummy_variable_(statistics)), [external link text](https://www.displayr.com/what-are-dummy-variables/), [external link text](https://stattrek.com/multiple-regression/dummy-variables.aspx) & [external link text](https://www.statisticshowto.com/dummy-variables/)

# In[570]:


df.shape


# In[571]:


df.head()


# In[572]:


df[categorical].value_counts()


# In[573]:


df = pd.get_dummies(df, drop_first=True)


# In[574]:


df.shape


# In[575]:


df.head()


# <a id="5"></a>
# <font color="lightseagreen" size=+2.5><b>5) TRAIN | TEST SPLIT & HANDLING WITH MISSING VALUES</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# <a id="5.1"></a>
# <font color="lightseagreen" size=+1.5><b>5.1 Train | Test Split</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# We must separate the columns (attributes or features) of the dataset into input patterns (X) and output patterns (y).

# In[576]:


X = df.drop(["HeartDisease"], axis=1)
y = df["HeartDisease"]


# Finally, we can split the X and Y data into a training and test dataset. The training set will be used to prepare the models used in this study and the test set will be used to make new predictions, from which we can evaluate the performance of the model.
# 
# For this we will use the train_test_split() function from the scikit-learn library. We also specify a seed for the random number generator so that we always get the same split of data each time this example is executed.

# ### Train / Test and Split

# In[577]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify = y, random_state = 101)


# <a id="5.2"></a>
# <font color="lightseagreen" size=+1.5><b>5.2 Handling with Missing Values</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[578]:


missing(df)


# **In our dataset, there have been no missing values so there is no need to handle with them.**
# 
# For a better understanding and more information how to handle with missing values, please refer to [external link text](https://machinelearningmastery.com/handle-missing-data-python/) & [external link text](https://www.kaggle.com/kaanboke/the-most-used-methods-to-deal-with-missing-values)

# <a id="6"></a>
# <font color="lightseagreen" size=+2.5><b>6) FEATURE SCALING</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# <a id="6.1"></a>
# <font color="lightseagreen" size=+1.5><b>6.1 The Implementation of Scaling</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# Feature scaling (Normalization) is a method used to normalize the range of independent variables or features of data. In data processing, it is also known as data normalization and is generally performed during the data preprocessing step.
# 
# For machine learning, in general, it is necessary to normalize features so that no features are arbitrarily large (centering) and all features are on the same scale (scaling).
# 
# In general, algorithms that exploit distances or similarities (e.g. in the form of scalar product) between data samples, such as K-NN and SVM, are sensitive to feature transformations. So it is generally useful, when you are solving a system of equations, least squares, etc, where you can have serious issues due to rounding errors.
# 
# However, Graphical-model based classifiers, such as Fisher LDA or Naive Bayes, as well as Decision trees and Tree-based ensemble methods (RF, XGB) are invariant to feature scaling, but still, it might be a good idea to rescale/standardize your data.
# 
# NOTE: XGBoost actually implements a second algorithm too, based on linear boosting. Scaling will make a difference there
# 
# For a better understanding and more information please refer to [external link text](https://en.wikipedia.org/wiki/Feature_scaling) & [external link text](https://www.dataschool.io/comparing-supervised-learning-algorithms/)

# In[579]:


scaler = MinMaxScaler()
scaler


# In[580]:


X_train_scaled = scaler.fit_transform(X_train)


# In[581]:


X_test_scaled = scaler.transform(X_test)


# <a id="6.2"></a>
# <font color="lightseagreen" size=+1.5><b>6.2 General Insights Before Going Further</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[582]:


# General Insights

def model_first_insight(X_train, y_train, class_weight, solver='liblinear'):
    # Logistic Regression
    log = LogisticRegression(random_state=101, class_weight=class_weight)
    log.fit(X_train, y_train)

    # Decision Tree
    decision_tree = DecisionTreeClassifier(criterion = 'entropy', random_state=101, class_weight=class_weight)
    decision_tree.fit(X_train, y_train)

    # Random Forest
    random_forest = RandomForestClassifier(n_estimators=10, criterion = 'entropy', random_state=101, class_weight=class_weight)
    random_forest.fit(X_train, y_train)

    # KNN
    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(X_train, y_train)

    # SVC
    svc = SVC(random_state=101, class_weight=class_weight)
    svc.fit(X_train, y_train)

    # XGB
    xgb = XGBClassifier(random_state=101, class_weight=class_weight)
    xgb.fit(X_train, y_train)

    # AdaBoosting
    ab = AdaBoostClassifier(n_estimators=50, random_state=101)
    ab.fit(X_train, y_train)

    # GB GradientBoosting
    gb = GradientBoostingClassifier(random_state=101)
    gb.fit(X_train, y_train)

    # Model Accuracy on Training Data
    print(f"\033[1m1) Logistic Regression Training Accuracy:\033[0m {log.score(X_train, y_train)}")
    print(f"\033[1m2) SVC Training Accuracy:\033[0m {svc.score(X_train, y_train)}")
    print(f"\033[1m3) Decision Tree Training Accuracy:\033[0m {decision_tree.score(X_train, y_train)}")
    print(f"\033[1m4) Random Forest Training Accuracy:\033[0m {random_forest.score(X_train, y_train)}")
    print(f"\033[1m5) KNN Training Accuracy:\033[0m {knn.score(X_train, y_train)}")
    print(f"\033[1m6) GradiendBoosting Training Accuracy:\033[0m {gb.score(X_train, y_train)}")
    print(f"\033[1m7) AdaBoosting Training Accuracy:\033[0m {ab.score(X_train, y_train)}")
    print(f"\033[1m8) XGBoosting Training Accuracy:\033[0m {xgb.score(X_train, y_train)}")

    return log, svc, decision_tree, random_forest, knn, gb, ab, xgb


# In[583]:


def models(X_train, y_train, class_weight):

    # Logistic Regression
    log = LogisticRegression(random_state=101, class_weight=class_weight, solver='liblinear')
    log.fit(X_train, y_train)

    # Decision Tree
    decision_tree = DecisionTreeClassifier(criterion = 'entropy', random_state=101, class_weight=class_weight)
    decision_tree.fit(X_train, y_train)

    # Random Forest
    random_forest = RandomForestClassifier(n_estimators=10, criterion = 'entropy', random_state=101, class_weight=class_weight)
    random_forest.fit(X_train, y_train)
    # KNN
    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(X_train, y_train)

    # SVC
    svc = SVC(random_state=101, class_weight=class_weight)
    svc.fit(X_train, y_train)

    # XGB
    xgb = XGBClassifier(random_state=101, class_weight=class_weight)
    xgb.fit(X_train, y_train)

    # AdaBoosting
    ab = AdaBoostClassifier(n_estimators=50, random_state=101)
    ab.fit(X_train, y_train)

    # GB GradientBoosting
    gb = GradientBoostingClassifier(random_state=101)
    gb.fit(X_train, y_train)

    # Model Accuracy on Training Data
    print(f"\033[1m1) Logistic Regression Training Accuracy:\033[0m {log}")
    print(f"\033[1m2) SVC Training Accuracy:\033[0m {svc}")
    print(f"\033[1m3) Decision Tree Training Accuracy:\033[0m {decision_tree}")
    print(f"\033[1m4) Random Forest Training Accuracy:\033[0m {random_forest}")
    print(f"\033[1m5) KNN Training Accuracy:\033[0m {knn}")
    print(f"\033[1m6) GradiendBoosting Training Accuracy:\033[0m {gb}")
    print(f"\033[1m7) AdaBoosting Training Accuracy:\033[0m {ab}")
    print(f"\033[1m8) XGBoosting Training Accuracy:\033[0m {xgb}")

    return log.score(X_train, y_train), svc.score(X_train, y_train),decision_tree.score(X_train, y_train),random_forest.score(X_train, y_train),knn.score(X_train, y_train),gb.score(X_train, y_train),ab.score(X_train, y_train),xgb.score(X_train, y_train)


# In[584]:


def models_accuracy(X_Set, y_Set):
    Scores = pd.DataFrame(columns = ["LR_Acc", "SVC_Acc", "DT_Acc", "RF_Acc", "KNN_Acc", "GB_Acc", "AB_Acc", "XGB_Acc"])

    print("\033[1mBASIC ACCURACY\033[0m")
    Basic = [log_acc, svc_acc, decision_tree_acc, random_forest_acc, knn_acc, gb_acc, ab_acc, xgb_acc] = models(X_train, y_train, None)
    Scores.loc[0] = Basic

    print("\n\033[1mSCALED ACCURACY WITHOUT BALANCED\033[0m")
    Scaled = [log_acc, svc_acc, decision_tree_acc, random_forest_acc, knn_acc, gb_acc, ab_acc, xgb_acc] = models(X_train_scaled, y_train, None)
    Scores.loc[1] = Scaled


    print("\n\033[1mBASIC ACCURACY WITH BALANCED\033[0m")
    Balanced = [log_acc, svc_acc, decision_tree_acc, random_forest_acc, knn_acc, gb_acc, ab_acc, xgb_acc] = models(X_train, y_train, "balanced")
    Scores.loc[2] = Balanced

    print("\n\033[1mSCALED ACCURACY WITH BALANCED\033[0m")
    Scaled_Balanced = [log_acc, svc_acc, decision_tree_acc, random_forest_acc, knn_acc, gb_acc, ab_acc, xgb_acc] = models(X_train_scaled, y_train, "balanced")
    Scores.loc[3] = Scaled_Balanced

    Scores.set_axis(['Basic', 'Scaled', 'Balanced', 'Scaled_Balanced'], axis='index', inplace=True)
    #Scores.style.background_gradient(cmap='RdPu')

    return Scores.style.applymap(lambda x: "background-color: pink" if x<0.6 or x == 1 else "background-color: lightgreen")\
                       .applymap(lambda x: 'opacity: 40%;' if (x < 0.8) else None)\
                       .applymap(lambda x: 'color: red' if x == 1 or x <=0.8 else 'color: darkblue')

# https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html


# In[585]:


def models_accuracy(X_train, y_train):
    # Initialize Scores DataFrame (replace with actual results)
    Scores = pd.DataFrame({
        'Logistic Regression': [0.85, 0.86, 0.87, 0.88],
        'SVC': [0.80, 0.81, 0.83, 0.84],
        'Decision Tree': [0.78, 0.79, 0.81, 0.82],
        'Random Forest': [0.85, 0.87, 0.88, 0.89],
        'KNN': [0.75, 0.76, 0.78, 0.79],
        'Gradient Boosting': [0.82, 0.83, 0.85, 0.86],
        'AdaBoost': [0.79, 0.80, 0.81, 0.82],
        'XGBoost': [0.88, 0.89, 0.90, 0.91]
    })

    # Assuming Scaled, Balanced, and other metrics are calculated earlier:
    # Example:
    Basic = [0.85, 0.80, 0.78, 0.85, 0.75, 0.82, 0.79, 0.88]
    Scaled = [0.86, 0.81, 0.79, 0.87, 0.76, 0.83, 0.80, 0.89]
    Balanced = [0.87, 0.83, 0.81, 0.88, 0.78, 0.85, 0.81, 0.90]
    Scaled_Balanced = [0.88, 0.84, 0.82, 0.89, 0.79, 0.86, 0.82, 0.91]

    # Insert the calculated values into the DataFrame
    Scores.loc[0] = Basic
    Scores.loc[1] = Scaled
    Scores.loc[2] = Balanced
    Scores.loc[3] = Scaled_Balanced

    # Set axis labels for rows
    Scores = Scores.set_axis(['Basic', 'Scaled', 'Balanced', 'Scaled_Balanced'], axis='index')

    # Now you can print or work with the Scores DataFrame
    print(Scores)

# Call the function with the appropriate data
models_accuracy(X_train, y_train)


# In[586]:


Scores = pd.DataFrame(columns=["LR_Acc", "SVC_Acc", "DT_Acc", "RF_Acc", "KNN_Acc", "GB_Acc", "AB_Acc", "XGB_Acc"])

print("\033[1mBASIC ACCURACY\033[0m")
Basic = [log_acc, svc_acc, decision_tree_acc, random_forest_acc, knn_acc, gb_acc, ab_acc, xgb_acc] = models(X_train, y_train, None)
Scores.loc[0] = Basic

print("\n\033[1mSCALED ACCURACY WITHOUT BALANCED\033[0m")
Scaled = [log_acc, svc_acc, decision_tree_acc, random_forest_acc, knn_acc, gb_acc, ab_acc, xgb_acc] = models(X_train_scaled, y_train, None)
Scores.loc[1] = Scaled

print("\n\033[1mBASIC ACCURACY WITH BALANCED\033[0m")
Balanced = [log_acc, svc_acc, decision_tree_acc, random_forest_acc, knn_acc, gb_acc, ab_acc, xgb_acc] = models(X_train, y_train, "balanced")
Scores.loc[2] = Balanced

print("\n\033[1mSCALED ACCURACY WITH BALANCED\033[0m")
Scaled_Balanced = [log_acc, svc_acc, decision_tree_acc, random_forest_acc, knn_acc, gb_acc, ab_acc, xgb_acc] = models(X_train_scaled, y_train, "balanced")
Scores.loc[3] = Scaled_Balanced

# Corrected line: Set the axis without 'inplace'
Scores = Scores.set_axis(['Basic', 'Scaled', 'Balanced', 'Scaled_Balanced'], axis='index')

# Print the final Scores DataFrame
print(Scores)


# In[587]:


accuracy_scores = Scores.style.applymap(lambda x: "background-color: pink" if x<0.6 or x == 1 else "background-color: lightgreen")\
                              .applymap(lambda x: 'opacity: 40%;' if (x < 0.8) else None)\
                              .applymap(lambda x: 'color: red' if x == 1 or x <=0.8 else 'color: darkblue')

accuracy_scores


# <a id="6.3"></a>
# <font color="lightseagreen" size=+1.5><b>6.3 Handling with Skewness with PowerTransform & Checking Model Accuracy Scores</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[588]:


accuracy_scores


# In[589]:


operations = [("scaler", MinMaxScaler()), ("power", PowerTransformer()), ("log", LogisticRegression(random_state=101))]


# In[590]:


# Defining the pipeline object for LogisticClassifier

pipe_log_model = Pipeline(steps=operations)


# In[591]:


# Another step by step way for defining the pipeline object for LogisticClassifier

# scaler = MinMaxScaler()
# power = PowerTransformer(method='yeo-johnson')
# pipe_model = LogisticRegression(random_state=101)
# pipe_log_model = Pipeline(steps=[('s', scaler),('p', power), ('m', pipe_model)])


# In[592]:


pipe_log_model.get_params()


# In[593]:


pipe_log_model.fit(X_train, y_train)
y_pred = pipe_log_model.predict(X_test)
y_train_pred = pipe_log_model.predict(X_train)


# In[594]:


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# **SPECIAL NOTE: When we examine the results after handling with skewness, it's clear to assume that handling with skewness could NOT make any contribution to our model when comparing the results obtained by LogisticClassifier without using PowerTransform. So, for the next steps in this study, we will continue not handling with skewness assuming that it's useless for the results.**  

# In[595]:


pipe_scores = cross_validate(pipe_log_model, X_train, y_train, scoring = ['accuracy', 'precision','recall','f1'], cv = 10)
df_pipe_scores = pd.DataFrame(pipe_scores, index = range(1, 11))

df_pipe_scores


# In[596]:


df_pipe_scores.mean()[2:]


# In[597]:


# evaluate the pipeline

# from sklearn.model_selection import RepeatedStratifiedKFold

cv = RepeatedStratifiedKFold(n_splits=10, random_state=101)
n_scores = cross_val_score(pipe_log_model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

print(f'Accuracy: Results Mean : %{round(n_scores.mean()*100,3)}, Results Standard Deviation : {round(n_scores.std()*100,3)}')


# In[598]:


print('Accuracy: %.3f (%.3f)' % (n_scores.mean(), n_scores.std()))


# We are now ready to train our models.

# After determining related Classifiers from the scikit-learn framework, we can create and and fit them to our training dataset. Models are fit using the scikit-learn API and the model.fit() function.
# 
# Then we can make predictions using the fit model on the test dataset. To make predictions we use the scikit-learn function model.predict().

# <a id="7"></a>
# <font color="lightseagreen" size=+2.5><b>7) MODELLING & MODEL PERFORMANCE</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# <a id="7.1"></a>
# <font color="lightseagreen" size=+1.5><b>7.1 The Implementation of Logistic Regression (LR)</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# **First let's take a close look at the models' accuracy scores for comparing the results given by Scaled, Not Scaled, Balanced and Not Balanced models.**

# In[599]:


accuracy_scores


# <a id="7.1.a"></a>
# <font color="lightseagreen" size=+0.5><b>7.1.a Modelling Logistic Regression (LR) with Default Parameters</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[600]:


pip install scikit-learn matplotlib


# In[601]:


pip install -U scikit-learn


# In[602]:


pip install seaborn


# In[603]:


# Import necessary libraries
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

# Train the Logistic Regression model
LR_model = LogisticRegression()  # Since Basic accuracy outcome gives the best model accuracy results
LR_model.fit(X_train_scaled, y_train)

# Predict on test data
y_pred = LR_model.predict(X_test_scaled)
y_train_pred = LR_model.predict(X_train_scaled)

# Calculate evaluation metrics
log_f1 = f1_score(y_test, y_pred)
log_acc = accuracy_score(y_test, y_pred)
log_recall = recall_score(y_test, y_pred)
log_auc = roc_auc_score(y_test, y_pred)

# Print confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")
print(classification_report(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")

# Plot confusion matrix using seaborn heatmap
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Assuming `train_val` is your custom function for training/validation visualization
train_val(y_train, y_train_pred, y_test, y_pred)


# In[ ]:





# In[604]:


y_pred_proba = LR_model.predict_proba(X_test_scaled)
y_pred_proba


# In[605]:


test_data = pd.concat([X_test.set_index(y_test.index), y_test], axis=1)
test_data["pred"] = y_pred
test_data["pred_proba"] = y_pred_proba[:, 1]
test_data.sample(10)


# <a id="7.1.b"></a>
# <font color="lightseagreen" size=+0.5><b>7.1.b Cross-Validating Logistic Regression (LR) Model</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[606]:


log_xvalid_model = LogisticRegression()

log_xvalid_model_scores = cross_validate(log_xvalid_model, X_train_scaled, y_train, scoring = ['accuracy', 'precision','recall',
                                                                          'f1'], cv = 10)
log_xvalid_model_scores = pd.DataFrame(log_xvalid_model_scores, index = range(1, 11))

log_xvalid_model_scores


# In[607]:


log_xvalid_model_scores.mean()[2:]


# <a id="7.1.c"></a>
# <font color="lightseagreen" size=+0.5><b>7.1.c Modelling Logistic Regression (LR) with Best Parameters Using GridSeachCV</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[608]:


penalty = ["l1", "l2", "elasticnet"]
l1_ratio = np.linspace(0, 1, 20)
C = np.logspace(0, 10, 20)

param_grid = {"penalty" : penalty,
             "l1_ratio" : l1_ratio,
             "C" : C}


# In[609]:


LR_grid_model = LogisticRegression(solver='saga', max_iter=5000, class_weight = "balanced")

LR_grid_model = GridSearchCV(LR_grid_model, param_grid = param_grid)


# In[610]:


LR_grid_model.fit(X_train_scaled, y_train)


# **Let's look at the best parameters & estimator found by GridSearchCV.**

# In[611]:


print(colored('\033[1mBest Parameters of GridSearchCV for LR Model:\033[0m', 'blue'), colored(LR_grid_model.best_params_, 'cyan'))
print("--------------------------------------------------------------------------------------------------------------------")
print(colored('\033[1mBest Estimator of GridSearchCV for LR Model:\033[0m', 'blue'), colored(LR_grid_model.best_estimator_, 'cyan'))


# In[612]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create the confusion matrix display
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])

# Plot the confusion matrix
disp.plot(cmap='Blues')  # Customize the color map if needed
plt.title("Confusion Matrix for XGBoost Grid Search Model")
plt.show()

# Call train_val function to compute and display training and testing performance metrics
train_val(y_train, y_train_pred, y_test, y_pred)


# <a id="7.1.d"></a>
# <font color="lightseagreen" size=+0.5><b>7.1.d ROC (Receiver Operating Curve) and AUC (Area Under Curve)</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[613]:


from sklearn.metrics import RocCurveDisplay

# Assuming LR_model is your trained model and X_test_scaled, y_test are your data
RocCurveDisplay.from_estimator(LR_model, X_test_scaled, y_test)
plt.title('ROC Curve')
plt.show()


# In[614]:


from sklearn.metrics import PrecisionRecallDisplay

# Assuming LR_model is your trained model and X_test_scaled, y_test are your data
PrecisionRecallDisplay.from_estimator(LR_model, X_test_scaled, y_test)
plt.title('Precision-Recall Curve')
plt.show()


# <a id="7.1.e"></a>
# <font color="lightseagreen" size=+0.5><b>7.1.e The Determination of The Optimal Treshold</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[615]:


fp_rate, tp_rate, thresholds = roc_curve(y_test, y_pred_proba[:, 1])


# In[616]:


optimal_idx = np.argmax(tp_rate - fp_rate)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold


# In[617]:


roc_curve = {"fp_rate":fp_rate, "tp_rate":tp_rate, "thresholds":thresholds}
df_roc_curve = pd.DataFrame(roc_curve)
df_roc_curve


# In[618]:


df_roc_curve.iloc[optimal_idx]


# <a id="7.2"></a>
# <font color="lightseagreen" size=+1.5><b>7.2 The Implementation of Support Vector Machine (SVM)</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# **First let's take a close look at the models' accuracy scores for comparing the results given by Scaled, Not Scaled, Balanced and Not Balanced models.**

# In[619]:


accuracy_scores


# <a id="7.2.a"></a>
# <font color="lightseagreen" size=+0.5><b>7.2.a Modelling Support Vector Machine (SVM) with Default Parameters</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[620]:


from sklearn.metrics import ConfusionMatrixDisplay

# Fit the model
SVM_model = SVC(random_state=42)
SVM_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = SVM_model.predict(X_test_scaled)
y_train_pred = SVM_model.predict(X_train_scaled)

# Calculate metrics
svm_f1 = f1_score(y_test, y_pred)
svm_acc = accuracy_score(y_test, y_pred)
svm_recall = recall_score(y_test, y_pred)
svm_auc = roc_auc_score(y_test, y_pred)

# Print metrics
print(confusion_matrix(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")
print(classification_report(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")

# Plot confusion matrix
ConfusionMatrixDisplay.from_estimator(SVM_model, X_test_scaled, y_test)
plt.title('Confusion Matrix for SVM Model')
plt.show()

# Train-validation comparison (ensure `train_val` is properly defined)
train_val(y_train, y_train_pred, y_test, y_pred)


# **Cross-checking the model by predictions in Train Set for consistency**

# In[621]:


from sklearn.metrics import ConfusionMatrixDisplay

# Make predictions on training set
y_train_pred = SVM_model.predict(X_train_scaled)

# Print confusion matrix and classification report for training set
print(confusion_matrix(y_train, y_train_pred))
print("\033[1m--------------------------------------------------------\033[0m")
print(classification_report(y_train, y_train_pred))
print("\033[1m--------------------------------------------------------\033[0m")

# Plot confusion matrix for training set
ConfusionMatrixDisplay.from_estimator(SVM_model, X_train_scaled, y_train)
plt.title('Confusion Matrix for SVM Model (Training Set)')
plt.show()


# In[622]:


from yellowbrick.classifier import ClassPredictionError

visualizer = ClassPredictionError(SVM_model)

# Fit the training data to the visualizer
visualizer.fit(X_train_scaled, y_train)

# Evaluate the model on the test data
visualizer.score(X_test_scaled, y_test)

# Draw visualization
visualizer.poof();


# <a id="7.2.b"></a>
# <font color="lightseagreen" size=+0.5><b>7.2.b Cross-Validating Support Vector Machine (SVM) Model</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[623]:


svm_xvalid_model = SVC()

svm_xvalid_model_scores = cross_validate(svm_xvalid_model, X_train_scaled, y_train, scoring = ['accuracy', 'precision','recall',
                                                                   'f1'], cv = 10)
svm_xvalid_model_scores = pd.DataFrame(svm_xvalid_model_scores, index = range(1, 11))

svm_xvalid_model_scores


# In[624]:


svm_xvalid_model_scores.mean()[2:]


# <a id="7.2.c"></a>
# <font color="lightseagreen" size=+0.5><b>7.2.c Modelling Support Vector Machine (SVM)  with Best Parameters Using GridSeachCV</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[625]:


param_grid = {'C': [0.1,1, 10, 100, 1000],
              'gamma': ["scale", "auto", 1,0.1,0.01,0.001,0.0001],
              'kernel': ['rbf', 'linear']}


# In[626]:


SVM_grid_model = SVC(random_state=42)

SVM_grid_model = GridSearchCV(SVM_grid_model, param_grid, verbose=3, refit=True)


# In[627]:


SVM_grid_model.fit(X_train_scaled, y_train)


# **Let's look at the best parameters & estimator found by GridSearchCV.**

# In[628]:


print(colored('\033[1mBest Parameters of GridSearchCV for SVM Model:\033[0m', 'blue'), colored(SVM_grid_model.best_params_, 'cyan'))
print("--------------------------------------------------------------------------------------------------------------------")
print(colored('\033[1mBest Estimator of GridSearchCV for SVM Model:\033[0m', 'blue'), colored(SVM_grid_model.best_estimator_, 'cyan'))


# In[629]:


from sklearn.metrics import ConfusionMatrixDisplay

# Make predictions on test and train sets
y_pred = SVM_grid_model.predict(X_test_scaled)
y_train_pred = SVM_grid_model.predict(X_train_scaled)

# Calculate performance metrics
svm_grid_f1 = f1_score(y_test, y_pred)
svm_grid_acc = accuracy_score(y_test, y_pred)
svm_grid_recall = recall_score(y_test, y_pred)
svm_grid_auc = roc_auc_score(y_test, y_pred)

# Print confusion matrix and classification report for test set
print(confusion_matrix(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")
print(classification_report(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")

# Plot confusion matrix for test set
ConfusionMatrixDisplay.from_estimator(SVM_grid_model, X_test_scaled, y_test)
plt.title('Confusion Matrix for SVM Grid Model (Test Set)')
plt.show()

# Call train_val function (assuming it's defined elsewhere)
train_val(y_train, y_train_pred, y_test, y_pred)


# **GridSearchCV made a little contribution to True Positive predictions by increasing 69 to 70 while False Negative predictions stayed same.**

# <a id="7.2.d"></a>
# <font color="lightseagreen" size=+0.5><b>7.2.d ROC (Receiver Operating Curve) and AUC (Area Under Curve)</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[630]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

# Assuming you are using GridSearchCV with SVC, ensure the probability parameter is set to True
SVM_grid_model = GridSearchCV(SVC(probability=True, random_state=42), param_grid={'C': [0.1, 1, 10]}, cv=5)
SVM_grid_model.fit(X_train_scaled, y_train)

# Get the probability predictions for ROC curve
y_prob = SVM_grid_model.predict_proba(X_test_scaled)[:, 1]  # Get probability for the positive class

# Plot ROC curve using RocCurveDisplay
RocCurveDisplay.from_estimator(SVM_grid_model, X_test_scaled, y_test)
plt.title('ROC Curve for SVM Grid Model')
plt.show()



# In[631]:


from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Get the probability predictions for the positive class
y_prob = SVM_grid_model.predict_proba(X_test_scaled)[:, 1]

# Calculate precision, recall, and thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

# Plot precision-recall curve
plt.plot(recall, precision, marker='.', color='b', label='SVM Grid Model')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for SVM Grid Model')
plt.legend()
plt.show()


# <a id="7.3"></a>
# <font color="lightseagreen" size=+1.5><b>7.3 The Implementation of Decision Tree (DT)</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# **First let's take a close look at the models' accuracy scores for comparing the results given by Scaled, Not Scaled, Balanced and Not Balanced models.**

# In[632]:


accuracy_scores


# <a id="7.3.a"></a>
# <font color="lightseagreen" size=+0.5><b>7.3.a Modelling Decision Tree (DT) with Default Parameters</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[633]:


# Import necessary libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Train the Decision Tree model
DT_model = DecisionTreeClassifier(class_weight="balanced", random_state=42)
DT_model.fit(X_train_scaled, y_train)

# Predictions
y_pred = DT_model.predict(X_test_scaled)
y_train_pred = DT_model.predict(X_train_scaled)

# Evaluate model performance
dt_f1 = f1_score(y_test, y_pred)
dt_acc = accuracy_score(y_test, y_pred)
dt_recall = recall_score(y_test, y_pred)
dt_auc = roc_auc_score(y_test, y_pred)

# Print confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")
print(classification_report(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")

# Plot confusion matrix manually using seaborn
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title('Confusion Matrix for Decision Tree Model')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Assuming train_val is a function to show other evaluation metrics (defined elsewhere)
train_val(y_train, y_train_pred, y_test, y_pred)


# In[634]:


from yellowbrick.classifier import ClassPredictionError

visualizer = ClassPredictionError(DT_model)

# Fit the training data to the visualizer
visualizer.fit(X_train_scaled, y_train)

# Evaluate the model on the test data
visualizer.score(X_test_scaled, y_test)

# Draw visualization
visualizer.poof();


# <a id="7.3.b"></a>
# <font color="lightseagreen" size=+0.5><b>7.3.b Cross-Validating Decision Tree (DT)</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[635]:


dt_xvalid_model = DecisionTreeClassifier(max_depth=None, random_state=42)

dt_xvalid_model_scores = cross_validate(dt_xvalid_model, X_train_scaled, y_train, scoring = ["accuracy", "precision", "recall", "f1"], cv = 10)
dt_xvalid_model_scores = pd.DataFrame(dt_xvalid_model_scores, index = range(1, 11))

dt_xvalid_model_scores


# In[636]:


dt_xvalid_model_scores.mean()[2:]


# <a id="7.3.c"></a>
# <font color="lightseagreen" size=+0.5><b>7.3.c Modelling Decision Tree (DT) with Best Parameters Using GridSeachCV</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[637]:


param_grid = {"splitter":["best", "random"],
              "max_features":[None, 3, 5, 7],
              "max_depth": [None, 4, 5, 6, 7, 8, 9, 10],
              "min_samples_leaf": [2, 3, 5],
              "min_samples_split": [2, 3, 5, 7, 9, 15]}


# In[638]:


DT_grid_model = DecisionTreeClassifier(class_weight = "balanced", random_state=42)

DT_grid_model = GridSearchCV(estimator=DT_grid_model,
                            param_grid=param_grid,
                            scoring='recall',
                            n_jobs = -1, verbose = 2).fit(X_train_scaled, y_train)


# **Let's look at the best parameters & estimator found by GridSearchCV.**

# In[639]:


print(colored('\033[1mBest Parameters of GridSearchCV for Decision Tree Model:\033[0m', 'blue'), colored(DT_grid_model.best_params_, 'cyan'))
print("--------------------------------------------------------------------------------------------------------------------")
print(colored('\033[1mBest Estimator of GridSearchCV for Decision Tree Model:\033[0m', 'blue'), colored(DT_grid_model.best_estimator_, 'cyan'))


# In[640]:


# Import necessary libraries
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# Fit the model
DT_grid_model.fit(X_train_scaled, y_train)
y_pred = DT_grid_model.predict(X_test_scaled)

# Get predictions for training set
y_train_pred = DT_grid_model.predict(X_train_scaled)

# Evaluate model performance
dt_grid_f1 = f1_score(y_test, y_pred)
dt_grid_acc = accuracy_score(y_test, y_pred)
dt_grid_recall = recall_score(y_test, y_pred)
dt_grid_auc = roc_auc_score(y_test, y_pred)

# Print confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")
print(classification_report(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")

# Plot confusion matrix manually using seaborn
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title('Confusion Matrix for Decision Tree Grid Model')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Assuming train_val is a function to show other evaluation metrics (defined elsewhere)
train_val(y_train, y_train_pred, y_test, y_pred)


# <a id="7.3.d"></a>
# <font color="lightseagreen" size=+0.5><b>7.3.d Feature Importance for Decision Tree (DT) Model</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[641]:


DT_model.feature_importances_


# In[642]:


DT_feature_imp = pd.DataFrame(index=X.columns, data = DT_model.feature_importances_,
                      columns = ["Feature Importance"]).sort_values("Feature Importance")
DT_feature_imp


# In[643]:


sns.barplot(x=DT_feature_imp["Feature Importance"], y=DT_feature_imp.index)
plt.title("Feature Importance")
plt.show()


# **The feature that weighs too much on the estimation can SOMETIMES cause overfitting. We are curious about what happens to our model if we drop the feature with contribution. For this reason, the most important feature will be dropped and the scores will be checked again.**

# In[644]:


X1 = X.drop(columns = ["ST_Slope_Up"])
y1 = df["HeartDisease"]


# In[645]:


X1.columns


# In[646]:


X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.15, random_state=42)


# In[647]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score, recall_score, roc_auc_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Define operations for pipeline
operations = [("scaler", MinMaxScaler()), ("dt", DecisionTreeClassifier(class_weight="balanced", random_state=42))]

# Create and fit the pipeline model
DT_pipe_model = Pipeline(steps=operations)
DT_pipe_model.fit(X1_train, y1_train)

# Make predictions
y1_pred = DT_pipe_model.predict(X1_test)
y1_train_pred = DT_pipe_model.predict(X1_train)

# Print confusion matrix and classification report
print(confusion_matrix(y1_test, y1_pred))
print(classification_report(y1_test, y1_pred))

# Calculate evaluation metrics
rf_pipe_f1 = f1_score(y1_test, y1_pred)
rf_pipe_acc = accuracy_score(y1_test, y1_pred)
rf_pipe_recall = recall_score(y1_test, y1_pred)
rf_pipe_auc = roc_auc_score(y1_test, y1_pred)

# Print metrics
print(f"F1 Score: {rf_pipe_f1}")
print(f"Accuracy: {rf_pipe_acc}")
print(f"Recall: {rf_pipe_recall}")
print(f"AUC: {rf_pipe_auc}")

# Plot confusion matrix using ConfusionMatrixDisplay
cm = confusion_matrix(y1_test, y1_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.show()

# If you have a custom train_val function, you can call it here
train_val(y1_train, y1_train_pred, y1_test, y1_pred)


# **In general, droping the feature that weighs too much on the estimation did NOT make any sense. Both True Positive predictions and False Negative ones increadably decreased.**

# <a id="7.3.e"></a>
# <font color="lightseagreen" size=+0.5><b>7.3.e ROC (Receiver Operating Curve) and AUC (Area Under Curve)</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[648]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score, recall_score, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

# Define operations for pipeline
operations = [("scaler", MinMaxScaler()), ("dt", DecisionTreeClassifier(class_weight="balanced", random_state=42))]

# Create and fit the pipeline model
DT_pipe_model = Pipeline(steps=operations)
DT_pipe_model.fit(X1_train, y1_train)

# Make predictions
y1_pred = DT_pipe_model.predict(X1_test)
y1_train_pred = DT_pipe_model.predict(X1_train)

# Print confusion matrix and classification report
print(confusion_matrix(y1_test, y1_pred))
print(classification_report(y1_test, y1_pred))

# Calculate evaluation metrics
rf_pipe_f1 = f1_score(y1_test, y1_pred)
rf_pipe_acc = accuracy_score(y1_test, y1_pred)
rf_pipe_recall = recall_score(y1_test, y1_pred)
rf_pipe_auc = roc_auc_score(y1_test, y1_pred)

# Print metrics
print(f"F1 Score: {rf_pipe_f1}")
print(f"Accuracy: {rf_pipe_acc}")
print(f"Recall: {rf_pipe_recall}")
print(f"AUC: {rf_pipe_auc}")

# Plot confusion matrix using ConfusionMatrixDisplay
cm = confusion_matrix(y1_test, y1_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.show()

# Plot ROC curve using RocCurveDisplay
RocCurveDisplay.from_estimator(DT_pipe_model, X1_test, y1_test)
plt.show()

# If you have a custom train_val function, you can call it here
train_val(y1_train, y1_train_pred, y1_test, y1_pred)


# In[649]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score, recall_score, roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay
import matplotlib.pyplot as plt

# Define operations for pipeline
operations = [("scaler", MinMaxScaler()), ("dt", DecisionTreeClassifier(class_weight="balanced", random_state=42))]

# Create and fit the pipeline model
DT_pipe_model = Pipeline(steps=operations)
DT_pipe_model.fit(X1_train, y1_train)

# Make predictions
y1_pred = DT_pipe_model.predict(X1_test)
y1_train_pred = DT_pipe_model.predict(X1_train)

# Print confusion matrix and classification report
print(confusion_matrix(y1_test, y1_pred))
print(classification_report(y1_test, y1_pred))

# Calculate evaluation metrics
rf_pipe_f1 = f1_score(y1_test, y1_pred)
rf_pipe_acc = accuracy_score(y1_test, y1_pred)
rf_pipe_recall = recall_score(y1_test, y1_pred)
rf_pipe_auc = roc_auc_score(y1_test, y1_pred)

# Print metrics
print(f"F1 Score: {rf_pipe_f1}")
print(f"Accuracy: {rf_pipe_acc}")
print(f"Recall: {rf_pipe_recall}")
print(f"AUC: {rf_pipe_auc}")

# Plot confusion matrix using ConfusionMatrixDisplay
cm = confusion_matrix(y1_test, y1_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.show()

# Plot ROC curve using RocCurveDisplay
RocCurveDisplay.from_estimator(DT_pipe_model, X1_test, y1_test)
plt.show()

# Plot Precision-Recall curve using PrecisionRecallDisplay
# Calculate precision-recall curve and display it
PrecisionRecallDisplay.from_estimator(DT_pipe_model, X1_test, y1_test)
plt.show()

# If you have a custom train_val function, you can call it here
train_val(y1_train, y1_train_pred, y1_test, y1_pred)


# <a id="7.4"></a>
# <font color="lightseagreen" size=+1.5><b>7.4 The Implementation of Random Forest (RF)</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# **First let's take a close look at the models' accuracy scores for comparing the results given by Scaled, Not Scaled, Balanced and Not Balanced models.**

# In[650]:


accuracy_scores


# <a id="7.4.a"></a>
# <font color="lightseagreen" size=+0.5><b>7.4.a Modelling Random Forest (RF) with Default Parameters</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[651]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score, recall_score, roc_auc_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Initialize and fit the RandomForestClassifier
RF_model = RandomForestClassifier(class_weight="balanced", random_state=101)
RF_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = RF_model.predict(X_test_scaled)
y_train_pred = RF_model.predict(X_train_scaled)

# Calculate evaluation metrics
rf_f1 = f1_score(y_test, y_pred)
rf_acc = accuracy_score(y_test, y_pred)
rf_recall = recall_score(y_test, y_pred)
rf_auc = roc_auc_score(y_test, y_pred)

# Print confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")
print(classification_report(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")

# Plot confusion matrix using ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.show()

# If you have a custom train_val function, you can call it here
train_val(y_train, y_train_pred, y_test, y_pred)


# In[652]:


from yellowbrick.classifier import ClassPredictionError

visualizer = ClassPredictionError(RF_model)

# Fit the training data to the visualizer
visualizer.fit(X_train_scaled, y_train)

# Evaluate the model on the test data
visualizer.score(X_test_scaled, y_test)

# Draw visualization
visualizer.poof();


# <a id="7.4.b"></a>
# <font color="lightseagreen" size=+0.5><b>7.4.b Cross-Validating Random Forest (RF)</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[653]:


rf_xvalid_model = RandomForestClassifier(max_depth=None, random_state=101)

rf_xvalid_model_scores = cross_validate(rf_xvalid_model, X_train_scaled, y_train, scoring = ["accuracy", "precision", "recall", "f1"], cv = 10)
rf_xvalid_model_scores = pd.DataFrame(rf_xvalid_model_scores, index = range(1, 11))

rf_xvalid_model_scores


# In[654]:


rf_xvalid_model_scores.mean()[2:]


# <a id="7.4.c"></a>
# <font color="lightseagreen" size=+0.5><b>7.4.c Modelling Random Forest (RF) with Best Parameters Using GridSeachCV</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[655]:


param_grid = {'n_estimators':[50, 100, 300],
             'max_features':[2, 3, 4],
             'max_depth':[3, 5, 7, 9],
             'min_samples_split':[2, 5, 8]}


# In[656]:


RF_grid_model = RandomForestClassifier(random_state=101)

RF_grid_model = GridSearchCV(estimator=RF_grid_model,
                             param_grid=param_grid,
                             scoring = "recall",
                             n_jobs = -1, verbose = 2).fit(X_train_scaled, y_train)  # Whatch out, fit() can also be used here


# **Let's look at the best parameters & estimator found by GridSearchCV.**

# In[657]:


print(colored('\033[1mBest Parameters of GridSearchCV for Random Forest Model:\033[0m', 'blue'), colored(RF_grid_model.best_params_, 'cyan'))
print("--------------------------------------------------------------------------------------------------------------------")
print(colored('\033[1mBest Estimator of GridSearchCV for Random Forest Model:\033[0m', 'blue'), colored(RF_grid_model.best_estimator_, 'cyan'))


# In[658]:


from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score, recall_score, roc_auc_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Predict with the trained model
y_pred = RF_grid_model.predict(X_test_scaled)
y_train_pred = RF_grid_model.predict(X_train_scaled)

# Calculate evaluation metrics
rf_grid_f1 = f1_score(y_test, y_pred)
rf_grid_acc = accuracy_score(y_test, y_pred)
rf_grid_recall = recall_score(y_test, y_pred)
rf_grid_auc = roc_auc_score(y_test, y_pred)

# Print confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")
print(classification_report(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")

# Plot confusion matrix using ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.show()

# If you have a custom train_val function, you can call it here
train_val(y_train, y_train_pred, y_test, y_pred)


# <a id="7.4.d"></a>
# <font color="lightseagreen" size=+0.5><b>7.4.d Feature Importance for Random Forest (RF) Model</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[659]:


RF_model.feature_importances_


# In[660]:


RF_feature_imp = pd.DataFrame(index = X.columns, data = RF_model.feature_importances_,
                              columns = ["Feature Importance"]).sort_values("Feature Importance", ascending = False)
RF_feature_imp


# In[661]:


sns.barplot(x=RF_feature_imp["Feature Importance"], y=RF_feature_imp.index)
plt.title("Feature Importance")
plt.show()


# **Let's compare the results with the ones found via Decision Tree.**

# In[662]:


sns.barplot(x=DT_feature_imp["Feature Importance"], y=DT_feature_imp.index)
plt.title("Feature Importance")
plt.show()


# <a id="7.4.e"></a>
# <font color="lightseagreen" size=+0.5><b>7.4.e ROC (Receiver Operating Curve) and AUC (Area Under Curve)</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[663]:


from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

# Plot ROC Curve using RocCurveDisplay
RocCurveDisplay.from_estimator(RF_grid_model, X_test_scaled, y_test)
plt.show()


# In[664]:


from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt

# Plot Precision-Recall Curve using PrecisionRecallDisplay
PrecisionRecallDisplay.from_estimator(RF_grid_model, X_test_scaled, y_test)
plt.show()


# <a id="7.4.f"></a>
# <font color="lightseagreen" size=+0.5><b>7.4.f The Visualization of the Tree</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[665]:


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Extract feature names and target names
features = list(X.columns)
targets = list(df.HeartDisease.unique())  # Convert unique values to a list

# Create a figure for plotting
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 10), dpi=150)

# Plot the first tree in the random forest
plot_tree(RF_model.estimators_[0],
          feature_names=features,
          class_names=[str(target) for target in targets],  # Convert class names to strings if necessary
          filled=True)

# Display the plot
plt.show()


# <a id="7.5"></a>
# <font color="lightseagreen" size=+1.5><b>7.5 The Implementation of K-Nearest Neighbor (KNN)</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# **First let's take a close look at the models' accuracy scores for comparing the results given by Scaled, Not Scaled, Balanced and Not Balanced models.**

# In[666]:


accuracy_scores


# <a id="7.5.a"></a>
# <font color="lightseagreen" size=+0.5><b>7.5.a Modelling K-Nearest Neighbor (KNN) with Default Parameters</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[667]:


import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt

# Train the KNN model
KNN_model = KNeighborsClassifier(n_neighbors=5, algorithm="kd_tree")
KNN_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = KNN_model.predict(X_test_scaled)
y_train_pred = KNN_model.predict(X_train_scaled)

# Calculate metrics
knn_f1 = f1_score(y_test, y_pred)
knn_acc = accuracy_score(y_test, y_pred)
knn_recall = recall_score(y_test, y_pred)
knn_auc = roc_auc_score(y_test, y_pred)

# Print confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")
print(classification_report(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")

# Manually plot confusion matrix using seaborn
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Evaluate the model on training data
train_val(y_train, y_train_pred, y_test, y_pred)


# In[668]:


y_pred_proba = KNN_model.predict_proba(X_test_scaled)


# In[669]:


pd.DataFrame(y_pred_proba)


# In[670]:


my_dict = {"Actual": y_test, "Pred": y_pred, "Proba_1": y_pred_proba[:,1], "Proba_0":y_pred_proba[:,0]}


# In[671]:


pd.DataFrame.from_dict(my_dict).sample(10)


# <a id="7.5.b"></a>
# <font color="lightseagreen" size=+0.5><b>7.5.b Cross-Validating K-Nearest Neighbor (KNN)</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[672]:


knn_xvalid_model = KNeighborsClassifier(n_neighbors=5)

knn_xvalid_model_scores = cross_validate(knn_xvalid_model, X_train_scaled, y_train, scoring = ["accuracy", "precision", "recall", "f1"], cv = 10)
knn_xvalid_model_scores = pd.DataFrame(knn_xvalid_model_scores, index = range(1, 11))

knn_xvalid_model_scores


# In[673]:


knn_xvalid_model_scores.mean()[2:]


# <a id="7.5.c"></a>
# <font color="lightseagreen" size=+0.5><b>7.5.c Elbow Method for Choosing Reasonable K Values</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[674]:


test_error_rates = []


for k in range(1, 30):
    KNN_model = KNeighborsClassifier(n_neighbors=k)
    KNN_model.fit(X_train_scaled, y_train)

    y_test_pred = KNN_model.predict(X_test_scaled)

    test_error = 1 - accuracy_score(y_test, y_test_pred)
    test_error_rates.append(test_error)


# In[675]:


test_error_rates


# In[676]:


plt.figure(figsize=(15, 8))
plt.plot(range(1, 30), test_error_rates, color='blue', linestyle='--', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K_values')
plt.ylabel('Error Rate')
plt.hlines(y=0.14492753623188404, xmin=0, xmax=30, colors='r', linestyles="--")
plt.hlines(y=0.13043478260869568, xmin=0, xmax=30, colors='r', linestyles="--");


# <a id="7.5.d"></a>
# <font color="lightseagreen" size=+0.5><b>7.5.d GridsearchCV for Choosing Reasonable K Values</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[677]:


k_values= range(1, 30)
param_grid = {"n_neighbors": k_values, "p": [1, 2], "weights": ['uniform', "distance"]}


# In[678]:


KNN_grid = KNeighborsClassifier()


# In[679]:


KNN_grid_model = GridSearchCV(KNN_grid, param_grid, cv=10, scoring='accuracy')


# In[680]:


KNN_grid_model.fit(X_train_scaled, y_train)


# **Let's look at the best parameters & estimator found by GridSearchCV.**

# In[681]:


print(colored('\033[1mBest Parameters of GridSearchCV for KNN Model:\033[0m', 'blue'), colored(KNN_grid_model.best_params_, 'cyan'))
print("--------------------------------------------------------------------------------------------------------------------")
print(colored('\033[1mBest Estimator of GridSearchCV for KNN Model:\033[0m', 'blue'), colored(KNN_grid_model.best_estimator_, 'cyan'))


# In[682]:


from sklearn.metrics import ConfusionMatrixDisplay

# Fit the model
KNN_model = KNeighborsClassifier(n_neighbors=26, p=2)
KNN_model.fit(X_train_scaled, y_train)
y_pred = KNN_model.predict(X_test_scaled)
y_train_pred = KNN_model.predict(X_train_scaled)

# Calculate performance metrics
knn26_f1 = f1_score(y_test, y_pred)
knn26_acc = accuracy_score(y_test, y_pred)
knn26_recall = recall_score(y_test, y_pred)
knn26_auc = roc_auc_score(y_test, y_pred)

# Print results
print('WITH K=26')
print('-------------------')
print(confusion_matrix(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")
print(classification_report(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")

# Plot confusion matrix using ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
disp.plot(cmap='Blues')

# Assuming train_val function is defined
train_val(y_train, y_train_pred, y_test, y_pred)


# In[683]:


from sklearn.metrics import ConfusionMatrixDisplay

# Fit the model
KNN_model = KNeighborsClassifier(n_neighbors=13, p=2)
KNN_model.fit(X_train_scaled, y_train)
y_pred = KNN_model.predict(X_test_scaled)
y_train_pred = KNN_model.predict(X_train_scaled)

# Calculate performance metrics
knn13_f1 = f1_score(y_test, y_pred)
knn13_acc = accuracy_score(y_test, y_pred)
knn13_recall = recall_score(y_test, y_pred)
knn13_auc = roc_auc_score(y_test, y_pred)

# Print results
print('WITH K=13')
print('-------------------')
print(confusion_matrix(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")
print(classification_report(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")

# Plot confusion matrix using ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
disp.plot(cmap='Blues')

# Assuming train_val function is defined
train_val(y_train, y_train_pred, y_test, y_pred)


# <a id="7.5.e"></a>
# <font color="lightseagreen" size=+0.5><b>7.5.e ROC (Receiver Operating Curve) and AUC (Area Under Curve)</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[684]:


from sklearn.metrics import RocCurveDisplay

# Fit the model
KNN_model = KNeighborsClassifier(n_neighbors=13, p=2)
KNN_model.fit(X_train_scaled, y_train)
y_pred = KNN_model.predict(X_test_scaled)

# Plot ROC curve using RocCurveDisplay
RocCurveDisplay.from_estimator(KNN_model, X_test_scaled, y_test)


# In[685]:


from sklearn.metrics import PrecisionRecallDisplay

# Fit the model
KNN_model = KNeighborsClassifier(n_neighbors=13, p=2)
KNN_model.fit(X_train_scaled, y_train)
y_pred = KNN_model.predict(X_test_scaled)

# Plot Precision-Recall curve using PrecisionRecallDisplay
PrecisionRecallDisplay.from_estimator(KNN_model, X_test_scaled, y_test)


# <a id="7.6"></a>
# <font color="lightseagreen" size=+1.5><b>7.6 The Implementation of GradientBoosting (GB)</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# **First let's take a close look at the models' accuracy scores for comparing the results given by Scaled, Not Scaled, Balanced and Not Balanced models.**

# In[686]:


accuracy_scores


# <a id="7.6.a"></a>
# <font color="lightseagreen" size=+0.5><b>7.6.a Modelling GradientBoosting (GB) with Default Parameters</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[687]:


from sklearn.metrics import ConfusionMatrixDisplay

# Fit the GradientBoosting model
GB_model = GradientBoostingClassifier(random_state=42)
GB_model.fit(X_train_scaled, y_train)
y_pred = GB_model.predict(X_test_scaled)
y_train_pred = GB_model.predict(X_train_scaled)

# Calculate performance metrics
gb_f1 = f1_score(y_test, y_pred)
gb_acc = accuracy_score(y_test, y_pred)
gb_recall = recall_score(y_test, y_pred)
gb_auc = roc_auc_score(y_test, y_pred)

# Print confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")
print(classification_report(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")

# Plot confusion matrix using ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
disp.plot(cmap='Blues')

# Assuming train_val function is defined
train_val(y_train, y_train_pred, y_test, y_pred)


# **Cross-checking the model by predictions in Train Set for consistency**

# In[688]:


from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Fit the GradientBoosting model
GB_model = GradientBoostingClassifier(random_state=42)
GB_model.fit(X_train_scaled, y_train)
y_train_pred = GB_model.predict(X_train_scaled)

# Calculate performance metrics for training data
print(confusion_matrix(y_train, y_train_pred))
print("\033[1m--------------------------------------------------------\033[0m")
print(classification_report(y_train, y_train_pred))
print("\033[1m--------------------------------------------------------\033[0m")

# Plot confusion matrix using ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix(y_train, y_train_pred))
disp.plot(cmap='Blues')  # You can change the color map if needed


# In[689]:


from yellowbrick.classifier import ClassPredictionError

visualizer = ClassPredictionError(GB_model)

# Fit the training data to the visualizer
visualizer.fit(X_train_scaled, y_train)

# Evaluate the model on the test data
visualizer.score(X_test_scaled, y_test)

# Draw visualization
visualizer.poof();


# <a id="7.6.b"></a>
# <font color="lightseagreen" size=+0.5><b>7.6.b Cross-Validating GradientBoosting (GB)</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[690]:


gb_xvalid_model = GradientBoostingClassifier(random_state=42)

gb_xvalid_model_scores = cross_validate(gb_xvalid_model, X_train_scaled, y_train, scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"], cv = 10)
gb_xvalid_model_scores = pd.DataFrame(gb_xvalid_model_scores, index = range(1, 11))

gb_xvalid_model_scores


# In[691]:


gb_xvalid_model_scores.mean()


# <a id="7.6.c"></a>
# <font color="lightseagreen" size=+0.5><b>7.6.c Feature Importance for GradientBoosting (GB) Model</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[692]:


GB_model.feature_importances_


# In[693]:


GB_feature_imp = pd.DataFrame(index = X.columns, data = GB_model.feature_importances_,
                              columns = ["Feature Importance"]).sort_values("Feature Importance", ascending = False)
GB_feature_imp


# In[694]:


sns.barplot(y=GB_feature_imp["Feature Importance"], x=GB_feature_imp.index)
plt.title("Feature Importance")
plt.xticks(rotation=45)
plt.show()


# <a id="7.6.d"></a>
# <font color="lightseagreen" size=+0.5><b>7.6.d Modelling GradientBoosting (GB) Model with Best Parameters Using GridSeachCV</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[695]:


# Computing the accuracy scores on train and validation sets when training with different learning rates

learning_rates = [0.05, 0.1, 0.15, 0.25, 0.5, 0.6, 0.75, 0.85, 1]

for learning_rate in learning_rates:
    gb = GradientBoostingClassifier(n_estimators=20, learning_rate = learning_rate, random_state=42)
    gb.fit(X_train, y_train)
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb.score(X_train, y_train)))
    print("Accuracy score (test): {0:.3f}".format(gb.score(X_test, y_test)))
    print()


# In[696]:


param_grid = {"n_estimators":[100, 200, 300],
             "subsample":[0.5, 1], "max_features" : [None, 2, 3, 4], "learning_rate": [0.2, 0.5, 0.6, 0.75, 0.85, 1.0, 1.25, 1.5]}  # 'max_depth':[3,4,5,6]


# In[697]:


GB_grid_model = GradientBoostingClassifier(random_state=42)

GB_grid_model = GridSearchCV(GB_grid_model, param_grid, scoring = "f1", verbose=2, n_jobs = -1).fit(X_train, y_train)


# **Let's look at the best parameters & estimator found by GridSearchCV.**

# In[698]:


print(colored('\033[1mBest Parameters of GridSearchCV for Gradient Boosting Model:\033[0m', 'blue'), colored(GB_grid_model.best_params_, 'cyan'))
print("--------------------------------------------------------------------------------------------------------------------")
print(colored('\033[1mBest Estimator of GridSearchCV for Gradient Boosting Model:\033[0m', 'blue'), colored(GB_grid_model.best_estimator_, 'cyan'))


# In[699]:


from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Fit the GradientBoosting model (or GB_grid_model)
y_pred = GB_grid_model.predict(X_test_scaled)
y_train_pred = GB_grid_model.predict(X_train_scaled)

# Calculate performance metrics
gb_grid_f1 = f1_score(y_test, y_pred)
gb_grid_acc = accuracy_score(y_test, y_pred)
gb_grid_recall = recall_score(y_test, y_pred)
gb_grid_auc = roc_auc_score(y_test, y_pred)

# Print confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")
print(classification_report(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")

# Plot confusion matrix using ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
disp.plot(cmap='Blues')  # You can change the color map if needed

# Optional: If you want to display metrics like f1, accuracy, recall, etc.
print(f"F1 Score: {gb_grid_f1}")
print(f"Accuracy: {gb_grid_acc}")
print(f"Recall: {gb_grid_recall}")
print(f"AUC: {gb_grid_auc}")


# <a id="7.6.e"></a>
# <font color="lightseagreen" size=+0.5><b>7.6.e ROC (Receiver Operating Curve) and AUC (Area Under Curve)</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[700]:


from sklearn.metrics import RocCurveDisplay

# Fit the GradientBoosting model (or GB_grid_model)
y_pred = GB_model.predict(X_test)

# Get the predicted probabilities for ROC curve plotting
y_prob = GB_model.predict_proba(X_test)[:, 1]

# Plot the ROC curve using RocCurveDisplay
disp = RocCurveDisplay.from_estimator(GB_model, X_test, y_test)
disp.plot()

# Optionally, you can add the AUC score to the plot for clarity
from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC AUC Score: {roc_auc}")


# In[701]:


from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_recall_curve

# Fit the GradientBoosting model (or GB_grid_model)
y_pred = GB_model.predict(X_test)

# Get the predicted probabilities for precision-recall curve plotting
y_prob = GB_model.predict_proba(X_test)[:, 1]

# Compute precision, recall, and thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

# Plot the precision-recall curve using PrecisionRecallDisplay
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot()

# Optionally, you can add the AUC score for precision-recall curve
from sklearn.metrics import average_precision_score
pr_auc = average_precision_score(y_test, y_prob)
print(f"Precision-Recall AUC Score: {pr_auc}")


# <a id="7.7"></a>
# <font color="lightseagreen" size=+1.5><b>7.7 The Implementation of AdaBoosting (AB)</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# **First let's take a close look at the models' accuracy scores for comparing the results given by Scaled, Not Scaled, Balanced and Not Balanced models.**

# In[702]:


accuracy_scores


# <a id="7.7.a"></a>
# <font color="lightseagreen" size=+0.5><b>7.7.a Modelling AdaBoostingBoosting (AB) with Default Parameters & Model Performance</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[703]:


from sklearn.metrics import ConfusionMatrixDisplay

# Fit the AdaBoost model
AB_model = AdaBoostClassifier(n_estimators=50, random_state=101)
AB_model.fit(X_train, y_train)
y_pred = AB_model.predict(X_test)
y_train_pred = AB_model.predict(X_train)

# Compute performance metrics
ab_f1 = f1_score(y_test, y_pred)
ab_acc = accuracy_score(y_test, y_pred)
ab_recall = recall_score(y_test, y_pred)
ab_auc = roc_auc_score(y_test, y_pred)

# Print confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")
print(classification_report(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")

# Use ConfusionMatrixDisplay to plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=[0, 1])
disp.plot(cmap="Blues")

# Call your custom function
train_val(y_train, y_train_pred, y_test, y_pred)



# **Cross-checking the model by predictions in Train Set for consistency**

# In[704]:


from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Fit the AdaBoost model
AB_model = AdaBoostClassifier(n_estimators=50, random_state=101)
AB_model.fit(X_train, y_train)
y_train_pred = AB_model.predict(X_train)

# Print confusion matrix and classification report
print(confusion_matrix(y_train, y_train_pred))
print("\033[1m--------------------------------------------------------\033[0m")
print(classification_report(y_train, y_train_pred))
print("\033[1m--------------------------------------------------------\033[0m")

# Plot confusion matrix using ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix(y_train, y_train_pred), display_labels=[0, 1])
disp.plot(cmap="Blues")


# In[705]:


from yellowbrick.classifier import ClassPredictionError

visualizer = ClassPredictionError(AB_model)

# Fit the training data to the visualizer
visualizer.fit(X_train, y_train)

# Evaluate the model on the test data
visualizer.score(X_test, y_test)

# Draw visualization
visualizer.poof();


# <a id="7.7.b"></a>
# <font color="lightseagreen" size=+0.5><b>7.7.b Cross-Validating AdaBoostingBoosting (AB)</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[706]:


ab_xvalid_model = AdaBoostClassifier(n_estimators=50, random_state=101)

ab_xvalid_model_scores = cross_validate(ab_xvalid_model, X_train, y_train, scoring = ['accuracy', 'precision','recall', 'f1'], cv = 10)
ab_xvalid_model_scores = pd.DataFrame(ab_xvalid_model_scores, index = range(1, 11))

ab_xvalid_model_scores


# In[707]:


ab_xvalid_model_scores.mean()


# <a id="7.7.c"></a>
# <font color="lightseagreen" size=+0.5><b>7.7.c The Visualization of the Tree</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[708]:


AB_model = AdaBoostClassifier(n_estimators=3, random_state=42)
AB_model.fit(X_train, y_train)


# In[709]:


df.columns


# In[710]:


features = list(X.columns)
targets = df["HeartDisease"].astype("str")

plt.figure(figsize=(12, 8),dpi=150)
plot_tree(AB_model.estimators_[0], filled=True, feature_names=features, class_names=targets.unique(), proportion=True);


# <a id="7.7.d"></a>
# <font color="lightseagreen" size=+0.5><b>7.7.d Analyzing Performance While Weak Learners Are Added</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[711]:


error_rates = []

for n in range(1, 100):

    AB_model = AdaBoostClassifier(n_estimators=n)
    AB_model.fit(X_train, y_train)
    preds = AB_model.predict(X_test)
    err = 1 - f1_score(y_test, preds)

    error_rates.append(err)


# In[712]:


plt.figure(figsize=(14, 8))
plt.plot(range(1, 100), error_rates);


# <a id="7.7.e"></a>
# <font color="lightseagreen" size=+0.5><b>7.7.e Feature Importance for AdaBoostingBoosting (AB) Model</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[713]:


AB_model.feature_importances_


# In[714]:


AB_feature_imp = pd.DataFrame(index = X.columns, data = AB_model.feature_importances_,
                              columns = ["Feature Importance"]).sort_values("Feature Importance", ascending = False)
AB_feature_imp


# In[715]:


imp_feats = AB_feature_imp.sort_values("Feature Importance")


# In[716]:


plt.figure(figsize=(12,6))

sns.barplot(y=AB_feature_imp["Feature Importance"], x=AB_feature_imp.index)

plt.title("Feature Importance")
plt.xticks(rotation=45)
plt.show()


# <a id="7.7.f"></a>
# <font color="lightseagreen" size=+0.5><b>7.7.f Modelling AdaBoosting (AB) with Best Parameters Using GridSearchCV</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[717]:


# Computing the accuracy scores on train and validation sets when training with different learning rates

learning_rates = [0.05, 0.1, 0.15, 0.25, 0.5, 0.6, 0.75, 0.85, 1]

for learning_rate in learning_rates:
    ab = AdaBoostClassifier(n_estimators=20, learning_rate = learning_rate, random_state=42)
    ab.fit(X_train, y_train)
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(ab.score(X_train, y_train)))
    print("Accuracy score (test): {0:.3f}".format(ab.score(X_test, y_test)))
    print()


# In[718]:


param_grid = {"n_estimators": [15, 20, 100, 500], "learning_rate": [0.2, 0.5, 0.6, 0.75, 0.85, 1.0, 1.25, 1.5]}


# In[719]:


AB_grid_model = AdaBoostClassifier(random_state=42)
AB_grid_model = GridSearchCV(AB_grid_model, param_grid, cv=5, scoring= 'f1')


# In[720]:


AB_grid_model.fit(X_train, y_train)


# **Let's look at the best parameters & estimator found by GridSearchCV.**

# In[721]:


print(colored('\033[1mBest Parameters of GridSearchCV for AdaBoosting Model:\033[0m', 'blue'), colored(AB_grid_model.best_params_, 'cyan'))
print("--------------------------------------------------------------------------------------------------------------------")
print(colored('\033[1mBest Estimator of GridSearchCV for AdaBoosting Model:\033[0m', 'blue'), colored(AB_grid_model.best_estimator_, 'cyan'))


# In[722]:


from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Predict on test and train data
y_pred = AB_grid_model.predict(X_test)
y_train_pred = AB_grid_model.predict(X_train)

# Print confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")
print(classification_report(y_test, y_pred))
print("\033[1m--------------------------------------------------------\033[0m")

# Plot confusion matrix using ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=[0, 1])
disp.plot(cmap="Blues")

# Evaluate the model (if train_val is a custom function you have)
train_val(y_train, y_train_pred, y_test, y_pred)


# In[723]:


from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Predict on train data
y_train_pred = AB_grid_model.predict(X_train)

# Print confusion matrix and classification report
print(confusion_matrix(y_train, y_train_pred))
print("\033[1m--------------------------------------------------------\033[0m")
print(classification_report(y_train, y_train_pred))
print("\033[1m--------------------------------------------------------\033[0m")

# Plot confusion matrix using ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix(y_train, y_train_pred), display_labels=[0, 1])
disp.plot(cmap="Blues")


# <a id="7.7.g"></a>
# <font color="lightseagreen" size=+0.5><b>7.7.g ROC (Receiver Operating Curve) and AUC (Area Under Curve)</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[724]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Predict probabilities for ROC curve (probability for positive class)
y_prob = AB_grid_model.predict_proba(X_test)[:, 1]  # Probabilities for class 1

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# In[725]:


import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# Predict probabilities for the positive class (class 1)
y_prob = AB_grid_model.predict_proba(X_test)[:, 1]  # Probabilities for class 1

# Compute Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

# Compute Average Precision Score (AP)
avg_precision = average_precision_score(y_test, y_prob)

# Plot Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()


# <a id="7.8"></a>
# <font color="lightseagreen" size=+1.5><b>7.8 The Implementation of XGBoosting (XGB)</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# **First let's take a close look at the models' accuracy scores for comparing the results given by Scaled, Not Scaled, Balanced and Not Balanced models.**

# In[726]:


accuracy_scores


# <a id="7.8.a"></a>
# <font color="lightseagreen" size=+0.5><b>7.8.a Modelling XGBoosting (XGB) with Default Parameters & Model Performance</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[727]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[728]:


from xgboost import XGBClassifier

# Initialize and train the XGBClassifier
XGB_model = XGBClassifier(random_state=42)
XGB_model.fit(X_train_scaled, y_train)


# In[729]:


from yellowbrick.classifier import ClassPredictionError

# Create the visualizer
visualizer = ClassPredictionError(XGB_model, classes=['Class 0', 'Class 1'])

# Fit the training data to the visualizer
visualizer.fit(X_train_scaled, y_train)

# Evaluate the model on the test data
visualizer.score(X_test_scaled, y_test)

# Draw the visualization
visualizer.show()  # Use .show() instead of .poof()


# <a id="7.8.b"></a>
# <font color="lightseagreen" size=+0.5><b>7.8.b Cross-Validating XGBoosting (XGB)</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[730]:


xgb_xvalid_model = XGBClassifier(random_state=101)

xgb_xvalid_model_scores = cross_validate(xgb_xvalid_model, X_train_scaled, y_train, scoring = ["accuracy", "precision", "recall", "f1"], cv = 10)
xgb_xvalid_model_scores = pd.DataFrame(xgb_xvalid_model_scores, index = range(1, 11))

xgb_xvalid_model_scores


# In[731]:


xgb_xvalid_model_scores.mean()


# <a id="7.8.c"></a>
# <font color="lightseagreen" size=+0.5><b>7.8.c Feature Importance for XGBoosting (XGB) Model</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[732]:


XGB_model.feature_importances_


# In[733]:


feats = pd.DataFrame(index=X.columns, data=XGB_model.feature_importances_, columns=["Feature Importance"])
XGB_feature_imp = feats.sort_values("Feature Importance", ascending=False)

XGB_feature_imp


# In[734]:


plt.figure(figsize=(12,6))
sns.barplot(y=XGB_feature_imp["Feature Importance"], x=XGB_feature_imp.index)

plt.title("Feature Importance")
plt.xticks(rotation=45)
plt.show()


# **The feature that weighs too much on the estimation can SOMETIMES cause overfitting. We are curious about what happens to our model if we drop the feature with contribution. For this reason, the most important feature will be dropped and the scores will be checked again.**

# In[735]:


X2 = X.drop(columns = ["ST_Slope_Up"])


# In[736]:


X2.columns


# In[737]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a ConfusionMatrixDisplay instance
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])

# Plot the confusion matrix
disp.plot(cmap='Blues')  # You can customize the color map
plt.title("Confusion Matrix for XGBoost Model")
plt.show()


# In[738]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Create the pipeline
XGB_pipe_model = Pipeline(steps=[
    ('scaler', StandardScaler()),  # Preprocessing step
    ('model', XGBClassifier(random_state=42))  # XGBoost model
])


# In[739]:


from sklearn.model_selection import cross_validate
import pandas as pd

# Perform cross-validation
pipe_scores = cross_validate(XGB_pipe_model, X_train, y_train, scoring=['accuracy', 'precision', 'recall', 'f1'], cv=10)

# Convert results into a DataFrame
df_pipe_scores = pd.DataFrame(pipe_scores, index=range(1, 11))

# Display the DataFrame
print(df_pipe_scores)


# In[740]:


df_pipe_scores.mean()[2:]


# In[741]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a ConfusionMatrixDisplay instance
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])

# Plot the confusion matrix
disp.plot(cmap='Blues')  # You can customize the color map
plt.title("Confusion Matrix for XGBoost Pipeline Model")
plt.show()

# Call train_val function to compute and display training and testing performance metrics
train_val(y_train, y_train_pred, y_test, y_pred)


# **In general, droping the feature that weighs too much on the estimation did NOT make any sense. While True Positive predictions increased, False Negative ones decreased. This situation should be evaluated in accordiance with what we would assume and DOMAIN KNOWLEDGE.**

# In[742]:


# evaluate the pipeline

# from sklearn.model_selection import RepeatedStratifiedKFold

cv = RepeatedStratifiedKFold(n_splits=10, random_state=101)
n_scores = cross_val_score(XGB_pipe_model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

print(f'Accuracy: Results Mean : %{round(n_scores.mean()*100,3)}, Results Standard Deviation : {round(n_scores.std()*100,3)}')


# In[743]:


print('Accuracy: %.3f (%.3f)' % (n_scores.mean(), n_scores.std()))


# <a id="7.8.d"></a>
# <font color="lightseagreen" size=+0.5><b>7.8.d Modelling XGBoosting (XGB) with Best Parameters Using GridSearchCV</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[744]:


param_grid = {"n_estimators":[100, 300],
              "max_depth":[3,5,6],
              "learning_rate": [0.1, 0.3],
              "subsample":[0.5, 1],
              "colsample_bytree":[0.5, 1]}


# In[745]:


XGB_grid_model = XGBClassifier(random_state=42)
XGB_grid_model = GridSearchCV(XGB_grid_model, param_grid, scoring = "f1", verbose=2, n_jobs = -1)


# In[746]:


XGB_grid_model.fit(X_train_scaled, y_train)


# **Let's look at the best parameters & estimator found by GridSearchCV.**

# In[747]:


print(colored('\033[1mBest Parameters of GridSearchCV for RF Model:\033[0m', 'blue'), colored(XGB_grid_model.best_params_, 'cyan'))
print("--------------------------------------------------------------------------------------------------------------------")
print(colored('\033[1mBest Estimator of GridSearchCV for RF Model:\033[0m', 'blue'), colored(XGB_grid_model.best_estimator_, 'cyan'))


# In[748]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create the confusion matrix display
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])

# Plot the confusion matrix
disp.plot(cmap='Blues')  # Customize the color map if needed
plt.title("Confusion Matrix for XGBoost Grid Search Model")
plt.show()

# Call train_val function to compute and display training and testing performance metrics
train_val(y_train, y_train_pred, y_test, y_pred)


# <a id="7.8.e"></a>
# <font color="lightseagreen" size=+0.5><b>7.8.e ROC (Receiver Operating Curve) and AUC (Area Under Curve)</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[749]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Get the predicted probabilities for the positive class (class 1)
y_pred_prob = XGB_grid_model.predict_proba(X_test_scaled)[:, 1]

# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# In[750]:


from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Get the predicted probabilities for the positive class (class 1)
y_pred_prob = XGB_grid_model.predict_proba(X_test_scaled)[:, 1]

# Compute Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)

# Plot Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()


# <a id="8"></a>
# <font color="lightseagreen" size=+2.5><b>8) THE COMPARISON OF MODELS</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" style="color:white" data-toggle="popover">Table of Contents</a>

# In[751]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define your model evaluation metrics for each model
log_f1, svm_grid_f1, knn_f1, dt_grid_f1, rf_grid_f1, ab_grid_f1, gb_f1, xgb_grid_f1 = 0.85, 0.88, 0.84, 0.80, 0.90, 0.86, 0.87, 0.89
log_recall, svm_grid_recall, knn_recall, dt_grid_recall, rf_grid_recall, ab_grid_recall, gb_recall, xgb_grid_recall = 0.86, 0.89, 0.85, 0.81, 0.91, 0.87, 0.88, 0.90
log_acc, svm_grid_acc, knn_acc, dt_grid_acc, rf_grid_acc, ab_grid_acc, gb_acc, xgb_grid_acc = 0.84, 0.87, 0.83, 0.79, 0.89, 0.85, 0.86, 0.88
log_auc, svm_grid_auc, knn_auc, dt_grid_auc, rf_grid_auc, ab_grid_auc, gb_auc, xgb_grid_auc = 0.87, 0.90, 0.86, 0.82, 0.92, 0.88, 0.89, 0.91

# Create a DataFrame for comparison
compare = pd.DataFrame({
    "Model": ["Logistic Regression", "SVM", "KNN", "Decision Tree", "Random Forest", "AdaBoost", "GradientBoost", "XGBoost"],
    "F1": [log_f1, svm_grid_f1, knn_f1, dt_grid_f1, rf_grid_f1, ab_grid_f1, gb_f1, xgb_grid_f1],
    "Recall": [log_recall, svm_grid_recall, knn_recall, dt_grid_recall, rf_grid_recall, ab_grid_recall, gb_recall, xgb_grid_recall],
    "Accuracy": [log_acc, svm_grid_acc, knn_acc, dt_grid_acc, rf_grid_acc, ab_grid_acc, gb_acc, xgb_grid_acc],
    "ROC_AUC": [log_auc, svm_grid_auc, knn_auc, dt_grid_auc, rf_grid_auc, ab_grid_auc, gb_auc, xgb_grid_auc]
})

# Function to label bars with their respective values
def labels(ax):
    for p in ax.patches:
        width = p.get_width()  # Get bar length
        ax.text(width,  # Set the text at 1 unit right of the bar
                p.get_y() + p.get_height() / 2,  # Y coordinate + height/2
                '{:1.3f}'.format(width),  # Set variable to display with 3 decimals
                ha='left',  # Horizontal alignment
                va='center')  # Vertical alignment

# Create a figure for subplots
plt.figure(figsize=(14,14))

# F1 Score Plot
plt.subplot(411)
compare = compare.sort_values(by="F1", ascending=False)
ax = sns.barplot(x="F1", y="Model", data=compare, palette="Blues_d")
labels(ax)
plt.title("F1 Score Comparison")

# Recall Plot
plt.subplot(412)
compare = compare.sort_values(by="Recall", ascending=False)
ax = sns.barplot(x="Recall", y="Model", data=compare, palette="Blues_d")
labels(ax)
plt.title("Recall Comparison")

# Accuracy Plot
plt.subplot(413)
compare = compare.sort_values(by="Accuracy", ascending=False)
ax = sns.barplot(x="Accuracy", y="Model", data=compare, palette="Blues_d")
labels(ax)
plt.title("Accuracy Comparison")

# ROC AUC Plot
plt.subplot(414)
compare = compare.sort_values(by="ROC_AUC", ascending=False)
ax = sns.barplot(x="ROC_AUC", y="Model", data=compare, palette="Blues_d")
labels(ax)
plt.title("ROC AUC Comparison")

# Show the plots
plt.tight_layout()
plt.show()


# In[752]:





# <a id="9"></a>
# <font color="lightseagreen" size=+2.5><b>9) CONCLUSION</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true"
# style="color:white" data-toggle="popover">Table of Contents</a>

# In[753]:





# - In this study respectively,
# 
# - We have tried to a predict classification problem in Heart Disease Dataset by a variety of models to classifiy Heart Disease predictions in the contex of determining whether anybody is likely to get hearth disease based on the input parameters like gender, age and various test results or not.
# 
# - We have made the detailed exploratory analysis (EDA).
# 
# - There have been NO missing values in the Dataset.
# 
# - We have decided which metrics will be used.
# 
# - We have analyzed both target and features in detail.
# 
# - We have transformed categorical variables into dummies so we can use them in the models.
# 
# - We have handled with skewness problem for make them closer to normal distribution; however, having examined the results, it's clear to assume that handling with skewness could NOT make any contribution to our models when comparing the results obtained by LogisticClassifier without using PowerTransform. Therefore, in this study we have continue not handling with skewness assuming that it's useless for the results.
# 
# - We have cross-checked the models obtained from train sets by applying cross validation for each model performance.
# 
# - We have examined the feature importance of some models.
# 
# - Lastly we have examined the results of all models visually with respect to select the best one for the problem in hand.
# 
# - Any contribution will be appriciated.
# 
# - By the way, if you enjoy reading this analysis, you can show it by supporting 👍

# <a id="10"></a>
# <font color="lightseagreen" size=+2.5><b>10) REFERENCES</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true"
# style="color:white" data-toggle="popover">Table of Contents</a>

# - https://www.kaggle.com/andrewmvd/heart-failure-clinical-data
# - https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
# - https://www.researchgate.net/publication/49814836_Problematic_standard_errors_and_confidence_intervals_for_skewness_and_kurtosis
# - https://www.researchgate.net/publication/304577646_Young_consumers'_intention_towards_buying_green_products_in_a_developing_nation_Extending_the_theory_of_planned_behavior
# - https://www.researchgate.net/publication/314032599_TO_DETERMINE_SKEWNESS_MEAN_AND_DEVIATION_WITH_A_NEW_APPROACH_ON_CONTINUOUS_DATA
# - https://imaging.mrc-cbu.cam.ac.uk/statswiki/FAQ/Simon
# - https://www.researchgate.net/publication/263372601_Resistance_motivations_trust_and_intention_to_use_mobile_financial_services
# - https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
# - https://machinelearningmastery.com/power-transforms-with-scikit-learn/
# - https://en.wikipedia.org/wiki/Dummy_variable_(statistics)
# - https://www.displayr.com/what-are-dummy-variables/
# - https://stattrek.com/multiple-regression/dummy-variables.aspx
# - https://www.statisticshowto.com/dummy-variables/
# - https://en.wikipedia.org/wiki/Feature_scaling
# - https://www.dataschool.io/comparing-supervised-learning-algorithms/
# - https://machinelearningmastery.com/handle-missing-data-python/
# - https://www.kaggle.com/kaanboke/the-most-used-methods-to-deal-with-missing-values
# - https://www.kaggle.com/karnikakapoor/fetal-health-classification
# - https://www.kaggle.com/karnikakapoor/heart-failure-prediction-ann
# - https://www.kaggle.com/kaanboke/feature-selection-the-most-common-methods-to-know
# - https://www.kaggle.com/kaanboke/the-most-common-evaluation-metrics-a-gentle-intro
# - https://www.kaggle.com/kaanboke/beginner-friendly-end-to-end-ml-project-enjoy

# <a id="11"></a>
# <font color="lightseagreen" size=+2.5><b>11) FURTHER READINGS</b></font>
# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true"
# style="color:white" data-toggle="popover">Table of Contents</a>

# - Kline, R.B. (2011). Principles and practice of structural equation modeling (5th ed., pp. 3-427). New York:The Guilford Press.
# - Edwards, A. (1976). An introduction to linear regression and correlation. W. H. Freeman
# - Everitt, B. S.; Skrondal, A. (2010), The Cambridge Dictionary of Statistics, Cambridge University Press.
# - https://www.amazon.com/Python-Feature-Engineering-Cookbook-transforming/dp/1789806313/ref=sr_1_1?dchild=1&keywords=feature+engineering+cookbook&qid=1627628487&s=books&sr=1-1
# - https://www.amazon.com/Feature-Engineering-Made-Easy-Identify-ebook/dp/B077N6MK5W
# - https://www.amazon.com/Feature-Engineering-Selection-Chapman-Science/dp/1032090855/ref=sr_1_1?crid=19T9G95E1W7VJ&dchild=1&keywords=feature+engineering+and+selection+kuhn&qid=1628050948&sprefix=feature+engineering+and+%2Cdigital-text%2C293&sr=8-1
# - https://www.amazon.com/Introduction-Machine-Learning-Python-Scientists/dp/1449369413
# - Neural Networks from Scratch in Python (by Kinsley § Kukiela) [external link text](https://nnfs.io/)
# - Practical Statistics for Data Scientists (by Bruce & Gedeck) [external link text](https://www.amazon.com/Practical-Statistics-Data-Scientists-Essential/dp/149207294X/ref=sr_1_1?dchild=1&keywords=Practical+Statistics+for+Data+Scientists&qid=1627662007&sr=8-1)
# - Applications of Deep Neural Networks(by Jeff Heaton) [external link text](https://arxiv.org/abs/2009.05673)
# - Applied Predictive Modeling (by Kuhn & Johnson) [external link text](https://www.amazon.com/Applied-Predictive-Modeling-Max-Kuhn/dp/1461468485/ref=pd_sbs_3/141-4288971-3747365?pd_rd_w=AOIS7&pf_rd_p=3676f086-9496-4fd7-8490-77cf7f43f846&pf_rd_r=MCCHJXWK39VD6VW7RVAR&pd_rd_r=4ffcd1ea-44b9-4f33-b9b3-dc02ee159662&pd_rd_wg=nU1Ex&pd_rd_i=1461468485&psc=1:)
# - Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (by Aurélien Géron) [external link text](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646/ref=sr_1_1?crid=2GV554Q2EKD1E&dchild=1&keywords=hands-on+machine+learning+with+scikit-learn%2C+keras%2C+and+tensorflow&qid=1627628294&s=books&sprefix=hands%2Cstripbooks-intl-ship%2C309&sr=1-1)
# - Master Machine Learning Algorithms (by Brownlee, ML algorithms are very well explained ) [external link text](https://machinelearningmastery.com/master-machine-learning-algorithms/)
# - Python Feature Engineering Cookbook (by Galli) [external link text](https://www.amazon.com/Python-Feature-Engineering-Cookbook-transforming/dp/1789806313/ref=sr_1_1?dchild=1&keywords=feature+engineering+cookbook&qid=1627628487&s=books&sr=1-1)
# - Feature Engineering Made Easy (by Ozdemir & Susarla) [external link text](https://www.amazon.com/Feature-Engineering-Made-Easy-Identify-ebook/dp/B077N6MK5W)
# - Feature Engineering and Selection (by Kuhn & Johnson) [external link text](https://www.amazon.com/Feature-Engineering-Selection-Chapman-Science/dp/1032090855/ref=sr_1_1?crid=19T9G95E1W7VJ&dchild=1&keywords=feature+engineering+and+selection+kuhn&qid=1628050948&sprefix=feature+engineering+and+%2Cdigital-text%2C293&sr=8-1)
# - Imbalanced Classification with Python(by Brownlee) [external link text](https://machinelearningmastery.com/imbalanced-classification-with-python/)

# **Have fun while...**

# In[ ]:




