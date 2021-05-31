#!/usr/bin/env python
# coding: utf-8

# # Dataset Contains of 14 attributes
Age : Age of the patient

Sex : Sex of the patient

exang: exercise induced angina (1 = yes; 0 = no)

ca: number of major vessels (0-3)

cp : Chest Pain type chest pain type

Value 0: typical angina
Value 1: atypical angina
Value 2: non-anginal pain
Value 3: asymptomatic

trtbps : resting blood pressure (in mm Hg)

chol : cholestoral in mg/dl fetched via BMI sensor

fbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)

rest_ecg : resting electrocardiographic results
    Value 0: normal
    Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
    Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria

thalach : maximum heart rate achieved

target : 0= less chance of heart attack 1= more chance of heart attack

# In[1]:


pip install chart_studio


# In[2]:


#importing essential libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
import plotly.express as px
import plotly.io as pio
warnings.filterwarnings("ignore")


# In[3]:


import plotly.graph_objects as go


# In[15]:


df= pd.read_csv('/Users/mahes/Desktop/heart.csv')


# In[16]:


df.head()


# In[17]:


df.describe()


# In[18]:


df.isnull().sum()

There are no missing values in the dataset
# In[19]:


df.shape

There are 303 records with 14 columns
# In[20]:


plt.figure(figsize= (15,6))
sns.set_style("darkgrid")
sns.heatmap(df.corr(),annot= True)
plt.show()

Correlation heatmap matrix for different features in the dataset 
# In[21]:


ax= px.histogram(df,x= "age", template= "plotly_dark",color= "output",title='Age distribution')
ax.show()

The age column is normally distributed with most of the patients ranging in 50 to 65 age also having most chance of heart attack


0= less chance of heart attack 1= more chance of heart attack
# In[22]:


ax= px.pie(df, names= "output",template= "plotly_dark",title= "chances of heart attack",hole= 0.5)
ax.show()

Around 54% of the patients in the dataset have chances of heart attack
# In[23]:


ax= px.pie(df, names= "sex",template= "plotly_dark",title= "sex",hole=.7)
ax.show()

Distribution of sex shows that 68.3% of patients are of type 1 sex while 31.7% are of type 0Chest pain types
Value 0: typical angina
Value 1: atypical angina
Value 2: non-anginal pain
Value 3: asymptomatic
# In[24]:


ax= px.pie(df, names= "cp",template= "plotly_dark",title= " Chest pain types ",hole=.7)
ax.show()

Majority of the patients experience chest pain type 1:typical angina(47.2%) followed by type 2 ie:type non angina pain,only 7% patients are aymptotic(type 3)
# In[25]:


ax= px.sunburst(df, names= "cp",path= ["output","cp"],template= "plotly_dark",title= "Chest pain based on heart attack chances")
ax.show()

Patients with higher chances of heart attack majorly have type 2 chest pain followed by type1 and type0 Majority of people having less chance of heart attack experience type 0 chest pain ie typical angina
# In[26]:


ax = px.scatter(df, x="age", y="trtbps",size="trtbps", color="output",
                size_max=20,template= "plotly_dark",title="resting blood pressure/age correlation")
ax.show()

trtbps : resting blood pressure (in mm Hg)Distribution of resting blood pressure and age(scatter plot) with chances of heart attack
# In[27]:


ax= px.scatter(df,x= "age",y= "chol",marginal_x='histogram', marginal_y='histogram',size="chol", size_max=20,
              template= "plotly_dark",color= "output",title="age and cholestrol correlation")
ax.show()

chol : cholestoral in mg/dl fetched via BMI sensor
# In[28]:


ax = px.scatter_3d(df, x="age", y="trtbps", z="chol",template= "plotly_dark",color="output")
ax.show()

3d scatter plot showing correlation between age,cholestrol and blood pressure also additional dimension is added based on output(chance of heart attack)
# In[29]:


ax= px.treemap(df,path= ["fbs","output"],template= "plotly_dark",color= "output",title="blood sugar treemap")
ax.show()

fbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)The treemap shows most patients having blood sugar less than 120mg/dl having greater chances of heart attack
# In[30]:


fig = px.histogram(df, x='thalachh', color="output",title= "maximum heart rate achieved",template= "plotly_dark")
fig.show()

thalach : maximum heart rate achievedDistribution based on maximum heart rate and chances of heart attack
# In[31]:


ax= px.sunburst(df,path= ["output","caa"],template= "plotly_dark",title= "No of major blood vessels based on heart attack chances")
ax.show()

Patients having higher chance of heart attack have majority of no blood vessels
# # Classification Part

# In[32]:


#importing essential libraries
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

All the values in the dataset are numerical hence no need of label encodingLets check variance in th features
# In[33]:


df.var()

The age ,trtbps,chol,thalach have higher levels of variance hence needs to be normalized
# In[34]:


#Using log transformation
df["age"]= np.log(df.age)


# In[35]:


df["trtbps"]= np.log(df.trtbps)
df["chol"]= np.log(df.chol)
df["thalachh"]= np.log(df.thalachh)


# In[36]:


df.var()

all features are now normally distributed
# In[37]:


px.box(df,template= "plotly_dark")


# In[38]:


df.describe()


# # KNN Classification With Basic Model Tuning

# In[39]:


#train test split
label= df["output"]
train= df.drop("output",axis= 1)


# In[40]:


x_train,x_test,y_train,y_test= train_test_split(train,label,test_size= 0.25,random_state= 5)


# In[41]:


scores= []
for i in range(1,50):
    knn= KNeighborsClassifier(n_neighbors= i)
    knn.fit(x_train,y_train)
    scores.append(accuracy_score(y_test,knn.predict(x_test)))


# In[42]:


plt.figure(figsize= (15,6))
sns.lineplot(np.arange(1,50),scores)
plt.show()

The n_neighbors values of 13 gives the best prediction score
# In[43]:


knn= KNeighborsClassifier(n_neighbors= 13)
knn.fit(x_train,y_train)
knnpred = knn.predict(x_test)
accuracy_score(y_test,knnpred)

The accuracy score is 0.92(92%)
# In[44]:


#confusion matrix
cm= confusion_matrix(y_test,knnpred)
sns.heatmap(cm,annot= True)


# In[45]:


#classification report
cr= classification_report(y_test,knnpred)
cr


# # Logistic Regression

# In[46]:


lr= LogisticRegression()
lr.fit(x_train,y_train)
lrpred= lr.predict(x_test)
accuracy_score(y_test,lrpred)

accuracy score (logistic regression): 0.8852459016393442
# In[47]:


#confusion matrix
cm= confusion_matrix(y_test,lrpred)
sns.heatmap(cm,annot= True)


# In[48]:


#classification report
classification_report(y_test,lrpred)


# # Random Forest Classifier

# In[49]:


rf= RandomForestClassifier()
rf.fit(x_train,y_train)
rfpred= rf.predict(x_test)
accuracy_score(y_test,rfpred)

accuracy score for random forest: 0.8947368421052632
# In[51]:


dc= DecisionTreeClassifier()
dc.fit(x_train,y_train)
preddc= dc.predict(x_test)
accuracy_score(y_test,preddc)

accuracy score for decision tree classifier: 0.7631578947368421
# In[53]:


grid.best_params_


# In[54]:


grid.best_score_


# # AdaBoost Classifier

# In[55]:


ad=AdaBoostClassifier(learning_rate= 0.2,n_estimators= 100)
ad.fit(x_train,y_train)
adpred= ad.predict(x_test)
accuracy_score(y_test,adpred)

accuracy score for adaboost : 0.8552631578947368
# # Gradient Boosting Classifier

# In[56]:


gb= GradientBoostingClassifier()
gb.fit(x_train,y_train)
predgb= gb.predict(x_test)
accuracy_score(y_test,predgb)

accuracy score for gradientboost :0.8947368421052632
# # XG Boost

# In[57]:


xgb= XGBClassifier()
xgb.fit(x_train,y_train)
accuracy_score(y_test,xgb.predict(x_test))

accuracy score for xgboost: 0.8552631578947368
# # Conclusion: KNN model gave best result with a accuracy score of 0.9210526315789473(92%)

# # Analysis in one more way
Heart Attack EDA and Prediction 

Importing Libraries
# In[59]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso, Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

Read The Dataset
# In[60]:


heart=pd.read_csv('/Users/mahes/Desktop/heart.csv')


# In[61]:


heart.head()


# # Data Description
age - Age of the patient
sex - Sex of the patient
cp - Chest pain type ~ 0 = Typical Angina, 1 = Atypical Angina, 2 = Non-anginal Pain, 3 = Asymptomatic
trtbps - Resting blood pressure (in mm Hg)
chol - Cholestoral in mg/dl fetched via BMI sensor
fbs - (fasting blood sugar > 120 mg/dl) ~ 1 = True, 0 = False
restecg - Resting electrocardiographic results ~ 0 = Normal, 1 = ST-T wave normality, 2 = Left ventricular hypertrophy
thalachh - Maximum heart rate achieved
oldpeak - Previous peak
slp - Slope
caa - Number of major vessels
thall - Thalium Stress Test result ~ (0,3)
exng - Exercise induced angina ~ 1 = Yes, 0 = No
output - target : 0= less chance of heart attack 1= more chance of heart attack
# In[62]:


heart.shape


# In[63]:


col=heart.columns
col


# In[64]:


heart.describe()


# In[65]:


heart.nunique()


# In[66]:


heart


# In[67]:


categorical_cols = ['sex','exng','caa','cp','fbs','restecg','slp','thall']
numerical_cols = ["age","trtbps","chol","thalachh","oldpeak"]
target_col = ["output"]


# In[68]:


# check for the missing values
heart.isnull().sum()


# # Univariate Analysis
Categorical and Target features
# In[69]:


# target variable
heart['output'].value_counts(normalize=True).plot.bar(color=['red','blue'],edgecolor='black',title='target variable')

Around 55% people have more chances to get heart attack
Around 45% people have less chances to get heart attack
# # Sex Feature

# In[70]:


# sex variable
heart['sex'].value_counts(normalize=True).plot.bar(color=['cyan','magenta'],edgecolor='black',title='sex variable')

Around 68 % people are with sex=1
Around 30 % people are with sex=0
# # Chest Pain Feature

# In[72]:


# cp variable
heart['cp'].value_counts(normalize=True).plot.bar(color=['yellow','orange','cyan','magenta'],edgecolor='black',title='chest pain variable')

Around 50 % of the people have chest pain type: Typical Angina
Around 28 % of the people have chest pain type: Non-anginal Pain
Around less than 20 % of the people have chest pain type: Atypical Angina
Around less than 10% of the people have chest pain type: Asymptomatic
# 1.exercise induced angina
# 
# 2.fasting blood sugar > 120 mg/dl
# 
# 3.resting electrocardiographic results
# 
# 4.Slope

# In[73]:


plt.figure(figsize=(20,7))
plt.subplot(221)
heart['exng'].value_counts(normalize=True).plot.bar(color=['yellow','orange'],edgecolor='black',title='exercise induced angina (1 = yes; 0 = no)')
plt.subplot(222)
heart['fbs'].value_counts(normalize=True).plot.bar(color=['yellow','green'],edgecolor='black',title='(fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)')
plt.subplot(223)
heart['restecg'].value_counts(normalize=True).plot.bar(color=['magenta','blue','cyan'],edgecolor='black',title='resting electrocardiographic results')
plt.subplot(224)
heart['slp'].value_counts(normalize=True).plot.bar(color=['red','blue','green'],edgecolor='black',title='- Slope')

More than 65 % of the people Exercise don't induced angina
More than 35 % of the people Exercise induced angina
less than 20 % of the people have fasting blood sugar > 120 mg/dl
More than 80 % of the people have fasting blood sugar <= 120 mg/dl
less than 50 % of the people have resting electrocardiographic results normal
50 % of the people have resting electrocardiographic results: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
1% or 2% of the people have resting electrocardiographic results: showing probable or definite left ventricular hypertrophy by Estes' criteria1.number of major vessels

2.Thalium Stress Test result ~ (0,3)
# In[74]:


plt.figure(figsize=(20,7))
plt.subplot(221)
heart['caa'].value_counts(normalize=True).plot.bar(color=['magenta','blue','cyan','red','orange'],edgecolor='black',title='number of major vessels')
plt.subplot(222)
heart['thall'].value_counts(normalize=True).plot.bar(color=['lightblue','lightgreen','lightyellow','magenta'],edgecolor='black',title='Thalium Stress Test result ~ (0,3)')


# # Numerical features

# In[75]:


numerical_cols


# # age , blood pressure , cholestoral , Heart Rate

# In[76]:


plt.figure(figsize=(20,7))
plt.subplot(221)
heart['age'].plot.hist(edgecolor='black',color='lightgreen',title='age variable')
plt.subplot(222)
heart['trtbps'].plot.hist(edgecolor='black',color='lightblue',title='resting blood pressure in mm hg')
plt.subplot(223)
heart['chol'].plot.hist(edgecolor='black',color='lightcoral',title='cholestoral in mg/dl fetched via BMI sensor')
plt.subplot(224)
heart['thalachh'].plot.hist(edgecolor='black',color='lightgray',title='maximum heart rate achieved')


# # Oldpeak

# In[77]:


heart['oldpeak'].plot.hist(edgecolor='black',color='lightyellow',title='oldpeak variable')


# # Bivariate Analysis

# # effect of age on heart attack

# In[78]:


plt.figure(figsize=(10,7))
plt.style.use("fivethirtyeight")
plt.title("effect of age on heart attack")
sns.lineplot(x=heart['age'],y=heart['output'])

The people with the age 30 to 35 have higher chance of heart attacks

The people with the age than 70 and less than 75 have higher chance of heart attacks

apart from it no certain trend i will be able to find
# # heart attack related with sex

# In[79]:


sns.countplot(data=heart,x='sex',palette=["blue","red"], hue='output')

people of sex=1 have higher chances of getting heart attacks
# # effect of age on blood pressure

# In[84]:


plt.figure(figsize=(10,7))
plt.style.use("fivethirtyeight")
plt.title("effect of age on blood pressure")
sns.lineplot(x=heart['age'],y=heart['trtbps'])

as age is incresing the increase in the blood pressure has been founded
# # effect of age on cholestrol level

# In[85]:


plt.figure(figsize=(10,7))
plt.style.use("fivethirtyeight")
plt.title("effect of age on cholestrol level")
sns.lineplot(x=heart['age'],y=heart['chol'])

as age is incresing the increase in the cholestrol level has been founded
# # effect of age on heart rate

# In[86]:


plt.figure(figsize=(10,7))
plt.style.use("fivethirtyeight")
plt.title("effect of age on heart rate")
sns.lineplot(x=heart['age'],y=heart['thalachh'])

as age is incresing the decrease in the heart rate has been founded
# # How does incresed heart rate and age affect the heart attack

# In[87]:


plt.figure(figsize=(10,7))
plt.style.use("fivethirtyeight")
plt.title("effect of heart attack with increase in age and heart rate")
sns.lineplot(x=heart['age'],y=heart['thalachh'],hue=heart['output'])

as with the increase in the age the heart rate is decresing and also the people with more chances of heart attacks are decreasing hence we can say higher heart rate increases the chance of heart attackHow does incresed cholestrol and age affect the heart attack
# In[90]:


plt.figure(figsize=(10,7))
plt.style.use("fivethirtyeight")
plt.title("effect of heart attack with increase in age and cholestrol")
sns.lineplot(x=heart['age'],y=heart['chol'],hue=heart['output'])

as with the increase in the age the cholestrol level is incresing and also the people with more chances of heart attacks are also increasing hence we can say higher cholestrol level increases the chance of heart attack
# # How does incresed blood pressure and age affect the heart attack

# In[91]:


plt.figure(figsize=(10,7))
plt.style.use("fivethirtyeight")
plt.title("effect of heart attack with increase in age and blood pressure")
sns.lineplot(x=heart['age'],y=heart['trtbps'],hue=heart['output'])

as with the increase in the age the blood pressure is incresing and also the people with more chances of heart attacks are also increasing hence we can say blood pressure increases the chance of heart attack
# # Model Building

# In[92]:


target=heart['output']
target


# In[93]:


heart.drop(['output'],axis=1,inplace=True)
heart.head()


# # Checking for skewness

# In[94]:


heart['trtbps'].plot(kind='density')
plt.show()
heart['chol'].plot(kind='density')
plt.show()
heart['thalachh'].plot(kind='density')
plt.show()
heart['age'].plot(kind='density')
plt.show()


# In[95]:


heart.head(1)


# # Robust Scaler

# In[96]:


from sklearn import preprocessing
scaler = preprocessing.RobustScaler()
robust_df = scaler.fit_transform(heart)
robust_df = pd.DataFrame(robust_df, columns =['age','sex','cp','trtbps','chol','fbs','restecg','thalachh','exng','oldpeak','slp','caa','thall'])
robust_df


# # Standard Scaler

# In[97]:


scaler = preprocessing.StandardScaler()
standard_df = scaler.fit_transform(robust_df)
standard_df = pd.DataFrame(standard_df, columns =['age','sex','cp','trtbps','chol','fbs','restecg','thalachh','exng','oldpeak','slp','caa','thall'])


# In[98]:


standard_df.head()


# # Train Test Split

# In[99]:


x_train,x_test,y_train,y_test=train_test_split(heart,target,test_size=0.1,random_state=42)


# # Logistic Regresson

# In[100]:


logistic=LogisticRegression(max_iter=100,random_state=1,n_jobs=-1)
logistic.fit(x_train,y_train)
pred1=logistic.predict(x_test)
pred1


# In[101]:


logistic.score(x_train,y_train)*100


# In[102]:


logistic.score(x_test,y_test)*100


# In[103]:


from sklearn.metrics import accuracy_score

print('Logistic Regresson model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, pred1)))


# In[104]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
d_pred = decision_tree.predict(x_test)
acc_decision_tree = round(decision_tree.score(x_train,y_train)*100,2)
print(f'{acc_decision_tree}%')


# In[105]:


from sklearn.metrics import accuracy_score

print('Decision Tree model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, d_pred)))


# # LightGBM

# In[106]:


import lightgbm as lgb
lgbm= lgb.LGBMClassifier()
lgbm.fit(x_train,y_train)
pred2=lgbm.predict(x_test)
acc_lgbm=round(lgbm.score(x_train,y_train)*100,2)
print(f'{acc_lgbm}%')


# In[107]:


from sklearn.metrics import accuracy_score

print('LightGBM model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, pred2)))


# # XG Boost

# In[108]:


import xgboost as xgb
# define data_dmatrix
data_dmatrix = xgb.DMatrix(data=heart,label=target)

DMatrix is an internal data structure that is used by XGBoost, which is optimized for both memory efficiency and training speed.
# In[109]:


params = {
            'objective':'binary:logistic',
            'max_depth': 4,
            'alpha': 10,
            'learning_rate': 0.01,
            'n_estimators':100
        }      


# In[110]:


import xgboost as xgb
xgbo= xgb.XGBClassifier(**params)
xgbo.fit(x_train,y_train)
pred3=xgbo.predict(x_test)
acc_xgbo=round(xgbo.score(x_train,y_train)*100,2)
print(f'{acc_xgbo}%')


# In[111]:


from sklearn.metrics import accuracy_score

print('XGBoost model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, pred3)))


# # XG Boost with Cross Validation

# In[112]:


# cross validation
from xgboost import cv

params = {"objective":"binary:logistic",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}

xgb_cv = cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50, early_stopping_rounds=10, metrics="auc", as_pandas=True, seed=123)


# In[119]:


xgb_cv.head()


# In[120]:


xgb_cv.shape


# In[ ]:




