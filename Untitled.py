#!/usr/bin/env python
# coding: utf-8

# In[87]:


#کتابخانه ها
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.svm import SVC

#خواندن داده و اطلاعات اولیه آن

df = pd.read_csv('full_data.csv')
df.info()


# In[43]:


df.head()


# In[44]:


df['smoking_status'].unique()


# In[45]:


df['smoking_status'].value_counts()


# In[46]:


#df[stroke['smoking_status']=='never smoked']
df[['stroke','ever_married']][df['smoking_status']=='never smoked']


# In[47]:


#نمودارها
ax = sns.countplot(x="gender", data=df,)
plt.title("Female = 1 & Male = 0")


# In[48]:


plt.hist(df['age'],bins=10,edgecolor='black')
plt.title("age")
plt.hist(df['age'],bins=10,edgecolor='black')


# In[49]:


ax = sns.countplot(x="hypertension", data=df,)
plt.title("0 = Doesn't have hypertension & 1 = Have hypertension")


# In[50]:


ax = sns.countplot(x="heart_disease", data=df,)
plt.title("0 = Doesn't have heart_disease & 1 = Have heart_disease")


# In[51]:


ax = sns.countplot(x="ever_married", data=df,)
plt.title(" not married or married")


# In[52]:


df['work_type'].unique()


# In[53]:


ax = sns.countplot(x="work_type", data=df,)
plt.title("Private or Self-employed or  Govt_job or children")


# In[54]:


ax = sns.countplot(x="Residence_type", data=df,)
plt.title("urban or rural")


# In[55]:


plt.hist(df['avg_glucose_level'],bins=10,edgecolor='black')
plt.title("avg_glucose_level")


# In[56]:


plt.hist(df['bmi'],bins=10,edgecolor='black')
plt.title("bmi")


# In[57]:


df['smoking_status'].unique()


# In[58]:


ax = sns.countplot(x="smoking_status", data=df,)
plt.title("formerly smoked or never smoked or  smokes or Unknown")


# In[59]:


ax = sns.countplot(x="stroke", data=df,)
plt.title("  dont have stroke = 0 & have stroke = 1")


# In[60]:


#تبدیل داده های رشته ای به اعداد
df["gender"] = df["gender"].replace("Female", 1).replace("Male", 0)
df["ever_married"] = df["ever_married"].replace("Yes", 1).replace("No", 0)
df["work_type"] = df["work_type"].replace("Private", 0).replace("Self-employed", 1).replace("children", 2).replace("Govt_job", 3)
df["Residence_type"] = df["Residence_type"].replace("Urban", 1).replace("Rural", 0)
df["smoking_status"] = df["smoking_status"].replace("never smoked", 0).replace("formerly smoked", 1).replace("smokes", 2).replace("Unknown", 3)


# In[61]:


#نمایش هیت مپ داده ها
corr=df.corr()
sns.heatmap(corr, vmin=-1, vmax=1,annot=True, annot_kws={'fontsize':5}, cmap='coolwarm')
plt.figure(figsize=(40,40),dpi=300)


# In[62]:


#نرمالایز کردن داده ها
data_normal=df.copy()
data_normal.head()


# In[63]:


#نرمالایز کردن داده ها
data_normal["age"]=data_normal["age"]/data_normal["age"].max()
data_normal["avg_glucose_level"]=data_normal["avg_glucose_level"]/data_normal["avg_glucose_level"].max()
data_normal["bmi"]=data_normal["bmi"]/data_normal["bmi"].max()
data_normal["smoking_status"]=data_normal["smoking_status"]/data_normal["smoking_status"].max()


# In[64]:


data_normal.head()


# In[65]:


#train و test
X=data_normal.drop('stroke', axis='columns')
Y=data_normal['stroke']


# In[66]:


X_train , X_test , Y_train , Y_test=train_test_split(X,Y,test_size=0.3 , random_state=42)


# In[67]:


X_train.shape


# In[68]:


#ساخت درخت تصمیم

dt1=DecisionTreeClassifier(max_depth=4)


# In[69]:


dt1.get_params()


# In[70]:


#فیت کردن درخت تصمیم روی داده ها
dt1.fit(X_train,Y_train)
Y_pred=dt1.predict(X_test)
acc_test=accuracy_score(Y_test,Y_pred)
acc_test


# In[71]:


#نمایش درخت تصمیم
plt.figure("decision tree" , figsize=[24,24])
plot_tree(dt1,fontsize=10 , filled=True)
plt.tight_layout
plt.show()


# In[78]:


#ساخت درخت تصمیم CART 

dt2=DecisionTreeClassifier(criterion='gini',max_depth=3)


# In[79]:


#فیت کردن درخت تصمیم روی داده ها
dt2.fit(X_train,Y_train)
Y_pred=dt2.predict(X_test)
acc_test=accuracy_score(Y_test,Y_pred)
acc_test


# In[80]:


# cart نمایش درخت تصمیم
plt.figure("decision tree" , figsize=[24,24])
plot_tree(dt2,fontsize=10 , filled=True)
plt.tight_layout
plt.show()


# In[81]:


#ساخت درخت تصمیم C4 

dt3=DecisionTreeClassifier(criterion="entropy",max_depth=5)


# In[82]:


#فیت کردن درخت تصمیم روی داده ها
dt3.fit(X_train,Y_train)
Y_pred=dt3.predict(X_test)
acc_test=accuracy_score(Y_test,Y_pred)
acc_test


# In[83]:


#c4 نمایش درخت تصمیم
plt.figure("decision tree" , figsize=[24,24])
plot_tree(dt3,fontsize=10 , filled=True)
plt.tight_layout
plt.show()


# In[94]:


#تابع هیت مپ

def plot_confusion_matrix(Y_true , y_pred , title):
    cm = confusion_matrix(Y_true , y_pred)
    ax = plt.axes()
    sns.heatmap(cm , annot=True , fmt="d" , cmap = "Blues" , ax = ax)
    ax.set_title(title)
    plt.show()


# In[84]:


#svm خطی

X_train , X_test , Y_train , Y_test=train_test_split(X,Y,test_size=0.3 , random_state=42)

svm_liner=SVC(kernel='linear',random_state=0)
svm_liner.fit(X_train,Y_train)
predictions_liner=svm_liner.predict(X_test)
accuracy_liner=accuracy_score(Y_test ,predictions_liner)
print(accuracy_liner)


# In[95]:


plot_confusion_matrix(Y_test , predictions_liner ,"confusion_matrix - liner"  )


# In[85]:


#svm هسته چند جمله ای

X_train , X_test , Y_train , Y_test=train_test_split(X,Y,test_size=0.3 , random_state=42)

svm_poly = SVC(kernel='poly' , degree = 5)
svm_poly.fit(X_train,Y_train)
predictions_poly=svm_poly.predict(X_test)
accuracy_poly=accuracy_score(Y_test ,predictions_poly)
print(accuracy_poly)


# In[96]:


plot_confusion_matrix(Y_test , predictions_poly ,"confusion_matrix -  poly"  )


# In[86]:


#svm هسته گاوس


X_train , X_test , Y_train , Y_test=train_test_split(X,Y,test_size=0.3 , random_state=42)

svm_rbf=SVC(kernel='rbf')
svm_rbf.fit(X_train,Y_train)
predictions_rbf=svm_rbf.predict(X_test)
accuracy_rbf=accuracy_score(Y_test ,predictions_rbf)
print(accuracy_rbf)


# In[97]:


plot_confusion_matrix(Y_test , predictions_rbf ,"confusion_matrix -  rbf"  )


# In[ ]:




