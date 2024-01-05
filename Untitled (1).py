#!/usr/bin/env python
# coding: utf-8

# In[74]:


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
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

#خواندن داده و اطلاعات اولیه آن

df = pd.read_csv('full_data.csv')
df.info()


# In[2]:


df.head()


# In[3]:


df['smoking_status'].unique()


# In[4]:


df['smoking_status'].value_counts()


# In[5]:


#df[stroke['smoking_status']=='never smoked']
df[['stroke','ever_married']][df['smoking_status']=='never smoked']


# In[6]:


#نمودارها
ax = sns.countplot(x="gender", data=df,)
plt.title("Female = 1 & Male = 0")


# In[7]:


plt.hist(df['age'],bins=10,edgecolor='black')
plt.title("age")
plt.hist(df['age'],bins=10,edgecolor='black')


# In[8]:


ax = sns.countplot(x="hypertension", data=df,)
plt.title("0 = Doesn't have hypertension & 1 = Have hypertension")


# In[9]:


ax = sns.countplot(x="heart_disease", data=df,)
plt.title("0 = Doesn't have heart_disease & 1 = Have heart_disease")


# In[10]:


ax = sns.countplot(x="ever_married", data=df,)
plt.title(" not married or married")


# In[11]:


df['work_type'].unique()


# In[12]:


ax = sns.countplot(x="work_type", data=df,)
plt.title("Private or Self-employed or  Govt_job or children")


# In[13]:


ax = sns.countplot(x="Residence_type", data=df,)
plt.title("urban or rural")


# In[14]:


plt.hist(df['avg_glucose_level'],bins=10,edgecolor='black')
plt.title("avg_glucose_level")


# In[15]:


plt.hist(df['bmi'],bins=10,edgecolor='black')
plt.title("bmi")


# In[16]:


df['smoking_status'].unique()


# In[17]:


ax = sns.countplot(x="smoking_status", data=df,)
plt.title("formerly smoked or never smoked or  smokes or Unknown")


# In[18]:


ax = sns.countplot(x="stroke", data=df,)
plt.title("  dont have stroke = 0 & have stroke = 1")


# In[19]:


#تبدیل داده های رشته ای به اعداد
df["gender"] = df["gender"].replace("Female", 1).replace("Male", 0)
df["ever_married"] = df["ever_married"].replace("Yes", 1).replace("No", 0)
df["work_type"] = df["work_type"].replace("Private", 0).replace("Self-employed", 1).replace("children", 2).replace("Govt_job", 3)
df["Residence_type"] = df["Residence_type"].replace("Urban", 1).replace("Rural", 0)
df["smoking_status"] = df["smoking_status"].replace("never smoked", 0).replace("formerly smoked", 1).replace("smokes", 2).replace("Unknown", 3)


# In[20]:


#نمایش هیت مپ داده ها
corr=df.corr()
sns.heatmap(corr, vmin=-1, vmax=1,annot=True, annot_kws={'fontsize':5}, cmap='coolwarm')
plt.figure(figsize=(40,40),dpi=300)


# In[21]:


#نرمالایز کردن داده ها
data_normal=df.copy()
data_normal.head()


# In[22]:


#نرمالایز کردن داده ها
data_normal["age"]=data_normal["age"]/data_normal["age"].max()
data_normal["avg_glucose_level"]=data_normal["avg_glucose_level"]/data_normal["avg_glucose_level"].max()
data_normal["bmi"]=data_normal["bmi"]/data_normal["bmi"].max()
data_normal["smoking_status"]=data_normal["smoking_status"]/data_normal["smoking_status"].max()


# In[23]:


data_normal.head()


# In[24]:


#train و test
X=data_normal.drop('stroke', axis='columns')
Y=data_normal['stroke']


# In[25]:


X_train , X_test , Y_train , Y_test=train_test_split(X,Y,test_size=0.3 , random_state=42)


# In[26]:


X_train.shape


# In[27]:


#ساخت درخت تصمیم

dt1=DecisionTreeClassifier(max_depth=4)


# In[28]:


dt1.get_params()


# In[29]:


#فیت کردن درخت تصمیم روی داده ها
dt1.fit(X_train,Y_train)
Y_pred=dt1.predict(X_test)
acc_test=accuracy_score(Y_test,Y_pred)
acc_test


# In[30]:


#نمایش درخت تصمیم
plt.figure("decision tree" , figsize=[24,24])
plot_tree(dt1,fontsize=10 , filled=True)
plt.tight_layout
plt.show()


# In[31]:


#ساخت درخت تصمیم CART 

dt2=DecisionTreeClassifier(criterion='gini',max_depth=3)


# In[32]:


#فیت کردن درخت تصمیم روی داده ها
dt2.fit(X_train,Y_train)
Y_pred=dt2.predict(X_test)
acc_test=accuracy_score(Y_test,Y_pred)
acc_test


# In[33]:


# cart نمایش درخت تصمیم
plt.figure("decision tree" , figsize=[24,24])
plot_tree(dt2,fontsize=10 , filled=True)
plt.tight_layout
plt.show()


# In[34]:


#ساخت درخت تصمیم C4 

dt3=DecisionTreeClassifier(criterion="entropy",max_depth=5)


# In[35]:


#فیت کردن درخت تصمیم روی داده ها
dt3.fit(X_train,Y_train)
Y_pred=dt3.predict(X_test)
acc_test=accuracy_score(Y_test,Y_pred)
acc_test


# In[36]:


#c4 نمایش درخت تصمیم
plt.figure("decision tree" , figsize=[24,24])
plot_tree(dt3,fontsize=10 , filled=True)
plt.tight_layout
plt.show()


# In[37]:


#تابع هیت مپ

def plot_confusion_matrix(Y_true , y_pred , title):
    cm = confusion_matrix(Y_true , y_pred)
    ax = plt.axes()
    sns.heatmap(cm , annot=True , fmt="d" , cmap = "Blues" , ax = ax)
    ax.set_title(title)
    plt.show()


# In[38]:


#svm خطی

X_train , X_test , Y_train , Y_test=train_test_split(X,Y,test_size=0.3 , random_state=42)

svm_liner=SVC(kernel='linear',random_state=0)
svm_liner.fit(X_train,Y_train)
predictions_liner=svm_liner.predict(X_test)
accuracy_liner=accuracy_score(Y_test ,predictions_liner)
print(accuracy_liner)


# In[39]:


plot_confusion_matrix(Y_test , predictions_liner ,"confusion_matrix - liner"  )


# In[40]:


#svm هسته چند جمله ای

X_train , X_test , Y_train , Y_test=train_test_split(X,Y,test_size=0.3 , random_state=42)

svm_poly = SVC(kernel='poly' , degree = 5)
svm_poly.fit(X_train,Y_train)
predictions_poly=svm_poly.predict(X_test)
accuracy_poly=accuracy_score(Y_test ,predictions_poly)
print(accuracy_poly)


# In[41]:


plot_confusion_matrix(Y_test , predictions_poly ,"confusion_matrix -  poly"  )


# In[42]:


#svm هسته گاوس


X_train , X_test , Y_train , Y_test=train_test_split(X,Y,test_size=0.3 , random_state=42)

svm_rbf=SVC(kernel='rbf')
svm_rbf.fit(X_train,Y_train)
predictions_rbf=svm_rbf.predict(X_test)
accuracy_rbf=accuracy_score(Y_test ,predictions_rbf)
print(accuracy_rbf)


# In[43]:


plot_confusion_matrix(Y_test , predictions_rbf ,"confusion_matrix -  rbf"  )


# In[50]:


#X = data_normal.copy()
data_normal.head()
#X = data_normal[['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status','stroke']]


# In[69]:


#kmean:


#انتخاب ستون های ویژگی برای خوشه بندی
X = data_normal[['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status','stroke']]

#تعداد خوشه ها را 2 الی 10 خوشه در نظر میگیریم
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    
#بر روی داده های خود فیت میکنیم
kmeans.fit(X)
    
#برچسب خوشه ها را دریافت میکنیم

labels = kmeans.labels_
    

#دریافت مرکز هر خوشه
centers = kmeans.cluster_centers_
    
#در اینجا اطلاعات خوشه ها را با استفاده از دیتا فریم کلاستر بدست میآوریم
cluster_info = pd.DataFrame({
            'Cluster': range(1, n_clusters + 1),
            'Num Members': pd.Series(labels).value_counts().sort_index(),
            'Center Values': [center.tolist() for center in centers],
            'SSE': kmeans.inertia_
    })

    #اطلاعات خوشه ها را چاپ میکنیم

print(f"Number of Clusters: {n_clusters}")
print(cluster_info)
print("\n" + "=" * 50 + "\n")




# In[73]:


#نمایش گرافیکی برای یک تعداد خوشخ خاص و دو گروه خاص

X = data_normal[['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status','stroke']]

#اینجا تعداد خوشه ها را جایگزین میکنیم
n_clusters = 10 
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)

kmeans.fit(X)

labels = kmeans.labels_

df['Cluster'] = labels

plt.figure(figsize=(10, 6))
for cluster in range(n_clusters):
    cluster_data = df[df['Cluster'] == cluster]
    
    #اینجا دو گروه دلخواه جهت نمایش انتخاب میکنیم
    
    plt.scatter(cluster_data['gender'], cluster_data['stroke'], label=f'Cluster {cluster + 1}')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, c='red', label='Cluster Centers')

plt.title(f'Clustering with {n_clusters} clusters')
plt.xlabel('gender')
plt.ylabel('stroke')
plt.legend()
plt.grid(True)
plt.show()


# In[80]:


X = data_normal[['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status','stroke']]

#اینجا تعداد خوشه ها را جایگزین میکنیم
n_clusters = 10 
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)

kmeans.fit(X)

labels = kmeans.labels_

df['Cluster'] = labels

plt.figure(figsize=(10, 6))
for cluster in range(n_clusters):
    cluster_data = df[df['Cluster'] == cluster]
    
    #اینجا دو گروه دلخواه جهت نمایش انتخاب میکنیم
    
    plt.scatter(cluster_data['age'], cluster_data['stroke'], label=f'Cluster {cluster + 1}')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, c='red', label='Cluster Centers')

plt.title(f'Clustering with {n_clusters} clusters')
plt.xlabel('age')
plt.ylabel('stroke')
plt.legend()
plt.grid(True)
plt.show()


# In[82]:


X = data_normal[['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status','stroke']]

#اینجا تعداد خوشه ها را جایگزین میکنیم
n_clusters = 10 
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)

kmeans.fit(X)

labels = kmeans.labels_

df['Cluster'] = labels

plt.figure(figsize=(10, 6))
for cluster in range(n_clusters):
    cluster_data = df[df['Cluster'] == cluster]
    
    #اینجا دو گروه دلخواه جهت نمایش انتخاب میکنیم
    
    plt.scatter(cluster_data['heart_disease'], cluster_data['stroke'], label=f'Cluster {cluster + 1}')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, c='red', label='Cluster Centers')

plt.title(f'Clustering with {n_clusters} clusters')
plt.xlabel('heart_disease')
plt.ylabel('stroke')
plt.legend()
plt.grid(True)
plt.show()


# In[81]:


X = data_normal[['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status','stroke']]

#اینجا تعداد خوشه ها را جایگزین میکنیم
n_clusters = 10 
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)

kmeans.fit(X)

labels = kmeans.labels_

df['Cluster'] = labels

plt.figure(figsize=(10, 6))
for cluster in range(n_clusters):
    cluster_data = df[df['Cluster'] == cluster]
    
    #اینجا دو گروه دلخواه جهت نمایش انتخاب میکنیم
    
    plt.scatter(cluster_data['hypertension'], cluster_data['stroke'], label=f'Cluster {cluster + 1}')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, c='red', label='Cluster Centers')

plt.title(f'Clustering with {n_clusters} clusters')
plt.xlabel('hypertension')
plt.ylabel('stroke')
plt.legend()
plt.grid(True)
plt.show()


# In[77]:


#DBSCAN:

#ویژگی های خوشه بندی
X = data_normal[['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status','stroke']]

#انتخاب تعداد خوشه ها که اینجا 2 الی 10 هست
for n_clusters in range(2, 11):
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    
 #فیت کردن داده ها روی الگوریتم

    dbscan.fit(X)

    labels = dbscan.labels_
#حساب کردن تعداد خوشه ها و دستورات اونا بدون در نظر گرفتن داده های پرت

    unique_clusters = np.unique(labels)
    num_clusters = len(unique_clusters) - (1 if -1 in labels else 0) 
    cluster_info = pd.DataFrame(columns=['Cluster', 'Num Members', 'Center Values', 'SSE'])

    for cluster in unique_clusters:
        if cluster == -1:
            continue  

        cluster_data = X[labels == cluster]
        num_members = len(cluster_data)
        center_values = np.mean(cluster_data, axis=0)
        sse = np.sum((cluster_data - center_values) ** 2)
        
#اطلاعات خوشه ها
        cluster_info.loc[len(cluster_info)] = [cluster, num_members, center_values.tolist(), sse]

    print(f"Number of Clusters: {num_clusters}")
    print(cluster_info)
    print("\n" + "=" * 50 + "\n")


# In[79]:


#DBSCAN:

#این کد همون کد قبلیه ولی اطلاعات خلاصه تری مثل شماره خوشه، تعداد اعضای هر خوشه، مراکز خوشه و اس اس ای رو خلاصه تر نمایش میده
X = data_normal[['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status','stroke']]
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X)
labels = dbscan.labels_
cluster_info = pd.DataFrame(columns=['Cluster', 'Num Members', 'Center Values', 'SSE'])

for cluster in np.unique(labels):
    if cluster == -1:
        continue 
    cluster_data = X[labels == cluster]
    num_members = len(cluster_data)
    center_values = np.mean(cluster_data, axis=0)
    sse = np.sum((cluster_data - center_values) ** 2)

    cluster_info.loc[len(cluster_info)] = [cluster, num_members, center_values.tolist(), sse]
print("Cluster Information:")
print(cluster_info)


# In[ ]:




