#!/usr/bin/env python
# coding: utf-8

# # Тема “Обучение без учителя”
# ### Задание 1
# 
# 
# 
# 

# Импортируйте библиотеки pandas, numpy и matplotlib.
# Загрузите "Boston House Prices dataset" из встроенных наборов 
# данных библиотеки sklearn.
# Создайте датафреймы X и y из этих данных.
# Разбейте эти датафреймы на тренировочные (X_train, y_train) и тестовые (X_test, y_test)
# с помощью функции train_test_split так, чтобы размер тестовой выборки
# составлял 20% от всех данных, при этом аргумент random_state должен быть равен 42.
# Масштабируйте данные с помощью StandardScaler.
# Постройте модель TSNE на тренировочный данных с параметрами:
# n_components=2, learning_rate=250, random_state=42.
# Постройте диаграмму рассеяния на этих данных.
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.datasets import load_boston
boston = load_boston()
data = boston["data"]
feature_names = boston["feature_names"]


X = pd.DataFrame(data, columns=feature_names)

X.info()


# In[3]:


target = boston["target"]

y = pd.DataFrame(target, columns=["price"])
y.info()


# In[4]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape


# Масштабируйте данные с помощью StandardScaler. Постройте модель TSNE на тренировочный данных с параметрами: n_components=2, learning_rate=250, random_state=42. Постройте диаграмму рассеяния на этих данных.

# In[5]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), 
                        columns=X_train.columns, 
                        index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), 
                        columns=X_test.columns, 
                        index=X_test.index)

X_train_scaled.shape


# In[6]:


from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, learning_rate=250, random_state=42)
X_train_tsne = tsne.fit_transform(X_train_scaled)

X_train_tsne.shape


# In[7]:


X_train_tsne2 = pd.DataFrame(data = X_train_tsne)
X_train_tsne2.head()


# In[8]:


plt.scatter(X_train_tsne[:,0], X_train_tsne[:,1])
plt.show()


# # Задание 2
# С помощью KMeans разбейте данные из тренировочного набора на 3 кластера,
# используйте все признаки из датафрейма X_train.
# Параметр max_iter должен быть равен 100, random_state сделайте равным 42.
# Постройте еще раз диаграмму рассеяния на данных, полученных с помощью TSNE,
# и раскрасьте точки из разных кластеров разными цветами.
# Вычислите средние значения price и CRIM в разных кластерах.
# 

# In[9]:


from sklearn.cluster import KMeans

kmeans_3 = KMeans(n_clusters=3, max_iter=100, random_state=42)
labels_clast_3 = kmeans_3.fit_predict(X_train_scaled)


# In[10]:


plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=labels_clast_3)
plt.show()


# In[13]:


y_train[labels_clast_3 == 0].mean()


# In[12]:


y_train[labels_clast_3 == 1].mean()


# In[14]:


y_train[labels_clast_3 == 2].mean()


# In[16]:


X_train.loc[labels_clast_3 == 0, 'CRIM'].mean()


# In[17]:


X_train.loc[labels_clast_3 == 1, 'CRIM'].mean()


# In[18]:


X_train.loc[labels_clast_3 == 2, 'CRIM'].mean()


# # *Задание 3
# Примените модель KMeans, построенную в предыдущем задании,
# к данным из тестового набора.
# Вычислите средние значения price и CRIM в разных кластерах на тестовых данных.
# 

# In[19]:


labels_clast_3_test = kmeans_3.predict(X_test_scaled)


# In[22]:


y_test[labels_clast_3_test == 0].mean()


# In[23]:


y_test[labels_clast_3_test == 1].mean()


# In[24]:


y_test[labels_clast_3_test == 2].mean()


# In[28]:


X_test.loc[labels_clast_3_test == 0, 'CRIM'].mean()


# In[30]:


X_test.loc[labels_clast_3_test == 1, 'CRIM'].mean()


# In[29]:


X_test.loc[labels_clast_3_test == 2, 'CRIM'].mean()

