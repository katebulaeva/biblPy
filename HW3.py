#!/usr/bin/env python
# coding: utf-8

# # Тема “Обучение с учителем”
# 
# ## Задание 1
# 

# Импортируйте библиотеки pandas и numpy.
# Загрузите "Boston House Prices dataset" из встроенных наборов данных библиотеки sklearn. Создайте датафреймы X и y из этих данных.
# 
# 
# 

# In[1]:


import numpy as np
import pandas as pd

from sklearn.datasets import load_boston


# In[2]:


boston = load_boston()
data = boston["data"]
feature_names = boston["feature_names"]


X = pd.DataFrame(data, columns=feature_names)

X.info()


# In[3]:


target = boston["target"]

y = pd.DataFrame(target, columns=["price"])
y.info()


# Разбейте эти датафреймы на тренировочные (X_train, y_train) и тестовые (X_test, y_test) с помощью функции train_test_split так, чтобы размер тестовой выборки
# составлял 30% от всех данных, при этом аргумент random_state должен быть равен 42.

# In[4]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Создайте модель линейной регрессии под названием lr с помощью класса LinearRegression из модуля sklearn.linear_model.

# In[5]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# Обучите модель на тренировочных данных (используйте все признаки) и сделайте предсказание на тестовых.

# In[6]:


lr.fit(X_train, y_train)


# In[7]:


y_pred = lr.predict(X_test)

y_pred.shape


# In[8]:


check_test = pd.DataFrame({
    "y_test": y_test["price"],
    "y_pred": y_pred.flatten(),
})

check_test.head(10)


# Вычислите R2 полученных предказаний с помощью r2_score из модуля sklearn.metrics.

# In[9]:


from sklearn.metrics import r2_score


# In[10]:


r2=r2_score(check_test["y_pred"], check_test["y_test"])
r2


# # Задание 2
# 

# Создайте модель под названием model с помощью RandomForestRegressor из модуля sklearn.ensemble.
# Сделайте агрумент n_estimators равным 1000,
# max_depth должен быть равен 12 и random_state сделайте равным 42.
# Обучите модель на тренировочных данных аналогично тому, как вы обучали модель LinearRegression,
# но при этом в метод fit вместо датафрейма y_train поставьте y_train.values[:, 0],
# чтобы получить из датафрейма одномерный массив Numpy,
# так как для класса RandomForestRegressor в данном методе для аргумента y предпочтительно применение массивов вместо датафрейма.
# Сделайте предсказание на тестовых данных и посчитайте R2. Сравните с результатом из предыдущего задания.
# Напишите в комментариях к коду, какая модель в данном случае работает лучше.
# 

# In[11]:


from sklearn.ensemble import RandomForestRegressor


# In[12]:


model = RandomForestRegressor(n_estimators=1000, max_depth=12, random_state=42)

model.fit(X_train, y_train.values[:, 0])


# In[13]:


y_pred = model.predict(X_test)
y_pred.shape


# In[14]:


check_test = pd.DataFrame({
 "y_test": y_test["price"],
 "y_pred": y_pred.flatten(),
})
check_test.head(10)


# In[15]:


r2_2=r2_score(check_test["y_pred"], check_test["y_test"])
r2_2


# R2 у RandomForestRegressor выше, следовательно эта модель в данном случае делает более точные предсказания.

# # *Задание 3
# 

# Вызовите документацию для класса RandomForestRegressor,
# найдите информацию об атрибуте feature_importances_.
# С помощью этого атрибута найдите сумму всех показателей важности,
# установите, какие два признака показывают наибольшую важность.
# 

# In[19]:


get_ipython().run_line_magic('pinfo', 'RandomForestRegressor')


# feature_importances_ : ndarray of shape (n_features,)
#     The impurity-based feature importances.
#     The higher, the more important the feature.
#     The importance of a feature is computed as the (normalized)
#     total reduction of the criterion brought by that feature.  It is also
#     known as the Gini importance.
# 
#     Warning: impurity-based feature importances can be misleading for
#     high cardinality features (many unique values). See
#     :func:`sklearn.inspection.permutation_importance` as an alternative.
# 

# In[21]:


print(model.feature_importances_)


# In[23]:


model.feature_importances_.sum()


# In[25]:


max_idx1=model.feature_importances_.argmax()
max_idx1


# In[ ]:





# In[ ]:





# In[ ]:




