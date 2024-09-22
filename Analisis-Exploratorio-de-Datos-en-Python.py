#!/usr/bin/env python
# coding: utf-8

# # Análisis Exploratorio de Datos (EDA)

# ## 0. Introducción

# ### Qué es un EDA
# 
# - Comprender mejor un conjunto de datos:
# - Identificar patrones, detectar anomalías, verificar suposiciones y resumir sus principales características
# 
# <img src="http://sharpsightlabs.com/wp-content/uploads/2016/05/1_data-analysis-for-ML_how-we-use-dataAnalysis_2016-05-16.png" />
# 
# ### DATASET IRIS
# 
# El dataset Iris, introducido por Ronald Fisher en 1936, es un clásico conjunto de datos multiclase usado para tareas de clasificación. 
# 
# Contiene 150 muestras de tres especies de flores (Iris setosa, Iris virginica e Iris versicolor), con cuatro características:
# 
#   1. longitud de sépalos
#   2. anchura de sépalos 
#   3. Longitud de pétalos
#   4. Anchura de pétalos 
# 
# <img src="https://media.licdn.com/dms/image/D4D12AQF5vivFTAdZjQ/article-cover_image-shrink_600_2000/0/1700911428185?e=2147483647&v=beta&t=RaJufpE5-ZMvIMZFVTy4dNtvnKHVgmThtTORx-_qu6Q"/>
# 
# ### Objetivo
# 
# El objetivo principal es construir un modelo que, usando estas características, clasifique correctamente las especies de flores con la mayor precisión posible.

# ## 1. Inicialización y Preparación de Datos

# ### 1.1 Imports

# In[4]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# ### 1.2 Ajustes iniciales

# In[6]:


# Comenta esta linea si tus visualizaciones no se ven
get_ipython().run_line_magic('matplotlib', 'inline')

# Configurar estilo de plt
plt.style.use("bmh")
# Configurar estilo de Seaborn
sns.set(style="whitegrid")


# ## 2. Carga y Preprocesamiento del Dataset

# ### 2.1 Lectura de datos

# In[7]:


df = pd.read_csv('Iris.csv')
df


# ### 2.2 Preprocesamiento

# In[8]:


#Renombramos las variables para falicitar su uso
df.rename({'SepalLengthCm':'sep_l',
           'SepalWidthCm':'sep_a',
           'PetalLengthCm':'pet_l',
           'PetalWidthCm':'pet_a',
           'Species':'especie'},
          axis=1, inplace=True)


# ## 3. Limpieza de Datos

# In[9]:


df.drop('Id', axis=1, inplace=True)
df.head(7)


# ## 4. Análisis Descriptivo

# In[10]:


df.head(8)


# #### Dimensiones del Dataset

# In[11]:


df.shape


# #### Información General del Dataset

# In[18]:


df.info()


# In[19]:


df.describe()


# #### Recuento de la variable a predecir

# In[14]:


df['especie'].value_counts()


# In[15]:


couns = df['especie'].value_counts()
sns.countplot (data= df, x='especie', palette='pastel')


# In[16]:


# Crear el gráfico de barras
counts =  df['especie'].value_counts()
# Crear el gráfico de barras usando Seaborn con colores diferentes
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='especie', palette='pastel')
plt.ylabel('Número de Muestras')
plt.title('Número de Muestras por Especie')
plt.xticks(rotation=45)
plt.show()


# ### Comprobamos las variables nulas

# In[17]:


df.isnull().sum(axis=0)


# ## 5. Analisis Univariante

# ### 5.1 Longitud del Sépalo

# In[19]:


# 1. Histograma con KDE
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
sns.histplot(df['sep_l'], kde=True, bins=20, color='purple')
plt.title('Histograma con KDE - Longitud del sépalo')
plt.xlabel('Longitud del sépalo (cm)')
plt.ylabel('Frecuencia')

# 2. Boxplot
plt.subplot(1, 3, 2)
sns.boxplot(y=df['sep_l'], color='lightgreen')
plt.title('Boxplot - Longitud del sépalo')
plt.xlabel('Longitud del sépalo (cm)')
plt.ylabel('')

# 3. Violin plot
plt.subplot(1, 3, 3)
sns.violinplot(y=df['sep_l'], color='lightblue')
plt.title('Violin plot - Longitud del sépalo')
plt.xlabel('Longitud del sépalo (cm)')
plt.ylabel('')

# Mostrar los gráficos
plt.tight_layout()
plt.show()


# ### 5.2 Ancho del Sépalo

# In[20]:


# 1. Histograma con KDE
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
sns.histplot(df['sep_a'], kde=True, bins=20, color='purple')
plt.title('Histograma con KDE - Ancho del sépalo')
plt.xlabel('Ancho del sépalo (cm)')
plt.ylabel('Frecuencia')

# 2. Boxplot
plt.subplot(1, 3, 2)
sns.boxplot(y=df['sep_a'], color='lightgreen')
plt.title('Boxplot - Ancho del sépalo')
plt.xlabel('Ancho del sépalo (cm)')
plt.ylabel('')

# 3. Violin plot
plt.subplot(1, 3, 3)
sns.violinplot(y=df['sep_a'], color='lightblue')
plt.title('Violin plot - Ancho del sépalo')
plt.xlabel('Ancho del sépalo (cm)')
plt.ylabel('')

# Mostrar los gráficos
plt.tight_layout()
plt.show()


# ### 5.3 Longitud del pétalo

# In[21]:


# 1. Histograma con KDE
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
sns.histplot(df['pet_l'], kde=True, bins=20, color='purple')
plt.title('Histograma con KDE - Longitud del pétalo')
plt.xlabel('Longitud del pétalo (cm)')
plt.ylabel('Frecuencia')

# 2. Boxplot
plt.subplot(1, 3, 2)
sns.boxplot(y=df['pet_l'], color='lightgreen')
plt.title('Boxplot - Longitud del pétalo')
plt.xlabel('Longitud del pétalo (cm)')
plt.ylabel('')

# 3. Violin plot
plt.subplot(1, 3, 3)
sns.violinplot(y=df['pet_l'], color='lightblue')
plt.title('Violin plot - Longitud del pétalo')
plt.xlabel('Longitud del pétalo (cm)')
plt.ylabel('')

# Mostrar los gráficos
plt.tight_layout()
plt.show()


# ### 5.4 Ancho del pétalo

# In[22]:


# 1. Histograma con KDE
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
sns.histplot(df['sep_a'], kde=True, bins=20, color='purple')
plt.title('Histograma con KDE - Ancho del pétalo')
plt.xlabel('Ancho del pétalo (cm)')
plt.ylabel('Frecuencia')

# 2. Boxplot
plt.subplot(1, 3, 2)
sns.boxplot(y=df['sep_a'], color='lightgreen')
plt.title('Boxplot - Ancho del pétalo')
plt.xlabel('Ancho del pétalo (cm)')
plt.ylabel('')

# 3. Violin plot
plt.subplot(1, 3, 3)
sns.violinplot(y=df['sep_a'], color='lightblue')
plt.title('Violin plot - Ancho del pétalo')
plt.xlabel('Ancho del pétalo (cm)')
plt.ylabel('')

# Mostrar los gráficos
plt.tight_layout()
plt.show()


# ## 6. Analisis Multivariante

# ### 6.1 Longitud de Sépalo y Especie

# In[23]:


# Crear una figura con una cuadrícula de 2x2 subgráficos
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# 1. Histograma de Longitud del Sépalo por Especie
sns.histplot(data=df, x='sep_l', hue='especie', multiple='stack', palette='Set1', bins=20, ax=axs[0, 0])
axs[0, 0].set_title('Histograma de Longitud del Sépalo por Especie')
axs[0, 0].set_xlabel('Longitud del Sépalo (cm)')
axs[0, 0].set_ylabel('Frecuencia')

# 2. Diagrama de Caja (Boxplot) de Longitud del Sépalo por Especie
sns.boxplot(data=df, x='especie', y='sep_l', palette='Set1', ax=axs[0, 1])
axs[0, 1].set_title('Diagrama de Caja de Longitud del Sépalo por Especie')
axs[0, 1].set_xlabel('Especie')
axs[0, 1].set_ylabel('Longitud del Sépalo (cm)')

# 3. Gráfico de Violín de Longitud del Sépalo por Especie
sns.violinplot(data=df, x='especie', y='sep_l', palette='Set1', ax=axs[1, 0])
axs[1, 0].set_title('Gráfico de Violín de Longitud del Sépalo por Especie')
axs[1, 0].set_xlabel('Especie')
axs[1, 0].set_ylabel('Longitud del Sépalo (cm)')

# 4. Gráfico de Dispersión (Scatter Plot) de Longitud del Sépalo por Especie
sns.scatterplot(data=df, x='sep_l', y='pet_l', hue='especie', palette='Set1', style='especie', ax=axs[1, 1])
axs[1, 1].set_title('Gráfico de Dispersión de Longitud del Sépalo y Longitud del Pétalo por Especie')
axs[1, 1].set_xlabel('Longitud del Sépalo (cm)')
axs[1, 1].set_ylabel('Longitud del Pétalo (cm)')
axs[1, 1].legend(title='Especie')

# Ajustar el espacio entre subgráficos
plt.tight_layout()
plt.show()



# ### 6.2 Longitud de Sépalo y Especie

# In[24]:


# Crear una figura con una cuadrícula de 2x2 subgráficos
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# 1. Histograma de Longitud del Sépalo por Especie
sns.histplot(data=df, x='sep_a', hue='especie', multiple='stack', palette='Set1', bins=20, ax=axs[0, 0])
axs[0, 0].set_title('Histograma de Longitud del Sépalo por Especie')
axs[0, 0].set_xlabel('Longitud del Sépalo (cm)')
axs[0, 0].set_ylabel('Frecuencia')

# 2. Diagrama de Caja (Boxplot) de Longitud del Sépalo por Especie
sns.boxplot(data=df, x='especie', y='sep_a', palette='Set1', ax=axs[0, 1])
axs[0, 1].set_title('Diagrama de Caja de Longitud del Sépalo por Especie')
axs[0, 1].set_xlabel('Especie')
axs[0, 1].set_ylabel('Longitud del Sépalo (cm)')

# 3. Gráfico de Violín de Longitud del Sépalo por Especie
sns.violinplot(data=df, x='especie', y='sep_a', palette='Set1', ax=axs[1, 0])
axs[1, 0].set_title('Gráfico de Violín de Longitud del Sépalo por Especie')
axs[1, 0].set_xlabel('Especie')
axs[1, 0].set_ylabel('Longitud del Sépalo (cm)')

# 4. Gráfico de Dispersión (Scatter Plot) de Longitud del Sépalo por Especie
sns.scatterplot(data=df, x='sep_a', y='pet_l', hue='especie', palette='Set1', style='especie', ax=axs[1, 1])
axs[1, 1].set_title('Gráfico de Dispersión de Longitud del Sépalo y Longitud del Pétalo por Especie')
axs[1, 1].set_xlabel('Longitud del Sépalo (cm)')
axs[1, 1].set_ylabel('Longitud del Pétalo (cm)')
axs[1, 1].legend(title='Especie')

# Ajustar el espacio entre subgráficos
plt.tight_layout()
plt.show()



# ### 6.3 Relación entre todas las variables

# In[25]:


# 1. Pairplot: muestra la relación entre todas las variables
b


# ### 6.4 Correlacion entre variables

# In[57]:


df.corr()


# In[56]:


# 2. Heatmap de la correlación entre variables
plt.figure(figsize=(8, 6))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Mapa de calor de correlación entre variables')
plt.show()


# ### 6.5 Relación entre 3 variables

# In[59]:


# 3. Scatterplot 3D: mostrar la relación entre 3 variables
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Crear scatterplot 3D usando 3 dimensiones: longitud del sépalo, longitud del pétalo y ancho del pétalo
scatter = ax.scatter(
    df['sep_l'], 
    df['pet_l'], 
    df['pet_a'], 
    c=pd.Categorical(df['especie']).codes, 
    cmap='Set1'
)

# Etiquetas
ax.set_xlabel('Longitud del Sépalo (cm)')
ax.set_ylabel('Longitud del Pétalo (cm)')
ax.set_zlabel('Ancho del Pétalo (cm)')
plt.title('Scatterplot 3D - Longitud y ancho del sépalo y pétalo')

# Mostrar gráfico
plt.show()


# ## 7. Conclusiones
