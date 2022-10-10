# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _kg_hide-input=true _kg_hide-output=true
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# -

# **TABLE OF CONTENTS (CHECK RIGHT WINDOW) & DON'T FORGET TO UPVOTE IF IT'S USEFULL :)**

# # Import necessary libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# # First look at files

# + _kg_hide-output=true
# !pip install tables
metadataset=pd.read_csv('../input/open-problems-multimodal/metadata.csv')
#train_cite=pd.read_hdf('../input/open-problems-multimodal/train_cite_inputs.h5')
train_cite_target=pd.read_hdf('../input/open-problems-multimodal/train_cite_targets.h5')
# -

metadataset

train_cite_target

# # EDA

# **Firstly - unique values for each column**

unique_dict={}
for i in metadataset.columns:
    unique_dict[i]=len(metadataset[i].unique())
unique_dict

# **Now we need to take a look at values distributions**

metadataset.hist(figsize=(8,8))
plt.show()

# **Common info & scalar metrics about dataset**

metadataset.info()

metadataset.describe()

# **Correlation matrix**

metadataset.corr()

# **Features pairplot**

# # Distribution plots

sns.displot(metadataset['day'],kind='kde')
plt.show()

# **Day bar distributions**

sns.barplot(metadataset['day'].unique(),metadataset['day'].value_counts())
plt.show()

# **Donor bar counts**

sns.barplot(metadataset['donor'].unique(),metadataset['donor'].value_counts())
plt.show()

# **Cell type distributions**

sns.barplot(metadataset['cell_type'].unique(),metadataset['cell_type'].value_counts())
plt.show()

# **Technology type**

sns.barplot(metadataset['technology'].unique(),metadataset['technology'].value_counts())
plt.show()

# # Feature-to-feature dependency

# **Cell type bar distributions for each technology type**

fig, axs = plt.subplots(1, 2)
technology_list=metadataset['technology'].unique()
for i in range(2):
    axs[i].set_title(technology_list[i])
    axs[i].bar(metadataset[metadataset['technology']==technology_list[i]]['cell_type'].unique(), metadataset[metadataset['technology']==technology_list[i]]['cell_type'].value_counts())
plt.show()

# **Percentage pie diagram for cell types**

# +

plt.figure(figsize=(6,6))
plt.title('Cell type circle diagramm')
plt.pie(metadataset['cell_type'].value_counts(), labels=metadataset['cell_type'].unique())
plt.show()
# -

# **Cell types for each day**

fig, axs = plt.subplots(1, 4)
day_list=metadataset['day'].unique()
for i in range(4):
    axs[i].set_title(day_list[i])
    axs[i].bar(metadataset[metadataset['day']==day_list[i]]['cell_type'].unique(), metadataset[metadataset['day']==day_list[i]]['cell_type'].value_counts())
fig.show()

# # Train cite target EDA

train_cite_target

train_cite_target.info()

train_cite_target.describe()

train_cite_target.corr()

# **WORK IN PROCESS :)**
