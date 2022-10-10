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

# !pip install --quiet tables

# + _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19"
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
        
pd.options.display.max_columns = 228942 

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# -

train_cite_targets = pd.read_hdf('/kaggle/input/open-problems-multimodal/train_cite_targets.h5')
train_cite_targets.shape



train_cite_targets.head()

import matplotlib.pyplot as plt
import plotly.express as px

features =['CD86', 'CD274','CD270']
fig = px.scatter_matrix(train_cite_targets,dimensions= features )
fig.show()


train_cite_inputs = pd.read_hdf('/kaggle/input/open-problems-multimodal/train_cite_inputs.h5',start=0, stop = 1000)
train_cite_inputs.shape



train_multi_targets = pd.read_hdf('/kaggle/input/open-problems-multimodal/train_multi_targets.h5',start=0, stop = 1000)
train_multi_targets.shape

train_multi_targets.info



train_multi_inputs = pd.read_hdf('/kaggle/input/open-problems-multimodal/train_multi_inputs.h5',start=0, stop = 1000)

train_cite_targets.head()

train_cite_inputs.head()

train_multi_targets.head()

train_multi_inputs.head()

'''
/kaggle/input/open-problems-multimodal/test_multi_inputs.h5
/kaggle/input/open-problems-multimodal/test_cite_inputs.h5
'''

'''
/kaggle/input/open-problems-multimodal/train_cite_inputs.h5
/kaggle/input/open-problems-multimodal/train_multi_targets.h5
/kaggle/input/open-problems-multimodal/train_multi_inputs.h5
/kaggle/input/open-problems-multimodal/train_cite_targets.h5
'''

metadata= pd.read_csv('/kaggle/input/open-problems-multimodal/metadata.csv')

metadata.head()

metadata.shape

metadata.groupby(['day'])['day'].count()

metadata.groupby(['cell_type'])['cell_type'].count()

metadata.groupby(['donor'])['donor'].count()

metadata.shape






































































































