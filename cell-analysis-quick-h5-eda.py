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

# + [markdown] papermill={"duration": 0.005517, "end_time": "2022-08-16T16:47:30.500487", "exception": false, "start_time": "2022-08-16T16:47:30.494970", "status": "completed"} tags=[]
# # Open Problems in Cell Analyis: Quick EDA

# + [markdown] papermill={"duration": 0.003797, "end_time": "2022-08-16T16:47:30.508673", "exception": false, "start_time": "2022-08-16T16:47:30.504876", "status": "completed"} tags=[]
# <div style="color:white;
#        display:fill;
#        border-radius:5px;
#        background-color:#ffffe6;
#        font-size:120%;">
#     <p style="padding: 10px;
#           color:black;">
# Let's take a look at CITEseq inputs first
#     </p>
# </div>

# + _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" papermill={"duration": 1.299045, "end_time": "2022-08-16T16:47:31.811745", "exception": false, "start_time": "2022-08-16T16:47:30.512700", "status": "completed"} tags=[]
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
# %matplotlib inline

# + papermill={"duration": 14.858439, "end_time": "2022-08-16T16:47:46.674563", "exception": false, "start_time": "2022-08-16T16:47:31.816124", "status": "completed"} tags=[]
# !pip install --quiet tables

# + papermill={"duration": 30.355956, "end_time": "2022-08-16T16:48:17.034893", "exception": false, "start_time": "2022-08-16T16:47:46.678937", "status": "completed"} tags=[]
import os
os.makedirs('/kaggle/working/inputs', exist_ok=True)
# Circumvent read-only issues
# !cp ../input/open-problems-multimodal/train_cite_inputs.h5 '/kaggle/working/inputs'

# + papermill={"duration": 170.030916, "end_time": "2022-08-16T16:51:07.075752", "exception": false, "start_time": "2022-08-16T16:48:17.044836", "status": "completed"} tags=[]
with pd.HDFStore('/kaggle/working/inputs/train_cite_inputs.h5') as data:
    shape = data['/train_cite_inputs'].shape
    print(f"There are {shape[0]} cell IDs and {shape[1]} columns (!)")
    selected_columns = data['/train_cite_inputs'].columns[:40]
    # We select only 50 cells for starters
    df = data['/train_cite_inputs'][selected_columns].head(50)
    
df.head()

# + papermill={"duration": 13.777328, "end_time": "2022-08-16T16:51:20.857820", "exception": false, "start_time": "2022-08-16T16:51:07.080492", "status": "completed"} tags=[]
# !pip install --quiet joypy

# + papermill={"duration": 2.883592, "end_time": "2022-08-16T16:51:23.746413", "exception": false, "start_time": "2022-08-16T16:51:20.862821", "status": "completed"} tags=[]
import pandas as pd
import joypy
import numpy as np

def color_gradient(x=0.0, start=(0, 0, 0), stop=(1, 1, 1)):
    r = np.interp(x, [0, 1], [start[0], stop[0]])
    g = np.interp(x, [0, 1], [start[1], stop[1]])
    b = np.interp(x, [0, 1], [start[2], stop[2]])
    return (r, g, b)

joypy.joyplot(
              df,
    title="Cell distribution by gene",overlap=4,
              colormap=lambda x: color_gradient(x, start=(153/256, 255/256, 204/256),
                                                stop=(204/256, 102/256, 255/256)),
              linecolor='black', linewidth=.5,
             figsize=(7,12),);



# + papermill={"duration": 1.114301, "end_time": "2022-08-16T16:51:24.868014", "exception": false, "start_time": "2022-08-16T16:51:23.753713", "status": "completed"} tags=[]
corr = df.corr()
plt.figure(figsize=(12,8));
sns.heatmap(corr, cmap="viridis");

# + [markdown] papermill={"duration": 0.008467, "end_time": "2022-08-16T16:51:24.885640", "exception": false, "start_time": "2022-08-16T16:51:24.877173", "status": "completed"} tags=[]
# ----
# <div style="color:white;
#        display:fill;
#        border-radius:5px;
#        background-color:#ffffe6;
#        font-size:120%;">
#     <p style="padding: 10px;
#           color:black;">
# Now we get to the CITEseq targets
#     </p>
# </div>

# + papermill={"duration": 2.640314, "end_time": "2022-08-16T16:51:27.534570", "exception": false, "start_time": "2022-08-16T16:51:24.894256", "status": "completed"} tags=[]
os.makedirs('/kaggle/working/labels', exist_ok=True)
# !cp ../input/open-problems-multimodal/train_cite_targets.h5 '/kaggle/working/labels'

# + papermill={"duration": 0.680986, "end_time": "2022-08-16T16:51:28.224251", "exception": false, "start_time": "2022-08-16T16:51:27.543265", "status": "completed"} tags=[]
with pd.HDFStore('/kaggle/working/labels/train_cite_targets.h5') as data:
    shape = data['/train_cite_targets'].shape
    print(f"There are {shape[0]} cell IDs and {shape[1]} columns (!)")
    selected_columns = data['/train_cite_targets'].columns[:40]
    # We select only 50 cells for starters
    df_targets = data['/train_cite_targets'][selected_columns].head(50)

# + papermill={"duration": 0.045025, "end_time": "2022-08-16T16:51:28.278194", "exception": false, "start_time": "2022-08-16T16:51:28.233169", "status": "completed"} tags=[]
df_targets.head()

# + papermill={"duration": 2.871673, "end_time": "2022-08-16T16:51:31.159271", "exception": false, "start_time": "2022-08-16T16:51:28.287598", "status": "completed"} tags=[]
joypy.joyplot(
              df_targets,
    title="Cell distribution by surface protein",overlap=4,
              colormap=lambda x: color_gradient(x, start=(153/256, 255/256, 204/256),
                                                stop=(204/256, 102/256, 255/256)),
              linecolor='black', linewidth=.5,
             figsize=(7,12),);

# + papermill={"duration": 0.010599, "end_time": "2022-08-16T16:51:31.181741", "exception": false, "start_time": "2022-08-16T16:51:31.171142", "status": "completed"} tags=[]

