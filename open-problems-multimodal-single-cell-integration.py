# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: saturn (Python 3)
#     language: python
#     name: python3
# ---

# + _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" papermill={"duration": null, "end_time": null, "exception": false, "start_time": "2022-08-17T13:00:12.630415", "status": "running"} tags=[]
# !pip install -q tables

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
print("Done")

# + [markdown] papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "pending"} tags=[]
# # Constants

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "pending"} tags=[]
INPUT_DIR = "../input/open-problems-multimodal"

EVALUATION_DIR = os.path.join(INPUT_DIR, "evaluation_ids.csv")
METADATA_DIR = os.path.join(INPUT_DIR, "metadata.csv")
SUBMISSION_DIR = os.path.join(INPUT_DIR, "sample_submission.csv")

MULTIOME_TRAIN_INPUTS = os.path.join(INPUT_DIR,"train_multi_inputs.h5")
MULTIOME_TRAIN_TARGETS = os.path.join(INPUT_DIR,"train_multi_targets.h5")
MULTIOME_TEST_INPUTS = os.path.join(INPUT_DIR,"test_multi_inputs.h5")
CITE_TRAIN_INPUTS = os.path.join(INPUT_DIR,"train_cite_inputs.h5")
CITE_TRAIN_TARGETS = os.path.join(INPUT_DIR,"train_cite_targets.h5")
CITE_TEST_INPUTS = os.path.join(INPUT_DIR,"test_cite_inputs.h5")
SUBMISSION_PATH = os.path.join(INPUT_DIR,"sample_submission.csv")
EVALUATION_IDS = os.path.join(INPUT_DIR,"evaluation_ids.csv")

START = int(1e4)
STOP = START+10000

ROW_ID = "row_id"
TARGET = "target"
GENE_ID_INT = "gene_id_int"
GENE_ID = "gene_id"

print("Done")


# + [markdown] papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "pending"} tags=[]
# # Functions 

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "pending"} tags=[]
def data_description(df):
    print("Data description")
    print(f"Total number of records {df.shape[0]}")
    print(f'number of features {df.shape[1]}\n\n')
    columns = df.columns
    data_type = []
    
    # Get the datatype of features
    for col in df.columns:
        data_type.append(df[col].dtype)
        
    n_uni = df.nunique()
    # Number of NaN values
    n_miss = df.isna().sum()
    
    names = list(zip(columns, data_type, n_uni, n_miss))
    variable_desc = pd.DataFrame(names, columns=["Name","Type","Unique levels","Missing"])
    print(variable_desc)


# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "pending"} tags=[]
def transform_df(df, column, int_col, drop_col):
    df = pd.DataFrame(df, columns = [column]).reset_index()
    df[int_col] = df[drop_col].apply(lambda x: int(x.replace("-","").replace(".","")[-8:],34)).astype(int)
    df.drop([drop_col], axis = 1, inplace = True)
    return df


# + [markdown] papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "pending"} tags=[]
# # Data preparation

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "pending"} tags=[]
train_cite_targ = pd.read_hdf(CITE_TRAIN_TARGETS)
metadata = pd.read_csv(METADATA_DIR)

train_multi_targ = pd.read_hdf(MULTIOME_TRAIN_TARGETS, start=START, stop=STOP)
print("Done")

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "pending"} tags=[]
metadata.head()

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "pending"} tags=[]
train_cite_targ.head()

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "pending"} tags=[]
data_description(train_cite_targ)
print("Done")

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "pending"} tags=[]
train_multi_targ.head()

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "pending"} tags=[]
data_description(train_multi_targ)

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "pending"} tags=[]
multi_gene_id_mean = train_multi_targ.mean()
multi_gene_id_mean


# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "pending"} tags=[]
cite_gene_id_mean = train_cite_targ.mean()
cite_gene_id_mean

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "pending"} tags=[]
cite_gene_id_mean.index

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "pending"} tags=[]
multi_gene_id_mean.index

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "pending"} tags=[]
gene_id_mean = list(cite_gene_id_mean.index) + list(multi_gene_id_mean.index)
gene_id = pd.DataFrame(gene_id_mean, columns = [GENE_ID])
gene_id.head()

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "pending"} tags=[]
gene_id[GENE_ID_INT] = gene_id[GENE_ID].apply(lambda x : int(x.replace("-", "").replace(".","")[-8:], 34)).astype(int)
gene_id.head()

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "pending"} tags=[]
data_description(gene_id)

# + [markdown] papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "pending"} tags=[]
# # Submission

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "pending"} tags=[]
submission = pd.read_csv(SUBMISSION_PATH, usecols = [ROW_ID])
data_description(submission)

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "pending"} tags=[]
evaluation = pd.read_csv(EVALUATION_IDS, usecols=[ROW_ID, GENE_ID])
evaluation[GENE_ID_INT] = evaluation[GENE_ID].apply(lambda x: int(x.replace('-', '').replace('.', '')[-8:],34)).astype(int)
evaluation.drop([GENE_ID], axis=1, inplace=True)
data_description(evaluation)

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "pending"} tags=[]
evaluation.head()

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "pending"} tags=[]
submission = submission.merge(evaluation, how = "left", on = ROW_ID)
data_description(submission)

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "pending"} tags=[]
cite_gene_id_mean = transform_df(cite_gene_id_mean, TARGET, GENE_ID_INT, GENE_ID)
cite_gene_id_mean.head()

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "pending"} tags=[]
multi_gene_id_mean = transform_df(multi_gene_id_mean, TARGET, GENE_ID_INT, GENE_ID)
multi_gene_id_mean.head()

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "pending"} tags=[]
completed_gene_id_mean = pd.concat([cite_gene_id_mean, multi_gene_id_mean])
data_description(completed_gene_id_mean)

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "pending"} tags=[]
submission = submission.merge(completed_gene_id_mean, how = "left", on = GENE_ID_INT)
data_description(submission)

# + tags=[]
submission

# + papermill={"duration": null, "end_time": null, "exception": null, "start_time": null, "status": "pending"} tags=[]
submission[[ROW_ID, TARGET]].to_csv('submission.csv', index=False)
