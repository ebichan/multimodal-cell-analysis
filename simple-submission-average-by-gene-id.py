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

# + [markdown] papermill={"duration": 0.008209, "end_time": "2022-08-16T14:50:10.470965", "exception": false, "start_time": "2022-08-16T14:50:10.462756", "status": "completed"} tags=[]
# # Summary
#
# Focusing only on gene_id, submit the average value of gene_id.
#
# In this case, cell_id is not used.

# + [markdown] papermill={"duration": 0.006364, "end_time": "2022-08-16T14:50:10.484148", "exception": false, "start_time": "2022-08-16T14:50:10.477784", "status": "completed"} tags=[]
# # Data preparation
#
# I referred to [@peterholderrieth](https://www.kaggle.com/peterholderrieth)'s notebook. (https://www.kaggle.com/code/peterholderrieth/getting-started-data-loading)

# + papermill={"duration": 17.539406, "end_time": "2022-08-16T14:50:28.030259", "exception": false, "start_time": "2022-08-16T14:50:10.490853", "status": "completed"} tags=[]
# !pip install --quiet tables

# + papermill={"duration": 0.01656, "end_time": "2022-08-16T14:50:28.053755", "exception": false, "start_time": "2022-08-16T14:50:28.037195", "status": "completed"} tags=[]
import os
import pandas as pd

# + papermill={"duration": 0.024554, "end_time": "2022-08-16T14:50:28.085479", "exception": false, "start_time": "2022-08-16T14:50:28.060925", "status": "completed"} tags=[]
os.listdir("/kaggle/input/open-problems-multimodal/")

# + _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" papermill={"duration": 0.019604, "end_time": "2022-08-16T14:50:28.112120", "exception": false, "start_time": "2022-08-16T14:50:28.092516", "status": "completed"} tags=[]
DATA_DIR = "/kaggle/input/open-problems-multimodal/"

SUBMISSON = os.path.join(DATA_DIR,"sample_submission.csv")

EVALUATION_IDS = os.path.join(DATA_DIR,"evaluation_ids.csv")

FP_CELL_METADATA = os.path.join(DATA_DIR,"metadata.csv")

FP_CITE_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_cite_inputs.h5")
FP_CITE_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_cite_targets.h5")
FP_CITE_TEST_INPUTS = os.path.join(DATA_DIR,"test_cite_inputs.h5")

FP_MULTIOME_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_multi_inputs.h5")
FP_MULTIOME_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_multi_targets.h5")
FP_MULTIOME_TEST_INPUTS = os.path.join(DATA_DIR,"test_multi_inputs.h5")

FP_SUBMISSION = os.path.join(DATA_DIR,"sample_submission.csv")
FP_EVALUATION_IDS = os.path.join(DATA_DIR,"evaluation_ids.csv")

# + [markdown] papermill={"duration": 0.006688, "end_time": "2022-08-16T14:50:28.125732", "exception": false, "start_time": "2022-08-16T14:50:28.119044", "status": "completed"} tags=[]
# ## Citeseq

# + papermill={"duration": 0.834784, "end_time": "2022-08-16T14:50:28.967446", "exception": false, "start_time": "2022-08-16T14:50:28.132662", "status": "completed"} tags=[]
df_cite_train_y = pd.read_hdf('../input/open-problems-multimodal/train_cite_targets.h5')
df_cite_train_y.head()

# + papermill={"duration": 0.051541, "end_time": "2022-08-16T14:50:29.026101", "exception": false, "start_time": "2022-08-16T14:50:28.974560", "status": "completed"} tags=[]
cite_gene_id_mean = df_cite_train_y.mean()
cite_gene_id_mean

# + [markdown] papermill={"duration": 0.00704, "end_time": "2022-08-16T14:50:29.040476", "exception": false, "start_time": "2022-08-16T14:50:29.033436", "status": "completed"} tags=[]
# ## Multiome

# + papermill={"duration": 0.017279, "end_time": "2022-08-16T14:50:29.065206", "exception": false, "start_time": "2022-08-16T14:50:29.047927", "status": "completed"} tags=[]
START = int(1e4)
STOP = START+10000

# + papermill={"duration": 8.721852, "end_time": "2022-08-16T14:50:37.794395", "exception": false, "start_time": "2022-08-16T14:50:29.072543", "status": "completed"} tags=[]
df_multi_train_y = pd.read_hdf(FP_MULTIOME_TRAIN_TARGETS, start=START, stop=STOP)
df_multi_train_y.info()

# + papermill={"duration": 0.650552, "end_time": "2022-08-16T14:50:38.452522", "exception": false, "start_time": "2022-08-16T14:50:37.801970", "status": "completed"} tags=[]
multi_gene_id_mean = df_multi_train_y.mean()
multi_gene_id_mean

# + [markdown] papermill={"duration": 0.007134, "end_time": "2022-08-16T14:50:38.467163", "exception": false, "start_time": "2022-08-16T14:50:38.460029", "status": "completed"} tags=[]
# ## Convert gene_id to int (to save memory)

# + papermill={"duration": 0.020733, "end_time": "2022-08-16T14:50:38.495553", "exception": false, "start_time": "2022-08-16T14:50:38.474820", "status": "completed"} tags=[]
cite_gene_id_mean.index

# + papermill={"duration": 0.021044, "end_time": "2022-08-16T14:50:38.524196", "exception": false, "start_time": "2022-08-16T14:50:38.503152", "status": "completed"} tags=[]
multi_gene_id_mean.index

# + papermill={"duration": 0.028699, "end_time": "2022-08-16T14:50:38.560789", "exception": false, "start_time": "2022-08-16T14:50:38.532090", "status": "completed"} tags=[]
_ = list(cite_gene_id_mean.index) + list(multi_gene_id_mean.index)
gene_id = pd.DataFrame(_, columns=['gene_id'])
gene_id

# + papermill={"duration": 0.048913, "end_time": "2022-08-16T14:50:38.617713", "exception": false, "start_time": "2022-08-16T14:50:38.568800", "status": "completed"} tags=[]
gene_id['gene_id_int'] = gene_id['gene_id'].apply(lambda x: int(x.replace('-', '').replace('.', '')[-8:],34)).astype(int)
gene_id['gene_id_int'].value_counts()

# + [markdown] papermill={"duration": 0.00794, "end_time": "2022-08-16T14:50:38.634010", "exception": false, "start_time": "2022-08-16T14:50:38.626070", "status": "completed"} tags=[]
# # Create submit file

# + papermill={"duration": 15.251682, "end_time": "2022-08-16T14:50:53.893796", "exception": false, "start_time": "2022-08-16T14:50:38.642114", "status": "completed"} tags=[]
df_sample_submission = pd.read_csv(SUBMISSON, usecols=['row_id'])
df_sample_submission.info()

# + papermill={"duration": 123.710955, "end_time": "2022-08-16T14:52:57.613042", "exception": false, "start_time": "2022-08-16T14:50:53.902087", "status": "completed"} tags=[]
df_evaluation = pd.read_csv(EVALUATION_IDS, usecols=['row_id', 'gene_id'])
df_evaluation['gene_id_int'] = df_evaluation['gene_id'].apply(lambda x: int(x.replace('-', '').replace('.', '')[-8:],34)).astype(int)
df_evaluation.drop(['gene_id'], axis=1, inplace=True)
df_evaluation.info()

# + papermill={"duration": 68.409644, "end_time": "2022-08-16T14:54:06.031114", "exception": false, "start_time": "2022-08-16T14:52:57.621470", "status": "completed"} tags=[]
df_sample_submission = df_sample_submission.merge(df_evaluation, how='left', on='row_id')
df_sample_submission.info()

# + papermill={"duration": 0.024195, "end_time": "2022-08-16T14:54:06.064986", "exception": false, "start_time": "2022-08-16T14:54:06.040791", "status": "completed"} tags=[]
cite_gene_id_mean = pd.DataFrame(cite_gene_id_mean, columns=['target']).reset_index()
cite_gene_id_mean['gene_id_int'] = cite_gene_id_mean['gene_id'].apply(lambda x: int(x.replace('-', '').replace('.', '')[-8:],34)).astype(int)
cite_gene_id_mean.drop(['gene_id'], axis=1, inplace=True)

# + papermill={"duration": 0.045686, "end_time": "2022-08-16T14:54:06.119309", "exception": false, "start_time": "2022-08-16T14:54:06.073623", "status": "completed"} tags=[]
multi_gene_id_mean = pd.DataFrame(multi_gene_id_mean, columns=['target']).reset_index()
multi_gene_id_mean['gene_id_int'] = multi_gene_id_mean['gene_id'].apply(lambda x: int(x.replace('-', '').replace('.', '')[-8:],34)).astype(int)
multi_gene_id_mean.drop(['gene_id'], axis=1, inplace=True)

# + papermill={"duration": 0.029167, "end_time": "2022-08-16T14:54:06.156799", "exception": false, "start_time": "2022-08-16T14:54:06.127632", "status": "completed"} tags=[]
cite_multi_gene_id_mean = pd.concat([cite_gene_id_mean, multi_gene_id_mean])
cite_multi_gene_id_mean.info()

# + papermill={"duration": 20.552243, "end_time": "2022-08-16T14:54:26.717442", "exception": false, "start_time": "2022-08-16T14:54:06.165199", "status": "completed"} tags=[]
df_sample_submission = df_sample_submission.merge(cite_multi_gene_id_mean, how='left', on='gene_id_int')
df_sample_submission.info()

# + papermill={"duration": 125.265601, "end_time": "2022-08-16T14:56:31.992675", "exception": false, "start_time": "2022-08-16T14:54:26.727074", "status": "completed"} tags=[]
df_sample_submission[['row_id', 'target']].to_csv('submission.csv', index=False)
