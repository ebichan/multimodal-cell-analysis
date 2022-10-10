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

# + [markdown] papermill={"duration": 0.01644, "end_time": "2022-08-16T00:11:59.185194", "exception": false, "start_time": "2022-08-16T00:11:59.168754", "status": "completed"} tags=[]
# # Multimodal Single-Cell Integration Competition: Data Exploration and Visualization

# + [markdown] papermill={"duration": 0.013534, "end_time": "2022-08-16T00:11:59.213323", "exception": false, "start_time": "2022-08-16T00:11:59.199789", "status": "completed"} tags=[]
# ## 1. Setup Notebook

# + [markdown] papermill={"duration": 0.013999, "end_time": "2022-08-16T00:11:59.241037", "exception": false, "start_time": "2022-08-16T00:11:59.227038", "status": "completed"} tags=[]
# ### 1.1. Import packages
#
#

# + papermill={"duration": 14.545259, "end_time": "2022-08-16T00:12:13.800278", "exception": false, "start_time": "2022-08-16T00:11:59.255019", "status": "completed"} tags=[]
#If you see a urllib warning running this cell, go to "Settings" on the right hand side, 
#and turn on internet. Note, you need to be phone verified.
# !pip install --quiet tables

# + papermill={"duration": 1.212697, "end_time": "2022-08-16T00:12:15.027157", "exception": false, "start_time": "2022-08-16T00:12:13.814460", "status": "completed"} tags=[]
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# + [markdown] papermill={"duration": 0.014158, "end_time": "2022-08-16T00:12:15.055636", "exception": false, "start_time": "2022-08-16T00:12:15.041478", "status": "completed"} tags=[]
# ### 1.2. Set filepaths

# + papermill={"duration": 0.03117, "end_time": "2022-08-16T00:12:15.100922", "exception": false, "start_time": "2022-08-16T00:12:15.069752", "status": "completed"} tags=[]
os.listdir("/kaggle/input/open-problems-multimodal/")

# + _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" papermill={"duration": 0.024812, "end_time": "2022-08-16T00:12:15.140701", "exception": false, "start_time": "2022-08-16T00:12:15.115889", "status": "completed"} tags=[]
DATA_DIR = "/kaggle/input/open-problems-multimodal/"
FP_CELL_METADATA = os.path.join(DATA_DIR,"metadata.csv")

FP_CITE_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_cite_inputs.h5")
FP_CITE_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_cite_targets.h5")
FP_CITE_TEST_INPUTS = os.path.join(DATA_DIR,"test_cite_inputs.h5")

FP_MULTIOME_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_multi_inputs.h5")
FP_MULTIOME_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_multi_targets.h5")
FP_MULTIOME_TEST_INPUTS = os.path.join(DATA_DIR,"test_multi_inputs.h5")

FP_SUBMISSION = os.path.join(DATA_DIR,"sample_submission.csv")
FP_EVALUATION_IDS = os.path.join(DATA_DIR,"evaluation_ids.csv")

# + [markdown] papermill={"duration": 0.013866, "end_time": "2022-08-16T00:12:15.169429", "exception": false, "start_time": "2022-08-16T00:12:15.155563", "status": "completed"} tags=[]
# ## 2. Load and Visualize Data

# + [markdown] papermill={"duration": 0.013969, "end_time": "2022-08-16T00:12:15.197575", "exception": false, "start_time": "2022-08-16T00:12:15.183606", "status": "completed"} tags=[]
# ### 2.1. Cell Metadata

# + [markdown] papermill={"duration": 0.013891, "end_time": "2022-08-16T00:12:15.225819", "exception": false, "start_time": "2022-08-16T00:12:15.211928", "status": "completed"} tags=[]
# The metadata of our dataset comes is data about the cells. To understand the different groups of cells, let's first review how the experiment was conducted (see figure below):
# 1. On the first day (*day 1*), hemapoetic stem cells are cultured in a dish with liquids that trigger the differentation of these cells into blood cells.
# 2. On subsequent *days 2,3,4,7,10* some of the cells are removed and split into two subgroups `CITE` and `MULTIOME`.
# 3. Each of these assays (technologies) gives us two readouts per single cell: 
#     1. CITEseq measures gene expression (RNA) and surface protein levels.
#     2. Multiome measures gene expression (RNA) and chromatin accessibility (via ATACseq).
#
# This experiment was repeated for 4 different donors of hemapoetic stem cells. The metadata gives information about day, donor, cell type,
# and technology. `cell_id` is a unique cell identifier and has no meaning beyond its purpose as a cell id.
#
# ![Dataset_Kaggle_structure_small.jpeg](attachment:8ac4d726-390d-4a7e-8a31-0be3c7df2739.jpeg)

# + papermill={"duration": 0.422622, "end_time": "2022-08-16T00:12:15.662929", "exception": false, "start_time": "2022-08-16T00:12:15.240307", "status": "completed"} tags=[]
df_cell = pd.read_csv(FP_CELL_METADATA)
df_cell

# + [markdown] papermill={"duration": 0.015014, "end_time": "2022-08-16T00:12:15.693558", "exception": false, "start_time": "2022-08-16T00:12:15.678544", "status": "completed"} tags=[]
# **NOTE:** the cell type is hidden for the test set of the multiome as this can reveal information about the RNA.

# + [markdown] papermill={"duration": 0.01505, "end_time": "2022-08-16T00:12:15.723857", "exception": false, "start_time": "2022-08-16T00:12:15.708807", "status": "completed"} tags=[]
# **Let's split the cells by technology**

# + papermill={"duration": 0.089122, "end_time": "2022-08-16T00:12:15.828058", "exception": false, "start_time": "2022-08-16T00:12:15.738936", "status": "completed"} tags=[]
df_cell_cite = df_cell[df_cell.technology=="citeseq"]
df_cell_multi = df_cell[df_cell.technology=="multiome"]

# + [markdown] papermill={"duration": 0.014506, "end_time": "2022-08-16T00:12:15.857234", "exception": false, "start_time": "2022-08-16T00:12:15.842728", "status": "completed"} tags=[]
# **Number of cells per group:**
#
# The number of cells in each group is relatively constant, around 7500 cells per donor and day.

# + papermill={"duration": 0.609353, "end_time": "2022-08-16T00:12:16.481647", "exception": false, "start_time": "2022-08-16T00:12:15.872294", "status": "completed"} tags=[]
fig, axs = plt.subplots(1,2,figsize=(12,6))
df_cite_cell_dist = df_cell_cite.set_index("cell_id")[["day","donor"]].value_counts().to_frame()\
                .sort_values("day").reset_index()\
                .rename(columns={0:"# cells"})
sns.barplot(data=df_cite_cell_dist, x="day",hue="donor",y="# cells", ax=axs[0])
axs[0].set_title("Number of cells measured with CITEseq")

df_multi_cell_dist = df_cell_multi.set_index("cell_id")[["day","donor"]].value_counts().to_frame()\
                .sort_values("day").reset_index()\
                .rename(columns={0:"# cells"})
sns.barplot(data=df_multi_cell_dist, x="day",hue="donor",y="# cells", ax=axs[1])
axs[1].set_title("Number of cells measured with Multiome")
plt.show()

# + [markdown] papermill={"duration": 0.014574, "end_time": "2022-08-16T00:12:16.511391", "exception": false, "start_time": "2022-08-16T00:12:16.496817", "status": "completed"} tags=[]
# ### 2.2. Citeseq

# + [markdown] papermill={"duration": 0.014603, "end_time": "2022-08-16T00:12:16.541052", "exception": false, "start_time": "2022-08-16T00:12:16.526449", "status": "completed"} tags=[]
# For CITEseq, the task is to predict surface protein levels ("targets") from RNA expression levels ("inputs" of the model).

# + [markdown] papermill={"duration": 0.014627, "end_time": "2022-08-16T00:12:16.570557", "exception": false, "start_time": "2022-08-16T00:12:16.555930", "status": "completed"} tags=[]
# **Inputs:** For the RNA counts, each row corresponds to a cell and each column to a gene. The column format for a gene is given by `{EnsemblID}_{GeneName}` where `EnsemblID` refers to the [Ensembl Gene ID](https://www.ebi.ac.uk/training/online/courses/ensembl-browsing-genomes/navigating-ensembl/investigating-a-gene/#:~:text=Ensembl%20gene%20IDs%20begin%20with,of%20species%20other%20than%20human) and `GeneName` to the gene name.

# + papermill={"duration": 99.326515, "end_time": "2022-08-16T00:13:55.911960", "exception": false, "start_time": "2022-08-16T00:12:16.585445", "status": "completed"} tags=[]
df_cite_train_x = pd.read_hdf(FP_CITE_TRAIN_INPUTS)
df_cite_test_x = pd.read_hdf(FP_CITE_TEST_INPUTS)
df_cite_train_x.head()

# + [markdown] papermill={"duration": 0.016763, "end_time": "2022-08-16T00:13:55.946076", "exception": false, "start_time": "2022-08-16T00:13:55.929313", "status": "completed"} tags=[]
# **Targets:** For the surface protein levels, each row corresponds to a cell and each column to a protein:

# + papermill={"duration": 0.763113, "end_time": "2022-08-16T00:13:56.725963", "exception": false, "start_time": "2022-08-16T00:13:55.962850", "status": "completed"} tags=[]
df_cite_train_y = pd.read_hdf(FP_CITE_TRAIN_TARGETS)
df_cite_train_y.head()

# + [markdown] papermill={"duration": 0.01683, "end_time": "2022-08-16T00:13:56.760476", "exception": false, "start_time": "2022-08-16T00:13:56.743646", "status": "completed"} tags=[]
# **Donor and cell types:** The train data consists of both gene expression (RNA) and surface protein data for days 2,3,4 for donors 1-3 (donor IDs: `32606`,`13176`, and `31800`), the public test data consists of RNA for days 2,3,4 for donor 4 (donor ID: `27678`) and the private test data consists data from day 7 from all donors.

# + papermill={"duration": 0.162585, "end_time": "2022-08-16T00:13:56.940305", "exception": false, "start_time": "2022-08-16T00:13:56.777720", "status": "completed"} tags=[]
train_cells = df_cite_train_x.index.to_list()    
test_cells = df_cite_test_x.index.to_list()                                                     
df_cell_cite["split"] = ""
df_cell_cite.loc[df_cell_cite.cell_id.isin(train_cells),"split"] = "train"
df_cell_cite.loc[df_cell_cite.cell_id.isin(test_cells),"split"] = "test"

df_cell_cite[["split","day","donor"]].value_counts().to_frame().sort_values(["split","day","donor"]).rename(columns={0: "n cells"})

# + [markdown] papermill={"duration": 0.017932, "end_time": "2022-08-16T00:13:56.975616", "exception": false, "start_time": "2022-08-16T00:13:56.957684", "status": "completed"} tags=[]
# ### 2.3. Multiome
#
# For the Multiome data set, the task is to predict RNA levels ("targets") from ATACseq.

# + [markdown] papermill={"duration": 0.017777, "end_time": "2022-08-16T00:13:57.011966", "exception": false, "start_time": "2022-08-16T00:13:56.994189", "status": "completed"} tags=[]
# **Inputs:** for the ATACseq data, each row corresponds to a cell and each column to a fragment of a gene.
#
# <font fontsize=20 color="red"> **NOTE**: to save memory, we only read an excerpt from the ATACseq data!

# + papermill={"duration": 4.493496, "end_time": "2022-08-16T00:14:01.523307", "exception": false, "start_time": "2022-08-16T00:13:57.029811", "status": "completed"} tags=[]
START = int(1e5)
STOP = START+1000

df_multi_train_x = pd.read_hdf(FP_MULTIOME_TRAIN_INPUTS,start=START,stop=STOP)
df_multi_test_x = pd.read_hdf(FP_MULTIOME_TEST_INPUTS,start=START,stop=STOP)
df_multi_train_x.head()

# + [markdown] papermill={"duration": 0.023693, "end_time": "2022-08-16T00:14:01.567054", "exception": false, "start_time": "2022-08-16T00:14:01.543361", "status": "completed"} tags=[]
# **Targets:** the RNA count data is in similar shape as the RNA count data from CITEseq:

# + papermill={"duration": 0.863258, "end_time": "2022-08-16T00:14:02.449732", "exception": false, "start_time": "2022-08-16T00:14:01.586474", "status": "completed"} tags=[]
df_multi_train_y = pd.read_hdf(FP_MULTIOME_TRAIN_TARGETS, start=START, stop=STOP)
df_multi_train_y.head()

# + [markdown] papermill={"duration": 0.017977, "end_time": "2022-08-16T00:14:02.485955", "exception": false, "start_time": "2022-08-16T00:14:02.467978", "status": "completed"} tags=[]
# **Donor and cell types:** The train data consists of both gene expression (RNA) and ATACseq data for days 2,3,4,7 for donors 1-3 (donor IDs: `32606`,`13176`, and `31800`), the public test data consists of RNA for days 2,3,4,7 for donor 4 (donor ID: `27678`) and the private test data consists data from day 7 from all donors.
#
# <font fontsize=20 color="red"> **NOTE**: Uncomment the below cell if you have loaded the full ATACseq data!

# + papermill={"duration": 0.027309, "end_time": "2022-08-16T00:14:02.531503", "exception": false, "start_time": "2022-08-16T00:14:02.504194", "status": "completed"} tags=[]
# train_cells = df_multi_train_y.index.to_list()    
# test_cells = df_multi_test_y.index.to_list()                                                     
# df_cell_multi["split"] = ""
# df_cell_multi.loc[df_cell_multi.cell_id.isin(train_cells),"split"] = "train"
# df_cell_multi.loc[df_cell_multi.cell_id.isin(test_cells),"split"] = "test"

# df_cell_multi[["split","day","donor"]].value_counts().to_frame().sort_values(["split","day","donor"]).rename(columns={0: "n cells"})
