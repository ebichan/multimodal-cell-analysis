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

# + [markdown] _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" papermill={"duration": 0.010157, "end_time": "2022-08-18T15:02:49.813780", "exception": false, "start_time": "2022-08-18T15:02:49.803623", "status": "completed"} tags=[]
# # 1. Overview
# The goal of this competition is to better understand the relationship between different modalities in cells. The goal of this notebook is to gain a better understanding of the associated data. This equips us with the knowledge needed to make good decisions about model design and data layout.
#
# **This is a work in progress. If any aspect needs clarification, please let me know. My understanding of genetics is very limited. Feel free to point out anything that is false.**
#
# <div style="color:white;display:fill;
#             background-color:#3bb2d6;font-size:200%;">
#     <p style="padding: 4px;color:white;"><b>1.1 What do we want to learn?</b></p>
# </div>
#
# During transcription in cells, there is a known flow of information. DNA must be accessible to produce RNA. Produced RNA is used as a template to build proteins. Therefore, one could assume that we can use knowledge about the accessibility of DNA to predict future states of RNA and that we could use knowledge about RNA to predict the concentration of proteins in the future. In this challenge, we want to learn more about this relationship between DNA, RNA, and proteins. We thus need to capture information about three distinct properties of a cell:
# * chromatin accessibility
# * gene expression
# * surface protein levels
#
# <div style="color:white;display:fill;
#             background-color:#3bb2d6;font-size:200%;">
#     <p style="padding: 4px;color:white;"><b>1.2 How are those three properties of a cell presented?</b></p>
# </div>
#
# Before we have a look at how the information about those properties of a cell is laid out, we must note that the methods used to obtain the data do not capture all properties at once. We have two distinct methods for testing. The first one is the "10x Chromium Single Cell Multiome ATAC + Gene Expression" short "multiome" test. The second one is the "10x Genomics Single Cell Gene Expression with Feature Barcoding technology" short "citeseq" test.
#
# With the multiome test, we can measure **chromatin accessibility and gene expression**. With the citeseq test, we can measure **gene expression and surface protein levels**.
#
# Therefore, we will have data about chromatin accessibility and surface protein levels once (from multiome and citeseq, respectively). And we will have data about gene expression two times, one from each test. With that out of the way, let's dive into how the data is actually presented.
#
# <div style="color:white;display:fill;
#             background-color:#3bb2d6;font-size:200%;">
#     <p style="padding: 4px;color:white;"><b>1.3 Imports</b></p>
# </div>

# + _kg_hide-input=true _kg_hide-output=true papermill={"duration": 14.547772, "end_time": "2022-08-18T15:03:04.370227", "exception": false, "start_time": "2022-08-18T15:02:49.822455", "status": "completed"} tags=[]
# installs
# !pip install --quiet tables

# imports
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# set paths
DATA_DIR = "../input/open-problems-multimodal/"
FP_CELL_METADATA = os.path.join(DATA_DIR,"metadata.csv")

FP_CITE_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_cite_inputs.h5")
FP_CITE_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_cite_targets.h5")
FP_CITE_TEST_INPUTS = os.path.join(DATA_DIR,"test_cite_inputs.h5")

FP_MULTIOME_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_multi_inputs.h5")
FP_MULTIOME_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_multi_targets.h5")
FP_MULTIOME_TEST_INPUTS = os.path.join(DATA_DIR,"test_multi_inputs.h5")

FP_SUBMISSION = os.path.join(DATA_DIR,"sample_submission.csv")
FP_EVALUATION_IDS = os.path.join(DATA_DIR,"evaluation_ids.csv")

# + [markdown] papermill={"duration": 0.008242, "end_time": "2022-08-18T15:03:04.387254", "exception": false, "start_time": "2022-08-18T15:03:04.379012", "status": "completed"} tags=[]
# # 2. Data
#
# <div style="color:white;display:fill;
#             background-color:#3bb2d6;font-size:200%;">
#     <p style="padding: 4px;color:white;"><b>2.1 Chromatin accessibility data</b></p>
# </div>
#
# First we will have a look at the data about chromatin accessibility:

# + papermill={"duration": 4.739406, "end_time": "2022-08-18T15:03:09.135216", "exception": false, "start_time": "2022-08-18T15:03:04.395810", "status": "completed"} tags=[]
# Loading the whole dataset into pandas exceeds the memory,
# therefore we define start and stop values
START = int(1e4)
STOP = START+1000

df_multi_train_x = pd.read_hdf(FP_MULTIOME_TRAIN_INPUTS,start=START,stop=STOP)
df_multi_train_x.head()

# + [markdown] papermill={"duration": 0.008556, "end_time": "2022-08-18T15:03:09.153791", "exception": false, "start_time": "2022-08-18T15:03:09.145235", "status": "completed"} tags=[]
# As we can see, each individual cell is identified by a cell_id. We then have 228942 columns that are named something like "STUFF:NUMBER-NUMBER". STUFF is actually the name of a chromosome, while the numbers are a range indicating where the gene starts and ends. Let's have a look at what kind of chromosomes we have:

# + papermill={"duration": 0.105934, "end_time": "2022-08-18T15:03:09.268729", "exception": false, "start_time": "2022-08-18T15:03:09.162795", "status": "completed"} tags=[]
print(sorted(list({i[:i.find(':')] for i in df_multi_train_x.columns})))

# + [markdown] papermill={"duration": 0.008609, "end_time": "2022-08-18T15:03:09.286371", "exception": false, "start_time": "2022-08-18T15:03:09.277762", "status": "completed"} tags=[]
# We actually find the chromosomes we expect, namely chr1-chr22, the 22 chromosomes humans have (called autosomes), and also chrX and chrY, being the gender-specific chromosomes. What about the ones starting with KI and GL? According to a quick internet search, those are unplaced genes. They most likely are part of the human genome, but we don't know yet on which chromosome they are. Noteworthy at this point is that the number of protein-coding genes in humans is estimated to be between 19.9k and 21.3k. Therefore, it looks like we have measurements of much more than just the protein-coding genes.
#
# Next, we check the range of the values we have.

# + papermill={"duration": 2.045636, "end_time": "2022-08-18T15:03:11.342281", "exception": false, "start_time": "2022-08-18T15:03:09.296645", "status": "completed"} tags=[]
# first call to min/max gives us the min/max in each column. 
# Than we min/max again to get total min/max
print(f"Values range from {df_multi_train_x.min().min()} to {df_multi_train_x.max().max()}")

# + [markdown] papermill={"duration": 0.008893, "end_time": "2022-08-18T15:03:11.361863", "exception": false, "start_time": "2022-08-18T15:03:11.352970", "status": "completed"} tags=[]
# So let's summarize what we have learned about the data corresponding to the accessibility of DNA so far:
#
# * We have chromatin accessibility measurements for approximately 106k cells in total.
# * We measure how accessible certain genes are in each cell, approximately 229k genes per cell.
# * Accessibility is given in numbers from 0.0 to ~18. We do not know the upper bound because we have not looked at all the data yet.

# + [markdown] papermill={"duration": 0.008811, "end_time": "2022-08-18T15:03:11.380036", "exception": false, "start_time": "2022-08-18T15:03:11.371225", "status": "completed"} tags=[]
# What else do we want to know about chromatin accessibility data?
#
# * How many values are non-zero for each cell?
# * What is the standard deviation and what is the average non-zero value?
# * How many significant figures does the test produces? What data format is appropriate for stroing the values?
#
# First, we will have a closer look at the chromatin accessibility values of each cell.

# + papermill={"duration": 10.585682, "end_time": "2022-08-18T15:03:21.974717", "exception": false, "start_time": "2022-08-18T15:03:11.389035", "status": "completed"} tags=[]
# get data about non-zero values
min_cells_non_zero = df_multi_train_x.gt(0).sum(axis=1).min()
max_cells_non_zero = df_multi_train_x.gt(0).sum(axis=1).max()
sum_non_zero_values = df_multi_train_x.sum().sum()
count_non_zero_values = df_multi_train_x.gt(0).sum().sum()
average_non_zero_per_gene = df_multi_train_x[df_multi_train_x.gt(0)].count(axis = 1).mean()

print(f"Each cell has at least {min_cells_non_zero} genes with non-zero accessibility values and a maximum of {max_cells_non_zero}.")
print(f"On average there are {round(average_non_zero_per_gene)} genes with non-zero accessibility values in each cell.")
print(f"The average non-zero value is about {sum_non_zero_values / count_non_zero_values:.2f}.")

# investigate standard deviation of features
std_dev_of_genes = df_multi_train_x.std()

# ignore genes that are only accessible in a single cell
std_dev_of_genes_without_singles = std_dev_of_genes[df_multi_train_x.gt(0).sum().gt(1)]
print(f"The standard deviation of our features is between {std_dev_of_genes_without_singles.min():.2f} and {std_dev_of_genes_without_singles.max():.2f}.\nThe average standard deviation is {std_dev_of_genes_without_singles.mean():.2f}")

# + [markdown] papermill={"duration": 0.009449, "end_time": "2022-08-18T15:03:21.993557", "exception": false, "start_time": "2022-08-18T15:03:21.984108", "status": "completed"} tags=[]
# That's already good information about what we can expect from our features for the first problem. To even better understand how many features we have for each sample, we will plot the number of cells per feature count.

# + papermill={"duration": 1.127536, "end_time": "2022-08-18T15:03:23.130841", "exception": false, "start_time": "2022-08-18T15:03:22.003305", "status": "completed"} tags=[]
s = df_multi_train_x.gt(0).sum(axis = 1)
counts = s.groupby(lambda x: s[x] // 300).count()
counts.index = counts.index * 300

fig, ax = plt.subplots()
ax.plot(counts.index, counts.values)
ax.set_xlabel('number of accessible genes')
ax.set_ylabel('number of cells')
plt.show()

# + [markdown] papermill={"duration": 0.009241, "end_time": "2022-08-18T15:03:23.149898", "exception": false, "start_time": "2022-08-18T15:03:23.140657", "status": "completed"} tags=[]
# As we can see, the majority of our cells have between 2K and 7K accessible genes.
#
# We now have quite a good understanding of the chromatin accessibility measured with the multiome test. We continue with investigating gene expression features.

# + [markdown] papermill={"duration": 0.009226, "end_time": "2022-08-18T15:03:23.168787", "exception": false, "start_time": "2022-08-18T15:03:23.159561", "status": "completed"} tags=[]
# <div style="color:white;display:fill;
#             background-color:#3bb2d6;font-size:200%;">
#     <p style="padding: 4px;color:white;"><b>2.2 Gene expression data</b></p>
# </div>
#
# As mentioned before we have two datasets containing gene expression data. We will first look at the data from the multiome test.
#
# ### 2.2.1 Gene expression from multiome
# We would actually be able to load the whole dataset at once, but we will only look at the part corresponding to the already seen X values for now.

# + papermill={"duration": 0.789283, "end_time": "2022-08-18T15:03:23.967791", "exception": false, "start_time": "2022-08-18T15:03:23.178508", "status": "completed"} tags=[]
df_multi_train_y = pd.read_hdf(FP_MULTIOME_TRAIN_TARGETS, start=START, stop=STOP)
df_multi_train_y.head()

# + [markdown] papermill={"duration": 0.010169, "end_time": "2022-08-18T15:03:23.989022", "exception": false, "start_time": "2022-08-18T15:03:23.978853", "status": "completed"} tags=[]
# As we can see, we have 23418 values that our model will need to predict. But what exactly are those values?

# + papermill={"duration": 0.039346, "end_time": "2022-08-18T15:03:24.039240", "exception": false, "start_time": "2022-08-18T15:03:23.999894", "status": "completed"} tags=[]
print(sorted(list({i[:10] for i in df_multi_train_y.columns})))
print(df_multi_train_y.columns.str.len().unique().item())

# + [markdown] papermill={"duration": 0.010317, "end_time": "2022-08-18T15:03:24.061180", "exception": false, "start_time": "2022-08-18T15:03:24.050863", "status": "completed"} tags=[]
# As we can see, all of the features start with ENSG and then 5 zeroes. What we have here is called the Ensambl ID. The general form is ENS(species)(object type)(identifier).(version).
# ENS tells us that we are looking at an ensembl ID. The species part is empty for human genes by convention. The object type is G for gene. It looks like the identifier is always 11 decimals long. And it looks like we don't have any version specifications in our data.
#
# We will now check for similar properties than before.

# + papermill={"duration": 1.294256, "end_time": "2022-08-18T15:03:25.365588", "exception": false, "start_time": "2022-08-18T15:03:24.071332", "status": "completed"} tags=[]
print(f"Values for gene expression range from {df_multi_train_y.min().min():.2f} to {df_multi_train_y.max().max():.2f}")

# get data about non-zero values
min_cells_non_zero_y = df_multi_train_y.gt(0).sum(axis=1).min()
max_cells_non_zero_y = df_multi_train_y.gt(0).sum(axis=1).max()
sum_non_zero_values_y = df_multi_train_y.sum().sum()
count_non_zero_values_y = df_multi_train_y.gt(0).sum().sum()
average_non_zero_per_gene_y = df_multi_train_y[df_multi_train_y.gt(0)].count(axis = 1).mean()

print(f"Each cell has at least {min_cells_non_zero_y} genes with non-zero gene expression values and a maximum of {max_cells_non_zero_y}.")
print(f"On average there are {round(average_non_zero_per_gene_y)} genes with non-zero gene expression values in each cell.")
print(f"The average non-zero value for gene expression is about {sum_non_zero_values_y / count_non_zero_values_y:.2f}.")

# investigate standard deviation of features
std_dev_of_genes_y = df_multi_train_y.std()

# ignore genes that are only accessible in a single cell
std_dev_of_genes_without_singles_y = std_dev_of_genes_y[df_multi_train_y.gt(0).sum().gt(1)]
print(f"The standard deviation of gene expression values is between {std_dev_of_genes_without_singles_y.min():.2f} and {std_dev_of_genes_without_singles_y.max():.2f}.\nThe average standard deviation is {std_dev_of_genes_without_singles_y.mean():.2f}")

# + [markdown] papermill={"duration": 0.010462, "end_time": "2022-08-18T15:03:25.386696", "exception": false, "start_time": "2022-08-18T15:03:25.376234", "status": "completed"} tags=[]
# We can see that the range of gene expression values is smaller than that for chromatin accessibility, but the standard deviation is higher. This might be important for the design of our model.
#
# Even though this information will probably not influence the design of our model, let's still have a look at how many genes are expressed in cells. Just because it is interesting.

# + papermill={"duration": 0.302902, "end_time": "2022-08-18T15:03:25.700423", "exception": false, "start_time": "2022-08-18T15:03:25.397521", "status": "completed"} tags=[]
s = df_multi_train_y.gt(0).sum(axis = 1)
counts = s.groupby(lambda x: s[x] // 100).count()
counts.index = counts.index * 100

fig, ax = plt.subplots()
ax.plot(counts.index, counts.values)
ax.set_xlabel('number of genes expressed')
ax.set_ylabel('number of cells')
plt.show()

# + [markdown] papermill={"duration": 0.010697, "end_time": "2022-08-18T15:03:25.722441", "exception": false, "start_time": "2022-08-18T15:03:25.711744", "status": "completed"} tags=[]
# Having an overview of gene expression data obtained by the multiome test, we will now compare it to those obtained by the citeseq test.
#
# ### 2.2.2 Gene expression from citeseq

# + papermill={"duration": 0.876315, "end_time": "2022-08-18T15:03:26.609926", "exception": false, "start_time": "2022-08-18T15:03:25.733611", "status": "completed"} tags=[]
df_cite_train_x = pd.read_hdf(FP_CITE_TRAIN_INPUTS,start=START,stop=STOP)
df_cite_train_x.head()

# + [markdown] papermill={"duration": 0.010938, "end_time": "2022-08-18T15:03:26.633157", "exception": false, "start_time": "2022-08-18T15:03:26.622219", "status": "completed"} tags=[]
# The first thing we notice is that the start of our gene_id looks much like what we have seen in the multiome data, but there is a new suffix. So what is it about the suffix?
#
# Checking the Ensembl ID of gene_id on [ensembl.org](https://www.ensembl.org/Homo_sapiens/Gene/Summary?db=core;g=ENSG00000121410;r=19:58345178-58353492) (in this case for ENSG00000121410) we see that the suffix is actually the name of the gene. As we will see in the next code cell, the gene_id is unique even without this suffix, so it looks like redundant information for now.

# + papermill={"duration": 0.037664, "end_time": "2022-08-18T15:03:26.682481", "exception": false, "start_time": "2022-08-18T15:03:26.644817", "status": "completed"} tags=[]
gene_ids_multiome = set(df_multi_train_y.columns)
print(f"Different Gene IDs in multiome: {len(gene_ids_multiome)}")
#for now we just keep the stem of the gene_id
gene_ids_citeseq = set([i[:i.find("_")] for i in df_cite_train_x.columns])
print(f"Different Gene IDs in citeseq: {len(gene_ids_citeseq)}")

# + [markdown] papermill={"duration": 0.010915, "end_time": "2022-08-18T15:03:26.704871", "exception": false, "start_time": "2022-08-18T15:03:26.693956", "status": "completed"} tags=[]
# As mentioned, stripping of the suffix still produces unique names.
#
# Let's check for overlap in both datasets about gene expression.

# + papermill={"duration": 0.035536, "end_time": "2022-08-18T15:03:26.751704", "exception": false, "start_time": "2022-08-18T15:03:26.716168", "status": "completed"} tags=[]
print(f"Elements in Set Union: {len(gene_ids_citeseq | gene_ids_multiome)}")
print(f"Elements in Set Intersection: {len(gene_ids_citeseq & gene_ids_multiome)}")
print(f"multiome has {len(gene_ids_multiome - gene_ids_citeseq)} unique gene ids.")
print(f"Citeseq has {len(gene_ids_citeseq - gene_ids_multiome)} unique gene ids.")

# + [markdown] papermill={"duration": 0.010915, "end_time": "2022-08-18T15:03:26.775203", "exception": false, "start_time": "2022-08-18T15:03:26.764288", "status": "completed"} tags=[]
# Even though we have a huge intersection, there are quite a few genes unique to each test.
#
# We will now again get information about distribution of values:

# + papermill={"duration": 1.232082, "end_time": "2022-08-18T15:03:28.018606", "exception": false, "start_time": "2022-08-18T15:03:26.786524", "status": "completed"} tags=[]
print(f"Values for gene expression range from {df_cite_train_x.min().min():.2f} to {df_cite_train_x.max().max():.2f}")

# get data about non-zero values
min_cells_non_zero_y = df_cite_train_x.gt(0).sum(axis=1).min()
max_cells_non_zero_y = df_cite_train_x.gt(0).sum(axis=1).max()
sum_non_zero_values_y = df_cite_train_x.sum().sum()
count_non_zero_values_y = df_cite_train_x.gt(0).sum().sum()
average_non_zero_per_gene_y = df_cite_train_x[df_cite_train_x.gt(0)].count(axis = 1).mean()

print(f"Each cell has at least {min_cells_non_zero_y} genes with non-zero gene expression values and a maximum of {max_cells_non_zero_y}.")
print(f"On average there are {round(average_non_zero_per_gene_y)} genes with non-zero gene expression values in each cell.")
print(f"The average non-zero value for gene expression is about {sum_non_zero_values_y / count_non_zero_values_y:.2f}.")

# investigate standard deviation of features
std_dev_of_genes_y = df_cite_train_x.std()

# ignore genes that are only accessible in a single cell
std_dev_of_genes_without_singles_y = std_dev_of_genes_y[df_cite_train_x.gt(0).sum().gt(1)]
print(f"The standard deviation of gene expression values is between {std_dev_of_genes_without_singles_y.min():.2f} and {std_dev_of_genes_without_singles_y.max():.2f}.\nThe average standard deviation is {std_dev_of_genes_without_singles_y.mean():.2f}")

# + [markdown] papermill={"duration": 0.011341, "end_time": "2022-08-18T15:03:28.041818", "exception": false, "start_time": "2022-08-18T15:03:28.030477", "status": "completed"} tags=[]
# We can see that the gene expression features obtained by the citeseq and the multiome test are quite similar. Values are in the same range and also the standard deviation and average non-zero values are of comparable size. For now, we only had a look at part of the data, so it will be interesting to see if this holds for all the data. We will get to that later.
#
# For now, let's have a look at how many genes are expressed in individual cells.

# + papermill={"duration": 0.303311, "end_time": "2022-08-18T15:03:28.356861", "exception": false, "start_time": "2022-08-18T15:03:28.053550", "status": "completed"} tags=[]
s = df_cite_train_x.gt(0).sum(axis = 1)
counts = s.groupby(lambda x: s[x] // 300).count()
counts.index = counts.index * 300

fig, ax = plt.subplots()
ax.plot(counts.index, counts.values)
ax.set_xlabel('number of genes expressed')
ax.set_ylabel('number of cells')
plt.show()

# + [markdown] papermill={"duration": 0.011821, "end_time": "2022-08-18T15:03:28.380961", "exception": false, "start_time": "2022-08-18T15:03:28.369140", "status": "completed"} tags=[]
# This sums up our investigation of the gene expression data obtained by the citeseq test. We saw that the data is comparable to that obtained by multiome. One key difference is that both have unique genes that are only measured in one test. Later, we will also investigate if the comparability of the data is due to preceding normalization of the raw data obtained by the tests and also address the question of what an expression value of e. g. 2.4 actually means.

# + [markdown] papermill={"duration": 0.012007, "end_time": "2022-08-18T15:03:28.405673", "exception": false, "start_time": "2022-08-18T15:03:28.393666", "status": "completed"} tags=[]
# <div style="color:white;display:fill;
#             background-color:#3bb2d6;font-size:200%;">
#     <p style="padding: 4px;color:white;"><b>2.3 Surface protein level data</b></p>
# </div>
#
# Lastly, we will have a look at the surface protein levels data gathered by citeseq.

# + papermill={"duration": 0.760015, "end_time": "2022-08-18T15:03:29.177729", "exception": false, "start_time": "2022-08-18T15:03:28.417714", "status": "completed"} tags=[]
df_cite_train_y = pd.read_hdf(FP_CITE_TRAIN_TARGETS)
df_cite_train_y.head()

# + [markdown] papermill={"duration": 0.011687, "end_time": "2022-08-18T15:03:29.202183", "exception": false, "start_time": "2022-08-18T15:03:29.190496", "status": "completed"} tags=[]
# Compared to what we have seen so far, the number of columns in this data set is quite small. We have measurements of 140 features per cell. Most of the names start with CD, which is short for "Cluster of differentiation". CDs are used to classify surface molecules a cell expresses. This information can then be used to get an idea of what kind of cell is present, or what function this cell is supposed to serve in the body (I am not sure if my understanding here is even remotely accurate).
#
# Let's forget about the biological view for a moment and focus on the data science centric view. As we will see in the next cell, we have no zero values in this dataset, and thus much of the computation we did before (counting non-zero values, for example) is pointless. We will look at other features:

# + papermill={"duration": 0.375806, "end_time": "2022-08-18T15:03:29.589899", "exception": false, "start_time": "2022-08-18T15:03:29.214093", "status": "completed"} tags=[]
print(f"Measurements of surface protein levels range from {df_cite_train_y.min().min():.2f} to {df_cite_train_y.max().max():.2f}.")
print(f"The average value is {df_cite_train_y.mean().mean():.2f}.")
print(f"The standard deviation of surface protein levels is between {df_cite_train_y.std().min():.2f} and {df_cite_train_y.std().max():.2f}.")
print(f"The average standard deviation is {df_cite_train_y.std().mean():.2f}.")

# + [markdown] papermill={"duration": 0.011989, "end_time": "2022-08-18T15:03:29.614810", "exception": false, "start_time": "2022-08-18T15:03:29.602821", "status": "completed"} tags=[]
# We would also like to know if the absence of zero values could be due to inaccuracy in measurements. The following code checks for that. ATTENTION: As of right now, the threshold is completely arbitrary and will be revised when I know how accurate the test is!

# + papermill={"duration": 10.431082, "end_time": "2022-08-18T15:03:40.058257", "exception": false, "start_time": "2022-08-18T15:03:29.627175", "status": "completed"} tags=[]
threshold = 0.1
df_cite_train_y.applymap(lambda x: abs(x)).gt(threshold).sum(axis = 1)
print(f"Each cell has between {df_cite_train_y.applymap(lambda x: abs(x)).gt(threshold).sum(axis = 1).min()} and {df_cite_train_y.applymap(lambda x: abs(x)).gt(threshold).sum(axis = 1).max()} measurements with absolute values over {threshold}.")

# + [markdown] papermill={"duration": 0.012023, "end_time": "2022-08-18T15:03:40.083641", "exception": false, "start_time": "2022-08-18T15:03:40.071618", "status": "completed"} tags=[]
# It does not look like inaccuracy in measurements is the reason for absence of 0 values, but as said, this needs to be checked!

# + [markdown] papermill={"duration": 0.011923, "end_time": "2022-08-18T15:03:40.107775", "exception": false, "start_time": "2022-08-18T15:03:40.095852", "status": "completed"} tags=[]
# <div style="color:white;display:fill;
#             background-color:#3bb2d6;font-size:200%;">
#     <p style="padding: 4px;color:white;"><b>X Notes</b></p>
# </div>
#
# There are still some open questions in the text we need to address. Also, we want to get an understanding of the accuracy of the values and thus how many bits we will take for storage of data in the final data format.
