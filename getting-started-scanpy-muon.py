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

# + [markdown] papermill={"duration": 0.021941, "end_time": "2022-08-16T19:03:36.940583", "exception": false, "start_time": "2022-08-16T19:03:36.918642", "status": "completed"} tags=[]
# Relevant tutorials:
# - [scRNA-seq (scanpy)](https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html)
# - [multiome (muon)](https://muon-tutorials.readthedocs.io/en/latest/single-cell-rna-atac/index.html)
# - [CITE-seq (muon)](https://muon-tutorials.readthedocs.io/en/latest/cite-seq/1-CITE-seq-PBMC-5k.html)

# + papermill={"duration": 32.569811, "end_time": "2022-08-16T19:04:09.524851", "exception": false, "start_time": "2022-08-16T19:03:36.955040", "status": "completed"} tags=[]
# !pip install tables
# !pip install muon

# + papermill={"duration": 22.642704, "end_time": "2022-08-16T19:04:32.184707", "exception": false, "start_time": "2022-08-16T19:04:09.542003", "status": "completed"} tags=[]
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import mudata
from mudata import AnnData, MuData
import scanpy as sc
import muon as mu

# + papermill={"duration": 0.02692, "end_time": "2022-08-16T19:04:32.229068", "exception": false, "start_time": "2022-08-16T19:04:32.202148", "status": "completed"} tags=[]
DATA_DIR = Path("../input/open-problems-multimodal")

# + [markdown] papermill={"duration": 0.016621, "end_time": "2022-08-16T19:04:32.262942", "exception": false, "start_time": "2022-08-16T19:04:32.246321", "status": "completed"} tags=[]
# For demonstration purposes, only first 1000 cells will be loaded. For full data, free tier on Kaggle might not be enough.

# + papermill={"duration": 0.024364, "end_time": "2022-08-16T19:04:32.304340", "exception": false, "start_time": "2022-08-16T19:04:32.279976", "status": "completed"} tags=[]
N = 1000

# + [markdown] papermill={"duration": 0.01745, "end_time": "2022-08-16T19:04:32.338922", "exception": false, "start_time": "2022-08-16T19:04:32.321472", "status": "completed"} tags=[]
# # CITE-seq
#
# 1. Load CITE-seq data into AnnData/MuData format.
# 2. Run PCA/UMAP to visualize the data.

# + [markdown] papermill={"duration": 0.016657, "end_time": "2022-08-16T19:04:32.372777", "exception": false, "start_time": "2022-08-16T19:04:32.356120", "status": "completed"} tags=[]
# ## anndata/mudata

# + papermill={"duration": 0.995018, "end_time": "2022-08-16T19:04:33.385280", "exception": false, "start_time": "2022-08-16T19:04:32.390262", "status": "completed"} tags=[]
rna_df = pd.read_hdf(DATA_DIR / "train_cite_inputs.h5", start=0, stop=N)
prot_df = pd.read_hdf(DATA_DIR / "train_cite_targets.h5", start=0, stop=N)

# # full data:
# rna_df = pd.read_hdf(DATA_DIR / "train_cite_inputs.h5")
# prot_df = pd.read_hdf(DATA_DIR / "train_cite_targets.h5")

# + papermill={"duration": 0.692682, "end_time": "2022-08-16T19:04:34.095731", "exception": false, "start_time": "2022-08-16T19:04:33.403049", "status": "completed"} tags=[]
rna = AnnData(csr_matrix(rna_df))
rna.obs_names = rna_df.index.values
rna.var_names = rna_df.columns.values

# + papermill={"duration": 0.035353, "end_time": "2022-08-16T19:04:34.148125", "exception": false, "start_time": "2022-08-16T19:04:34.112772", "status": "completed"} tags=[]
prot = AnnData(csr_matrix(prot_df))
prot.obs_names = prot_df.index.values
prot.var_names = prot_df.columns.values

# + papermill={"duration": 0.063268, "end_time": "2022-08-16T19:04:34.235390", "exception": false, "start_time": "2022-08-16T19:04:34.172122", "status": "completed"} tags=[]
cite = MuData({"rna": rna, "prot": prot})

# + papermill={"duration": 0.464532, "end_time": "2022-08-16T19:04:34.716999", "exception": false, "start_time": "2022-08-16T19:04:34.252467", "status": "completed"} tags=[]
metadata = pd.read_csv(DATA_DIR / "metadata.csv")
cite.obs = cite.obs.join(metadata.set_index("cell_id"))

cite.obs.donor = cite.obs.donor.astype("category")
cite.obs.cell_type = cite.obs.cell_type.astype("category")
cite.obs.day = cite.obs.day.astype("category")
cite.obs.technology = cite.obs.technology.astype("category")

# + papermill={"duration": 0.03365, "end_time": "2022-08-16T19:04:34.767924", "exception": false, "start_time": "2022-08-16T19:04:34.734274", "status": "completed"} tags=[]
with mudata.set_options(display_style="html", display_html_expand=0b100):
    display(cite)

# + [markdown] papermill={"duration": 0.016834, "end_time": "2022-08-16T19:04:34.802362", "exception": false, "start_time": "2022-08-16T19:04:34.785528", "status": "completed"} tags=[]
# ## [rna] scanpy

# + [markdown] papermill={"duration": 0.016828, "end_time": "2022-08-16T19:04:34.836326", "exception": false, "start_time": "2022-08-16T19:04:34.819498", "status": "completed"} tags=[]
# ### Feature selection

# + papermill={"duration": 2.134309, "end_time": "2022-08-16T19:04:36.988332", "exception": false, "start_time": "2022-08-16T19:04:34.854023", "status": "completed"} tags=[]
sc.pp.highly_variable_genes(rna, min_mean=0.05, max_mean=8, min_disp=0.5)
sc.pl.highly_variable_genes(rna)

# + papermill={"duration": 0.030674, "end_time": "2022-08-16T19:04:37.037533", "exception": false, "start_time": "2022-08-16T19:04:37.006859", "status": "completed"} tags=[]
np.sum(rna.var.highly_variable)

# + papermill={"duration": 0.34659, "end_time": "2022-08-16T19:04:37.402610", "exception": false, "start_time": "2022-08-16T19:04:37.056020", "status": "completed"} tags=[]
rna.layers["data"] = rna.X.copy()
sc.pp.scale(rna)

# + [markdown] papermill={"duration": 0.01874, "end_time": "2022-08-16T19:04:37.439714", "exception": false, "start_time": "2022-08-16T19:04:37.420974", "status": "completed"} tags=[]
# ### PCA

# + papermill={"duration": 0.383003, "end_time": "2022-08-16T19:04:37.841017", "exception": false, "start_time": "2022-08-16T19:04:37.458014", "status": "completed"} tags=[]
sc.tl.pca(rna)

# + papermill={"duration": 0.470842, "end_time": "2022-08-16T19:04:38.378687", "exception": false, "start_time": "2022-08-16T19:04:37.907845", "status": "completed"} tags=[]
mu.pl.embedding(cite, basis="rna:X_pca", color=["cell_type", "donor", "day"])

# + [markdown] papermill={"duration": 0.020378, "end_time": "2022-08-16T19:04:38.420667", "exception": false, "start_time": "2022-08-16T19:04:38.400289", "status": "completed"} tags=[]
# ### UMAP

# + papermill={"duration": 5.936573, "end_time": "2022-08-16T19:04:44.377840", "exception": false, "start_time": "2022-08-16T19:04:38.441267", "status": "completed"} tags=[]
sc.pp.neighbors(rna)
sc.tl.umap(rna, random_state=1)

# + papermill={"duration": 0.538394, "end_time": "2022-08-16T19:04:44.938808", "exception": false, "start_time": "2022-08-16T19:04:44.400414", "status": "completed"} tags=[]
mu.pl.embedding(cite, basis="rna:X_umap", color=["cell_type", "donor", "day"])

# + [markdown] papermill={"duration": 0.024411, "end_time": "2022-08-16T19:04:44.988225", "exception": false, "start_time": "2022-08-16T19:04:44.963814", "status": "completed"} tags=[]
# ## [prot] scanpy

# + papermill={"duration": 0.040083, "end_time": "2022-08-16T19:04:45.053283", "exception": false, "start_time": "2022-08-16T19:04:45.013200", "status": "completed"} tags=[]
prot.layers["data"] = prot.X.copy()
sc.pp.scale(prot)

# + [markdown] papermill={"duration": 0.024677, "end_time": "2022-08-16T19:04:45.102817", "exception": false, "start_time": "2022-08-16T19:04:45.078140", "status": "completed"} tags=[]
# ### PCA

# + papermill={"duration": 0.145825, "end_time": "2022-08-16T19:04:45.273675", "exception": false, "start_time": "2022-08-16T19:04:45.127850", "status": "completed"} tags=[]
sc.tl.pca(prot)

# + papermill={"duration": 0.478528, "end_time": "2022-08-16T19:04:45.862239", "exception": false, "start_time": "2022-08-16T19:04:45.383711", "status": "completed"} tags=[]
mu.pl.embedding(cite, basis="prot:X_pca", color=["cell_type", "donor", "day"])

# + papermill={"duration": 0.171498, "end_time": "2022-08-16T19:04:46.060182", "exception": false, "start_time": "2022-08-16T19:04:45.888684", "status": "completed"} tags=[]
sc.pp.neighbors(prot)

# + [markdown] papermill={"duration": 0.025649, "end_time": "2022-08-16T19:04:46.111957", "exception": false, "start_time": "2022-08-16T19:04:46.086308", "status": "completed"} tags=[]
# ## [rna+atac] muon

# + papermill={"duration": 0.068474, "end_time": "2022-08-16T19:04:46.206596", "exception": false, "start_time": "2022-08-16T19:04:46.138122", "status": "completed"} tags=[]
cite.update()

# + papermill={"duration": 0.035672, "end_time": "2022-08-16T19:04:46.268863", "exception": false, "start_time": "2022-08-16T19:04:46.233191", "status": "completed"} tags=[]
cite
# cite.write("/kaggle/tmp/train_cite.h5mu")

# + [markdown] papermill={"duration": 0.026121, "end_time": "2022-08-16T19:04:46.321571", "exception": false, "start_time": "2022-08-16T19:04:46.295450", "status": "completed"} tags=[]
# # Multiome
#
# 1. Load Multiome data into AnnData/MuData format.
# 2. Run PCA/UMAP to visualize the data.

# + [markdown] papermill={"duration": 0.026161, "end_time": "2022-08-16T19:04:46.374021", "exception": false, "start_time": "2022-08-16T19:04:46.347860", "status": "completed"} tags=[]
# ## anndata/mudata

# + papermill={"duration": 4.73779, "end_time": "2022-08-16T19:04:51.138435", "exception": false, "start_time": "2022-08-16T19:04:46.400645", "status": "completed"} tags=[]
atac_df = pd.read_hdf(DATA_DIR / "train_multi_inputs.h5", start=0, stop=N)
rna_df = pd.read_hdf(DATA_DIR / "train_multi_targets.h5", start=0, stop=N)

# # full data:
# rna_df = pd.read_hdf(DATA_DIR / "train_multi_inputs.h5")
# prot_df = pd.read_hdf(DATA_DIR / "train_multi_targets.h5")

# + papermill={"duration": 0.622912, "end_time": "2022-08-16T19:04:51.788642", "exception": false, "start_time": "2022-08-16T19:04:51.165730", "status": "completed"} tags=[]
rna = AnnData(csr_matrix(rna_df))
rna.obs_names = rna_df.index.values
rna.var_names = rna_df.columns.values

# + papermill={"duration": 6.423975, "end_time": "2022-08-16T19:04:58.239088", "exception": false, "start_time": "2022-08-16T19:04:51.815113", "status": "completed"} tags=[]
atac = AnnData(csr_matrix(atac_df))
atac.obs_names = atac_df.index.values
atac.var_names = atac_df.columns.values

# + papermill={"duration": 0.304106, "end_time": "2022-08-16T19:04:58.586358", "exception": false, "start_time": "2022-08-16T19:04:58.282252", "status": "completed"} tags=[]
multiome = MuData({"rna": rna, "atac": atac})

# + papermill={"duration": 0.470274, "end_time": "2022-08-16T19:04:59.089583", "exception": false, "start_time": "2022-08-16T19:04:58.619309", "status": "completed"} tags=[]
metadata = pd.read_csv(DATA_DIR / "metadata.csv")
multiome.obs = multiome.obs.join(metadata.set_index("cell_id"))
multiome.obs.donor = multiome.obs.donor.astype("category")
multiome.obs.cell_type = multiome.obs.cell_type.astype("category")
multiome.obs.day = multiome.obs.day.astype("category")
multiome.obs.technology = multiome.obs.technology.astype("category")

# + papermill={"duration": 0.048577, "end_time": "2022-08-16T19:04:59.172531", "exception": false, "start_time": "2022-08-16T19:04:59.123954", "status": "completed"} tags=[]
with mudata.set_options(display_style="html", display_html_expand=0b100):
    display(multiome)

# + [markdown] papermill={"duration": 0.027966, "end_time": "2022-08-16T19:04:59.229320", "exception": false, "start_time": "2022-08-16T19:04:59.201354", "status": "completed"} tags=[]
# ## [rna] scanpy

# + [markdown] papermill={"duration": 0.02703, "end_time": "2022-08-16T19:04:59.283992", "exception": false, "start_time": "2022-08-16T19:04:59.256962", "status": "completed"} tags=[]
# ### Feature selection

# + papermill={"duration": 1.978156, "end_time": "2022-08-16T19:05:01.289309", "exception": false, "start_time": "2022-08-16T19:04:59.311153", "status": "completed"} tags=[]
sc.pp.highly_variable_genes(rna, min_mean=0.2, max_mean=6, min_disp=0.8)
sc.pl.highly_variable_genes(rna)

# + papermill={"duration": 0.043233, "end_time": "2022-08-16T19:05:01.362506", "exception": false, "start_time": "2022-08-16T19:05:01.319273", "status": "completed"} tags=[]
np.sum(rna.var.highly_variable)

# + papermill={"duration": 0.678218, "end_time": "2022-08-16T19:05:02.075173", "exception": false, "start_time": "2022-08-16T19:05:01.396955", "status": "completed"} tags=[]
rna.layers["data"] = rna.X.copy()
sc.pp.scale(rna)

# + [markdown] papermill={"duration": 0.033204, "end_time": "2022-08-16T19:05:02.142366", "exception": false, "start_time": "2022-08-16T19:05:02.109162", "status": "completed"} tags=[]
# ### PCA

# + papermill={"duration": 0.476053, "end_time": "2022-08-16T19:05:02.651692", "exception": false, "start_time": "2022-08-16T19:05:02.175639", "status": "completed"} tags=[]
sc.tl.pca(rna)

# + papermill={"duration": 0.484201, "end_time": "2022-08-16T19:05:03.225379", "exception": false, "start_time": "2022-08-16T19:05:02.741178", "status": "completed"} tags=[]
mu.pl.embedding(cite, basis="rna:X_pca", color=["cell_type", "donor", "day"])

# + [markdown] papermill={"duration": 0.046061, "end_time": "2022-08-16T19:05:03.307673", "exception": false, "start_time": "2022-08-16T19:05:03.261612", "status": "completed"} tags=[]
# ### UMAP

# + papermill={"duration": 3.540446, "end_time": "2022-08-16T19:05:06.880122", "exception": false, "start_time": "2022-08-16T19:05:03.339676", "status": "completed"} tags=[]
sc.pp.neighbors(rna)
sc.tl.umap(rna, random_state=1)

# + papermill={"duration": 0.696907, "end_time": "2022-08-16T19:05:07.612604", "exception": false, "start_time": "2022-08-16T19:05:06.915697", "status": "completed"} tags=[]
mu.pl.embedding(multiome, basis="rna:X_umap", color=["cell_type", "donor", "day"])

# + [markdown] papermill={"duration": 0.06144, "end_time": "2022-08-16T19:05:07.749398", "exception": false, "start_time": "2022-08-16T19:05:07.687958", "status": "completed"} tags=[]
# ## [atac] scanpy

# + [markdown] papermill={"duration": 0.062933, "end_time": "2022-08-16T19:05:07.872449", "exception": false, "start_time": "2022-08-16T19:05:07.809516", "status": "completed"} tags=[]
# ### Feature selection

# + papermill={"duration": 16.724635, "end_time": "2022-08-16T19:05:24.660323", "exception": false, "start_time": "2022-08-16T19:05:07.935688", "status": "completed"} tags=[]
sc.pp.highly_variable_genes(atac, min_mean=0.5, max_mean=8, min_disp=1.0)
sc.pl.highly_variable_genes(atac)

# + papermill={"duration": 0.047232, "end_time": "2022-08-16T19:05:24.744969", "exception": false, "start_time": "2022-08-16T19:05:24.697737", "status": "completed"} tags=[]
np.sum(atac.var.highly_variable)

# + [markdown] papermill={"duration": 0.035627, "end_time": "2022-08-16T19:05:24.816937", "exception": false, "start_time": "2022-08-16T19:05:24.781310", "status": "completed"} tags=[]
# ### PCA

# + papermill={"duration": 20.285253, "end_time": "2022-08-16T19:05:45.137194", "exception": false, "start_time": "2022-08-16T19:05:24.851941", "status": "completed"} tags=[]
mu.atac.tl.lsi(atac)

# + papermill={"duration": 0.759416, "end_time": "2022-08-16T19:05:45.931644", "exception": false, "start_time": "2022-08-16T19:05:45.172228", "status": "completed"} tags=[]
mu.pl.embedding(multiome, basis="atac:X_lsi", color=["cell_type", "donor", "day"])

# + [markdown] papermill={"duration": 0.038733, "end_time": "2022-08-16T19:05:46.010294", "exception": false, "start_time": "2022-08-16T19:05:45.971561", "status": "completed"} tags=[]
# ### UMAP

# + papermill={"duration": 3.012494, "end_time": "2022-08-16T19:05:49.062646", "exception": false, "start_time": "2022-08-16T19:05:46.050152", "status": "completed"} tags=[]
sc.pp.neighbors(atac, use_rep="X_lsi")
sc.tl.umap(atac, random_state=1)

# + papermill={"duration": 0.539336, "end_time": "2022-08-16T19:05:49.640160", "exception": false, "start_time": "2022-08-16T19:05:49.100824", "status": "completed"} tags=[]
mu.pl.embedding(multiome, basis="atac:X_umap", color=["cell_type", "donor", "day"])

# + [markdown] papermill={"duration": 0.04233, "end_time": "2022-08-16T19:05:49.730925", "exception": false, "start_time": "2022-08-16T19:05:49.688595", "status": "completed"} tags=[]
# ## [rna+atac] muon

# + papermill={"duration": 0.333515, "end_time": "2022-08-16T19:05:50.107582", "exception": false, "start_time": "2022-08-16T19:05:49.774067", "status": "completed"} tags=[]
multiome.update()

# + papermill={"duration": 0.054825, "end_time": "2022-08-16T19:05:50.205195", "exception": false, "start_time": "2022-08-16T19:05:50.150370", "status": "completed"} tags=[]
multiome
# multiome.write("/kaggle/tmp/train_multiome.h5mu")
