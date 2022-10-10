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

# # Loading and Visualizing scRNA-seq and scATAC-seq Data

# + tags=[]
import h5py
import hdf5plugin
import tables

import pandas as pd
import numpy as np
import scipy

import scanpy as sc
import anndata as ad

# + tags=[]
sc.set_figure_params(dpi=150)

# + tags=[]
base_dir = '/home/jovyan/kaggle/input/open-problems-multimodal/'
filenames = [
    'test_cite_inputs', 
    'test_multi_inputs', 
    'train_cite_inputs',
    'train_cite_targets',
    'train_multi_inputs',
    'train_multi_targets',
]

# + tags=[]
metadata_df = pd.read_csv('/home/jovyan/kaggle/input/open-problems-multimodal/metadata.csv')
metadata_df = metadata_df.set_index('cell_id')
# -

# The ATAC-seq data is stored in very large matrices (50,000-100,000 rows and over 220,000 columns). Although these matrices are very sparse, the .h5 format does not natively support sparse formats. We could load the ATAC-seq data into memory and then sparsify it, but this would take over 200GB of RAM. Instead we can load the dense ATAC-seq data in chunks, convert these chunks into a sparse format, then stack the chunks together.
#
# The scRNA-seq data matrices are much less sparse (only 78% of entries are zero) so we don't load them in this way.

# + tags=[]
adatas = {}
chunk_size = 10000

for filename in filenames:
    print(f'loading {filename}.h5')
    filepath = base_dir + filename + '.h5'
    
    h5_file = h5py.File(filepath)
    h5_data = h5_file[filename]
    
    features = h5_data['axis0'][:]
    cell_ids = h5_data['axis1'][:]
    
    features = features.astype(str)
    cell_ids = cell_ids.astype(str)
    
    technology = metadata_df.loc[cell_ids, 'technology'].unique().item()
    
    if technology == 'multiome':
        sparse_chunks = []
        n_cells = h5_data['block0_values'].shape[0]
        
        for chunk_indices in np.array_split(np.arange(n_cells), 100):
            chunk = h5_data['block0_values'][chunk_indices]
            sparse_chunk = scipy.sparse.csr_matrix(chunk)
            sparse_chunks.append(sparse_chunk)
            
        X = scipy.sparse.vstack(sparse_chunks)
    elif technology == 'citeseq':
        X = h5_data['block0_values'][:]
        
    adata = ad.AnnData(
        X=X,
        obs=metadata_df.loc[cell_ids],
        var=pd.DataFrame(index=features),
    )
    
    adatas[filename] = adata
# -

adatas

# ## scATAC-seq Data

# + tags=[]
atac_adata = ad.concat([adatas['test_multi_inputs'], adatas['train_multi_inputs']])

# + tags=[]
sc.pp.pca(atac_adata, zero_center=None)

# + tags=[]
sc.pp.neighbors(atac_adata)

# + tags=[]
sc.tl.umap(atac_adata)

# + tags=[]
sc.pl.umap(atac_adata, color=['day', 'cell_type'])
# -

# ## scRNA-seq Data

# + tags=[]
rna_adata = ad.concat([adatas['test_cite_inputs'], adatas['train_cite_inputs']])

# + tags=[]
sc.pp.pca(rna_adata)

# + tags=[]
sc.pp.neighbors(rna_adata)

# + tags=[]
sc.tl.umap(rna_adata)

# + tags=[]
sc.pl.umap(rna_adata, color=['day', 'cell_type'])
