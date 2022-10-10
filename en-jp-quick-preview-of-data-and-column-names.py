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

# + [markdown] papermill={"duration": 0.009613, "end_time": "2022-08-18T08:31:22.086684", "exception": false, "start_time": "2022-08-18T08:31:22.077071", "status": "completed"} tags=[]
# # Data Preview
# We cannot see a preview of the .h5 format files on the competition page, so check the first few rows at first.
#
# The following notebook was used for data loading.
#
# コンペページでは、.h5ファイルのプレビューが見れないので、ひとまず先頭行を確認します。
#
# データの読み込みについては、下記のnotebookを参考にしました。
#
# [Getting Started - Data Loading created @ Peter Holderrieth](https://www.kaggle.com/code/peterholderrieth/getting-started-data-loading)

# + [markdown] papermill={"duration": 0.007418, "end_time": "2022-08-18T08:31:22.103376", "exception": false, "start_time": "2022-08-18T08:31:22.095958", "status": "completed"} tags=[]
# # Data files
# 9 files are provided.
#
# 9つのファイルが提供されています。
# 1. metadata.csv
# >*     cell_id - A unique identifier for each observed cell. / 観測された各セルに一意な識別子。
# >*     donor - An identifier for the four cell donors. / 4人の細胞提供者の識別子。
# >*     day - The day of the experiment the observation was made. / 実験観察が行われた日付。
# >*     technology - Either citeseq or multiome. / "citeseq"か"multiome"のいずれか。
# >*     cell_type - One of the  following cell types or else hidden. / 下記のセルタイプのいずれか、またはそれ以外が"hidden"。
# >>* MasP = Mast Cell Progenitor
# >>* MkP = Megakaryocyte Progenitor
# >>* NeuP = Neutrophil Progenitor
# >>* MoP = Monocyte Progenitor
# >>* EryP = Erythrocyte Progenitor
# >>* HSC = Hematoploetic Stem Cell
# >>* BP = B-Cell ProgenitorMasP = Mast Cell Progenitor
# >>* MkP = Megakaryocyte Progenitor
# >>* NeuP = Neutrophil Progenitor
# >>* MoP = Monocyte Progenitor
# >>* EryP = Erythrocyte Progenitor
# >>* HSC = Hematoploetic Stem Cell
# >>* BP = B-Cell Progenitor
#     
# 2. train_multi_inputs.h5
# 3. test_multi_inputs.h5
# >* train/test_multi_inputs.h5 - ATAC-seq peak counts transformed with TF-IDF using the default log(TF) * log(IDF) output (chromatin accessibility), with rows corresponding to cells and columns corresponding to the location of the genome whose level of accessibility is measured, here identified by the genomic coordinates on reference genome GRCh38 provided in the 10x References - 2020-A (July 7, 2020).
# >* train/test_multi_inputs.h5 - ATAC-seqのピーク数をデフォルトのlog(TF) * log(IDF)出力でTF-IDF変換したもの（クロマチンアクセス性）。行は細胞、列はアクセス性のレベルが測定されたゲノムの位置に対応し、ここでは10x References - 2020-A (July 7, 2020) で提供された参照ゲノムGRCh38のゲノム座標で特定されています。
#
# 4. train_multi_targets.h5
# >* train_multi_labels.h5 - RNA gene expression levels as library-size normalized and log1p transformed counts for the same cells.
# >* train_multi_labels.h5 - RNA遺伝子の発現量は、同じ細胞のライブラリーサイズで正規化し、log1p変換したカウント値。
#
# 5. train_cite_inputs.h5
# 6. test_cite_inputs.h5
# >* train/test_cite_inputs.h5 - RNA library-size normalized and log1p transformed counts (gene expression levels), with rows corresponding to cells and columns corresponding to genes given by {gene_name}_{gene_ensemble-ids}.
# >* train/test_cite_inputs.h5 - RNAライブラリーサイズを正規化し、log1p変換したカウント（遺伝子発現量）。行は細胞、列は{gene_name}_{gene_ensemble-ids}で指定した遺伝子に対応する。
# 7. train_cite_targets.h5
# >* train_cite_labels.h5 - Surface protein levels for the same cells that have been dsb normalized.
# >* train_cite_labels.h5 - dsbで正規化された同じ細胞の表面タンパク質レベル。
#
# 8. evaluation_ids.csv
# >* Identifies the labels from the test set to be evaluated. It provides a join key from the cell_id / gene_id identifiers of the label matrix to the row_id needed for the submission file.
# >* 評価対象のテストセットからラベルを特定する。ラベルマトリックスのcell_id / gene_id 識別子から、提出ファイルに必要なrow_idへの結合キーを提供します。
#
# 9. sample_submission.csv
# >* A sample submission file in the correct format. See the Evaluation page for more information.
# >* 正しい形式の投稿ファイルのサンプルです。詳しくは「評価」のページをご覧ください。
#
# www.DeepL.com/Translator（無料版）で翻訳しました。

# + [markdown] papermill={"duration": 0.007009, "end_time": "2022-08-18T08:31:22.117682", "exception": false, "start_time": "2022-08-18T08:31:22.110673", "status": "completed"} tags=[]
# # Data Preview
#
# Only the first few lines are read, as memory will be insufficient.
#
# メモリが不足するため、先頭の数行のみ読み込みます。

# + papermill={"duration": 0.022154, "end_time": "2022-08-18T08:31:22.147159", "exception": false, "start_time": "2022-08-18T08:31:22.125005", "status": "completed"} tags=[]
num_rows = 5

# + papermill={"duration": 14.116701, "end_time": "2022-08-18T08:31:36.271284", "exception": false, "start_time": "2022-08-18T08:31:22.154583", "status": "completed"} tags=[]
#If you see a urllib warning running this cell, go to "Settings" on the right hand side, 
#and turn on internet. Note, you need to be phone verified.
# !pip install --quiet tables

# + papermill={"duration": 1.061361, "end_time": "2022-08-18T08:31:37.340312", "exception": false, "start_time": "2022-08-18T08:31:36.278951", "status": "completed"} tags=[]
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "/kaggle/input/open-problems-multimodal/"
FP_CELL_METADATA = os.path.join(DATA_DIR,"metadata.csv")  # 1

FP_CITE_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_cite_inputs.h5")  # 2
FP_CITE_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_cite_targets.h5")  # 4
FP_CITE_TEST_INPUTS = os.path.join(DATA_DIR,"test_cite_inputs.h5")  # 3

FP_MULTIOME_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_multi_inputs.h5")  # 5
FP_MULTIOME_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_multi_targets.h5")  # 7
FP_MULTIOME_TEST_INPUTS = os.path.join(DATA_DIR,"test_multi_inputs.h5")  # 6

FP_SUBMISSION = os.path.join(DATA_DIR,"sample_submission.csv")  # 8
FP_EVALUATION_IDS = os.path.join(DATA_DIR,"evaluation_ids.csv")  # 9

# + papermill={"duration": 0.069307, "end_time": "2022-08-18T08:31:37.416971", "exception": false, "start_time": "2022-08-18T08:31:37.347664", "status": "completed"} tags=[]
df_1_metadata = pd.read_csv(FP_CELL_METADATA, nrows=num_rows)
df_1_metadata.head(num_rows)

# + papermill={"duration": 1.14288, "end_time": "2022-08-18T08:31:38.567571", "exception": false, "start_time": "2022-08-18T08:31:37.424691", "status": "completed"} tags=[]
df_2_train_multi_inputs = pd.read_hdf(FP_MULTIOME_TRAIN_INPUTS, stop=num_rows)
df_2_train_multi_inputs.head(num_rows)

# + papermill={"duration": 0.978819, "end_time": "2022-08-18T08:31:39.554839", "exception": false, "start_time": "2022-08-18T08:31:38.576020", "status": "completed"} tags=[]
df_3_test_multi_inputs = pd.read_hdf(FP_MULTIOME_TEST_INPUTS, stop=num_rows)
df_3_test_multi_inputs.head(num_rows)

# + papermill={"duration": 0.217533, "end_time": "2022-08-18T08:31:39.781884", "exception": false, "start_time": "2022-08-18T08:31:39.564351", "status": "completed"} tags=[]
df_4_train_multi_targets = pd.read_hdf(FP_MULTIOME_TRAIN_TARGETS, stop=num_rows)
df_4_train_multi_targets.head(num_rows)

# + papermill={"duration": 0.211384, "end_time": "2022-08-18T08:31:40.002315", "exception": false, "start_time": "2022-08-18T08:31:39.790931", "status": "completed"} tags=[]
df_5_train_cite_inputs = pd.read_hdf(FP_CITE_TRAIN_INPUTS, stop=num_rows)
df_5_train_cite_inputs.head(num_rows)

# + papermill={"duration": 0.202708, "end_time": "2022-08-18T08:31:40.214623", "exception": false, "start_time": "2022-08-18T08:31:40.011915", "status": "completed"} tags=[]
df_6_test_cite_inputs = pd.read_hdf(FP_CITE_TEST_INPUTS, stop=num_rows)
df_6_test_cite_inputs.head(num_rows)

# + papermill={"duration": 0.13689, "end_time": "2022-08-18T08:31:40.361612", "exception": false, "start_time": "2022-08-18T08:31:40.224722", "status": "completed"} tags=[]
df_7_train_cite_targets = pd.read_hdf(FP_CITE_TRAIN_TARGETS, stop=num_rows)
df_7_train_cite_targets.head(num_rows)

# + papermill={"duration": 0.044067, "end_time": "2022-08-18T08:31:40.416491", "exception": false, "start_time": "2022-08-18T08:31:40.372424", "status": "completed"} tags=[]
df_8_sample_submission = pd.read_csv(FP_SUBMISSION, nrows=num_rows)
df_8_sample_submission.head(num_rows)

# + papermill={"duration": 0.042744, "end_time": "2022-08-18T08:31:40.469625", "exception": false, "start_time": "2022-08-18T08:31:40.426881", "status": "completed"} tags=[]
df_9_evaluation_ids = pd.read_csv(FP_EVALUATION_IDS, nrows=num_rows)
df_9_evaluation_ids.head(num_rows)

# + [markdown] papermill={"duration": 0.010681, "end_time": "2022-08-18T08:31:40.490952", "exception": false, "start_time": "2022-08-18T08:31:40.480271", "status": "completed"} tags=[]
# # Column Name Patterns
# ### Check for patterns in column names by replacing numbers in column names with "@" symbols
# ### カラム名の数字を「@」記号に置き換えることにより、カラム名のパターンを確認

# + [markdown] papermill={"duration": 0.01137, "end_time": "2022-08-18T08:31:40.512946", "exception": false, "start_time": "2022-08-18T08:31:40.501576", "status": "completed"} tags=[]
# For Multiome, the number of columns is very large, but the pattern is limited.
#
# Multiomeについて、カラム数は非常に多いが、パターンは限られている。

# + papermill={"duration": 0.019971, "end_time": "2022-08-18T08:31:40.543538", "exception": false, "start_time": "2022-08-18T08:31:40.523567", "status": "completed"} tags=[]
print(len(df_2_train_multi_inputs.columns))

# + papermill={"duration": 1.064177, "end_time": "2022-08-18T08:31:41.619175", "exception": false, "start_time": "2022-08-18T08:31:40.554998", "status": "completed"} tags=[]
import re
cols_replace_digit = []
for col in df_2_train_multi_inputs.columns:
    cols_replace_digit.append(re.sub(r'\d', "@", col))
set(cols_replace_digit)

# + papermill={"duration": 0.020065, "end_time": "2022-08-18T08:31:41.650063", "exception": false, "start_time": "2022-08-18T08:31:41.629998", "status": "completed"} tags=[]
print(len(df_3_test_multi_inputs.columns))

# + papermill={"duration": 1.085, "end_time": "2022-08-18T08:31:42.745766", "exception": false, "start_time": "2022-08-18T08:31:41.660766", "status": "completed"} tags=[]
import re
cols_replace_digit = []
for col in df_3_test_multi_inputs.columns:
    cols_replace_digit.append(re.sub(r'\d', "@", col))
set(cols_replace_digit)

# + papermill={"duration": 0.021137, "end_time": "2022-08-18T08:31:42.778087", "exception": false, "start_time": "2022-08-18T08:31:42.756950", "status": "completed"} tags=[]
print(len(df_4_train_multi_targets.columns))

# + papermill={"duration": 0.109755, "end_time": "2022-08-18T08:31:42.899094", "exception": false, "start_time": "2022-08-18T08:31:42.789339", "status": "completed"} tags=[]
cols_replace_digit = []
for col in df_4_train_multi_targets.columns:
    cols_replace_digit.append(re.sub(r'\d', "@", col))
set(cols_replace_digit)

# + [markdown] papermill={"duration": 0.011151, "end_time": "2022-08-18T08:31:42.921379", "exception": false, "start_time": "2022-08-18T08:31:42.910228", "status": "completed"} tags=[]
# CITEseq has many variations of column name patterns
#
# CITEseqは、カラム名のパターンのバリエーションが多い

# + papermill={"duration": 0.020552, "end_time": "2022-08-18T08:31:42.953232", "exception": false, "start_time": "2022-08-18T08:31:42.932680", "status": "completed"} tags=[]
print(len(df_5_train_cite_inputs.columns))

# + papermill={"duration": 0.128532, "end_time": "2022-08-18T08:31:43.092674", "exception": false, "start_time": "2022-08-18T08:31:42.964142", "status": "completed"} tags=[]
cols_replace_digit = []
for col in df_5_train_cite_inputs.columns:
    cols_replace_digit.append(re.sub(r'\d', "@", col))
set(cols_replace_digit)

# + papermill={"duration": 0.024474, "end_time": "2022-08-18T08:31:43.129600", "exception": false, "start_time": "2022-08-18T08:31:43.105126", "status": "completed"} tags=[]
len(set(cols_replace_digit))

# + papermill={"duration": 0.023968, "end_time": "2022-08-18T08:31:43.165974", "exception": false, "start_time": "2022-08-18T08:31:43.142006", "status": "completed"} tags=[]
print(len(df_7_train_cite_targets.columns))

# + papermill={"duration": 0.026329, "end_time": "2022-08-18T08:31:43.205291", "exception": false, "start_time": "2022-08-18T08:31:43.178962", "status": "completed"} tags=[]
cols_replace_digit = []
for col in df_7_train_cite_targets.columns:
    cols_replace_digit.append(re.sub(r'\d', "@", col))
set(cols_replace_digit)

# + papermill={"duration": 0.012184, "end_time": "2022-08-18T08:31:43.230124", "exception": false, "start_time": "2022-08-18T08:31:43.217940", "status": "completed"} tags=[]


# + papermill={"duration": 0.01281, "end_time": "2022-08-18T08:31:43.255675", "exception": false, "start_time": "2022-08-18T08:31:43.242865", "status": "completed"} tags=[]


# + [markdown] papermill={"duration": 0.012726, "end_time": "2022-08-18T08:31:43.281199", "exception": false, "start_time": "2022-08-18T08:31:43.268473", "status": "completed"} tags=[]
# # 
