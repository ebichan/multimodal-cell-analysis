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

# + _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" papermill={"duration": 0.021858, "end_time": "2022-08-16T11:34:32.124940", "exception": false, "start_time": "2022-08-16T11:34:32.103082", "status": "completed"} tags=[]
import pandas as pd 

# + papermill={"duration": 16.426905, "end_time": "2022-08-16T11:34:48.554086", "exception": false, "start_time": "2022-08-16T11:34:32.127181", "status": "completed"} tags=[]
sample_submission = pd.read_csv('../input/open-problems-multimodal/sample_submission.csv')
sample_submission.head()

# + papermill={"duration": 85.493816, "end_time": "2022-08-16T11:36:14.050209", "exception": false, "start_time": "2022-08-16T11:34:48.556393", "status": "completed"} tags=[]
sample_submission.to_csv('submission.csv', index = 0)

# + papermill={"duration": 0.001869, "end_time": "2022-08-16T11:36:14.054537", "exception": false, "start_time": "2022-08-16T11:36:14.052668", "status": "completed"} tags=[]

