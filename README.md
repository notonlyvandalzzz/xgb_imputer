# xgb_imputer v.0.1
Simple XGB imputer for tabular data
#### Created as part of data preprocessing during Song Popularity prediction competition at Kaggle  
# How to use:
```python
from xgbimputer import fill_feats
# all code below related to song competition

miss_cat = ['key'] # categorical columns
miss_cont = ['song_duration_ms', 'acousticness', # continious columns
    'danceability', 'energy', 'instrumentalness', 
    'liveness', 'loudness']
nomiss_feats = list(test_df.columns[~test_df.columns.isin(miss_cat + miss_disc + 'id' + 'song_popularity')])

train_imp, test_imp = fill_feats(train_df, test_df, miss_cont, miss_cat, nomiss_feats)
```
# Some notice:
It works via threatening non-NA columns and target column as feats, and one column with NAs as target.  
So amount of NAs should be reasonable, to allow model to find dependencies in non-missed rows.  
Also the more non-NA cols original data has the better inner imputer model gonna be built.  
# ToDo:
1. Code refactoring 
2. More flexibility to input data format
3. Basic hyperparameter autotuning
