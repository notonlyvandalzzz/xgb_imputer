from xgboost import XGBRegressor,XGBClassifier
import pandas as pd
import numpy as np
# the code is dirty and somewhat inneficient
# but it works

def fill_feats(tr_org, ts_org, disc_feats, cat_feats, ok_feats, xgb_lr=0.1):
    combined = pd.concat([ts_org.drop('id', axis=1), tr_org.drop(['id', 'song_popularity'], axis=1)], axis=0)
    tr_ed = tr_org.copy()
    ts_ed = ts_org.copy()
    for tgt_col in disc_feats:
        tmp_df = combined[combined.columns[combined.columns.isin(ok_feats + [tgt_col])]]
        tmp_train = tmp_df[~tmp_df[tgt_col].isnull()]
        tmp_xgb = XGBRegressor(learning_rate=xgb_lr)
        tmp_x = tmp_train[tmp_train.columns[tmp_train.columns.isin(ok_feats)]]
        tmp_y = tmp_train[tmp_train.columns[~tmp_train.columns.isin(ok_feats)]]
        tmp_xgb.fit(tmp_x, tmp_y)
        tr_ed[tgt_col + '_imp'] = tr_ed[ok_feats].apply(lambda x: tmp_xgb.predict(np.array(x).reshape((1,-1)))[0], axis=1)
        tr_ed[target_col] = tr_ed[[tgt_col, tgt_col + '_imp']].apply(lambda row: row[tgt_col + '_imp'] if pd.isnull(row[tgt_col]) else row[tgt_col], axis=1)
        ts_ed[tgt_col + '_imp'] = ts_ed[ok_feats].apply(lambda x: tmp_xgb.predict(np.array(x).reshape((1,-1)))[0], axis=1)
        ts_ed[target_col] = ts_ed[[tgt_col, tgt_col + '_imp']].apply(lambda row: row[tgt_col + '_imp'] if pd.isnull(row[tgt_col]) else row[tgt_col], axis=1)
    for tgt_col in cat_feats:
        tmp_df = combined[combined.columns[combined.columns.isin(ok_feats + [tgt_col])]]
        tmp_train = tmp_df[~tmp_df[tgt_col].isnull()]
        tmp_xgb = XGBClassifier(learning_rate=xgb_lr)
        tmp_x = tmp_train[tmp_train.columns[tmp_train.columns.isin(ok_feats)]]
        tmp_y = tmp_train[tmp_train.columns[~tmp_train.columns.isin(ok_feats)]]
        tmp_xgb.fit(tmp_x, tmp_y)
        tr_ed[tgt_col + '_imp'] = tr_ed[ok_feats].apply(lambda x: tmp_xgb.predict(np.array(x).reshape((1,-1)))[0], axis=1)
        tr_ed[target_col] = tr_ed[[tgt_col, tgt_col + '_imp']].apply(lambda row: row[tgt_col + '_imp'] if pd.isnull(row[tgt_col]) else row[tgt_col], axis=1)
        ts_ed[tgt_col + '_imp'] = ts_ed[ok_feats].apply(lambda x: tmp_xgb.predict(np.array(x).reshape((1,-1)))[0], axis=1)
        ts_ed[target_col] = ts_ed[[tgt_col, tgt_col + '_imp']].apply(lambda row: row[tgt_col + '_imp'] if pd.isnull(row[tgt_col]) else row[tgt_col], axis=1)
    return tr_ed, ts_ed
