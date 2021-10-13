#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 12:32:20 2021

@author: bblanchard006
"""

import pandas as pd
import os

os.chdir('/Users/bblanchard006/Desktop/SMU/QTW/Week 3/Summer 2021')

df = pd.read_csv('diabetic_data.csv')

target_labels = df['readmitted'].unique().tolist()

target_mod = {
        'NO':0,
        '>30':0,
        '<30':1
}

df['readmitted_binary'] = df['readmitted'].map(target_mod)
df['readmitted_binary'].unique()

df.to_csv('diabetic_data_mod.csv', index=False)

# Quick demo in sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ind_vars = [
    'admission_type_id',
    'discharge_disposition_id',
    'admission_source_id',
    'num_lab_procedures',
]

X = df[ind_vars].values
y = df['readmitted_binary'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

clf = LogisticRegression(random_state=0).fit(X_train, y_train)
pred_result = clf.predict_proba(X_test)

df_w_preds = pd.concat([df,pd.DataFrame(pred_result)], axis=1)
df_w_preds.to_csv('sklearn_log_reg.csv', index=False)

