#!/usr/bin/env python
# coding: utf-8

# In[193]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
import pickle
'''
このプログラムはCSVファイルから読み込んだデータから機械学習を行い、結果を出力する。

分析手法：ロジスティック回帰分析
テストデータ：CreateTestDataによって生成されたCSVファイル
目的変数：在職(1 or 0)
説明変数：勤務時間と年齢

'''

#　定数
TR_CSV_PLACE = "./Data/Train_Data.csv"
RS_CSV_PLACE = "./Data/Test_Data.csv"
SAVE_MODEL = "./Data/LogisticModel.sav"


# In[194]:


# LogisticRegressionクラスのインスタンスを作成
lreg = LogisticRegression()

tr_df = pd.read_csv(TR_CSV_PLACE)
test_df = pd.read_csv(RS_CSV_PLACE)

# 説明変数の読み込み
X_train = tr_df[['年齢', '勤務時間']].values
# 目的変数の読み込み
Y_train = tr_df['在職'].values

# ロジスティック回帰モデルの作成

try:
    lr = pickle.load(open(SAVE_MODEL, 'rb'))
except Exception as e:
    print('Error!')
    lr = LogisticRegression(C=1000, random_state=0)
    
# 学習させる
lr.fit(X_train, Y_train)

# 学習モデルの保存
pickle.dump(lr, open(SAVE_MODEL, 'wb'))


# In[195]:


from sklearn.metrics import accuracy_score, precision_score, recall_score

# テストデータの読み込み
X_test = test_df[['年齢', '勤務時間']].values
Y_test = test_df['在職'].values

# 作成したモデルを元にした予測の実行
predict = lr.predict(X_test)

# 結果の出力
#print(accuracy_score(Y_test, predict), precision_score(Y_test, predict), recall_score(Y_test, predict))

print("正解率(Accuracy):", '{:.2f}'.format(accuracy_score(Y_test, predict)*100),"%",  sep="")
print("適合率(Precsion):", '{:.2f}'.format(precision_score(Y_test, predict)*100),"%",  sep="")
print("再現率（Recall）:", '{:.2f}'.format(recall_score(Y_test, predict)*100),"%",  sep="")





