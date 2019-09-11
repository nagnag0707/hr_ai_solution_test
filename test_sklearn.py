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


# In[196]:


from matplotlib.colors import ListedColormap

def plot_regions(clf, X, y):
    """ モデルが学習した領域をプロット """
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.3),
                           np.arange(x2_min, x2_max, 0.3))

    Z = clf.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=ListedColormap(('red', 'blue')))

    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    plt.scatter(x=X[y == 0, 0], y=X[y == 0, 1], alpha=0.8, c='red')
    plt.scatter(x=X[y == 1, 0], y=X[y == 1, 1], alpha=0.8, c='blue')


# In[197]:


plot_regions(lr, X_train, Y_train);


# In[ ]:





# In[ ]:




