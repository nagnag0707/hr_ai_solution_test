#!/usr/bin/env python
# coding: utf-8

# In[750]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

'''

このプログラムは機械学習用のテストデータを方向付けしランダムに生成します。
1件のデータ形式は下記

連番, 名前, 名前（カナ), 年齢, 性別, 勤務時間, 在職

--------------------------
def edit_dataframe(df)

この関数はテストデータの勤務時間に対して正規分布を用いた
任意のデータを生成し挿入します。生成データは下記パラメータに準拠します。

MEAN : 平均値
STD  : 標準偏差
--------------------------

def retirement(df)

この関数は勤務時間に比例してランダムで在職の真偽値を変更させます。
退職判定は下記の計算式を元に判定を行います。

((勤務時間 - 平均勤務時間) / 標準偏差 * 10 + 50) / 任意のパラメータ
 =  偏差値 / 任意の係数です。
 
1 / (偏差値 / 任意の係数)の確率で退職判定となります。

def save_csv(df)
定数 SAVE_PLACEにcsv形式で保存します。


'''


#　定数
CSV_PLACE = "./5000_Datas.csv"
SAVE_PLACE = "./Data"
FILE_NAME = "Train_Data.csv" # "Test_Data.csv"
MEAN = 250  # 生成するテストデータの平均値
STD  = 48   # 生成するテストデータの標準偏差
RET  = 10   # 偏差値 / RET 分の 1の確率で退職させる
DEBUG = True # デバッグ用にデータを表示させるか


# In[751]:


df = pd.read_csv(CSV_PLACE)
print("読み込みデータ件数は", len(df), "件です。", sep="")


# In[752]:



df = df.assign(在職=1)
df.head(3)


# In[753]:


def edit_dataframe(df):
    # データ件数のカウントを行う
    count_df = df.count()['連番']
    
    # ランダムデータの生成 
    random_data = np.random.normal(MEAN, STD, count_df)
    
    for index, row in df.iterrows():
        df.iat[index, 5] = random_data[index]
    
    print("平均値 :", MEAN, sep="")
    print("標準偏差:", STD, sep="")
    print(count_df, "件のデータを生成しました。", sep="")
    


# In[754]:


edit_dataframe(df)


# In[763]:


def retirement(df):
    mean = df['勤務時間'].mean()
    std  = df['勤務時間'].std()
    
    age_mean = df['年齢'].mean()
    
    for index, row in df.iterrows():
        ret = RET
        # 勤務時間の取得
        worktime = df.iat[index, 5]
        
        # 年齢の取得
        age = df.iat[index, 4]
        # 勤務時間が平均以上なら退職確率を上げる
        if worktime > mean:
            # 若い人は早めに見切り付けがち
            ret = ret / (5 * (age_mean / age))
                
        
        # 退職確率の計算
        probability = ((worktime - mean) / std * 10 + 50) / ret / 100
        # 在職の確率の計算
        tenure = 1 - probability
        
        # 退職判定
        df.iat[index, 6] = np.random.choice(2, p=[probability, tenure])
        
        if DEBUG == True:
            print("------------------")
            print(row["氏名"], "さんの退職確率は:",'{:.1f}'.format(probability*100), "%です。", sep="")
             
            if df.iat[index, 6] == 0:
                print('退職しました...')


# In[764]:


retirement(df)


# In[757]:


df.describe()


# In[758]:


fig, ax = plt.subplots()

ax.hist(df['年齢'], bins=30)


# In[759]:


fig, ax = plt.subplots()

# 年齢 x , 勤務時間 を yとした散布図の描画

ax.scatter(df[df['在職'] == 1]['年齢'], df[df['在職'] == 1]['勤務時間'], s=10, c='b')
ax.scatter(df[df['在職'] == 0]['年齢'], df[df['在職'] == 0]['勤務時間'], s=10, c='r')
plt.title('Scatter')


# In[762]:


# 勤務時間のヒストグラム
fig, ax = plt.subplots()

ax.hist(df[df['在職'] == 1]['勤務時間'] , bins=40, color="blue")
ax.hist(df[df['在職'] == 0]['勤務時間'] , bins=40, color="red")
plt.title('histgram')


# In[761]:


def save_csv(df):
    # SAVE_PLACEにCSV出力を行う
    df.to_csv(SAVE_PLACE + "/" + FILE_NAME , sep=",")
    
save_csv(df)


# In[476]:





# In[ ]:





# In[ ]:





# In[ ]:




