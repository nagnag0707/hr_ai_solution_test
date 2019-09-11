# hr_ai_solution_test
自動生成された個人情報データをもとに、機械学習によるロジスティック回帰分析を用いた予想を行います。
JupyterNotebook環境での動作を想定しており、予め個人情報データ(csvファイル)を生成する必要があります。

-- ファイル構成 と ファイル別概要--
CreateTestDatas.py : ベースとなるCSVに意図的なデータの偏りを付加する。
test_sklearn.py    : 付加されたCSVファイルから学習と予測を行う。

-- 事前準備 --
個人情報データは下記URLより生成可能です。
https://hogehoge.tk/personal/generator/

連番,氏名,氏名（カタカナ）,性別,年齢,乱数(勤務時間として想定される値を選択)
のCSVファイルを作成し、1行目の"乱数"を"勤務時間"に変更してください。

-- 使い方 --

JupyterNotebook環境にプログラムを配置し設定値を各自変更の上、実行してください。

【CreateTestDatas.py】
トレーニング用データとテスト用データを生成する必要があります。
テスト用データ : Test_Data.csv
トレーニング用データ : Train_Data.csv

で作成するとスムーズにtest_slearnで実行可能です。

設定値の詳細は以下
  CSV_PLACE = "./任意の場所.csv"
  SAVE_PLACE = "./Data"
  FILE_NAME = "Train_Data.csv" # "Test_Data.csv"
  MEAN = 250  # 生成するテストデータの平均値
  STD  = 48   # 生成するテストデータの標準偏差
  RET  = 10   # 偏差値 / RET 分の 1の確率で退職させる
  DEBUG = True # デバッグ用にデータを表示させるか

【test_sklearn.py】

ロジスティック回帰分析を用いて、分析を行います。
出力される結果は
 正解率(Accuracy)
 適合率(Precsion)
 再現率（Recall）
になります。プロットが必要な方はソースコードを各自追加して下さい。

設定値の詳細は以下
TR_CSV_PLACE = "./Data/Train_Data.csv"　# トレーニング用データのファイル格納場所
RS_CSV_PLACE = "./Data/Test_Data.csv"   # テスト用データのファイル格納場所
SAVE_MODEL = "./Data/LogisticModel.sav" # 学習モデルの保存先
