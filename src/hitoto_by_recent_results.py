"""
実行コマンド
$python3 hitoto_by_recent_results.py ../data/resultsData_20170709.jl 
訓練データ：2015年の試合結果データ
           └入力：n-5〜n-1の試合結果データ（得点数、失点数、勝ち点）
           └出力：n試合目の試合結果（試合結果）
テストデータ：2016年の試合結果データ
           └入力：n-5〜n-1の試合結果データ（得点数、失点数、勝ち点）
           └出力：n試合目の試合結果（試合結果）
予測対象データ：直近の試合結果データ
"""
# coding: utf-8
import os
import sys
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
import json
import codecs
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer
from common.load_results_data import load_results_data, _get_start_results

#(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
# 試合結果ファイルのロード
recent_num = 5
onlyWin = True
# 訓練データのロード
(x_train, t_train) = load_results_data('../data/result2015_J1.json', onlyWin)
(tmp_x_train, tmp_t_train) = load_results_data('../data/result2015_J2.json', onlyWin)
x_train = np.r_[x_train, tmp_x_train]
t_train = np.r_[t_train, tmp_t_train]
# テストデータのロード
(x_test, t_test) = load_results_data('../data/result2016_J1.json', onlyWin)
(tmp_x_test, tmp_t_test) = load_results_data('../data/result2016_J2.json', onlyWin)
x_test = np.r_[x_test, tmp_x_test]
t_test = np.r_[t_test, tmp_t_test]

# Dropuoutの有無、割り合いの設定 ========================
use_dropout = True  # Dropoutなしのときの場合はFalseに
dropout_ratio = 0.2
# ====================================================
# Batch Normalizationの有無
use_batchnorm = True

#network = MultiLayerNetExtend(input_size=3 * recent_num, hidden_size_list=[5, 5, 5, 5, 5, 5],
#                              output_size=18, use_dropout=use_dropout, dropout_ration=dropout_ratio)
network = MultiLayerNetExtend(input_size=3 * recent_num, hidden_size_list=[5, 5, 5, 5, 5, 5],
                              output_size=3, use_dropout=use_dropout, dropout_ration=dropout_ratio, use_batchnorm=use_batchnorm)

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=301, mini_batch_size=100,
                  optimizer='adagrad', optimizer_param={'lr': 0.01}, verbose=True)
trainer.train()
train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

# 推論処理
# 推論対象データの読み込み
args = sys.argv
#試合結果ファイルの読込
recent_results = {}
f = open(args[1])
for line in codecs.open(args[1],'r', 'utf-8'):
    results = json.loads(line)
    recent_results.update(results)

# チーム毎に直近recent_num試合のデータと試合結果のデータを取得
print("-------予測結果--------")
predict_data = np.zeros((0,3 * recent_num),dtype=np.uint8)

for team, results in recent_results.items():
    # recent_num節までのデータを取得
    recent_results = _get_start_results(results)
    recent_results = np.reshape(recent_results,(1,3 * recent_num))
    # 試合結果の予測
    predict_result = trainer.network.predict(recent_results, train_flg=False) 
    #print(str(team) + ":")
    predict_result = np.argmax(predict_result, axis=1)
    #print(predict_result)
    target = results[5]
    print(str(target['date']) + ":" + str(team) + " - " + str(target['team']) + "@"+ str(target['sta']) + "-->" + str(predict_result))

# グラフの描画==========
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
