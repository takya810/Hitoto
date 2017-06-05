# coding: utf-8
"""
訓練データ（教師データ含む):train_data/team_specific_data
テストデータ：無し
コマンドライ引数に推論対象ファイルを指定
引数のデータ形式../data/orig_toto_data/*.txtと同じ

"""

import sys, os
sys.path.append(os.pardir)

import numpy as np
from common.load_toto_data import load_toto_data
from common.load_toto_data import load_predict_data
from two_layer_net import TwoLayerNet

# データの読み込み
#(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
(x_train, t_train) = load_toto_data()

#print(x_train)
#print(t_train)

network = TwoLayerNet(input_size=3, hidden_size=2, output_size=3)

iters_num = 1000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = int(max(train_size / batch_size, 1))

train_acc = network.accuracy(x_train, t_train)
print(str(-1) + ":" + str(train_acc))

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 勾配
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    # 更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        #test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        #test_acc_list.append(test_acc)
        print(str(i) + ":" + str(train_acc))
        #print(train_acc, test_acc)
        
# 推論処理
# 推論対象データファイルの読み込み
args = sys.argv
predict_data = load_predict_data(file_name=args[1])
print("predict_data")
print(predict_data)
# 推論
y = network.predict(predict_data)
# 推論結果出力
print("推論結果")
print(y)
