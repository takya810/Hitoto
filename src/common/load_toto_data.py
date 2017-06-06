# coding: utf-8

import numpy as np

# データファイル情報
key_file = {
    'train_data_file_path':'../data/train_data/team_specific_data/',
    'test_data_file_path':'../data/test_data/team_specific_data/',
    'team_list':'teamList.dat'
}

# トトデータを読み込む
def load_toto_data():
    toto_data = {
        'train_data':np.zeros((0,3)),
        'train_label':np.zeros((0,1)),
        'test_data':np.zeros((0,3)),
        'test_label':np.zeros((0,1))
    }

    # 対象チームリストの読み込み
    #test_file_path = key_file('train_data_file_path')  + key_file('team_list');
    
    # データ読み込み
    toto_data['train_data'],toto_data['train_label'] = load_data(path=key_file['train_data_file_path'], name=key_file['team_list'])
    # 正規化
    toto_data['train_data'] = toto_data['train_data'].astype(np.float32)
    toto_data['train_data'] /= 100.0
        
    #toto_data['test_data'],toto_data['test_label'] = load_data(test_file_path,file_name)
    
    # one-hot表現に変換
    toto_data['train_label'] = _change_one_hot_label(toto_data['train_label'])
    
    return (toto_data['train_data'],toto_data['train_label'])
    #return (toto_data['train_data'],toto_data['train_label']),(toto_data['test_data'],toto_data['test_label'])

def load_data(path, name):
    file_name = path + name
    
    file = open(file_name)
    team_list = file.read().split('\n')
    
    data = np.zeros((0,3))
    label = np.zeros((0,1),dtype=np.uint8)
    
    for line in team_list:

        if(line != ""):
            file_name = path + line + ".dat"
            tmp_data = np.genfromtxt(file_name, delimiter=" ", usecols=(2,3,4))
            tmp_label = np.genfromtxt(file_name, delimiter=" ", usecols=(1), dtype=np.uint8)

            if(tmp_data.shape == (3,)):
                tmp_data = np.reshape(tmp_data,(1,3))
            tmp_label = np.reshape(tmp_label,(-1,1))    
            
            # データの連結
            data = np.r_[data,tmp_data]
            label= np.r_[label,tmp_label]
    
    return data,label


def _change_one_hot_label(X):
    
    T = np.zeros((X.size, 3))
    for idx, row in enumerate(T):
        #0,1,2以外の場合0を設定する
        if(X[idx] > 2 or X[idx] < 0): 
            row[0] = 1
        else:
            row[X[idx]] = 1
    return T


def load_predict_data(file_name):
    
    file = open(file_name)
    team_list = file.read().split('\n')
    
    data = np.zeros((0,3))

    tmp_data = np.genfromtxt(file_name, delimiter=" ", usecols=(6,7,8))
    
    if(tmp_data.shape == (3,)):
        tmp_data = np.reshape(tmp_data,(1,3))
        
    # データの連結
    data = np.r_[data,tmp_data]

    #正規化
    data = data.astype(np.float32)
    data /= 100.0

    return data
