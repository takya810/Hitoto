
# coding: utf-8
"""
JSONの試合結果データを読み込み訓練データに変換
変換前データ：result2016.json
        チーム毎の各節の試合結果が格納されている
変換後データ：numpy配列
        データ：n-5〜n-1節のスタッツ
                └結果（勝分負)
                └得点数
                └失点数
        ラベル：第n節の試合結果
                └試合結果、得点数、失点数から17クラスに分類
"""

import sys
import json
import numpy as np

# 直近の試合数定義
recent_num = 5

"""
recent_num節までの試合結果を返す
"""
def _get_start_results(results):
    # 直近5試合結果格納用変数
    recent_results = np.zeros((0,3))
    for result in results[:recent_num]:
        # 試合結果をnumpy配列に変換
        result_array = _convert_numpy(result)
        # 追加
        recent_results = np.r_[recent_results, result_array]
    # 1次元に変換
    
    recent_results = np.ravel(recent_results)
    recent_results = recent_results.astype(np.uint8)
    return recent_results

"""
試合結果、得点数、失点数をnumpy配列にして返す
"""
def _convert_numpy(result):
    # 初期値の定義
    stats = np.zeros((1,3))
    # 得点数の取得
    goal_for = result['result'].split('-')[0]
    stats[0][1] = goal_for
    # 失点数の取得
    goal_against = result['result'].split('-')[1]
    stats[0][2] = goal_against
    # 試合結果の取得
    if goal_for > goal_against:
        stats[0][0] = 1
    elif goal_for == goal_against:
        stats[0][0] = 0
    else:
        stats[0][0] = 2
        
    return stats

"""
勝敗結果を返す
0:引き分け
1:ホーム勝ち
2:ホーム負け
"""
def _get_match_label(result):
    # 試合結果変数の取得
    label = np.zeros((1,1))
    # 得点数の取得
    goal_for = int(result['result'].split('-')[0])
    # 失点数の取得
    goal_against = int(result['result'].split('-')[1])
    # 勝利の場合（得点数 > 失点数)
    if goal_for > goal_against:
        label[0] = 1
    elif goal_for == goal_against:
        label[0] = 0
    else:
        label[0] = 2

    label = label.astype(np.uint8)
    return label

"""
recent_num+1試合目の結果を返す。（訓練ラベル）
0〜6:勝利
7〜10:引き分け
11〜17:負け
"""
def _get_results_label(result):

    # 試合結果変数の取得
    label = np.zeros((1,1))
    # 得点数の取得
    goal_for = int(result['result'].split('-')[0])
    # 失点数の取得
    goal_against = int(result['result'].split('-')[1])
    # 勝利の場合（得点数 > 失点数)
    if goal_for > goal_against:
        # 1-0
        if goal_for == 1:
            label[0] = 0
        elif goal_for == 2:
            # 2-0
            if goal_against == 0:
                label[0] = 1
            # 2-1
            else:
                label[0] = 2
        elif goal_for >= 3:
            # 3-0
            if goal_against == 0:
                label[0] = 3
            # 3-1
            elif goal_against == 1:
                label[0] = 4
            # 3-2
            elif goal_against == 2:
                label[0] = 5
            # 両チームとも3点以上でホームチーム勝利
            else:
                label[0] = 6
    # 引き分け
    elif goal_for == goal_against:
        # 0-0
        if goal_for == 0:
            label[0] = 7
        # 1-1
        elif goal_for == 1:
            label[0] = 8
        # 2-2
        elif goal_for == 2:
            label[0] = 9
        # 3-3以上の引き分け
        else:
            label[0] = 10
    # ホームチーム負け
    else:
        if goal_for == 0:
            # 0-1
            if goal_against == 1:
                label[0] = 11
            # 0-2
            elif goal_against == 2:
                label[0] = 12
            # 0-3以上の負け
            else:
                label[0] = 13
        elif goal_for == 1:
            # 1-2
            if goal_against == 2:
                label[0] = 14
            # 1-3以上の負け
            else:
                label[0] = 15
        # 2-3以上の負け        
        elif goal_for == 2:
            label[0] = 16
        # 3-3以上の負け
        elif goal_for >= 3:
            label[0] = 17

    label = label.astype(np.uint8)
    return label

"""
one-hot表現に変換
"""
def _change_one_hot_label(X, outputSize):
    T = np.zeros((X.size, outputSize))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
        
    return T

"""
試合結果JSONファイルを読み込み、訓練データに変換して返す。
"""
def load_results_data(result_f, onlyWin):
    #試合結果ファイルの読込
    f = open(result_f)
    season_results = json.load(f)

    # 学習データ格納用変数
    train_data = np.zeros((0,3 * recent_num),dtype=np.uint8)
    train_label = np.zeros((0,1),dtype=np.uint8)

    # チーム毎に直近recent_num試合のデータと試合結果のデータを取得
    for team, results in season_results.items():
        # recent_num節までのデータを取得
        recent_results = _get_start_results(results)
        
        for result in results[recent_num:]:
            # 直近5試合の試合結果の次の試合のラベルを取得
            if onlyWin:
                label = _get_match_label(result)
            else:
                label = _get_results_label(result)
            # 訓練データに追加
            recent_results = np.reshape(recent_results,(1,3 * recent_num))
            train_data = np.r_[train_data, recent_results]
            train_label = np.r_[train_label, label]
            # 最も古いデータを破棄
            recent_results = np.roll(recent_results,-3)
            # 試合結果をnumpy配列に変換
            result_array = _convert_numpy(result)
            # 直近3試合の試合結果に更新
            recent_results[0][3 * recent_num -3] = int(result_array[0][0])
            recent_results[0][3 * recent_num -2] = int(result_array[0][1])
            recent_results[0][3 * recent_num -1] = int(result_array[0][2])
            
    # one-hot表現に変換
    if onlyWin:
        train_label = _change_one_hot_label(train_label,3)
    else:
        train_label = _change_one_hot_label(train_label,18)
    return train_data, train_label
