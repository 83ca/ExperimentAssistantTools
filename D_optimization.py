import argparse
import pandas as pd
import numpy as np

class GeneratedSamples(object):
    def __init__(self, csv_path:str) -> None:
        self.df = pd.read_csv(csv_path, index_col=0, header=0)

    def indexes(self):
        return list(range(self.df.shape[0]))

def D_optimization(input_csv_path:str, number_of_samples:int, number_of_random_searches:int, selected_csv_path:str, remaining_csv_path:str):
    generated_samples = GeneratedSamples(input_csv_path)
    best_d_optimal_value = []
    selected_sample_indexes = []

    # 実験条件の候補のインデックス
    all_indexes = generated_samples.indexes()

    # D最適化基準による最適化
    for i in range(number_of_random_searches):
        # ランダム選択
        new_selected_indexes = np.random.choice(all_indexes, number_of_samples)
        new_selected_samples = generated_samples.df.iloc[new_selected_indexes, :]

        # 標準化
        autoscaled_new_selected_samples = (new_selected_samples - new_selected_samples.mean()) / new_selected_samples.std()

        # D最適基準の計算
        xt_x = np.dot(autoscaled_new_selected_samples.T, autoscaled_new_selected_samples)
        d_optimal_value = np.linalg.det(xt_x)

        # D最適基準が最大値を上回った場合サンプル候補を更新
        if i == 0:
            best_d_optimal_value = d_optimal_value.copy()
            selected_sample_indexes = new_selected_indexes.copy()
            print("D:{0}, n={1}".format(d_optimal_value,i))
        elif best_d_optimal_value < d_optimal_value:
            best_d_optimal_value = d_optimal_value.copy()
            selected_sample_indexes = new_selected_indexes.copy()
            print("D:{0}, n={1}".format(d_optimal_value,i))

    # 選択されたサンプルのインデックスをlist型に変換
    selected_sample_indexes = list(selected_sample_indexes)

    # 選択されたサンプル
    selected_samples = generated_samples.df.iloc[selected_sample_indexes, :]

    # 選択されなかったサンプルのインデックス
    remaining_indexes = np.delete(all_indexes, selected_sample_indexes)
    # 選択されなかったサンプル
    remaining_samples = generated_samples.df.iloc[remaining_indexes, :]

    # 出力
    selected_samples.to_csv(selected_csv_path)
    remaining_samples.to_csv(remaining_csv_path)

    # 相関行列の出力
    print(selected_samples.corr())


if __name__ == '__main__':
    # 引数の読み込み
    parser = argparse.ArgumentParser()

    default_input_csv_path = "./generated_samples.csv"
    parser.add_argument("-i", default=default_input_csv_path, help="Input csv file. Default: {}".format(default_input_csv_path))

    default_number_of_samples = 30
    parser.add_argument("-ns", default=default_number_of_samples, help="Number of sample selecting. Default: {}".format(default_number_of_samples))

    default_number_of_trials = 10**3
    parser.add_argument("-nt", default=default_number_of_trials, help="Number of trials. Default: {}".format(default_number_of_trials))

    default_output_selected_csv_path = "./selected_samples.csv"
    parser.add_argument("-os", default=default_output_selected_csv_path, help="Output csv file(Selected). Default: {}".format(default_output_selected_csv_path))

    default_output_selected_csv_path = "./remaining_samples.csv"
    parser.add_argument("-ore", default=default_output_selected_csv_path, help="Output csv file(Remaining). Default: {}".format(default_output_selected_csv_path))

    args = parser.parse_args()

    # サンプル生成
    D_optimization(args.i, int(args.ns), int(args.nt), args.os, args.ore)
