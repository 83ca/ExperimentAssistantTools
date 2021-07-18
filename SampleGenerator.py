import argparse
import numpy as np
from numpy import matlib
import pandas as pd
from pandas.core.series import Series


class ExperimentSettings(object):
    def __init__(self, csv_path:str) -> None:
        self.df = pd.read_csv(csv_path, index_col=0, header=0)

    # 上限値
    def uppers(self) -> Series:
        return self.df.loc["upper"]

    # 下限値
    def lowers(self) -> Series:
        return self.df.loc["lower"]

    # 合計値設定グループ
    def sum_groups(self) -> Series:
        return self.df.loc["group with a total of desired_sum_of_components"]

    # 合計値設定グループ数
    def number_of_sum_groups(self) -> int:
        return self.sum_groups().astype(int).max()

    # 桁丸め
    def rounds(self) -> Series:
        return self.df.loc["rounding"]

    def is_rounding(self) -> bool:
        return "rounding" in self.df.index


def sample_generator(input_csv_path:str, number_of_generating_samples:int, output_path:str) -> pd.DataFrame:
    # 実験条件設定読み込み
    exp_setting = ExperimentSettings(input_csv_path)
    x_uppers = exp_setting.uppers()
    x_lowers = exp_setting.lowers()
    x_sum_groups = exp_setting.sum_groups()
    max_group_number = exp_setting.number_of_sum_groups()

    # 合計値を指定する特徴量における合計値
    desired_sum_of_components = 1

    # 0～1の一様乱数でサンプル生成
    x_generated = np.random.rand(number_of_generating_samples, len(exp_setting.df.columns))

    # 上限値から下限値までの間に変換
    x_generated = x_generated * (x_uppers.to_numpy() - x_lowers.to_numpy()) + x_lowers.values

    # 合計を desired_sum_of_components にする特徴量がある場合
    if max_group_number != 0:
        # 最大グループ数まで繰り返し
        for group_number in range(1, max_group_number + 1):

            # グループに属する条件のindexを取得
            components_indexes_in_group = np.where(x_sum_groups == group_number)[0]

            # 生成されたサンプル毎の合計値を取得(axis=1: 行)
            sum_of_generated_components_in_group = x_generated[:, components_indexes_in_group].sum(axis=1)

            # 合計値の配列をサンプル数[行]×1[列]に変形
            sum_of_generated_components_reshaped = np.reshape(sum_of_generated_components_in_group, (x_generated.shape[0], 1))
            # グループに属する条件数だけ列方向に複製
            components_converted = matlib.repmat(sum_of_generated_components_reshaped, 1, len(components_indexes_in_group))

            # 要素の最大値がdesired_sum_of_componentsとなるように変換
            x_generated[:, components_indexes_in_group] = x_generated[:, components_indexes_in_group] / components_converted * desired_sum_of_components

            # 範囲を超えるサンプルを消去
            deleting_sample_numbers, _ = np.where(x_generated > x_uppers.values)
            x_generated = np.delete(x_generated, deleting_sample_numbers, axis=0)
            deleting_sample_numbers, _ = np.where(x_generated < x_lowers.values)
            x_generated = np.delete(x_generated, deleting_sample_numbers, axis=0)

    # 数値の丸め込みをする場合
    if exp_setting.is_rounding():
        x_rounds = exp_setting.rounds()
        for i in range(x_generated.shape[1]):
            x_generated[:, i] = np.round(x_generated[:, i], int(x_rounds[i]))

    # 結果の保存
    df_x_generated = pd.DataFrame(x_generated, columns=exp_setting.df.columns)
    df_x_generated.to_csv(output_path)
    print("{} samples generated.".format(x_generated.shape[0]))
    return df_x_generated



if __name__ == '__main__':
    # 引数の読み込み
    parser = argparse.ArgumentParser()

    default_output_folder = "./data/"

    default_input_csv_path = "{}setting_of_generation.csv".format(default_output_folder)
    parser.add_argument("-i", default=default_input_csv_path, help="Input csv file. Default: {}".format(default_input_csv_path))

    default_number_of_samples = 10**6
    parser.add_argument("-n", default=default_number_of_samples, help="Number of samples. Default: {}".format(default_number_of_samples))

    default_output_csv_path = "{}generated_samples.csv".format(default_output_folder)
    parser.add_argument("-o", default=default_output_csv_path, help="Output csv file. Default: {}".format(default_output_csv_path))

    args = parser.parse_args()

    # サンプル生成
    x_generated = sample_generator(args.i, int(args.n), args.o)

