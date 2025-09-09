from typing import Any, Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xg
from matplotlib import pyplot as plt
from sklearn import ensemble, linear_model, neural_network, tree
from sklearn.model_selection import train_test_split


NAME_MODEL = {
    'artificial_neural_network': neural_network.MLPRegressor,
    'decision_tree': tree.DecisionTreeRegressor,
    'extreme_gradient_boosting': xg.XGBRegressor,
    'multivariate_linear_regression': linear_model.LinearRegression,
    'random_forest': ensemble.RandomForestRegressor,
}


class Regression:
    df: pd.DataFrame
    current_model: Any
    observation_len: int
    prediction_len: int
    split_random_state: int
    test_data: list
    test_target: list
    training_data: list
    training_target: list
    model_name: Literal[
        'artificial_neural_network',
        'decision_tree',
        'extreme_gradient_boosting',
        'multivariate_linear_regression',
        'random_forest',
    ]
    smoothed_data: pd.DataFrame

    def __init__(
            self,
            df: pd.DataFrame,
            **kwargs,
    ):
        self.df = df
        self.smoothed_data = None
        if 'observation_len' in kwargs:
            self.observation_len = kwargs.get('observation_len')

        if 'prediction_len' in kwargs:
            self.prediction_len = kwargs.get('prediction_len')

        if 'split_random_state' in kwargs:
            self.split_random_state = kwargs.get('split_random_state')

        if 'model_name' in kwargs:
            self.model_name = kwargs.get('model_name')

    def plot_ols_against_series(self):
        len_series = len(self.df)
        x = np.array(range(len_series))
        y = np.array(self.df)
        slope, intercept = np.polyfit(x=x, y=y, deg=1)

        self.df.plot()

        y_values = slope * x + intercept
        plt.plot(x, y_values)
        plt.show()

    def smooth(self, box_pts: int = 7):
        column_name = list(self.df.columns)[0]
        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(np.array(self.df.reset_index(drop=True).values.T.tolist()[0]), box, mode='same')

        copy_df = self.df.copy()
        copy_df[f'{column_name}_smooth'] = y_smooth
        self.smoothed_data = copy_df.drop(columns=[column_name])

    def plot_smoothed_against_origin(self):
        copy_df = self.df.copy()
        couple = copy_df.join(self.smoothed_data)
        couple.plot()

    def build_regression(self):
        data = self.df.dropna()
        variables = list(data.columns)
        y_series = data.loc[:, variables[0]]
        x_series = data.loc[:, variables[1:]]
        x_with_const = sm.add_constant(x_series)
        model = sm.OLS(y_series, x_with_const).fit()
        print(model.summary())

        y_series.plot()

        x = np.array(range(len(y_series)))
        y = np.array(np.matrix(x_with_const).dot(np.array(model.params)).T)
        plt.plot(x, y)
        plt.show()

    def split(self, smoothed: bool = False):
        self._split_data_for_training(
            data_len=self.observation_len,
            target_len=self.prediction_len,
            split_random_state=self.split_random_state,
            smoothed=smoothed,
        )

    def fit_model(self, input_dict: dict):
        """
        artificial_neural_network:
        https://scikit-learn.org/stable/modules/neural_networks_supervised.html

        decision_tree:
        https://scikit-learn.org/stable/modules/tree.html

        extreme_gradient_boosting:
        https://www.geeksforgeeks.org/machine-learning/xgboost-for-regression/

        multivariate_linear_regression:
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

        random_forest:
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor

        """
        self.current_model = NAME_MODEL.get(self.model_name)(**input_dict)
        self.current_model.fit(self.training_data, self.training_target)

    def scatter_plot(self):
        predicted_target = self.current_model.predict(self.test_data)
        plt.scatter(predicted_target, self.test_target)

        predicted_training_target = self.current_model.predict(self.training_data)
        plt.scatter(predicted_training_target, self.training_target, c='orange')
        plt.show()

    def training_score(self):
        return self.current_model.score(self.training_data, self.training_target)

    def test_score(self):
        return self.current_model.score(self.test_data, self.test_target)

    def print_score(self):
        print(f'Training score: {self.training_score()}')
        print(f'Testing score: {self.test_score()}')

    def _split_data_for_training(self, data_len: int, target_len: int, split_random_state: int, smoothed: bool = False):
        if smoothed:
            if self.smoothed_data is None:
                self.smooth(box_pts=4)

            data_to_split = self.smoothed_data
        else:
            data_to_split = self.df

        data_target_len = data_len + target_len
        series_len = len(data_to_split)
        data_list = list()
        target_list = list()
        for i in range(series_len - data_target_len):
            temp_data = data_to_split.reset_index(drop=True).iloc[i:i + data_len].values.T.tolist()[0]
            data_list.append(temp_data)
            temp_target_raw = data_to_split.reset_index(drop=True).iloc[i + data_len:i + data_target_len]
            temp_target = temp_target_raw.values.T.tolist()[0][0]
            target_list.append(temp_target)

        self.training_data, self.test_data, self.training_target, self.test_target = train_test_split(
            data_list,
            target_list,
            test_size=0.2,
            random_state=split_random_state
        )
