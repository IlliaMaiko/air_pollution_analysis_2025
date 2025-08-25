import random
from typing import Literal

import pandas as pd
from matplotlib import pyplot as plt

from core.files.extract_data import ExtractData
from core.analysis.regression import Regression
from enums import MODEL_NAMES, NAME_INPUT


def simple_linear_regression(series: pd.DataFrame):
    regression = Regression(df=series.dropna())
    regression.plot_ols_against_series()


def ols_regression(df: pd.DataFrame):
    regression = Regression(df=df)
    regression.build_regression()


def call_regression_model(
        df: pd.DataFrame,
        model_name: Literal[
            'decision_tree',
            'random_forest',
            'artificial_neural_network',
            'extreme_gradient_boosting',
            'multivariate_linear_regression'
        ],
        observation_len: int,
        prediction_len: int,
        split_random_state: int = 42,
        random_state: int = 42,
        hidden_layer_sizes: list[int] = (100, ),
):
    regression = Regression(
        df=df,
        observation_len=observation_len,
        prediction_len=prediction_len,
        model_name=model_name,
        split_random_state=split_random_state,
    )
    regression.split()
    input_dict = NAME_INPUT.get(model_name)

    if model_name == 'artificial_neural_network':
        input_dict.update({'random_state': random_state})
        input_dict.update({'hidden_layer_sizes': hidden_layer_sizes})

    regression.fit_model(input_dict=input_dict)
    post_actions(regression=regression)


def optimization_split_random_states(regression: Regression, split_random_state_iteration_len: int, input_dict: dict):
    best_score, best_split_random_state = 0, None
    try:
        for i in range(split_random_state_iteration_len):
            regression.split_random_state = i
            regression.split()
            regression.fit_model(input_dict=input_dict)
            test_score = regression.test_score()

            if best_score > test_score:
                continue

            best_score = test_score
            best_split_random_state = i
    finally:
        print(f'Current score: {best_score}, best_split_random_state: {best_split_random_state};')
        return best_score, best_split_random_state


def optimization_non_nn_split_random_states(
        df: pd.DataFrame,
        model_name: Literal[
            'decision_tree',
            'random_forest',
            'extreme_gradient_boosting',
            'multivariate_linear_regression'
        ],
        observation_len: int,
        prediction_len: int,
        split_random_state_iteration_len: int = 100,
):
    regression = Regression(
        df=df,
        observation_len=observation_len,
        prediction_len=prediction_len,
        model_name=model_name,
    )
    best_score, best_split_random_state = 0, None
    try:
        input_dict = NAME_INPUT.get(model_name)
        best_score, best_split_random_state = optimization_split_random_states(
            regression=regression,
            split_random_state_iteration_len=split_random_state_iteration_len,
            input_dict=input_dict,
        )
    finally:
        print(f'Current score: {best_score}, split_random_state_result: {best_split_random_state};')
        return best_score, best_split_random_state


def optimization_nn_random_states(
        df: pd.DataFrame,
        observation_len: int,
        prediction_len: int,
        random_state_iteration_len: int = 100,
        split_random_state_iteration_len: int = 100,
        hidden_layer_sizes: list[int] = (100, ),
):
    model_name = 'artificial_neural_network'
    regression = Regression(
        df=df,
        observation_len=observation_len,
        prediction_len=prediction_len,
        model_name=model_name,
    )
    best_score, best_random_state, best_split_random_state = 0, None, None
    test_score, test_split_random_state, j = 0, None, None

    try:
        for j in range(random_state_iteration_len):
            input_dict = NAME_INPUT.get(model_name)
            input_dict.update({'random_state': j})
            input_dict.update({'hidden_layer_sizes': hidden_layer_sizes})
            test_score, test_split_random_state = optimization_split_random_states(
                regression=regression,
                split_random_state_iteration_len=split_random_state_iteration_len,
                input_dict=input_dict,
            )

            print(f'Complete: {int(((j + 1) / random_state_iteration_len) * 100)}%; current best score: '
                  f'{best_score}, {best_split_random_state}, {best_random_state};')

            if best_score > test_score:
                continue

            best_score = test_score
            best_split_random_state = test_split_random_state
            best_random_state = j
    except Exception as error:
        print(error)
    finally:
        print(f'Current score: {test_score}, test_split_random_state: {test_split_random_state}, {j};')
        return best_score, best_split_random_state, best_random_state


def optimization_nn_structure(
        df: pd.DataFrame,
        observation_len: int,
        prediction_len: int,
        split_random_state: int = 1,
        random_state: int = 1,
        hidden_layer_base_size: int = 30,
        hidden_layer_base_quantity: int = 6,
        layer_size_iteration_len: int = 20,
        layer_quantity_iteration_len: int = 15,
):
    regression = Regression(
        df=df,
        observation_len=observation_len,
        prediction_len=prediction_len,
        split_random_state=split_random_state,
        model_name='artificial_neural_network',
    )
    best_score, best_hidden_layer_size, best_hidden_layer_quantity = 0, None, None
    test_score, i, j = 0, None, None
    hidden_layer_size, hidden_layer_quantity = None, None
    regression.split()

    try:
        for j in range(layer_size_iteration_len):
            hidden_layer_size = hidden_layer_base_size + j
            for i in range(layer_quantity_iteration_len):
                hidden_layer_quantity = hidden_layer_base_quantity + i
                hidden_layer_sizes = [hidden_layer_size for _ in range(hidden_layer_quantity)]
                input_dict = NAME_INPUT.get('artificial_neural_network')
                input_dict.update({'hidden_layer_sizes': hidden_layer_sizes})
                input_dict.update({'random_state': random_state})
                regression.fit_model(input_dict=input_dict)
                test_score = regression.test_score()
                if best_score > regression.test_score():
                    continue

                best_score = test_score
                best_hidden_layer_size = hidden_layer_size
                best_hidden_layer_quantity = hidden_layer_quantity

            print(f'Complete: {int(((j + 1) / layer_size_iteration_len) * 100)}%; current best score: {best_score}, '
                  f'{best_hidden_layer_size}, {best_hidden_layer_quantity};')
    finally:
        print(f'Current score: {test_score}, {hidden_layer_size}, {hidden_layer_quantity};')
        return best_score, best_hidden_layer_size, best_hidden_layer_quantity


def optimization_nn_random_structure(
        df: pd.DataFrame,
        observation_len: int,
        prediction_len: int,
        split_random_state: int = 1,
        random_state: int = 1,
        hidden_layer_base_size: int = 30,
        hidden_layer_base_quantity: int = 6,
        layer_size_iteration_len: int = 20,
        layer_quantity_iteration_len: int = 15,
):
    regression = Regression(
        df=df,
        observation_len=observation_len,
        prediction_len=prediction_len,
        split_random_state=split_random_state,
        model_name='artificial_neural_network',
    )
    best_score, best_hidden_layer_size, best_hidden_layer_quantity = 0, None, None
    test_score, best_hidden_layer_sizes = 0, None
    hidden_layer_size, hidden_layer_quantity, hidden_layer_sizes = None, None, None
    regression.split()

    try:
        for j in range(layer_size_iteration_len):
            hidden_layer_size = hidden_layer_base_size + j
            for i in range(layer_quantity_iteration_len):
                hidden_layer_quantity = hidden_layer_base_quantity + i
                hidden_layer_sizes = [random.randint(1, hidden_layer_base_size) for _ in range(hidden_layer_quantity)]
                input_dict = NAME_INPUT.get('artificial_neural_network')
                input_dict.update({'hidden_layer_sizes': hidden_layer_sizes})
                input_dict.update({'random_state': random_state})
                regression.fit_model(input_dict=input_dict)
                test_score = regression.test_score()
                if best_score > test_score:
                    continue

                best_score = test_score
                best_hidden_layer_size = hidden_layer_size
                best_hidden_layer_quantity = hidden_layer_quantity
                best_hidden_layer_sizes = hidden_layer_sizes
                print(best_hidden_layer_sizes)

            print(f'Complete: {int(((j + 1) / layer_size_iteration_len) * 100)}%; current best score: {best_score}, '
                  f'{best_hidden_layer_size}, {best_hidden_layer_quantity};')
    finally:
        print(f'Current score: {test_score}, {hidden_layer_size}, {hidden_layer_quantity};')
        print(hidden_layer_sizes)
        return best_score, best_hidden_layer_size, best_hidden_layer_quantity, best_hidden_layer_sizes


def optimization_nn(
        df: pd.DataFrame,
        observation_len: int,
        prediction_len: int,
        random_state_iteration_len: int = 50,
        split_random_state_iteration_len: int = 50,
        hidden_layer_base_size: int = 1,
        hidden_layer_base_quantity: int = 1,
        layer_size_iteration_len: int = 20,
        layer_quantity_iteration_len: int = 15,
):
    model_name = 'artificial_neural_network'
    regression = Regression(
        df=df,
        observation_len=observation_len,
        prediction_len=prediction_len,
        model_name=model_name,
    )
    best_random_state, best_split_random_state = None, None
    best_score, best_hidden_layer_size, best_hidden_layer_quantity = 0, None, None
    test_score, i, j = 0, None, None
    hidden_layer_size, hidden_layer_quantity = None, None

    try:
        for j in range(layer_size_iteration_len):
            hidden_layer_size = hidden_layer_base_size + j
            for i in range(layer_quantity_iteration_len):
                hidden_layer_quantity = hidden_layer_base_quantity + i
                hidden_layer_sizes = [hidden_layer_size for _ in range(hidden_layer_quantity)]
                input_dict = NAME_INPUT.get('artificial_neural_network')
                input_dict.update({'hidden_layer_sizes': hidden_layer_sizes})

                for random_state_index in range(random_state_iteration_len):
                    input_dict.update({'random_state': random_state_index})
                    test_score, test_split_random_state = optimization_split_random_states(
                        regression=regression,
                        split_random_state_iteration_len=split_random_state_iteration_len,
                        input_dict=input_dict,
                    )

                    if best_score > test_score:
                        continue

                    best_score = test_score
                    best_split_random_state = test_split_random_state
                    best_random_state = random_state_index
                    best_hidden_layer_size = hidden_layer_size
                    best_hidden_layer_quantity = hidden_layer_quantity

            print(f'Complete: {int(((j + 1) / layer_size_iteration_len) * 100)}%; current best score: {best_score}, '
                  f'{best_hidden_layer_size}, {best_hidden_layer_quantity};')
    except Exception as error:
        print(error)
    finally:
        print(f'Current score: {test_score}, {hidden_layer_size}, {hidden_layer_quantity};')
        return best_score, best_split_random_state, best_random_state, best_hidden_layer_size, best_hidden_layer_quantity


def post_actions(regression: Regression):
    regression.print_score()
    regression.scatter_plot()


def sandbox(data: pd.DataFrame):
    print(data.describe())
    print(data.corr())
    data.plot()
    plt.show()


def main():
    """
    Possible column values:
    'Амiак', 'Дiоксид азоту', 'Дiоксид сiрки', 'Завислі речовини', 'Оксид азоту', 'Оксид вуглецю', 'Сажа', 'Фенол',
     'Формальдегiд'

    Before the war - '1/27/2020':'12/5/2021'
    After the war started - '4/17/2023':
    Useful query example: .loc['1/27/2020':'12/5/2021', ['Амiак', 'Формальдегiд']]
    """
    data_df = ExtractData()
    data_df.extract_df_from_xls()
    sample_data = data_df.data.loc['4/17/2023':, ['Амiак']]

    result = None
    observation_len = 365
    prediction_len = 1
    hidden_layer_size = 40
    hidden_layer_sizes = [hidden_layer_size for _ in range(12)]
    split_random_state = 64
    random_state = 54
    call_regression_model(
        df=sample_data,
        model_name='artificial_neural_network',
        observation_len=observation_len,
        prediction_len=prediction_len,
        split_random_state=split_random_state,
        random_state=random_state,
        hidden_layer_sizes=hidden_layer_sizes,
    )

    # # Could take long time! (NN random states optimization)
    # result = optimization_nn_random_states(
    #     df=sample_data,
    #     observation_len=observation_len,
    #     prediction_len=prediction_len,
    #     hidden_layer_sizes=hidden_layer_sizes,
    #     split_random_state_iteration_len=50,
    #     random_state_iteration_len=50,
    # )

    # # Could take long time! (NN structure optimization)
    # split_random_state = 26
    # random_state = 26
    # result = optimization_nn_structure(
    #     df=sample_data,
    #     observation_len=observation_len,
    #     prediction_len=prediction_len,
    #     split_random_state=split_random_state,
    #     random_state=random_state,
    #     layer_size_iteration_len=40,
    #     layer_quantity_iteration_len=20,
    #     hidden_layer_base_size=150,
    #     hidden_layer_base_quantity=1,
    # )

    # # Could take long time! (NN random structure optimization)
    # split_random_state = 37
    # random_state = 26
    # result = optimization_nn_random_structure(
    #     df=sample_data,
    #     observation_len=observation_len,
    #     prediction_len=prediction_len,
    #     split_random_state=split_random_state,
    #     random_state=random_state,
    #     layer_size_iteration_len=10,
    #     layer_quantity_iteration_len=1,
    #     hidden_layer_base_size=5000,
    #     hidden_layer_base_quantity=13,
    # )

    # # Could take long time! (NN optimization)
    # result = optimization_nn(
    #     df=sample_data,
    #     observation_len=observation_len,
    #     prediction_len=prediction_len,
    #     split_random_state_iteration_len=20,
    #     random_state_iteration_len=20,
    #     layer_quantity_iteration_len=15,
    #     layer_size_iteration_len=40,
    # )

    print(result)

    # # Could take long time! (non NN random states optimization)
    # print('#################################################')
    # for value in list(data_df.data.columns):
    #     print(f'Data: {value}')
    #     temp_sample_data = data_df.data.loc['4/17/2023':, [value]]
    #     for model_name in MODEL_NAMES:
    #         if model_name == 'artificial_neural_network':
    #             continue
    #
    #         print(f'Model: {model_name}')
    #         result = optimization_non_nn_split_random_states(
    #             df=temp_sample_data,
    #             model_name=model_name,
    #             observation_len=observation_len,
    #             prediction_len=prediction_len,
    #             split_random_state_iteration_len=1000,
    #         )
    #
    #         print(result)
    #         print('#################################################')

    # ols_regression(df=sample_data)
    # sandbox(data=sample_data)


if __name__ == '__main__':
    main()
