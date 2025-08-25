import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

data_path = '\\data\\Dataset_air_pollution_Kharkiv_2020_2025_to_csv.csv'

MODEL_NAMES = [
    'artificial_neural_network',
    'decision_tree',
    'extreme_gradient_boosting',
    'multivariate_linear_regression',
    'random_forest',
]

hidden_layer_size = 2000
hidden_layer_sizes = [hidden_layer_size for _ in range(2)]

other_hidden_layer_sizes = [65, 428, 117, 429, 238, 233, 310, 290, 344, 115, 133, 236, 240, ]

NAME_INPUT = {
    'artificial_neural_network': {
        'random_state': 45,
        'max_iter': 2000,
        'tol': 1e-5,
        'hidden_layer_sizes': hidden_layer_size,
        'activation': 'relu',  # {'identity', 'logistic', 'tanh', 'relu'}, default='relu'
        'solver': 'adam',  # {'lbfgs', 'sgd', 'adam'}, default='adam'
        'loss': 'squared_error',  # {'squared_error', 'poisson'}, default='squared_error'
    },
    'decision_tree': {},
    'extreme_gradient_boosting': {'objective': 'reg:squarederror', 'n_estimators': 10},
    'multivariate_linear_regression': {},
    'random_forest': {},
}
