import numpy as np
import pandas as pd
from scipy.optimize import minimize
import gaps_config as gcfg
from functools import partial

class ConstraintHandler:
    WEIGHT_SUM_1 = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    BOUNDS = [gcfg.ASSET_WEIGHT_CONSTRAINTS[asset] for asset in gcfg.FN_ORDER]
    
    GROUP_BOUNDS = [
        {'type': 'ineq', 'fun': lambda w, idx=asset_indices: np.sum(w[idx]) - gcfg.GROUP_WEIGHT_CONSTRAINTS[group][0]}
        for group, asset_indices in gcfg.GROUP_TO_ASSETS_NUM.items()
    ] + [
        {'type': 'ineq', 'fun': lambda w, idx=asset_indices: gcfg.GROUP_WEIGHT_CONSTRAINTS[group][1] - np.sum(w[idx])}
        for group, asset_indices in gcfg.GROUP_TO_ASSETS_NUM.items()
    ]

    CONSTRAINTS = [WEIGHT_SUM_1] + GROUP_BOUNDS
    INITIAL_GUESS = np.array([1 / len(gcfg.FN_ORDER)] * len(gcfg.FN_ORDER))

    @staticmethod
    def _apply_constraints_to_row(objective_function, **objective_params):
        obj_func = partial(objective_function, **objective_params)
        result = minimize(
            obj_func, 
            ConstraintHandler.INITIAL_GUESS, 
            constraints=ConstraintHandler.CONSTRAINTS, 
            bounds=ConstraintHandler.BOUNDS, 
            method='SLSQP', 
            options={'disp': False}
        )
        return result.x
    
    @staticmethod
    def apply_constraints(weights_df, objective_function, **objective_params):
        optimized_weights_df = pd.DataFrame(index=weights_df.index, columns=weights_df.columns)
        for idx, row in weights_df.iterrows():
            optimized_weights = ConstraintHandler._apply_constraints_to_row(objective_function, original_weights=row.values, **objective_params)
            optimized_weights_df.loc[idx] = optimized_weights
        return optimized_weights_df

class Alpha:
    def __init__(self, author, weights_df):
        self.author = author
        self.weights_df = weights_df
        self._validate_format()
        self._validate_weights()
        self._validate_constraints()
        self._sort()

    def get_alpha(self):
        return self.weights_df
    
    def _validate_format(self):
        assert all(col in gcfg.FN_ORDER for col in self.weights_df.columns), "Invalid asset names"
        assert self.weights_df.index.is_monotonic_increasing, "Index is not sorted"

    def _validate_weights(self, tol=1e-6):
        assert np.isclose(self.weights_df.sum(axis=1), 1, atol=tol).all(), "Weights do not sum to approximately 1"

    def _validate_constraints(self):
        assert self.weights_df.apply(lambda col: col.between(*gcfg.ASSET_WEIGHT_CONSTRAINTS[col.name]), axis=0).all().all(), "Individual asset weight constraints violated"
        for group, assets in gcfg.GROUP_ASSETS.items():
            assert self.weights_df[assets].sum(axis=1).between(*gcfg.GROUP_WEIGHT_CONSTRAINTS[group]).all(), f"Group constraint violated for {group}"

    def _sort(self):
        self.weights_df = self.weights_df[gcfg.FN_ORDER]

    def to_numpy(self):
        return self.weights_df.to_numpy()

# Example objective functions
def obj_mse(weights, original_weights):
    return np.sum((weights - original_weights) ** 2)

def obj_max_sharpe(weights, expected_returns, cov_matrix, risk_free_rate):
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return -sharpe_ratio