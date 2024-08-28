import numpy as np
import pandas as pd
from scipy.optimize import minimize
import gaps_config as gcfg
from functools import partial

class ConstraintHandler:
    WEIGHT_SUM_1 = {'type': 'eq', 'fun': lambda w: np.sum(w) - 0.99} # Because of cash
    # BOUNDS = [gcfg.ASSET_WEIGHT_CONSTRAINTS[asset] for asset in gcfg.FN_ORDER]
    BOUNDS = list(gcfg.ASSET_WEIGHT_CONSTRAINTS.values())
    
    # GROUP_CONSTRAINTS = [
    #     {'type': 'ineq', 'fun': lambda w, idx=asset_indices,: np.sum(w[idx]) - gcfg.GROUP_WEIGHT_CONSTRAINTS[group][0]}
    #     for group, asset_indices in gcfg.GROUP_TO_ASSETS_NUM.items()
    # ] + [
    #     {'type': 'ineq', 'fun': lambda w, idx=asset_indices: gcfg.GROUP_WEIGHT_CONSTRAINTS[group][1] - np.sum(w[idx])}
    #     for group, asset_indices in gcfg.GROUP_TO_ASSETS_NUM.items()
    # ]

    # The above code has silent bug
    # gcfg.GROUP_WEIGHT_CONSTRAINTS[group][0] 값이 제대로 불러와지지 않음. 근데 또 오류없이 그냥 실행은 됨. 
    # GPT왈, lambda 와 list comprehension을 같이 쓸 때 생기는 Python closure 문제라고 함. 아직도 이해 안됨. 나중에 꼭 다시 알아보기. 어이가 없네.

    # GROUP_CONSTRAINTS = [
    #     {'type': 'ineq', 'fun': lambda w, idx=asset_indices, min_val=gcfg.GROUP_WEIGHT_CONSTRAINTS[group][0] : np.sum(w[idx]) - min_val}
    #     for group, asset_indices in gcfg.GROUP_TO_ASSETS_NUM.items()
    # ] + [
    #     {'type': 'ineq', 'fun': lambda w, idx=asset_indices, max_val=gcfg.GROUP_WEIGHT_CONSTRAINTS[group][1] : max_val - np.sum(w[idx])}
    #     for group, asset_indices in gcfg.GROUP_TO_ASSETS_NUM.items()
    # ]

    # The above code도, min_val은 제대로 불러와지나 max_val은 제대로 불러와지지 않음.
    # 도대체 원인을 모르겠음. 
    # 그래서 아래와 같이 list comprehension을 사용하지 않는 방법으로 수정함.

    GROUP_LOWER_CONSTRAINTS = []
    GROUP_UPPER_CONSTRAINTS = []
    for group, asset_indices in gcfg.GROUP_TO_ASSETS_NUM.items():
        min_val, max_val = gcfg.GROUP_WEIGHT_CONSTRAINTS[group]
        GROUP_LOWER_CONSTRAINTS.append({'type': 'ineq', 'fun': lambda w, idx=asset_indices, min_val=min_val: np.sum(w[idx]) - min_val})
        GROUP_UPPER_CONSTRAINTS.append({'type': 'ineq', 'fun': lambda w, idx=asset_indices, max_val=max_val: max_val - np.sum(w[idx])})
    
    GROUP_CONSTRAINTS = GROUP_LOWER_CONSTRAINTS + GROUP_UPPER_CONSTRAINTS

    # 이제야 정상 작동됨. 확실히 list comprehension과 lambda를 같이 쓸 때 생기는 이해할 수 없는 오류가 있는 것 같음. 

    CONSTRAINTS = [WEIGHT_SUM_1] + GROUP_CONSTRAINTS 
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
            # options={'disp': True}
        )
        return result.x
    
    # TODO: weights_df의 row가 all nan일 때 처리 안됨. 안되는 것이 당연. 다 떨구고 해야..
    @staticmethod
    def apply_constraints(weights_df, objective_function, **objective_params):
        optimized_weights_df = pd.DataFrame(index=weights_df.index, columns=weights_df.columns)
        for idx, row in weights_df.iterrows():
            optimized_weights = ConstraintHandler._apply_constraints_to_row(objective_function, original_weights=row.values, **objective_params) # TODO: Fix objective param not working
            # optimized_weights = ConstraintHandler._apply_constraints_to_row(objective_function, **objective_params)
            optimized_weights_df.loc[idx] = optimized_weights
        
        return optimized_weights_df.astype(float)

class Alpha:
    def __init__(self, author, weights_df, tol=1e-6):
        self.tol = tol
        assert len(author) <= 9, "Author name too long. It should be less than 10 characters"
        self.author = author
        self.weights_df = weights_df

        self._sort()

        self._validate_format()
        self._validate_weights()
        self._validate_constraints()
        
        self.dates = self.weights_df.index
        

    def get_alpha(self):
        return self.weights_df
    
    def _validate_format(self):
        assert all(col in gcfg.FN_ORDER for col in self.weights_df.columns), "Invalid asset names"
        assert self.weights_df.index.is_monotonic_increasing, "Index is not sorted"

    def _validate_weights(self):
        # assert np.isclose(self.weights_df.sum(axis=1), 1, atol=tol).all(), "Weights do not sum to approximately 1"
        assert np.isclose(self.weights_df.sum(axis=1), 0.99, atol=self.tol).all(), "Weights do not sum to approximately 0.99 (= 1 - 0.01 cash)"

    def _validate_constraints(self):
        def is_within_bounds(series, min_val, max_val):
            return np.isclose(series, min_val, atol=self.tol) | np.isclose(series, max_val, atol=self.tol) | ((series > min_val) & (series < max_val))

        for col in self.weights_df.columns:
            min_val, max_val = gcfg.ASSET_WEIGHT_CONSTRAINTS[col]
            if not is_within_bounds(self.weights_df[col], min_val, max_val).all():
                invalid_values = self.weights_df[~is_within_bounds(self.weights_df[col], min_val, max_val)]
                print(f"Failed column '{col}':\n{invalid_values}")
                assert False, f"Individual asset weight constraints violated in column '{col}' / {min_val} ~ {max_val} / with values:\n{invalid_values}"

        for group, assets in gcfg.GROUP_ASSETS.items():
            group_sums = self.weights_df[assets].sum(axis=1)
            min_val, max_val = gcfg.GROUP_WEIGHT_CONSTRAINTS[group]
            if not is_within_bounds(group_sums, min_val, max_val).all():
                invalid_group_values = group_sums[~is_within_bounds(group_sums, min_val, max_val)]
                print(f"Failed group '{group}':\n{invalid_group_values}")
                assert False, f"Group constraint violated for '{group}' / {min_val} ~ {max_val} / with values:\n{invalid_group_values}"

    def _sort(self):
        self.weights_df = self.weights_df[gcfg.FN_ORDER]

    def to_numpy(self):
        return self.weights_df.to_numpy()
    
    def reindex_dates(self, dates):
        # TODO: Fix this
        # assert set(dates).issubset(set(self.dates)), "Reindex dates not a subset of original dates"
        self.weights_df = self.weights_df.reindex(dates)

    def __repr__(self):
        return f"Alpha({self.author})"

# Example objective functions
def obj_mse(weights, original_weights):
    return np.sum( (weights - original_weights) ** 2 )

def obj_max_sharpe(weights, expected_returns, cov_matrix, risk_free_rate):
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return -sharpe_ratio