import gaps_config as gcfg

import pandas as pd
import numpy as np

from scipy.optimize import minimize

class Alpha:
    """
    The Alpha class is responsible for validating and managing an alpha weight DataFrame upon initialization. 
    The class ensures that the DataFrame adheres to the required format, weight constraints, and sorting rules.

    Attributes:
        author (str): The author of the alpha instance.
        weights_df (pd.DataFrame): The DataFrame containing asset weights.

    Methods:
        get_alpha(): Returns the validated weights DataFrame.
        _validate_format(): Validates the format of the weights DataFrame.
        _validate_weights(): Ensures that weights in the DataFrame sum to 1 for each row.
        _validate_constraints(): Validates individual asset and group weight constraints.
        _sort(): Sorts the DataFrame columns based on predefined asset aliases.
        to_numpy(): Converts the weights DataFrame to a NumPy array.
    """

    def __init__(self, author, weights_df):
        """
        Initializes the Alpha instance and validates the provided weights DataFrame.

        Parameters:
            author (str): The author of the alpha instance.
            weights_df (pd.DataFrame): The DataFrame containing asset weights.

        Raises:
            AssertionError: If any validation checks fail.
        """
        self.author = author
        self.weights_df = weights_df

        self._validate_format()
        self._validate_weights()
        self._validate_constraints()
        self._sort()

    def get_alpha(self):
        """
        Returns the validated weights DataFrame.

        Returns:
            pd.DataFrame: The validated weights DataFrame.
        """
        return self.weights_df
    
    def _validate_format(self):
        """
        Validates the format of the weights DataFrame.

        Checks:
            - The columns of the DataFrame must match the expected asset aliases.
            - The index of the DataFrame must be sorted in ascending order.

        Raises:
            AssertionError: If the format is invalid.
        """
        assert all(col in gcfg.FN_ORDER for col in self.weights_df.columns), "Invalid asset names"
        assert self.weights_df.index.is_monotonic_increasing, "Index is not sorted"

    def _validate_weights(self, tol=1e-6):
        """
        Ensures that weights in the DataFrame sum to 1 for each row.

        Raises:
            AssertionError: If any row's weights do not sum to 1.
        """
        assert np.isclose(self.weights_df.sum(axis=1), 1, atol=tol).all(), "Weights do not sum to approximately 1"

    def _validate_constraints(self):
        """
        Validates individual asset and group weight constraints.

        Individual Asset Constraints:
            Ensures each asset's weight is within specified constraints.

        Group Constraints:
            Ensures the sum of weights for asset groups is within specified constraints.

        Raises:
            AssertionError: If any constraints are violated.
        """
        # Individual asset constraints
        assert self.weights_df.apply(lambda col: col.between(*gcfg.ASSET_WEIGHT_CONSTRAINTS[col.name]), axis=0).all().all(), "Individual asset weight constraints violated"

        # Group constraints
        for group, assets in gcfg.GROUP_ASSETS.items():
            assert self.weights_df[assets].sum(axis=1).between(*gcfg.GROUP_WEIGHT_CONSTRAINTS[group]).all(), f"Group constraint violated for {group}"

    def _sort(self):
        """
        Sorts the DataFrame columns based on predefined asset aliases.
        """
        self.weights_df = self.weights_df[gcfg.FN_ORDER]

    def to_numpy(self):
        """
        Converts the weights DataFrame to a NumPy array.

        Returns:
            np.ndarray: The weights DataFrame as a NumPy array.
        """
        return self.weights_df.to_numpy()

class ConstraintHandler:
    WEIGHT_SUM_1 = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    BOUNDS = [gcfg.ASSET_WEIGHT_CONSTRAINTS[asset] for asset in gcfg.FN_ORDER]
    GROUP_BOUNDS = [
         {'type': 'ineq', 'fun': lambda w: np.sum(w[asset_indices_in_group]) - gcfg.GROUP_WEIGHT_CONSTRAINTS[group][0]} for group, asset_indices_in_group in gcfg.GROUP_TO_ASSETS_NUM.items()
        ] + [
         {'type': 'ineq', 'fun': lambda w: gcfg.GROUP_WEIGHT_CONSTRAINTS[group][1] - np.sum(w[asset_indices_in_group])} for group, asset_indices_in_group in gcfg.GROUP_TO_ASSETS_NUM.items()
            ]
    
    CONSTRAINTS = [WEIGHT_SUM_1] + BOUNDS + GROUP_BOUNDS

    INITIAL_GUESS = np.array([1/len(gcfg.FN_ORDER)] * len(gcfg.FN_ORDER))

    @staticmethod
    def obj_min_deviation(weights, original_weights):
        return np.sum(np.abs(weights - original_weights))
    
    @staticmethod
    def opt_truncate(weights, original_weights):
        return minimize(
            ConstraintHandler.obj_min_deviation, 
            ConstraintHandler.INITIAL_GUESS, 
            args=(original_weights,), 
            constraints=ConstraintHandler.CONSTRAINTS, 
            bounds=ConstraintHandler.BOUNDS, 
            method='SLSQP', 
            options={'disp': False}
            ).x

