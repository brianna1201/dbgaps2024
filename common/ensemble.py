import pandas as pd
import numpy as np

from abc import ABC, abstractmethod

# custom libs
from validator import Alpha, ConstraintHandler, obj_mse
from backtest import Backtest

class Ensemble(ABC):
    def __init__(self, 
                 alphas,
                 returns_df, 
                 bm=None, 
                 ):
        
        self.alphas = alphas
        self.returns_df = returns_df
        if bm:
            assert isinstance(bm, Alpha), "Benchmark must be an instance of Alpha"
            self.bm = bm
        self._reindex_dates()
        self._get_alpha_infos()

    def _reindex_dates(self):
        min_date = min([alpha.dates[0] for alpha in self.alphas])
        max_date = max([alpha.dates[-1] for alpha in self.alphas])

        assert min_date <= max_date, "No subset of dates found among alphas"

        self.subset_dates = self.returns_df.loc[min_date:max_date, :].index

        for alpha in self.alphas:
            alpha.reindex_dates(self.subset_dates)

        self.returns_df = self.returns_df.loc[self.subset_dates, :]
    
    def _get_alpha_infos(self):
        authors = [alpha.author for alpha in self.alphas] # ['Sam', 'Sam', 'John', 'Alice']
        alpha_names = np.empty_like(authors, dtype='S12')
        for author in np.unique(authors):
            author_mask = np.where(authors == author)
            alpha_names[author_mask] = [f"{author}_{i}" for i in range(len(author_mask))]
        
        # self.authors = np.unique(authors) # ['Sam', 'John', 'Alice']
        self.alpha_names = alpha_names # ['Sam_0', 'Sam_1', 'John_0', 'Alice_0']
        
    def get_alpha_infos(self):
        return self.alpha_names

    @abstractmethod
    def mix(self):
        raise NotImplementedError

class EqualWeight(Ensemble):
    def __init__(self, alphas, returns_df, bm=None):
        super().__init__(alphas, returns_df, bm)

    def mix(self):
        ensemble_weights_df = np.mean([alpha.to_numpy() for alpha in self.alphas], axis=0)
        ensemble_weights_df = pd.DataFrame(ensemble_weights_df, index=self.returns_df.index, columns=self.returns_df.columns)
        ensemble_weights_df.dropna(how='all', axis=0, inplace=True)
        ensemble_weights_df = ConstraintHandler.apply_constraints( # TODO: DRY. Put this in the parent class
            weights_df=ensemble_weights_df,
            objective_function=obj_mse,
        )

        return Alpha("Ensemble", ensemble_weights_df)
    
class CustomWeight(Ensemble):
    def __init__(self, alphas, returns_df, bm=None):
        super().__init__(alphas, returns_df, bm)

    def mix(self, alpha_weights):
        assert len(alpha_weights) == len(self.alphas), "Number of weights must match number of alphas"
        assert np.isclose(np.sum(alpha_weights), 1, atol=1e-6), "Weights must sum to 1"

        ensemble_weights_df = np.sum([alpha.to_numpy() * weight for alpha, weight in zip(self.alphas, alpha_weights)], axis=0)
        ensemble_weights_df = pd.DataFrame(ensemble_weights_df, index=self.returns_df.index, columns=self.returns_df.columns)
        ensemble_weights_df.dropna(how='all', axis=0, inplace=True)
        ensemble_weights_df = ConstraintHandler.apply_constraints(
            weights_df=ensemble_weights_df,
            objective_function=obj_mse,
        )

        return Alpha("Ensemble", ensemble_weights_df)

class AdaptiveWeight(Ensemble):
    def __init__(self, alphas, returns_df, bm=None):
        super().__init__(alphas, returns_df, bm)

    def mix(self, performance_window=63):
        assert performance_window < len(self.subset_dates), "Performance window must be smaller than number of dates"

        backtests = [Backtest(alpha, self.returns_df) for alpha in self.alphas]
        alpha_returns = [backtest.get_port_return() for backtest in backtests]

        alpha_returns_df = pd.concat(alpha_returns, axis=1)
        alpha_returns_df.columns = self.alpha_names

        alpha_weights_2d = alpha_returns_df.rolling(window=performance_window).mean().to_numpy() # lookback window will make it NaN for first performance_window days
        
        def normalize_weights(signals_2d, axis=1):
            # min max normalization to ensemble weights
            signals_min = np.nanmin(signals_2d, axis=axis, keepdims=True)
            signals_max = np.nanmax(signals_2d, axis=axis, keepdims=True)
            minmax_2d = (signals_2d - signals_min) / (signals_max - signals_min)
            
            # scale to sum to 1
            minmax_sum = np.nansum(minmax_2d, axis=axis, keepdims=True)
            sum_to_1 = minmax_2d / minmax_sum

            return sum_to_1
        
        alpha_weights_2d = normalize_weights(alpha_weights_2d, axis=1)

        stacked_alphas = np.stack([a.to_numpy() for a in self.alphas], axis=1)
        ensemble_weights_2d = np.einsum('ij, ijk -> ik', alpha_weights_2d, stacked_alphas)

        ensemble_weights_df = pd.DataFrame(ensemble_weights_2d, index=self.returns_df.index, columns=self.returns_df.columns)
        ensemble_weights_df.dropna(how='all', axis=0, inplace=True)

        ensemble_weights_df = ConstraintHandler.apply_constraints(
            weights_df=ensemble_weights_df,
            objective_function=obj_mse,
        )

        return Alpha("Ensemble", ensemble_weights_df)

            



        
