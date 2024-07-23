import pandas as pd
import numpy as np

import gaps_config as gcfg
import validator

class Backtest:
    def __init__(self, 
                 alpha,
                 returns_df, 
                 bm=None,
                 tx_cost=gcfg.TX_COST, 
                 ):
        
        assert isinstance(alpha, validator.Alpha), "alpha must be an instance of Alpha"

        self.returns_df = returns_df
        self.alpha_df = alpha.get_alpha()
        if bm:
            assert isinstance(bm, validator.Alpha), "Benchmark must be an instance of Alpha"
            self.bm_df = bm.get_alpha()
        
        self.tx_cost = tx_cost

        self._reindex_data()
        self._calculate()

    def _reindex_data(self):
        self.returns_df = self.returns_df.reindex(
            columns=self.alpha_df.columns,
            index=self.alpha_df.index,
        )

    def _calculate(self):
        self.port_return = self.returns_df.mul(self.alpha_df.shift(1)).sum(axis=1)
        self.cum_return = (1 + self.port_return).cumprod() - 1

    def get_port_return(self):
        return self.port_return
    
    def get_port_cum_return(self):
        return self.cum_return
    
    def plot_cum_return(self):
        self.cum_return.plot()
    

    


    
