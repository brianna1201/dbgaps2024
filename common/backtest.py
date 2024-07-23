import pandas as pd
import numpy as np

import gaps_config as gcfg
import validator

import quantstats as qs

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
        self._setup_quantstats()

    def _reindex_data(self):
        self.returns_df = self.returns_df.reindex(
            columns=self.alpha_df.columns,
            index=self.alpha_df.index,
        )

    def _calculate(self):
        self.port_return = self.returns_df.mul(self.alpha_df.shift(1)).sum(axis=1)
        self.cum_return = (1 + self.port_return).cumprod() - 1
    
    def _setup_quantstats(self):
        self.qs_stats = [stat for stat in dir(qs.stats) if stat[0] != '_']
        self.qs_plots = [plot for plot in dir(qs.plots) if plot[0] != '_']

    def get_port_return(self):
        return self.port_return
    
    def get_port_cum_return(self):
        return self.cum_return
    
    def plot_cum_return(self):
        self.cum_return.plot()

    ## quantstats

    def show_available_stats(self):
        return self.qs_stats
    
    def get_stat(self, stat_name):
        assert stat_name in self.qs_stats, f"Invalid stat name. Available stats are: {self.qs_stats}"
        
        stat_func = getattr(qs.stats, stat_name)

        return stat_func(self.port_return)

    def show_available_plots(self):
        return self.qs_plots

    def get_plot(self, plot_name):
        assert plot_name in self.qs_plots, f"Invalid plot name. Available plots are: {self.qs_plots}"
        
        plot_func = getattr(qs.reports._plots, plot_name)

        return plot_func(self.port_return, show=False)
    

    


    
