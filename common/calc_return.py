from return_calculator import *

def calc_return(data, ratio_df, cost, n_days_after=0):
    return return_calculator(data, ratio_df, cost, n_days_after).backtest_cumulative_return
def calc_return_contribution(data, ratio_df, cost, n_days_after=0):
    return return_calculator(data, ratio_df, cost, n_days_after).daily_ret_cntrbtn
