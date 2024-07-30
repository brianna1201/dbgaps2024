from pathlib import Path
import pandas as pd

import fn_config as fncfg

CWD = Path.cwd()
WORKSPACE_PATH = CWD.parent
COMMON_PATH = WORKSPACE_PATH / 'common'
DATA_PATH = WORKSPACE_PATH / 'data'

class FnDataLoader:
    def __init__(self, filename, use_alias=True):
        self.dataguide_df = pd.read_excel(DATA_PATH / filename, skiprows=8, header=[0, 1, 2, 3, 4, 5])
        
        self.to_multiindex(use_alias=use_alias)

    def to_multiindex(self, use_alias):
        if use_alias:
            self.dataguide_df.columns = pd.MultiIndex.from_tuples([self.map_alias(symbol_name, item_name) for symbol, symbol_name, kind, item, item_name, frequency in self.dataguide_df.columns])
        else:
            self.dataguide_df.columns = pd.MultiIndex.from_tuples([(symbol_name, item_name) for symbol, symbol_name, kind, item, item_name, frequency in self.dataguide_df.columns])

        self.dataguide_df.set_index(('Symbol Name', 'Item Name'), inplace=True)
        self.dataguide_df.index.name = 'Date'
        self.dataguide_df.columns = self.dataguide_df.columns.remove_unused_levels() # .drop() does not remove multiindex levels
    
    def map_alias(self, asset_name, item_name):
        asset_alias = fncfg.ASSET_ALIASES.get(asset_name, asset_name)
        item_alias = fncfg.ITEM_ALIASES.get(item_name, item_name)

        return asset_alias, item_alias

    def get_universe(self):
        return self.dataguide_df.columns.levels[0]
    
    def get_datafields(self):
        return self.dataguide_df.columns.levels[1]

    def get_data(self, datafield, dropna=True):
        if dropna:
            return self.dataguide_df.xs(datafield, level=1, axis=1).dropna()
        else:
            return self.dataguide_df.xs(datafield, level=1, axis=1)

    def __repr__(self):
        return f"FnDataLoader(filename='{self.filename}')"

    def __str__(self):
        return f"FnDataLoader with file '{self.filename}' containing {self.dataguide_df.shape[0]} rows and {self.dataguide_df.shape[1]} columns"