import fn_config

FN_ORDER = fn_config.ASSET_ALIASES.values()
ASSETS_TO_NUM = {asset: i for i, asset in enumerate(FN_ORDER)}

GROUP_ALIASES = {
    '국내주식': 'kor_stock',
    '해외주식': 'foreign_stock',
    '채권': 'bond',
    '원자재': 'commodity',
    '인버스': 'inverse',
    '통화': 'fx',
    '현금': 'cash'
}

ALIASES_GROUP = {v: k for k, v in GROUP_ALIASES.items()}

GROUP_ASSETS = {
    'kor_stock': ['kodex200', 'kosdaq150'],
    'foreign_stock': ['csi300', 'nikkei', 'euro50', 'sp500'],
    'bond': ['10y', 'midbond', 'hybond'],
    'commodity': ['oil', 'gold'],
    'inverse': ['kodexinv'],
    'fx': ['usd', 'usdinv'],
    'cash': ['shortterm', 'cash'],
}
GROUP_TO_ASSETS_NUM = {group: [ ASSETS_TO_NUM[asset] for asset in assets ] for group, assets in GROUP_ASSETS.items()}

ASSETS_GROUP = {asset: group for group, assets in GROUP_ASSETS.items() for asset in assets}

GROUP_WEIGHT_CONSTRAINTS = {
    'kor_stock': (0.1, 0.4),
    'foreign_stock': (0.1, 0.4),
    'bond': (0.2, 0.6),
    'commodity': (0.05, 0.2),
    'inverse': (0.0, 0.2),
    'fx': (0.0, 0.2),
    'cash': (0.01, 0.5),
}

ASSET_WEIGHT_CONSTRAINTS = {
    # kor_stock
    'kodex200': (0.0, 0.4),
    'kosdaq150': (0.0, 0.2),

    # foreign_stock
    'sp500': (0.0, 0.2),
    'euro50': (0.0, 0.2),
    'nikkei': (0.0, 0.2),
    'csi300': (0.0, 0.2),

    # bond
    '10y': (0.0, 0.5),
    'midbond': (0.0, 0.4),
    'hybond': (0.05, 0.4),

    # commodity
    'gold': (0.0, 0.15),
    'oil': (0.0, 0.15),

    # inverse
    'kodexinv': (0.0, 0.2),

    # fx
    'usd': (0.0, 0.2),
    'usdinv': (0.0, 0.2),

    # cash
    'shortterm': (0.0, 0.5),
    'cash': (0.01, 0.5),
}