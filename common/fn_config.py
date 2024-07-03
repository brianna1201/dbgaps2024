ASSET_ALIASES = {
    'KODEX 200': 'kodex200',
    'KODEX 200': 'kodex200',

    'TIGER 코스닥150': 'kosdaq150',
    'TIGER 코스닥150': 'kosdaq150',
    'TIGER 코스닥150': 'kosdaq150',
    'TIGER 차이나CSI300': 'csi300',

    'KOSEF 국고채10년': '10y',
    'KBSTAR 중기우량회사채': 'midbond',
    'TIGER 단기선진하이일드(합성 H)': 'hybond',

    'KODEX 골드선물(H)': 'gold',
    'TIGER 원유선물Enhanced(H)': 'oil',

    'KODEX 인버스': 'kodexinv',

    'KOSEF 미국달러선물': 'usd',
    'KOSEF 미국달러선물인버스': 'usdinv',
    
    'KOSEF 단기자금': 'shortterm',
    'KODEX 인버스': 'kodexinv',
}

ALIASES_ASSET = {v: k for k, v in ASSET_ALIASES.items()}

ITEM_ALIASES = {
    '거래대금(원)': 'moneyvolume',
    '수익률(%)': 'return',
    '수정고가(원)': 'adjhigh',
    '수정시가(원)': 'adjopen',
    '수정저가(원)': 'adjlow',
    '수정주가(원)': 'adjclose'
}

ALIASES_ITEM = {v: k for k, v in ITEM_ALIASES.items()}
