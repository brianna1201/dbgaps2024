import numpy as np
import pandas as pd
import os
import gzip, pickle
import matplotlib.pyplot as plt
import seaborn as sns
import quantstats as qs
from scipy.optimize import minimize
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def sort_item(data):

    """
    data를 읽고 item별로 정리
    """

    data.columns = data.columns.map(lambda x: x.split(".")[0])
    sort_data = data.T.reset_index(drop=False).set_index("Item Name").T

    new_data = {}
    for item_name in set(sort_data.columns):

        item_data = sort_data[item_name]
        item_data = item_data.T.set_index("index").T
        item_data.index.name = "date"
        item_data.columns.name = None

        if item_name != "시장구분":
            item_data = item_data.apply(pd.to_numeric)

        if "순매수대금(개인)" in item_name:
            item_name = "개인순매수"
        elif "순매수대금(기관계)" in item_name:
            item_name = "기관순매수"
        elif "순매수대금(등록외국인)" in item_name:
            item_name = "등록외국인순매수"
        elif "순매수대금(외국인계)" in item_name:
            item_name = "외인순매수"

        try:
            concat_data = new_data[item_name]
            concat_data = pd.concat([concat_data, item_data], axis=1)
            new_data.update({item_name: concat_data})
        except:
            new_data[item_name] = item_data
    return new_data
def get_excel_data(date, file_name):
    """ raw 파일 읽기

    Parameters
    ----------------------------
    date : 데이터를 가져온 날짜
    file_name : raw 파일명
    """

    original_path = os.getcwd()
    cnt_ = 0
    print(f'현재경로: {os.getcwd()}')
    while 'data' not in os.listdir(os.getcwd()):
        os.chdir('..')
        print(f'기준경로 변경 ==> {os.getcwd()}')
        if cnt_ > 100:
            print('경로설정확인요망')
            raise FileExistsError
    os.chdir('./data')
    print(f'경로변경완료: {os.getcwd()}')

    if not os.path.exists(f'./data/{date}/items'):
        print('##################### 데이터 정리하기 선택')
        file_path = os.path.join(os.path.join(os.getcwd(), f"{date}"),file_name)            
        data = pd.read_excel(file_path,
                        index_col=0,
                        header=8,
                        thousands=",",
                        skiprows=[9, 10, 11, 13],)
        new_data = sort_item(data)
        new_data = {key.split('(')[0]: value for key, value in new_data.items()}
        
        os.makedirs(f'./{date}/items', exist_ok=True)

        with pd.ExcelWriter(f'./{date}/items/new_{file_name}', engine='xlsxwriter') as writer:
            for i in new_data.keys():
                tmp = new_data[i]
                tmp.to_excel(writer, sheet_name=f'{i}', index=True)
            print("new_data 정리하기 완료")
    else:
        print('##################### 데이터 읽어오기 선택')
        file_path = f'./{date}/items/new_{file_name}'
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names

        new_data = {}
        for sheet in sheet_names:
            new_data[sheet] = pd.read_excel(file_path, sheet_name=sheet,index_col=0)
        print("new_data 읽어오기 완료")
    return new_data
# 엑셀 스프레드 시트에 자동 업데이트
def spreadsheet_update(df):
    # 엑셀 스프레드 시트에 업데이트
    scope = [
    'https://spreadsheets.google.com/feeds',
    'https://www.googleapis.com/auth/drive']

    json_file_name = '../bobae/dbgaps2024-35034d8d429c.json'
    credentials = ServiceAccountCredentials.from_json_keyfile_name(json_file_name, scope)
    gc = gspread.authorize(credentials)
    spreadsheet_url = 'https://docs.google.com/spreadsheets/d/1Uh5d2_jeajcxRyJxyb8QitqOTj8BhaIambfSH_V_xZQ/edit?usp=sharing'
    doc = gc.open_by_url(spreadsheet_url)
    worksheet = doc.worksheet('bobae')
    values = worksheet.get_all_values()
    # spreadsheet의 target탭의 값이 리밸런싱 값과 달라지면 업데이트
    if values != [df.columns.values.tolist()] + df.values.tolist():
        worksheet.update([df.columns.values.tolist()] + df.values.tolist())
        print('Target Raitio가 업데이트 되었습니다.')
    else:
        pass
    return worksheet

class Risk_Parity:
    """ Risk Parity Strategy

    Parameters
    ----------------------------
    price : 수정주가 데이터
    param : 공분산 주기
    """

    def __init__(self, price,param=12):

        # # 연율화 패러미터
        self.param = param

        # 일별 수익률
        self.rets = price.pct_change().dropna()

        # 공분산행렬
        cov = self.rets.rolling(self.param).cov().dropna() * self.param
        # cov = self.rets.cov().dropna()
        self.cov = cov.values.reshape(int(cov.shape[0]/cov.shape[1]), cov.shape[1], cov.shape[1])

        # 거래비용
        self.cost = 0.00015

    # RP
    def rp(self, cov):
        noa = cov.shape[1]
        init_guess = np.repeat(1/noa, noa)
        bounds = [
            (0.0, 1.0),    # A130730
            (0.0, 0.2),    # A138230
            (0.0, 0.2),    # A139660
            (0.0, 0.2),    # A114800
            (0.0, 0.15),   # A130680
            (0.0, 0.15),   # A132030
            (0.05, 0.40),  # A182490
            (0.0, 0.40),   # A136340
            (0.0, 0.50),   # A148070
            (0.0, 0.20),   # A192090
            (0.0, 0.20),   # A238720
            (0.0, 0.20),   # A195930
            (0.0, 0.20),   # A143850
            (0.0, 0.20),   # A232080
            (0.0, 0.40),   # A069500
        ]
        
        weights_sum_to_1 = {'type': 'eq',
                    'fun': lambda weights: np.sum(weights) - 1}
        sum_constraints = [
            {'type': 'ineq', 'fun': lambda weights: np.sum(weights[[13, 14]])- 0.10 },   # A232080 + A069500 >= 0.10
            {'type': 'ineq', 'fun': lambda weights: 0.40 - np.sum(weights[[13, 14]])},  # A232080 + A069500 <= 0.40
            {'type': 'ineq', 'fun': lambda weights: np.sum(weights[[9, 10, 11, 12]]) - 0.10},   # A192090+A238720+A195930+A143850 >= 0.10
            {'type': 'ineq', 'fun': lambda weights: 0.40 - np.sum(weights[[9, 10, 11, 12]])},  # A192090+A238720+A195930+A143850 <= 0.40
            {'type': 'ineq', 'fun': lambda weights: np.sum(weights[[6, 7, 8]]) - 0.20},   # A182490+A136340+A148070 >= 0.20
            {'type': 'ineq', 'fun': lambda weights: 0.60 - np.sum(weights[[6, 7, 8]])},  # A182490+A136340+A148070 <= 0.60
            {'type': 'ineq', 'fun': lambda weights: np.sum(weights[[4, 5]]) - 0.05},   # A130680+A132030 >= 0.05
            {'type': 'ineq', 'fun': lambda weights: 0.20 - np.sum(weights[[4, 5]])},  # A130680+A132030 <= 0.20
            {'type': 'ineq', 'fun': lambda weights: np.sum(weights[[0]]) - 0.01},   # A130730+Cash >= 0.01
            {'type': 'ineq', 'fun': lambda weights: 0.50 - np.sum(weights[[0]])},  # A130730+Cash <= 0.50
            {'type': 'ineq', 'fun': lambda weights: np.sum(weights[[1, 2]])},   # A138230+A139660 >= 0
            {'type': 'ineq', 'fun': lambda weights: 0.20 - np.sum(weights[[1, 2]])},  # A138230+A139660 <= 0.20
            {'type': 'ineq', 'fun': lambda weights: weights[3]},   # A114800 >= 0
            {'type': 'ineq', 'fun': lambda weights: 0.20 - weights[3]}  # A114800 <= 0.20
        ]
        constraints = [weights_sum_to_1] + sum_constraints
        target_risk = np.repeat(1/noa, noa)

        def msd_risk(weights, target_risk, cov):
            
            port_var = weights.T @ cov @ weights
            marginal_contribs = cov @ weights
            
            risk_contribs = np.multiply(marginal_contribs, weights.T) / port_var
            
            w_contribs = risk_contribs
            return ((w_contribs - target_risk)**2).sum()
        
        weights = minimize(msd_risk, 
                            init_guess,
                            args=(target_risk, cov), 
                            method='SLSQP',
                            constraints=constraints,
                            bounds=bounds)
        return weights.x
    
    # 거래비용 함수
    def transaction_cost(self, weights_df, rets_df, cost=0.00015):
        # 이전 기의 투자 가중치
        prev_weights_df = (weights_df.shift(1).fillna(0) * (1 + rets_df.iloc[self.param-1:,:])) \
        .div((weights_df.shift(1).fillna(0) * (1 + rets_df.iloc[self.param-1:,:])).sum(axis=1), axis=0)

        # 거래비용 데이터프레임
        cost_df = abs(weights_df - prev_weights_df) * cost
        cost_df.fillna(0, inplace=True)

        return cost_df

    # 백테스팅 실행 함수
    def run(self, cost):
        # 빈 딕셔너리
        backtest_dict = {}
        
        # 일별 수익률 데이터프레임
        rets = self.rets
        for i, index in enumerate(rets.index[engine.param-1:]):
            backtest_dict[index] = self.rp(self.cov[i])
        
        # 횡적 가중치 데이터프레임
        port_weights = pd.DataFrame(list(backtest_dict.values()), index=backtest_dict.keys(), columns=rets.columns)
        port_weights.fillna(0, inplace=True)

        # 자산 수익률
        pf_rets = port_weights.shift(1) * rets.iloc[self.param-1:,:]

        # 포트폴리오 수익률
        port_rets = pf_rets.sum(axis=1)
    
        # 거래비용 데이터프레임
        cost = self.transaction_cost(port_weights, rets)

        # 최종 포트폴리오 자산별 수익률
        port_asset_rets = port_weights.shift() * rets - cost

        # 최종 포트폴리오 수익률 
        port_rets = port_asset_rets.sum(axis=1)
        port_rets.index = pd.to_datetime(port_rets.index).strftime("%Y-%m-%d")

        return port_weights, port_asset_rets, port_rets
    
    # 성과분석 수행 함수
    def performance_analytics(self, port_weights, port_asset_rets, port_rets, BM, qs_report=False):
        
        # 자산별 투자 가중치
        plt.figure(figsize=(12, 7))
        port_weights['Cash'] = 1 - port_weights.sum(axis=1)
        plt.stackplot(port_weights.index, port_weights.T, labels=port_weights.columns)
        plt.title('Portfolio Weights')
        plt.xlabel('Date')
        plt.ylabel('Weights')
        plt.legend(loc='upper left')
        plt.show()

        # 자산별 누적 수익률
        plt.figure(figsize=(12, 7))
        plt.plot((1 + port_asset_rets).cumprod() - 1)
        plt.title('Underlying Asset Performance')
        plt.xlabel('Date')
        plt.ylabel('Returns')
        plt.legend(port_asset_rets.columns, loc='upper left')
        plt.show()

        # 포트폴리오 누적 수익률
        plt.figure(figsize=(12, 7))
        plt.plot((1 + port_rets).cumprod() - 1)
        plt.title('Portfolio Performance')
        plt.xlabel('Date')
        plt.ylabel('Returns')
        plt.show()

        # QuantStats 성과분석 리포트 작성
        if qs_report == True:
            port_rets.index = pd.to_datetime(port_rets.index)
            qs.reports.html(port_rets,BM, output=f'../bobae/rp_strategy_{date}.html')

    # 비중

if __name__ == "__main__":
    file_name = "DBGAPS2024_pricevolume.xlsx"
    # date = pd.Timestamp.now().strftime("%Y%m%d")
    # date = "20240626"
    date = input('리밸런싱 날짜를 입력해 주세요(format:yymmdd):')
    # 데이터 불러오기
    data = get_excel_data(date, file_name)

    # price 데이터
    price = data['수정주가']
    price.index = pd.to_datetime(price.index)

    # 리밸런싱
    rebal_dates = price.reset_index().assign(ym=lambda x:x['date'].dt.year.astype(str)+"-"+x['date'].dt.month.astype(str)).groupby('ym').last()['date'].sort_values()
    price_rebal = price.loc[rebal_dates].dropna(how='all', axis=1).dropna(how='all', axis=0)
    
    # 백테스팅 실행
    engine = Risk_Parity(price_rebal,param=12)
    res = engine.run(cost=0.00015)

    port_weights = res[0]
    port_asset_rets = res[1]
    port_rets = res[2]
    port_weights['Cash'] = 1 - port_weights.sum(1)

    # 벤치마크 : 동일비중 pf
    price_rebal_BM = price_rebal.stack().reset_index()
    price_rebal_BM.columns = ['date','종목코드','수정주가']
    price_rebal_BM['return'] = price_rebal_BM.groupby(by='종목코드')['수정주가'].shift(-1) / price_rebal_BM.groupby(by='종목코드')['수정주가'].shift(0) - 1 - 0.0015

    bm = price_rebal_BM.groupby('date')['return'].mean().reset_index()
    bm.columns = ['date', 'bm']
    bm = bm.set_index('date')

    # 백테스팅 결과 시각화
    engine.performance_analytics(port_weights, port_asset_rets, port_rets, bm, qs_report=True)
    # 포트폴리오 비중 엑셀
    port_weights.to_excel(f"../bobae/포트폴리오비중_{date}.xlsx")
    port_weights.iloc[-1].to_excel(f"../bobae/포트폴리오최근비중_{date}.xlsx")
    
    #TODO : 엑셀 스프레드 시트의 포트폴리오 비중 정리 자동화
    # 비율 전처리
    excel_weights = port_weights.copy()
    excel_weights = excel_weights.round(4)
    excel_weights = excel_weights.applymap(lambda x: f"{x:.4f}")
    excel_weights['Cash'] = 1 - excel_weights.sum(1)
    excel_weights = excel_weights.reset_index()
    excel_weights['index'] = excel_weights['index'].astype(str)
    
    spreadsheet_update(excel_weights)


