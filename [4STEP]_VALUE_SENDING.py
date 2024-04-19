import pandas as pd
import os
import requests
from datetime import datetime, timedelta
import schedule
import time

################################################################
##############################변수설정###########################
################################################################

# ustt_utc와 uend_utc 설정 2019-01-10부터 가능
todaydate = datetime.now()
yester = todaydate - timedelta(days=1)
yesterdate = yester.strftime("%Y-%m-%d")

n = 3  # 매수/매도 할 코인 개수
t = 1  # 매수/매도 주기
m = 100  # 코인 TOP 랭킹
d = 4  # d_day_price_change

# portion 값 설정
portion_values = {
    f'{d}_day_price_change': 0.7,
    '3_day_ma': 0,
    '20_day_ma': 0,
    '30_day_ma': 0,
    '50_day_ma': 0,
    '60_day_ma': 0,
    '100_day_ma': 0,
    '120_day_ma': 0,
    'RSI': 0.3,
    'drop_from_high': 0,
    'listing_period': 0
}


################################################################
##############################변수설정###########################
################################################################

def post_message(token, channel, text):
    response = requests.post("https://slack.com/api/chat.postMessage",
                             headers={"Authorization": "Bearer " + token},
                             data={"channel": channel, "text": text}
                             )
    print(response)


myToken = "xoxb-6969697834503-6996943830753-7Xc0scSNRe1xLbYgvS0D5Tz7"
today_date = datetime.now().strftime('%Y-%m-%d')

def calculate_n_score(row, portion_values):
    n_score = (row[f'{d}_day_price_change'] / 100) * portion_values[f'{d}_day_price_change'] + \
              ((row['3_day_ma'] - row['close']) / row['close']) * portion_values['3_day_ma'] + \
              ((row['20_day_ma'] - row['close']) / row['close']) * portion_values['20_day_ma'] + \
              ((row['30_day_ma'] - row['close']) / row['close']) * portion_values['30_day_ma'] + \
              ((row['50_day_ma'] - row['close']) / row['close']) * portion_values['50_day_ma'] + \
              ((row['60_day_ma'] - row['close']) / row['close']) * portion_values['60_day_ma'] + \
              ((row['100_day_ma'] - row['close']) / row['close']) * portion_values['100_day_ma'] + \
              ((row['120_day_ma'] - row['close']) / row['close']) * portion_values['120_day_ma'] + \
              (row['RSI'] / 100) * portion_values['RSI'] + \
              (row['drop_from_high'] / 100) * portion_values['drop_from_high'] + \
              ((760 - row['listing_period']) / 760) * portion_values['listing_period']
    return n_score

def jobs():
    # 파일 1에서 리스트 추출 및 DataFrame 생성
    file1_path = r'/Users/hyeonwhan.ko/Desktop/BIGDATA/TOP100(BINANCE).csv'
    file1_data = pd.read_csv(file1_path)

    # coin_list 생성 coin_1, coin_2,,, coin_n
    coin_list = [f'coin_{i + 1}' for i in range(n)]
    # coin_chng_list 생성 coin_1, coin_2,,, coin_n
    coin_chng_list = [f'coin_chng_{i + 1}' for i in range(n)]
    # coin_RSI_list 생성 coin_1, coin_2,,, coin_n
    coin_RSI_list = [f'coin_RSI_{i + 1}' for i in range(n)]
    # coin_num_list 생성 coin_num_1, coin_num_2,,, coin_num_n
    coin_close_list = [f'coin_close_{i + 1}' for i in range(n)]

    # portion 값의 합 계산
    portion_sum = sum(portion_values.values())

    # portion 값의 합이 1인지 확인하여 처리
    if abs(portion_sum - 1) < 1e-10:
        # 오늘 날짜 표 가지고오기
        coin_list_for_date = file1_data.loc[file1_data['Date'] == yesterdate, ['Date', 'RANK', 'Symbol']]
        # 열추가
        additional_columns = ['open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
                              'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                              'ignore', '3_day_ma', '20_day_ma', '30_day_ma', '50_day_ma', '60_day_ma',
                              '100_day_ma', '120_day_ma', '1_day_price_change', '2_day_price_change',
                              '3_day_price_change', '4_day_price_change',
                              '5_day_price_change', '6_day_price_change', '7_day_price_change', 'RSI', 'listing_period',
                              'drop_from_high']
        coin_list_for_date = pd.concat([coin_list_for_date, pd.DataFrame(columns=additional_columns)], axis=1)
        # 각 파일에서 값 채우기
        for index, row in coin_list_for_date.iterrows():
            symbol = row['Symbol']
            file2_path = rf'/Users/hyeonwhan.ko/Desktop/BIGDATA/COINRAW/{symbol}.csv'
            try:
                if os.path.exists(file2_path):
                    file2_data = pd.read_csv(file2_path)
                    # 파일 이름에서 심볼 정보 추출
                    symbol_from_filename = os.path.splitext(os.path.basename(file2_path))[0]
                    # 파일 2의 데이터에서 파일 1과의 조건을 통해 필터링
                    filtered_data = file2_data[file2_data['timestamp'] == yesterdate]
                    if not filtered_data.empty:
                        # 필터링된 데이터의 값을 파일 1의 해당 행에 추가
                        coin_list_for_date.loc[index, additional_columns] = filtered_data.iloc[0][
                            additional_columns].values
            except FileNotFoundError:
                continue

                # n스코어 계산
        # n_Score 열 추가 및 값 계산
        coin_list_for_date['n_Score'] = coin_list_for_date.apply(lambda row: calculate_n_score(row, portion_values), axis=1)
        # n_Score_Rank 열 추가 및 값 기입
        coin_list_for_date['n_Score_Rank'] = coin_list_for_date['n_Score'].rank(method='first', ascending=False)

        # 1~n 코인 리스트 확인
        for i in range(n):
            if not coin_list_for_date.empty:
                coin_list[i] = \
                    coin_list_for_date.loc[coin_list_for_date['n_Score_Rank'] == i + 1, 'Symbol'].values[0]
                coin_close_list[i] = \
                    round(coin_list_for_date.loc[coin_list_for_date['n_Score_Rank'] == i + 1, 'close'].values[0],1)
                coin_chng_list[i] = \
                    round(coin_list_for_date.loc[coin_list_for_date['n_Score_Rank'] == i + 1, f'{d}_day_price_change'].values[0],1)
                coin_RSI_list[i] = \
                    round(coin_list_for_date.loc[coin_list_for_date['n_Score_Rank'] == i + 1, f'RSI'].values[0],1)
            else:
                print(f"{yesterdate}에 해당하는 데이터가 없습니다. TOP100(BINANCE).csv파일을 확인하세요.")


        # btc_price가 저장된 데이터프레임
        btc_df = pd.read_csv(r'/Users/hyeonwhan.ko/Desktop/BIGDATA/COINRAW/BTC.csv')

        # yesterdate에 해당하는 btc_price를 가져오기
        btc_row = btc_df[btc_df['timestamp'] == yesterdate]

        btc_1chng = round(btc_row.iloc[0]['1_day_price_change'],1)
        btc_3chng = round(btc_row.iloc[0]['3_day_price_change'],1)
        btc_7chng = round(btc_row.iloc[0]['7_day_price_change'],1)
        btc_20MA = round(btc_row.iloc[0]['20MA_margin'],1)
        btc_50MA = round(btc_row.iloc[0]['50MA_margin'],1)
        btc_60MA = round(btc_row.iloc[0]['60MA_margin'],1)
        btc_120MA = round(btc_row.iloc[0]['120MA_margin'],1)

        post_message(myToken, "#rebalancing", f"* {today_date} 09:00 매수 가이드 ")
        post_message(myToken, "#rebalancing", f"코인 {n}개, {d}day {portion_values[f'{d}_day_price_change']}, RSI {portion_values['RSI']} 기준")
        post_message(myToken, "#rebalancing", f"TICKER  : {coin_list}")
        post_message(myToken, "#rebalancing", f"{d}day변화 : {coin_chng_list} [%]")
        post_message(myToken, "#rebalancing", f"RSI           : {coin_RSI_list}")
        post_message(myToken, "#rebalancing", f"* BTC 정보")
        post_message(myToken, "#rebalancing", f"1day:{btc_1chng},   3day:{btc_3chng},   7day:{btc_7chng} [%]")
        post_message(myToken, "#rebalancing", f" 20MA 기준 : {btc_20MA} $")
        post_message(myToken, "#rebalancing", f" 50MA 기준 : {btc_50MA} $")
        post_message(myToken, "#rebalancing", f" 60MA 기준 : {btc_60MA} $")
        post_message(myToken, "#rebalancing", f"120MA 기준 : {btc_120MA} $")

    else:
        print("portion_values 값을 확인하세요!")

# 매일 09:30에 job 함수 실행
schedule.every().day.at("09:30").do(jobs)

while True:
    schedule.run_pending()
    time.sleep(1)
