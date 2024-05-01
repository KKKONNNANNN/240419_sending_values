import os
import pandas as pd
from datetime import datetime, timedelta
import requests
import schedule
import time
import boto3
from botocore.exceptions import ClientError  # ClientError 임포트 추가
from io import StringIO   # io 모듈 임포트 추가
import hashlib
import hmac
from urllib.parse import urlencode
import math

# BINANCE API 엔드포인트와 키 설정
API_ENDPOINT = "https://api.binance.com/api/v3"
API_KEY = "API_KEY"
API_SECRET = "API_SECRET"

################################################################
##############################변수설정###########################
################################################################

n = 2  # 매수/매도 할 코인 개수
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

# 시작 일자 기입

def get_buy_sell_days():
    stt_date = datetime.now()
    # buy&sell days
    buy_sell_days = []
    while stt_date < datetime(2030, 12, 31):
        buy_sell_days.append(stt_date.strftime("%Y-%m-%d"))
        stt_date += timedelta(days=t)
    return buy_sell_days

buy_sell_days = get_buy_sell_days()

# SLACK 메시지 보내기 함수
def post_message(token, channel, text):
    response = requests.post("https://slack.com/api/chat.postMessage",
                             headers={"Authorization": "Bearer " + token},
                             data={"channel": channel, "text": text}
                             )
    print(response)


myToken = "SLAC_TOKEN"

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

# 서명 생성 함수
def generate_signature(params):
    query_string = urlencode(params)
    signature = hmac.new(API_SECRET.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
    return signature

# 특정 코인에 대한 계정 정보 가져오기 함수
def get_coin_balance(coin):
    params = {"timestamp": int(time.time() * 1000)}
    signature = generate_signature(params)
    params['signature'] = signature
    headers = {"X-MBX-APIKEY": API_KEY}
    response = requests.get(API_ENDPOINT + "/account", headers=headers, params=params)
    account_info = response.json()
    # 특정 코인에 대한 잔고 필터링
    coin_balance = next((asset for asset in account_info.get('balances', []) if asset.get('asset') == coin), None)
    if coin_balance is None:
        print(f"No balance found for {coin}.")
        post_message(myToken, "#rebalancing", f"코인 보유 정보 가져오기 에러발생 : {account_info}")
    return coin_balance


def get_coin_price(coin):
    # Binance API 엔드포인트
    endpoint = "https://api.binance.com/api/v3/ticker/price"
    # 요청 매개변수 설정
    params = {"symbol": coin + "USDT"}
    # API 요청 보내기
    response = requests.get(endpoint, params=params)
    # 응답 JSON 파싱
    response_json = response.json()
    try:
        # 응답 데이터에서 가격 정보 가져오기
        coin_price = float(response_json['price'])
    except KeyError:
        # 'price' 키가 없는 경우
        print(f"Warning: 'price' key not found in response for {coin}. Using default value.")
        post_message(myToken, "#rebalancing", f"코인 가격 가져오기 에러발생 : {response_json}")
        coin_price = 0  # 또는 다른 기본값 설정

    return coin_price

# 수 내림하기
def floor_to_decimal(number, decimal_places):
    factor = 10 ** decimal_places
    multiplied = number * factor
    floored = int(multiplied)
    result = floored / factor
    return result

# 10의 n승에서 내림하기
def floor_to_n(number,n):
    divided = number // (10**n)  # 100으로 나눈 몫을 구합니다.
    floored = divided * (10**n)  # 다시 100을 곱하여 100의 자리에서 내림한 값을 구합니다.
    return floored

# n값 구하기
def get_10n(number):
    if number == 0:
        return 0
    if number > 0:
        exponent = 0
        while number >= 10:
            number /= 10
            exponent += 1
        return exponent


# 주문 생성 함수
def create_order(symbol, side, quantity):
    params = {
        "symbol": symbol,
        "side": side,
        "type": "MARKET",
        "quantity": quantity,
        "timestamp": int(time.time() * 1000)
    }
    signature = generate_signature(params)
    params['signature'] = signature
    headers = {"X-MBX-APIKEY": API_KEY}
    response = requests.post(API_ENDPOINT + "/order", headers=headers, params=params)
    order_info = response.json()

    # 오류가 발생했을 경우 메시지 출력
    if "msg" in order_info:
        print("Error:", order_info["msg"])
        post_message(myToken, "#rebalancing", f"거래 중 오류 발생 : {order_info['msg']}")
    else :
        print(f"{symbol} {quantity}개 {side} 정상 진행")
        post_message(myToken, "#rebalancing", f"{symbol} {quantity}개 {side} 정상 진행")
    return order_info


# USDT로 coin 매수, ratio : USDT의 비율
def buy_coin(coin, ratio):
    coin_price = get_coin_price(coin)
    coin_balance_USDT_info = get_coin_balance("USDT")

    if coin_balance_USDT_info is not None:
        coin_balance_USDT = float(coin_balance_USDT_info['free'])
        buy_USDT = round(coin_balance_USDT * ratio * 0.999,3)
        buy_quantity = floor_to_decimal(buy_USDT / coin_price, get_10n(coin_price))
        if get_10n(coin_price) > 0:
            if buy_USDT > 10:
                print(f"*{coin} 매수.")
                post_message(myToken,"#rebalancing",f"*{coin} 매수. {buy_USDT}$")
                return create_order(coin + "USDT", "BUY", str(buy_quantity))
            else :
                print(f"* 보유 USDT : {round(coin_balance_USDT, 2)}, {ratio}배 가치가 10$ 미만입니다.")
                post_message(myToken, "#rebalancing", f"* 보유 USDT : {round(coin_balance_USDT, 2)}, {ratio}배 가치가 10$ 미만으로 매도 실패.")
        else:
            if buy_USDT > 10:
                print(f"*{coin} 매수.")
                post_message(myToken,"#rebalancing",f"*{coin} 매수. {buy_USDT}$")
                return create_order(coin + "USDT", "BUY", str(buy_quantity))
            else :
                print(f"* 보유 USDT : {round(coin_balance_USDT,2)}, {ratio}배 가치가 10$ 미만입니다.")
                post_message(myToken, "#rebalancing", f"* 보유 USDT : {round(coin_balance_USDT,2)}, {ratio}배 가치가 10$ 미만으로 매도 실패.")
    else:
        print("Failed to get USDT balance information.")

# coin 매도
def sell_coin(coin):
    coin_price = get_coin_price(coin)
    coin_balance_info = get_coin_balance(coin)

    if coin_balance_info is not None:
        coin_balance = float(coin_balance_info['free'])
        if get_10n(coin_price) > 0 :
            sell_quantity = floor_to_decimal(coin_balance * 0.999,get_10n(coin_price))
            if coin_price * coin_balance > 10:
                print(f"*{coin} 매도.")
                post_message(myToken, "#rebalancing", f"*{coin} 매도. 개수 : {sell_quantity} 가격 : {coin_price}")
                return create_order(coin + "USDT", "SELL", str(sell_quantity))
            else:
                print(f"* {coin}의 가치가 10$ 미만입니다.")
                post_message(myToken, "#rebalancing", f"*{coin}의 가치가 10$ 미만으로 매도 실패.")
        else :
            sell_quantity = floor_to_n(coin_balance * 0.999, get_10n(coin_price)*(-1))
            if coin_price * coin_balance > 10:
                print(f"*{coin} 매도.")
                post_message(myToken, "#rebalancing", f"*{coin} 매도. 개수 : {sell_quantity} 가격 : {coin_price}")
                return create_order(coin + "USDT", "SELL", str(sell_quantity))
            else:
                print(f"* {coin}의 가치가 10$ 미만입니다.")
                post_message(myToken, "#rebalancing", f"*{coin}의 가치가 10$ 미만으로 매도 실패.")
    else:
        print("Failed to get coin balance information.")

# 사용 예시
# sell_coin("LDO")
# buy_coin("LDO",0.1)

todaydate = datetime.now()
yester = todaydate - timedelta(days=1)
yesterdate = yester.strftime("%Y-%m-%d")
today_date = datetime.now().strftime("%Y-%m-%d")

coin_list_y = f"{yesterdate}_coin_list"
coin_close_list_y = f"{yesterdate}_coin_close_list"
coin_chng_list_y = f"{yesterdate}_coin_chng_list"
coin_RSI_list_y = f"{yesterdate}_coin_RSI_list"

# coin_list_y 생성 coin_1, coin_2,,, coin_n
coin_list_y = [f'{yesterdate}_coin_y_{i + 1}' for i in range(n)]
# coin_chng_list_y 생성 coin_1, coin_2,,, coin_n
coin_chng_list_y = [f'{yesterdate}_coin_chng_y_{i + 1}' for i in range(n)]
# coin_RSI_list_y 생성 coin_1, coin_2,,, coin_n
coin_RSI_list_y = [f'{yesterdate}_coin_RSI_y_{i + 1}' for i in range(n)]
# coin_num_list_y 생성 coin_num_1, coin_num_2,,, coin_num_n
coin_close_list_y = [f'{yesterdate}_coin_close_y_{i + 1}' for i in range(n)]

coin_list_y[0] = "TRX"
coin_list_y[1] = "LDO"

def jobs():
    # ustt_utc와 uend_utc 설정 2019-01-10부터 가능
    todaydate = datetime.now()
    yester = todaydate - timedelta(days=1)
    yesterdate = yester.strftime("%Y-%m-%d")
    today_date = datetime.now().strftime("%Y-%m-%d")

    # S3 클라이언트 생성
    aws_access_key_id = 'API_KEY'
    aws_secret_access_key = 'API_SECRET'
    bucket_name = '240419-sending-values'
    file1_key = 'raw-files/TOP100(BINANCE).csv'

    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

    # 파일 내용 가져오기
    response = s3.get_object(Bucket=bucket_name, Key=file1_key)
    file1_data = response['Body'].read().decode('utf-8')

    # StringIO를 사용하여 CSV 데이터를 pandas DataFrame으로 변환
    df = pd.read_csv(StringIO(file1_data))

    coin_list = f"{today_date}_coin_list"
    coin_close_list = f"{today_date}_coin_close_list"
    coin_chng_list = f"{today_date}_coin_chng_list"
    coin_RSI_list = f"{today_date}_coin_RSI_list"

    # coin_list 생성 coin_1, coin_2,,, coin_n
    coin_list = [f'{today_date}_coin_{i + 1}' for i in range(n)]
    # coin_chng_list 생성 coin_1, coin_2,,, coin_n
    coin_chng_list = [f'{today_date}_coin_chng_{i + 1}' for i in range(n)]
    # coin_chng_list 생성 coin_1, coin_2,,, coin_n
    coin_1chng_list = [f'{today_date}_coin_1chng_{i + 1}' for i in range(n)]
    # coin_chng_list 생성 coin_1, coin_2,,, coin_n
    coin_7chng_list = [f'{today_date}_coin_7chng_{i + 1}' for i in range(n)]
    # coin_RSI_list 생성 coin_1, coin_2,,, coin_n
    coin_RSI_list = [f'{today_date}_coin_RSI_{i + 1}' for i in range(n)]
    # coin_num_list 생성 coin_num_1, coin_num_2,,, coin_num_n
    coin_close_list = [f'{today_date}_coin_close_{i + 1}' for i in range(n)]

    # portion 값의 합 계산
    portion_sum = sum(portion_values.values())

    # portion 값의 합이 1인지 확인하여 처리
    if abs(portion_sum - 1) < 1e-10:
        # 오늘 날짜 표 가지고오기
        coin_list_for_date = df.loc[df['Date'] == yesterdate, ['Date', 'RANK', 'Symbol']]
        # 열추가
        additional_columns = ['open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
                              'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                              'ignore', '3_day_ma', '20_day_ma', '30_day_ma', '50_day_ma', '60_day_ma',
                              '100_day_ma', '112_day_ma', '120_day_ma', '1_day_price_change', '2_day_price_change',
                              '3_day_price_change', '4_day_price_change',
                              '5_day_price_change', '6_day_price_change', '7_day_price_change', 'RSI', 'listing_period',
                              'drop_from_high']
        coin_list_for_date = pd.concat([coin_list_for_date, pd.DataFrame(columns=additional_columns)], axis=1)
        # 각 파일에서 값 채우기

        for index, row in coin_list_for_date.iterrows():
            symbol = row['Symbol']
            file2_key = f'raw-files/COINRAW/{symbol}.csv'

            # 파일이 존재하는지 확인
            try:
                s3.head_object(Bucket=bucket_name, Key=file2_key)
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    print(f"File '{file2_key}' not found.")
                    continue

            response2 = s3.get_object(Bucket=bucket_name, Key=file2_key)
            file2_data = response2['Body'].read().decode('utf-8')
            df2 = pd.read_csv(StringIO(file2_data))

            try:
                # 파일 2의 데이터에서 해당 날짜의 데이터 추출
                filtered_data = df2[df2['timestamp'] == yesterdate]
                if not filtered_data.empty:
                    # 필터링된 데이터의 값을 coin_list_for_date의 해당 행에 추가
                    coin_list_for_date.loc[index, additional_columns] = filtered_data.iloc[0][additional_columns].values
            except FileNotFoundError:
                continue

                # n스코어 계산
        # n_Score 열 추가 및 값 계산
        coin_list_for_date['n_Score'] = coin_list_for_date.apply(lambda row: calculate_n_score(row, portion_values),
                                                                 axis=1)
        # n_Score_Rank 열 추가 및 값 기입
        coin_list_for_date['n_Score_Rank'] = coin_list_for_date['n_Score'].rank(method='first', ascending=False)

        # 1~n 코인 리스트 확인
        for i in range(n):
            if not coin_list_for_date.empty:
                coin_list[i] = \
                    coin_list_for_date.loc[coin_list_for_date['n_Score_Rank'] == i + 1, 'Symbol'].values[0]
                coin_close_list[i] = \
                    round(coin_list_for_date.loc[coin_list_for_date['n_Score_Rank'] == i + 1, 'close'].values[0], 1)
                coin_chng_list[i] = \
                    round(coin_list_for_date.loc[
                              coin_list_for_date['n_Score_Rank'] == i + 1, f'{d}_day_price_change'].values[0], 1)
                coin_1chng_list[i] = \
                    round(coin_list_for_date.loc[
                              coin_list_for_date['n_Score_Rank'] == i + 1, f'1_day_price_change'].values[0], 1)
                coin_7chng_list[i] = \
                    round(coin_list_for_date.loc[
                              coin_list_for_date['n_Score_Rank'] == i + 1, f'7_day_price_change'].values[0], 1)
                coin_RSI_list[i] = \
                    round(coin_list_for_date.loc[coin_list_for_date['n_Score_Rank'] == i + 1, f'RSI'].values[0], 1)
            else:
                print(f"{yesterdate}에 해당하는 데이터가 없습니다. TOP100(BINANCE).csv파일을 확인하세요.")

        # btc_price가 저장된 데이터프레임
        file_key_btc = 'raw-files/COINRAW/BTC.csv'
        response_btc = s3.get_object(Bucket=bucket_name, Key=file_key_btc)
        file_data_btc = response_btc['Body'].read().decode('utf-8')

        # StringIO를 사용하여 CSV 데이터를 pandas DataFrame으로 변환
        btc_df = pd.read_csv(StringIO(file_data_btc))

        # yesterdate에 해당하는 btc_price를 가져오기
        btc_row = btc_df[btc_df['timestamp'] == yesterdate]

        btc_1chng = round(btc_row.iloc[0]['1_day_price_change'], 1)
        btc_3chng = round(btc_row.iloc[0]['3_day_price_change'], 1)
        btc_7chng = round(btc_row.iloc[0]['7_day_price_change'], 1)
        btc_20MA = round(btc_row.iloc[0]['20MA_margin'], 1)
        btc_50MA = round(btc_row.iloc[0]['50MA_margin'], 1)
        btc_60MA = round(btc_row.iloc[0]['60MA_margin'], 1)
        btc_112MA = round(btc_row.iloc[0]['112MA_margin'], 1)
        btc_120MA = round(btc_row.iloc[0]['120MA_margin'], 1)

        # 거래변수 = 0
        exchange = 0

        sell_coin_list = []
        no_sell_coin_list = []
        buy_coin_list = []
        no_buy_coin_list = []

        # 112MA 양수, 매도/매수 진행
        if btc_112MA > 0:
            if today_date in buy_sell_days:
                # 어제 코인 매도 진행
                for i in range(n):
                    if coin_list_y:
                        if not coin_list_y[i] in coin_list:
                            # {yesterdate}_coin_list[i] 매도 진행,거래변수 +1
                            # no_sell_coin_list에 추가
                            sell_coin(coin_list_y[i])
                            exchange += 1
                            sell_coin_list.append(coin_list_y[i])

                        else:
                            # {yesterdate}_coin_list[i] 매도 미진행
                            # no_sell_coin_list에 추가
                            no_sell_coin_list.append(coin_list_y[i])
                            print(f"{coin_list_y[i]} 오늘과 동일한 coin. 매도 미진행.")
                            post_message(myToken, "#rebalancing", f"{coin_list_y[i]} 어제와 동일한 coin. 매도 미진행.")
                    else:
                        continue

                # 오늘 코인 매수 진행
                for i in range(n):
                    if not coin_list[i] in coin_list_y:
                        # {today_date}_coin_list[i] 매수, 1/거래변수 금액만큼
                        # buy_coin_list에 추가
                        # 거래변수 -1
                        buy_coin(coin_list[i], 1 / exchange)
                        buy_coin_list.append(coin_list[i])
                        exchange -= 1
                    else:
                        # {today_date}_coin_list[i] 매수 미진행
                        # no_buy_coin_list 에 추가
                        no_buy_coin_list.append(coin_list[i])
                        print(f"{coin_list[i]} 어제와 동일한 coin. 매수 미진행.")
                        post_message(myToken, "#rebalancing", f"{coin_list[i]} 어제와 동일한 coin. 매수 미진행.")
            # 오늘의 코인에 어제 코인 할당
            else:
                for i in range(n):
                    coin_list[i] = coin_list_y[i]
                    print("매도/매수날짜에 해당하지 않아 거래 미진행.")
                    post_message(myToken, "#rebalancing", f"매도/매수 날짜에 해당하지 않아 거래 미진행.")

        # 112MA 음수, 매도 진행
        else:
            for i in range(n):
                sell_coin(coin_list_y[i])
                print("112MA 미만으로, 매수는 진행하지 않음")
                post_message(myToken, "#rebalancing", f"* 112MA 미만으로, 매수는 진행하지 않음 ")

# 매일 10:20에 jobs 함수 실행
schedule.every().day.at("10:20").do(jobs)

while True:
    schedule.run_pending()
    time.sleep(1)
