import json
import os
import pandas as pd
import csv
import requests
from binance.client import Client
from datetime import datetime, timedelta
import schedule
import boto3
import time
import io
from botocore.exceptions import ClientError  # ClientError 임포트 추가
from io import StringIO   # io 모듈 임포트 추가
import hashlib
import hmac
from urllib.parse import urlencode
import math

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
    'listing_period': 0
}

################################################################
##############################변수설정###########################
################################################################


def load_data():
    with open('data.json', 'r') as f:
        loaded_data = json.load(f)
    return loaded_data


if __name__ == "__main__":
    loaded_data = load_data()  #데이터 로드
    # SLACK API 설정
    myToken = loaded_data['STOKEN']
    # Binance API 키 설정
    api_key = loaded_data['B_API']
    api_secret = loaded_data['B_SEC']
    client = Client(api_key, api_secret)
    # S3 클라이언트 생성
    aws_access_key_id = loaded_data['S3_API']
    aws_secret_access_key = loaded_data['S3_SEC']
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

    today_date = datetime.now().strftime('%Y%m%d')


def post_message(token, channel, text):
    response = requests.post("https://slack.com/api/chat.postMessage",
                             headers={"Authorization": "Bearer " + token},
                             data={"channel": channel, "text": text})
    print(response)


# binance에서 모든 symbol 얻기
def get_all_symbols():
    url = "https://api.binance.com/api/v3/exchangeInfo"
    response = requests.get(url)
    data = response.json()
    symbols = [symbol_info['symbol'] for symbol_info in data['symbols']]
    return symbols

# 끝자리 USDT인것들만 추출
def clean_symbols(symbols):
    cleaned_symbols = []
    for symbol in symbols:
        if symbol[-4:] == 'USDT':
            cleaned_symbols.append(symbol[:-4])
    return cleaned_symbols

# DOWN/BEAR/BULL/UP 거르기
def clean_symbols2(symbols):
    cleaned_symbols = []
    for symbol in symbols:
        if symbol[-4:] in ['DOWN', 'BEAR', 'BULL']:
            cleaned_symbols.append(symbol[:-4])
        elif symbol[-2:] == 'UP' and symbol != 'JUP':
            cleaned_symbols.append(symbol[:-2])
        else:
            cleaned_symbols.append(symbol)
    return cleaned_symbols

# 각 열의 계산 함수들
def calculate_3_day_ma(data):
    return data['close'].rolling(window=3).mean()


def calculate_20_day_ma(data):
    return data['close'].rolling(window=20).mean()


def calculate_30_day_ma(data):
    return data['close'].rolling(window=30).mean()


def calculate_50_day_ma(data):
    return data['close'].rolling(window=50).mean()


def calculate_60_day_ma(data):
    return data['close'].rolling(window=60).mean()


def calculate_100_day_ma(data):
    return data['close'].rolling(window=100).mean()

def calculate_112_day_ma(data):
    return data['close'].rolling(window=112).mean()

def calculate_120_day_ma(data):
    return data['close'].rolling(window=120).mean()


def calculate_1_day_price_change(data):
    if data['close'].iloc[-1] == 0:
        return 0
    return (data['close'].iloc[-1] - data['close'].iloc[-2]) / data['close'].iloc[-2] * 100


def calculate_2_day_price_change(data):
    if data['close'].iloc[-1] == 0:
        return 0
    return (data['close'].iloc[-1] - data['close'].iloc[-3]) / data['close'].iloc[-3] * 100


def calculate_3_day_price_change(data):
    if data['close'].iloc[-1] == 0:
        return 0
    return (data['close'].iloc[-1] - data['close'].iloc[-4]) / data['close'].iloc[-4] * 100


def calculate_4_day_price_change(data):
    if data['close'].iloc[-1] == 0:
        return 0
    return (data['close'].iloc[-1] - data['close'].iloc[-5]) / data['close'].iloc[-5] * 100


def calculate_5_day_price_change(data):
    if data['close'].iloc[-1] == 0:
        return 0
    return (data['close'].iloc[-1] - data['close'].iloc[-6]) / data['close'].iloc[-6] * 100


def calculate_6_day_price_change(data):
    if data['close'].iloc[-1] == 0:
        return 0
    return (data['close'].iloc[-1] - data['close'].iloc[-7]) / data['close'].iloc[-7] * 100


def calculate_7_day_price_change(data):
    if data['close'].iloc[-1] == 0:
        return 0
    return (data['close'].iloc[-1] - data['close'].iloc[-8]) / data['close'].iloc[-8] * 100


def calculate_rsi(data, window=14):
    if len(data) < window + 1:
        return None

    delta = data['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]


def calculate_listing_period(data):
    first_date = data['timestamp'].iloc[0]  # 이미 datetime 형식의 Timestamp 객체이므로 추가 변환이 필요하지 않음
    current_date = data['timestamp'].iloc[-1]
    return (current_date - first_date).days


#def calculate_drop_from_high(data):
#    if len(data) < 2:
#        return None

#    current_price = data['close'].iloc[-1]
#    max_price_before_current = data['close'].iloc[:-1].max()
#    if pd.isnull(max_price_before_current) or max_price_before_current == 0:
#        return None

    # 데이터프레임의 인덱스를 재설정하여 중복된 인덱스를 제거
#    data = data.reset_index(drop=True)

#    return (max_price_before_current - current_price) / max_price_before_current * 100


def calculate_20MA_margin(data):
        if data['close'].iloc[-1] == 0:
            return 0
        return (data['close'].iloc[-1] - data['20_day_ma'].iloc[-1])

def calculate_50MA_margin(data):
        if data['close'].iloc[-1] == 0:
            return 0
        return (data['close'].iloc[-1] - data['50_day_ma'].iloc[-1])

def calculate_60MA_margin(data):
        if data['close'].iloc[-1] == 0:
            return 0
        return (data['close'].iloc[-1] - data['60_day_ma'].iloc[-1])

def calculate_112MA_margin(data):
        if data['close'].iloc[-1] == 0:
            return 0
        return (data['close'].iloc[-1] - data['112_day_ma'].iloc[-1])

def calculate_120MA_margin(data):
        if data['close'].iloc[-1] == 0:
            return 0
        return (data['close'].iloc[-1] - data['120_day_ma'].iloc[-1])
    
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
              ((760 - row['listing_period']) / 760) * portion_values['listing_period']
    return n_score

# 서명 생성 함수
def generate_signature(params):
    query_string = urlencode(params)
    signature = hmac.new(api_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
    return signature

# 특정 코인에 대한 계정 정보 가져오기 함수
def get_coin_balance(coin):
    params = {"timestamp": int(time.time() * 1000)}
    signature = generate_signature(params)
    params['signature'] = signature
    headers = {"X-MBX-APIKEY": api_key}
    response = requests.get(API_ENDPOINT + "/account", headers=headers, params=params)
    account_info = response.json()
    # 특정 코인에 대한 잔고 필터링
    coin_balance = next((asset for asset in account_info.get('balances', []) if asset.get('asset') == coin), None)
    if coin_balance is None:
        print(f"No balance found for {coin}.")
        post_message(myToken, "#rebalancing", f"* 코인 보유 정보 가져오기 에러발생 : {account_info}")
        return 0
    
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
        post_message(myToken, "#rebalancing", f"* 코인 가격 가져오기 에러발생 : {response_json}")
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
    headers = {"X-MBX-APIKEY": api_key}
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
                post_message(myToken,"#rebalancing",f"* {coin} 매수. {buy_USDT}$")
                return create_order(coin + "USDT", "BUY", str(buy_quantity))
            else :
                print(f"* 보유 USDT : {round(coin_balance_USDT,2)}, {ratio}배 가치가 10$ 미만으로 {coin} 매수 미진행.")
                post_message(myToken, "#rebalancing", f"* 보유 USDT : {round(coin_balance_USDT,2)}, {ratio}배 가치가 10$ 미만으로 {coin} 매수 미진행.")
        else:
            if buy_USDT > 10:
                print(f"*{coin} 매수.")
                post_message(myToken,"#rebalancing",f"* {coin} 매수. {buy_USDT}$")
                return create_order(coin + "USDT", "BUY", str(buy_quantity))
            else :
                print(f"* 보유 USDT : {round(coin_balance_USDT,2)}, {ratio}배 가치가 10$ 미만으로 {coin} 매수 미진행.")
                post_message(myToken, "#rebalancing", f"* 보유 USDT : {round(coin_balance_USDT,2)}, {ratio}배 가치가 10$ 미만으로 {coin} 매수 미진행.")
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
                post_message(myToken, "#rebalancing", f"* {coin} 매도. 개수 : {sell_quantity} 가격 : {coin_price}")
                return create_order(coin + "USDT", "SELL", str(sell_quantity))
            else:
                print(f"* {coin}의 가치가 10$ 미만입니다.")
                post_message(myToken, "#rebalancing", f"* {coin}의 가치가 10$ 미만으로 매도 미진행.")
        else :
            sell_quantity = floor_to_n(coin_balance * 0.999, get_10n(coin_price)*(-1))
            if coin_price * coin_balance > 10:
                print(f"*{coin} 매도.")
                post_message(myToken, "#rebalancing", f"* {coin} 매도. 개수 : {sell_quantity} 가격 : {coin_price}")
                return create_order(coin + "USDT", "SELL", str(sell_quantity))
            else:
                print(f"* {coin}의 가치가 10$ 미만입니다.")
                post_message(myToken, "#rebalancing", f"* {coin}의 가치가 10$ 미만으로 매도 미진행.")
    else:
        print("Failed to get coin balance information.")
        
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


# 코인 리스트와 해당 코인의 값 리스트 생성
coin_columns = [f'Coin{i+1}' for i in range(n)]
coin_value_columns = [f'Coin{i+1}_value' for i in range(n)]

# 데이터프레임 생성

df_final = pd.DataFrame(columns=['Date'] + sum([list(pair) for pair in zip(coin_columns, coin_value_columns)], []) + ['USDT', 'Total_Asset', 'Multiple'])

coin_list_y[0] = "RNDR"
coin_list_y[1] = "RUNE"

def CODE() :
    ######## SYMBOL들 저장하기 #########
    all_symbols = get_all_symbols()
    cleaned_symbols = clean_symbols(all_symbols)
    cleaned_symbols = clean_symbols2(cleaned_symbols)
    cleaned_symbols = sorted(set(cleaned_symbols) - {''})
    #print(cleaned_symbols)
    print(f"{today_date} 코인 목록 가져오기 완료.")
    post_message(myToken, "#rebalancing", f"{today_date} 코인 목록 가져오기 완료.")
    
    ######## symbol_dfs[f'{SYMBOL}_DF'] 만들기 #########
    symbol_dfs = {}  # 빈 딕셔너리로 초기화
    
    for symbol in cleaned_symbols:
        try:
            symbol_with_usdt = f'{symbol}USDT'
    
            # 어제 날짜 구하기
            yesterday_date = datetime.now() - timedelta(days=1)
            yesterday_date_str = yesterday_date.strftime("%d %b, %Y")  # 어제 날짜를 포맷에 맞게 변환
    
            # 심볼에 대한 가격 데이터 가져오기
            klines = client.get_historical_klines(symbol_with_usdt, Client.KLINE_INTERVAL_1DAY, "1 Jan, 2017",
                                                  yesterday_date_str)
    
            # 가격 데이터를 DataFrame으로 변환
            data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                                 'quote_asset_volume', 'number_of_trades',
                                                 'taker_buy_base_asset_volume',
                                                 'taker_buy_quote_asset_volume', 'ignore'])
            
            # 'close' 열의 데이터 타입을 숫자형으로 변환
            data['close'] = pd.to_numeric(data['close'])
    
            # timestamp를 날짜 형식으로 변환
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    
            # 마지막으로 데이터가 입력된 날짜 다음 날부터 2024-12-31까지의 날짜 범위 생성(상장 폐지 반영)
            if not data.empty:
                last_data_date = data['timestamp'].iloc[-1]  # 마지막 데이터의 날짜
                date_range = pd.date_range(start=last_data_date + pd.Timedelta(days=1), end="2024-12-31", freq='D')
            else:
                date_range = pd.date_range(start="2017-01-01", end="2024-12-31", freq='D')  # 데이터가 없을 경우
    
            # 빈 데이터 생성
            empty_data = pd.DataFrame({'timestamp': date_range,
                                       'open': 0,
                                       'high': 0,
                                       'low': 0,
                                       'close': 0,
                                       'volume': 0,
                                       'close_time': 0,
                                       'quote_asset_volume': 0,
                                       'number_of_trades': 0,
                                       'taker_buy_base_asset_volume': 0,
                                       'taker_buy_quote_asset_volume': 0,
                                       'ignore': 0})
    
            final_data = pd.concat([data, empty_data])
    
            # {SYMBOL}_DF에 저장
            symbol_dfs[f'{symbol}_DF'] = final_data
            
        except Exception as e:
            print(f'Error processing {symbol_with_usdt}: {e}')
            # SLACK 메시지 보내기
            post_message(myToken, "#rebalancing", f"{today_date} 에러 발생.")
    print(f"{today_date} 각 코인 가격 업데이트 완료.")
    post_message(myToken, "#rebalancing", f"{today_date} 각 코인 가격 업데이트 완료.")
    
    #print(symbol_dfs['BTC_DF'])
    
    ######## 열 추가하기 #########
    
    # 추가할 열들
    additional_columns = [
        '3_day_ma', '20_day_ma', '30_day_ma', '50_day_ma', '60_day_ma', '100_day_ma', '112_day_ma', '120_day_ma',
        '1_day_price_change', '2_day_price_change', '3_day_price_change', '4_day_price_change', '5_day_price_change',
        '6_day_price_change', '7_day_price_change', 'RSI', 'listing_period'
    ]
    
    
    # 각 symbol에서 데이터 읽어오고 열 추가 및 데이터 입력
    for obj in cleaned_symbols:
        # DF 가져오기
        try:
            df = symbol_dfs[f'{obj}_DF']
        except Exception as e:
            print(f"파일 가져오기 실패: {e}")
            continue
    
        # 추가 열 생성 및 데이터 입력
        for column in additional_columns:
            df[column] = 0  # 초기화
    
        # close 값이 0인 경우 모든 데이터를 0으로 입력
        if df['close'].eq(0).all():
            df.loc[:, additional_columns] = 0
        else:
            # 각 열의 데이터 계산 및 입력
            df['timestamp'] = pd.to_datetime(df['timestamp'])  # timestamp 열을 날짜 형식으로 변환
            for index, row in df.iterrows():
                date_data = df.iloc[:index+1]  # 현재 날짜까지의 데이터 추출
    
                # 3일 이동평균선 계산
                if len(date_data) >= 3:
                    df.at[index, '3_day_ma'] = calculate_3_day_ma(date_data).iloc[-1]
    
                # 20일 이동평균선 계산
                if len(date_data) >= 20:
                    df.at[index, '20_day_ma'] = calculate_20_day_ma(date_data).iloc[-1]
    
                # 30일 이동평균선 계산
                if len(date_data) >= 30:
                    df.at[index, '30_day_ma'] = calculate_30_day_ma(date_data).iloc[-1]
    
                # 50일 이동평균선 계산
                if len(date_data) >= 50:
                    df.at[index, '50_day_ma'] = calculate_50_day_ma(date_data).iloc[-1]
    
                # 60일 이동평균선 계산
                if len(date_data) >= 60:
                    df.at[index, '60_day_ma'] = calculate_60_day_ma(date_data).iloc[-1]
    
                # 100일 이동평균선 계산
                if len(date_data) >= 100:
                    df.at[index, '100_day_ma'] = calculate_100_day_ma(date_data).iloc[-1]
    
                # 112일 이동평균선 계산
                if len(date_data) >= 112:
                    df.at[index, '112_day_ma'] = calculate_112_day_ma(date_data).iloc[-1]
    
                # 120일 이동평균선 계산
                if len(date_data) >= 120:
                    df.at[index, '120_day_ma'] = calculate_120_day_ma(date_data).iloc[-1]
    
                # 1일간 가격 상승률 계산
                if len(date_data) >= 2:
                    df.at[index, '1_day_price_change'] = calculate_1_day_price_change(date_data)
    
                # 2일간 가격 상승률 계산
                if len(date_data) >= 3:
                    df.at[index, '2_day_price_change'] = calculate_2_day_price_change(date_data)
    
                # 3일간 가격 상승률 계산
                if len(date_data) >= 4:
                    df.at[index, '3_day_price_change'] = calculate_3_day_price_change(date_data)
    
                # 4일간 가격 상승률 계산
                if len(date_data) >= 5:
                    df.at[index, '4_day_price_change'] = calculate_4_day_price_change(date_data)
    
                # 5일간 가격 상승률 계산
                if len(date_data) >= 6:
                    df.at[index, '5_day_price_change'] = calculate_5_day_price_change(date_data)
    
                # 6일간 가격 상승률 계산
                if len(date_data) >= 7:
                    df.at[index, '6_day_price_change'] = calculate_6_day_price_change(date_data)
    
                # 7일간 가격 상승률 계산
                if len(date_data) >= 8:
                    df.at[index, '7_day_price_change'] = calculate_7_day_price_change(date_data)
    
                # RSI 계산
                rsi = calculate_rsi(date_data)
                if rsi is not None:
                    df.at[index, 'RSI'] = rsi
    
                # 상장 기간 계산
                listing_period = calculate_listing_period(date_data)
                df.at[index, 'listing_period'] = listing_period
    
                # 고점 대비 하락한 비율 계산
    #            drop_ratio = calculate_drop_from_high(date_data)
    #            if drop_ratio is not None:
    #                df.at[index, 'drop_from_high'] = drop_ratio
       
        ## BTC에 20/50/60/112/120MA_margin 추가
        # BTC에 대한 추가 계산
        if obj == 'BTC':
    
            # 추가할 열들
            additional_columns_btc = [
                '20MA_margin', '50MA_margin', '60MA_margin', '112MA_margin', '120MA_margin'
            ]
    
            # 추가 열 생성 및 데이터 입력
            for column in additional_columns_btc:
                df[column] = 0  # 초기화
    
            # close 값이 0인 경우 모든 데이터를 0으로 입력
            if df['close'].eq(0).all():
                df.loc[:, additional_columns_btc] = 0
            else:
                # 각 열의 데이터 계산 및 입력
                df['timestamp'] = pd.to_datetime(df['timestamp'])  # timestamp 열을 날짜 형식으로 변환
                for index, row in df.iterrows():
                    date_data = df.iloc[:index+1]  # 현재 날짜까지의 데이터 추출
    
                    # 20일 이동평균선 마진 계산
                    if len(date_data) >= 20:
                        df.at[index, '20MA_margin'] = calculate_20MA_margin(date_data)
    
                    # 50일 이동평균선 마진 계산
                    if len(date_data) >= 50:
                        df.at[index, '50MA_margin'] = calculate_50MA_margin(date_data)
    
                    # 60일 이동평균선 마진 계산
                    if len(date_data) >= 60:
                        df.at[index, '60MA_margin'] = calculate_60MA_margin(date_data)
    
                    # 112일 이동평균선 마진 계산
                    if len(date_data) >= 112:
                        df.at[index, '112MA_margin'] = calculate_112MA_margin(date_data)
                    
                    # 120일 이동평균선 마진 계산
                    if len(date_data) >= 120:
                        df.at[index, '120MA_margin'] = calculate_120MA_margin(date_data)
    
        # 수정된 데이터를 DF로 저장
        try:
            symbol_dfs[f'{obj}_DF'] = df
            print(f'Saved {obj}_DF data.')
        except Exception as e:
            print(f"파일 DF 생성 실패: {e}")
            continue
    
    print("모든 코인 부가 정보 업데이트 완료.")
    post_message(myToken, "#rebalancing", f"{today_date} 모든 코인 부가 정보 업데이트 완료.")
    
    ######## DF 생성하고, 매도/매수하기 #########
    
    # BINANCE API 엔드포인트와 키 설정
    API_ENDPOINT = "https://api.binance.com/api/v3"
    # 시작 일자 기입
    stt_date = datetime.now().strftime("%Y-%m-%d")
    # buy_sell_days 생성
    def get_buy_sell_days():
        stt_date = datetime.now()
        # buy&sell days
        buy_sell_days = []
        while stt_date < datetime(2030, 12, 31):
            buy_sell_days.append(stt_date.strftime("%Y-%m-%d"))
            stt_date += timedelta(days=t)
        return buy_sell_days
    
    buy_sell_days = get_buy_sell_days()
    
    # ustt_utc와 uend_utc 설정 2019-01-10부터 가능
    todaydate = datetime.now()
    yester = todaydate - timedelta(days=1)
    yesterdate = yester.strftime("%Y-%m-%d")
    today_date = datetime.now().strftime("%Y-%m-%d")
    
    # S3 클라이언트 생성
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
                              '5_day_price_change', '6_day_price_change', '7_day_price_change', 'RSI', 'listing_period'
                              ]
        coin_list_for_date = pd.concat([coin_list_for_date, pd.DataFrame(columns=additional_columns)], axis=1)
        # 각 파일에서 값 채우기
    
        for index, row in coin_list_for_date.iterrows():
            symbol = row['Symbol']
            df2 = None
            try:
                df2 = symbol_dfs[f'{symbol}_DF']
                
            except KeyError:
                print(f"Warning: {symbol}_DF does not exist.")
    
            if df2 is not None :
                try:
                    # 파일 2의 데이터에서 해당 날짜의 데이터 추출
                    filtered_data = df2[df2['timestamp'] == yesterdate]
                    if not filtered_data.empty:
                        # 필터링된 데이터의 값을 coin_list_for_date의 해당 행에 추가
                        coin_list_for_date.loc[index, additional_columns] = filtered_data.iloc[0][additional_columns].values
                except FileNotFoundError:
                    coin_list_for_date.loc[index, additional_columns] = 0
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
        btc_df = symbol_dfs['BTC_DF']
    
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
        
        post_message(myToken, "#rebalancing", f"* {today_date} 09:00 매수 가이드 ")
        post_message(myToken, "#rebalancing", f"코인 {n}개, {d}day {portion_values[f'{d}_day_price_change']}, RSI {portion_values['RSI']} 기준")
        post_message(myToken, "#rebalancing", f"TICKER  : {coin_list}")
        post_message(myToken, "#rebalancing", f"1day변화 : {coin_1chng_list} [%]")        
        post_message(myToken, "#rebalancing", f"{d}day변화 : {coin_chng_list} [%]")
        post_message(myToken, "#rebalancing", f"7day변화 : {coin_7chng_list} [%]")        
        post_message(myToken, "#rebalancing", f"RSI           : {coin_RSI_list}")
        post_message(myToken, "#rebalancing", f"* BTC 정보")
        post_message(myToken, "#rebalancing", f"1day:{btc_1chng},   3day:{btc_3chng},   7day:{btc_7chng} [%]")
        post_message(myToken, "#rebalancing", f" 20MA 기준 : {btc_20MA} $")
        post_message(myToken, "#rebalancing", f" 50MA 기준 : {btc_50MA} $")
        post_message(myToken, "#rebalancing", f" 60MA 기준 : {btc_60MA} $")
        post_message(myToken, "#rebalancing", f" 112MA 기준 : {btc_112MA} $")
        post_message(myToken, "#rebalancing", f"120MA 기준 : {btc_120MA} $")
        post_message(myToken, "#rebalancing", f" ")        
    
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
                            post_message(myToken, "#rebalancing", f"* {coin_list_y[i]} 어제와 동일한 coin. 매도 미진행.")
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
                        post_message(myToken, "#rebalancing", f"* {coin_list[i]} 어제와 동일한 coin. 매수 미진행.")
            # 오늘의 코인에 어제 코인 할당
            else:
                for i in range(n):
                    coin_list[i] = coin_list_y[i]
                    print("매도/매수날짜에 해당하지 않아 거래 미진행.")
                    post_message(myToken, "#rebalancing", f"* 매도/매수 날짜에 해당하지 않아 거래 미진행.")
    
        # 112MA 음수, 매도 진행
        else:
            for i in range(n):
                sell_coin(coin_list_y[i])
                print("112MA 미만으로, 매수는 진행하지 않음")
                post_message(myToken, "#rebalancing", f"* 112MA 미만으로, 매수는 미진행. ")
                
    
        
        # 데이터프레임에 날짜 추가
        df_final.loc[len(df_final), 'Date'] = today_date
        
        # 각 코인과 가치, USDT, 총 자산 추가
        for i in range(n):
            coin_price = get_coin_price(coin_list[i])
            coin_balance_info = get_coin_balance(coin_list[i])
            df_final.loc[len(df_final)-1, coin_columns[i]] = coin_list[i]
            df_final.loc[len(df_final)-1, coin_value_columns[i]] = round(coin_price * float(coin_balance_info['free']) ,2)
            
        # USDT 값 추가
        usdt_balance_info = get_coin_balance("USDT")
        if usdt_balance_info is not None:
            usdt_balance = round(float(usdt_balance_info['free']),2)
            df_final.loc[len(df_final)-1, 'USDT'] = usdt_balance
        else:
            print("Failed to get USDT balance information.")
        
        # 총 자산 계산 및 추가
        total_asset = round(sum([df_final.loc[len(df_final) - 1, coin_value_columns[i]] for i in range(n)], df_final.loc[len(df_final) - 1, 'USDT']),2)
        df_final.loc[len(df_final) - 1, 'Total_Asset'] = total_asset
        
        # 현 자산 배수 계산
        multiple = round(total_asset / df_final.loc[0, 'Total_Asset'],2)
        df_final.loc[len(df_final) - 1, 'Multiple'] = multiple  
    
        post_message(myToken, "#rebalancing", f" ")      
        post_message(myToken, "#rebalancing", f"거래 {len(df_final)}일차 자산 : {total_asset} $")
        post_message(myToken, "#rebalancing", f"{stt_date} 대비 {multiple}배")
    
    
        # 데이터프레임 출력
        print(df_final)
                
        
        # 어제 코인 리스트에 오늘 코인 할당
        for i in range(n):
            coin_list_y[i] = coin_list[i]
