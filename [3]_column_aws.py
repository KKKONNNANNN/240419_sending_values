import os
import pandas as pd
from datetime import datetime
import requests
import schedule
import time
import boto3
import io  # io 모듈 임포트 추가
from binance.client import Client


# Binance API 키 설정
api_key = os.getenv('your-binance-key')
api_secret = os.getenv('your-binance-api-secret-key')
client = Client(api_key, api_secret)

# SLACK 메시지 보내기 함수
def post_message(token, channel, text):
    response = requests.post("https://slack.com/api/chat.postMessage",
                             headers={"Authorization": "Bearer " + token},
                             data={"channel": channel, "text": text}
                             )
    print(response)


myToken = "your-slack-token"
today_date = datetime.now().strftime('%Y%m%d')

# S3 클라이언트 생성
aws_access_key_id = 'your-S3-key'
aws_secret_access_key = 'your-secret-S3-key'
bucket_name = '240419-sending-values'
coin_raw_dir = 'raw-files/COINRAW'  # COINRAW 폴더 경로

s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

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


def calculate_drop_from_high(data):
    if len(data) < 2:
        return None

    current_price = data['close'].iloc[-1]
    max_price_before_current = data['close'].iloc[:-1].max()
    if pd.isnull(max_price_before_current) or max_price_before_current == 0:
        return None

    return (max_price_before_current - current_price) / max_price_before_current * 100

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

def calculate_120MA_margin(data):
        if data['close'].iloc[-1] == 0:
            return 0
        return (data['close'].iloc[-1] - data['120_day_ma'].iloc[-1])


# 추가할 열들
additional_columns = [
    '3_day_ma', '20_day_ma', '30_day_ma', '50_day_ma', '60_day_ma', '100_day_ma', '120_day_ma',
    '1_day_price_change', '2_day_price_change', '3_day_price_change', '4_day_price_change', '5_day_price_change',
    '6_day_price_change', '7_day_price_change',
    'RSI', 'listing_period', 'drop_from_high'
]

def process_add_symbols():
    # S3 버킷에서 파일 목록 가져오기
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=coin_raw_dir)
    
    # 각 symbol의 CSV 파일에서 데이터 읽어오고 열 추가 및 데이터 입력
    for obj in response.get('Contents', []):
        file_name = obj['Key']
        # 파일 이름에서 심볼 추출
        symbol = os.path.splitext(os.path.basename(file_name))[0]
        file_path = os.path.join(coin_raw_dir, os.path.basename(file_name))
    
        # S3에서 파일 가져오기
        try:
            response = s3.get_object(Bucket=bucket_name, Key=file_path)
            # 파일 내용 읽기
            file_content = response['Body'].read().decode('utf-8')
            # 파일 내용을 DataFrame으로 변환
            df = pd.read_csv(io.StringIO(file_content))
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
                date_data = df.loc[:index]  # 현재 날짜까지의 데이터 추출
    
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
                drop_ratio = calculate_drop_from_high(date_data)
                if drop_ratio is not None:
                    df.at[index, 'drop_from_high'] = drop_ratio
    
        # 수정된 데이터를 CSV 파일로 저장하여 다시 S3에 업로드
        try:
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            s3.put_object(Bucket=bucket_name, Key=file_path, Body=csv_buffer.getvalue())
            print(f'Saved {file_path} data to S3 bucket')
        except Exception as e:
            print(f"파일 업로드 실패: {e}")
            continue
    
        ## BTC에 20/50/60/120MA_margin 추가
        # BTC에 대한 추가 계산
        if symbol == 'BTC':
    
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
                    date_data = df.loc[:index]  # 현재 날짜까지의 데이터 추출
    
                    # 20일 이동평균선 마진 계산
                    if len(date_data) >= 20:
                        df.at[index, '20MA_margin'] = calculate_20MA_margin(date_data)
    
                    # 50일 이동평균선 마진 계산
                    if len(date_data) >= 50:
                        df.at[index, '50MA_margin'] = calculate_50MA_margin(date_data)
    
                    # 60일 이동평균선 마진 계산
                    if len(date_data) >= 60:
                        df.at[index, '60MA_margin'] = calculate_60MA_margin(date_data)
    
                    # 120일 이동평균선 마진 계산
                    if len(date_data) >= 120:
                        df.at[index, '120MA_margin'] = calculate_120MA_margin(date_data)
    
        # 수정된 데이터를 CSV 파일로 저장
        df.to_csv(csv_buffer, index=False)
        s3.put_object(Bucket=bucket_name, Key=file_path, Body=csv_buffer.getvalue())
    
    print("모든 코인 부가 정보 업데이트 완료.")
    post_message(myToken, "#rebalancing", f"{today_date} 모든 코인 부가 정보 업데이트 완료.")


# 매일 아침 9시 5분에 실행되도록 예약, 이거 최소 20분 걸린다.
schedule.every().day.at("09:05").do(process_add_symbols)

while True:
    schedule.run_pending()
    time.sleep(1)
