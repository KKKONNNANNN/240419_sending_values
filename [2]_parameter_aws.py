import os
import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta
import requests
import schedule
import time
import boto3
import io  # io 모듈 임포트 추가

# Binance API 키 설정
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET_KEY')
client = Client(api_key, api_secret)

# SLACK 메시지 보내기 함수
def post_message(token, channel, text):
    response = requests.post("https://slack.com/api/chat.postMessage",
                             headers={"Authorization": "Bearer " + token},
                             data={"channel": channel, "text": text}
                             )
    print(response)


myToken = "your-slack-token"


# 각 심볼에 대해 빈 데이터를 채워넣고 저장
def update_coin_data():
    # S3 클라이언트 생성
    aws_access_key_id = 'AWS_ACCESS_KEY'
    aws_secret_access_key = 'AWS_SECRET_ACCESS_KEY'
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    
    # CSV 파일 경로
    today_date = datetime.now().strftime('%Y%m%d')
    
    # S3 버킷 이름과 파일 경로 설정
    bucket_name = '240419-sending-values'
    file_key = 'raw-files/SYMBOL.csv'
    
    # S3에서 파일 가져오기
    try:
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        # 파일 내용 읽기
        file_content = response['Body'].read().decode('utf-8')
        # 파일 내용을 DataFrame으로 변환
        df = pd.read_csv(io.StringIO(file_content))
    except Exception as e:
        print(f"파일 가져오기 실패: {e}")
    
    for symbol in df['Symbol']:
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
    
            # 파일 이름 설정
            output_file_key = f'raw-files/COINRAW/{symbol}.csv'
    
            # DataFrame을 CSV 형식으로 변환하여 바이너리로 인코딩
            csv_buffer = io.StringIO()
            final_data = pd.concat([data, empty_data])
            final_data.to_csv(csv_buffer, index=False)
    
            # S3에 업로드
            s3.put_object(Bucket=bucket_name, Key=output_file_key, Body=csv_buffer.getvalue())
            print(f'Saved {symbol_with_usdt} data to S3 bucket with key: {output_file_key}')
    
        except Exception as e:
            print(f'Error processing {symbol_with_usdt}: {e}')
            # SLACK 메시지 보내기
            post_message(myToken, "#rebalancing", f"{today_date} 에러 발생.")
    
    post_message(myToken, "#rebalancing", f"{today_date} 각 코인 가격 업데이트 완료.")

# schedule을 사용하여 매일 아침 09:00에 함수 실행
schedule.every().day.at("09:00").do(update_coin_data)

while True:
    schedule.run_pending()
    time.sleep(1)
