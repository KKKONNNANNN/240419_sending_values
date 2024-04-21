import os
import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta
import requests
import schedule
import time
import boto3

# Binance API 키 설정
api_key = os.getenv('EkXMdOEzwJhYKIum3LRq4fF3HuY6vO3pUEqacLdnIdCxQ0vbrtVBxKkFpr8579TY')
api_secret = os.getenv('RkGGC4u3BbI3QIIZQl6Tq3HLk9kL1hRwIguvYr58ytfbWcxmsRxVFl1ozwsEjxfp')
client = Client(api_key, api_secret)

# SLACK 메시지 보내기 함수
def post_message(token, channel, text):
    response = requests.post("https://slack.com/api/chat.postMessage",
                             headers={"Authorization": "Bearer " + token},
                             data={"channel": channel, "text": text}
                             )
    print(response)
myToken = "my_slack_token"

# S3 클라이언트 생성
aws_access_key_id = 'Your_Access_Key_ID'
aws_secret_access_key = 'Your_Secret_Access_Key'
s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

# 각 심볼에 대해 빈 데이터를 채워넣고 저장
def process_symbols():
    
    # CSV 파일 경로
    today_date = datetime.now().strftime('%Y%m%d')
    input_csv_path = r'raw-files/SYMBOL.csv'
    output_dir = r'raw-files/COINRAW'

    # S3 버킷 이름과 파일 경로 설정
    bucket_name = 'your_bucket_name'
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
        return
    
    for symbol in df['Symbol']:
        try:
            symbol_with_usdt = f'{symbol}USDT'

            # 어제 날짜 구하기
            yesterday_date = datetime.now() - timedelta(days=1)
            yesterday_date_str = yesterday_date.strftime("%d %b, %Y")  # 어제 날짜를 포맷에 맞게 변환

            # 심볼에 대한 가격 데이터 가져오기
            klines = client.get_historical_klines(symbol_with_usdt, Client.KLINE_INTERVAL_1DAY, "1 Jan, 2017", yesterday_date_str)

            # 가격 데이터를 DataFrame으로 변환
            data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

            # timestamp를 날짜 형식으로 변환
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')

            # 마지막으로 데이터가 입력된 날짜 다음 날부터 2024-03-20까지의 날짜 범위 생성(상장 폐지 반영)
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

            # CSV 파일로 저장
            output_csv_path = os.path.join(output_dir, f'{symbol}.csv')
            final_data = pd.concat([data, empty_data])
            final_data.to_csv(output_csv_path, index=False)
            print(f'Saved {symbol_with_usdt} data to {output_csv_path}')
            post_message(myToken, "#rebalancing", f"{today_date} 각 코인 가격 업데이트 완료.")

        except Exception as e:
            print(f'Error processing {symbol_with_usdt}: {e}')
            # SLACK 메시지 보내기
            post_message(myToken, "#rebalancing", f"{today_date} 에러 발생.")

# 매일 아침 9시 1분에 실행되도록 예약
schedule.every().day.at("09:01").do(process_symbols)

while True:
    schedule.run_pending()
    time.sleep(1)
