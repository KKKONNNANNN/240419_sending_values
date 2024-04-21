import csv
import requests
from datetime import datetime
import schedule
import time
import boto3
import io

def get_all_symbols():
    url = "https://api.binance.com/api/v3/exchangeInfo"
    response = requests.get(url)
    data = response.json()
    symbols = [symbol_info['symbol'] for symbol_info in data['symbols']]
    return symbols

def clean_symbols(symbols):
    cleaned_symbols = []
    for symbol in symbols:
        if symbol[-4:] == 'USDT':
            cleaned_symbols.append(symbol[:-4])
    return cleaned_symbols

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

def save_symbols_to_s3(symbols):
    today_date = datetime.now().strftime('%Y%m%d')
    file_name = 'raw-files/SYMBOL.csv'
    s3_bucket_name = '240419-sending-values'

    # Access Key ID와 Secret Access Key 설정
    aws_access_key_id = 'access_key'
    aws_secret_access_key = 'secret_access_key'

    # 새로운 세션 생성
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    # 이 세션을 사용하여 S3 클라이언트 생성
    s3 = session.client('s3')

    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(['Symbol'])
    for symbol in symbols:
        writer.writerow([symbol])
    s3.put_object(Bucket=s3_bucket_name, Key=file_name, Body=csv_buffer.getvalue())
    print("모든 심볼이 S3에 저장되었습니다.")

def post_message_to_slack(token, channel, text):
    response = requests.post("https://slack.com/api/chat.postMessage",
                             headers={"Authorization": "Bearer " + token},
                             data={"channel": channel, "text": text})
    print(response)

def job():
    all_symbols = get_all_symbols()
    cleaned_symbols = clean_symbols(all_symbols)
    cleaned_symbols = clean_symbols2(cleaned_symbols)
    cleaned_symbols = sorted(set(cleaned_symbols) - {''})
    save_symbols_to_s3(cleaned_symbols)
    myToken = "my-slack-token"
    post_message_to_slack(myToken, "#rebalancing", f"{datetime.now().strftime('%Y%m%d')} 모든 코인 심볼 가져오기 완료.")

schedule.every().day.at("09:00").do(job)

while True:
    schedule.run_pending()
    time.sleep(1)
