import csv
import requests
from datetime import datetime
import schedule
import time

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

def save_symbols_to_csv(symbols):
    today_date = datetime.now().strftime('%Y%m%d')
    file_path = f'/Users/hyeonwhan.ko/Desktop/BIGDATA/{today_date}_SYMBOL.csv'
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Symbol'])
        for symbol in symbols:
            writer.writerow([symbol])
    print("모든 심볼이 CSV 파일에 저장되었습니다.")

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
    save_symbols_to_csv(cleaned_symbols)
    myToken = "xoxb-6969697834503-6996943830753-7Xc0scSNRe1xLbYgvS0D5Tz7"
    post_message_to_slack(myToken, "#rebalancing", f"{datetime.now().strftime('%Y%m%d')} 모든 코인 심볼 가져오기 완료.")

# 매일 아침 9시에 실행되도록 예약
schedule.every().day.at("09:00").do(job)

while True:
    schedule.run_pending()
    time.sleep(1)
