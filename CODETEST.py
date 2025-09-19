#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import math
import pandas as pd
from datetime import datetime, timedelta
from io import StringIO
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.exceptions import ClientError

import requests
from urllib.parse import urlencode
import hashlib
import hmac

from binance.client import Client
import schedule

################################################################
# =========================== 설정 =============================
################################################################

# 최적화 스위치 (#1~#6만)
FAST_MODE = True
MAX_WORKERS = 8           # 병렬 수집 스레드 수 (#2)
DAYS = 200                # klines 조회 일수 (#1)

# S3
S3_BUCKET = "250714-sending-values"
S3_KEY = "raw-files/TOP100(BINANCE).csv"

# Slack
SLACK_CHANNEL = "#rebalancing"

# 전략 파라미터 (원하는 값으로 바꿔 사용)
n = 2   # 매수/매도 할 코인 개수
d = 2   # d_day_price_change
t = 2   # 매수/매도 주기(일)
m = 100 # TOP 랭킹 상위 m만 작업 (#4)

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

API_ENDPOINT = "https://api.binance.com/api/v3"

# 이전 보유 코인 목록(전역). 처음엔 비어 있음; 길이는 n에 맞춤
coin_list_y = [None] * n

################################################################
# ========================= 유틸/포맷 ==========================
################################################################

def today_str():
    return datetime.now().strftime("%Y-%m-%d")

def yesterday_str():
    return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

def load_data():
    with open('data.json', 'r') as f:
        return json.load(f)

def post_message(token, channel, text):
    try:
        r = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": "Bearer " + token},
            data={"channel": channel, "text": text},
            timeout=10,
        )
        print("Slack:", r.status_code, r.text[:200])
    except Exception as e:
        print(f"Slack post failed: {e}")

def sanitize_symbol(sym: str) -> str:
    if sym is None:
        return ""
    s = str(sym).strip().upper()
    s = re.sub(r'[^A-Z0-9]', '', s)
    return s

################################################################
# ============== Binance 심볼/시세/서명/잔고 캐시 ===============
################################################################

_VALID_USDT_CACHE = None

def fetch_exchange_info():
    r = requests.get(API_ENDPOINT + "/exchangeInfo", timeout=15)
    r.raise_for_status()
    return r.json()

def _get_valid_usdt_pairs():
    info = fetch_exchange_info()
    return {s['symbol'] for s in info['symbols'] if s.get('status') == 'TRADING'}

def ensure_pair_exists(coin: str) -> str:
    global _VALID_USDT_CACHE
    c = sanitize_symbol(coin)
    if not c:
        raise ValueError("Empty/invalid coin symbol")
    pair = c + "USDT"
    if _VALID_USDT_CACHE is None:
        _VALID_USDT_CACHE = _get_valid_usdt_pairs()
    if pair not in _VALID_USDT_CACHE:
        raise ValueError(f"Unsupported or invalid pair: {pair}")
    return pair

def generate_signature(params, api_secret):
    query_string = urlencode(params)
    signature = hmac.new(api_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
    return signature

def fetch_all_prices():
    # (#5) 전체 시세 한 번에
    r = requests.get(API_ENDPOINT + "/ticker/price", timeout=15)
    r.raise_for_status()
    return r.json()

def get_account_balances(api_key, api_secret):
    # (#5) 잔고 한 번만
    params = {"timestamp": int(time.time() * 1000)}
    signature = generate_signature(params, api_secret)
    params['signature'] = signature
    headers = {"X-MBX-APIKEY": api_key}
    r = requests.get(API_ENDPOINT + "/account", headers=headers, params=params, timeout=15)
    r.raise_for_status()
    js = r.json()
    return {b['asset']: float(b.get('free', 0) or 0) for b in js.get('balances', [])}

################################################################
# ========================= 수학/지표 ===========================
################################################################

def safe_div(a, b):
    try:
        if b is None or b == 0 or (hasattr(b, "__float__") and pd.isna(b)):
            return 0.0
        return a / b
    except Exception:
        return 0.0

def safe_rel(ma, close):
    return safe_div(ma - close, close)

def calculate_n_score(row, portion_values, d):
    close = row.get('close', 0)
    return (
        (row.get(f'{d}_day_price_change', 0) / 100) * portion_values[f'{d}_day_price_change'] +
        safe_rel(row.get('3_day_ma', 0),   close) * portion_values['3_day_ma'] +
        safe_rel(row.get('20_day_ma', 0),  close) * portion_values['20_day_ma'] +
        safe_rel(row.get('30_day_ma', 0),  close) * portion_values['30_day_ma'] +
        safe_rel(row.get('50_day_ma', 0),  close) * portion_values['50_day_ma'] +
        safe_rel(row.get('60_day_ma', 0),  close) * portion_values['60_day_ma'] +
        safe_rel(row.get('100_day_ma', 0), close) * portion_values['100_day_ma'] +
        safe_rel(row.get('120_day_ma', 0), close) * portion_values['120_day_ma'] +
        (row.get('RSI', 0) / 100) * portion_values['RSI'] +
        ((760 - row.get('listing_period', 760)) / 760) * portion_values['listing_period']
    )

def compute_indicators_vectorized(df: pd.DataFrame):
    """
    (#3) 지표 벡터화 계산
    """
    if df.empty:
        cols = ['3_day_ma','20_day_ma','30_day_ma','50_day_ma','60_day_ma','100_day_ma','112_day_ma','120_day_ma',
                '1_day_price_change','2_day_price_change','3_day_price_change','4_day_price_change','5_day_price_change',
                '6_day_price_change','7_day_price_change','RSI','listing_period']
        for c in cols: df[c] = 0
        return df

    close = pd.to_numeric(df['close'], errors='coerce').fillna(0.0)
    df['close'] = close
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')

    # 이동평균
    df['3_day_ma']   = close.rolling(3).mean()
    df['20_day_ma']  = close.rolling(20).mean()
    df['30_day_ma']  = close.rolling(30).mean()
    df['50_day_ma']  = close.rolling(50).mean()
    df['60_day_ma']  = close.rolling(60).mean()
    df['100_day_ma'] = close.rolling(100).mean()
    df['112_day_ma'] = close.rolling(112).mean()
    df['120_day_ma'] = close.rolling(120).mean()

    # 변화율
    s = close
    df['1_day_price_change'] = (s - s.shift(1)) / s.shift(1) * 100
    df['2_day_price_change'] = (s - s.shift(3)) / s.shift(3) * 100
    df['3_day_price_change'] = (s - s.shift(4)) / s.shift(4) * 100
    df['4_day_price_change'] = (s - s.shift(5)) / s.shift(5) * 100
    df['5_day_price_change'] = (s - s.shift(6)) / s.shift(6) * 100
    df['6_day_price_change'] = (s - s.shift(7)) / s.shift(7) * 100
    df['7_day_price_change'] = (s - s.shift(8)) / s.shift(8) * 100

    # RSI (14)
    delta = s.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 상장 기간(일)
    df['listing_period'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.days

    df.fillna(0, inplace=True)
    return df

################################################################
# ======================== 주문/가격 유틸 =======================
################################################################

def floor_to_decimal(number, decimal_places):
    factor = 10 ** decimal_places
    return int(number * factor) / factor

def floor_to_n(number, n):
    divided = number // (10 ** n)
    return divided * (10 ** n)

def get_10n(number):
    if number == 0: return 0
    if number > 0:
        exponent = 0
        while number >= 10:
            number /= 10
            exponent += 1
        return exponent
    return 0

def get_coin_price_from_cache(coin, price_map):
    pair = ensure_pair_exists(coin)
    return float(price_map.get(pair, 0.0))

def create_order(symbol_pair, side, quantity, api_key, api_secret):
    # 실거래 주의. 테스트 때는 return으로 막아두세요.
    if not symbol_pair.endswith("USDT"):
        symbol_pair = ensure_pair_exists(symbol_pair.replace("USDT", ""))
    params = {
        "symbol": symbol_pair,
        "side": side,
        "type": "MARKET",
        "quantity": quantity,
        "timestamp": int(time.time() * 1000)
    }
    signature = generate_signature(params, api_secret)
    params['signature'] = signature
    headers = {"X-MBX-APIKEY": api_key}
    r = requests.post(API_ENDPOINT + "/order", headers=headers, params=params, timeout=15)
    return r.json()

def buy_coin(coin, ratio, api_key, api_secret, balances, price_map):
    price = get_coin_price_from_cache(coin, price_map)
    usdt_free = float(balances.get('USDT', 0.0))
    buy_usdt = round(usdt_free * ratio * 0.999, 3)

    if buy_usdt <= 10:
        return f"* 보유 USDT : {round(usdt_free, 2)}, 비율 {ratio:.3f} → 10$ 미만이라 매수 미진행."

    dec = get_10n(price)
    qty = floor_to_decimal(buy_usdt / price, dec) if dec > 0 else floor_to_decimal(buy_usdt / price, 0)

    pair = ensure_pair_exists(coin)
    info = create_order(pair, "BUY", str(qty), api_key, api_secret)
    if "msg" in info:
        return f"거래 중 오류 발생 : {info['msg']}"
    return f"{pair} {qty}개 BUY 정상 진행"

def sell_coin(coin, api_key, api_secret, balances, price_map):
    price = get_coin_price_from_cache(coin, price_map)
    free_amt = float(balances.get(coin, 0.0))
    value = price * free_amt

    if value <= 10:
        return f"* {coin}의 가치가 10$ 미만으로 매도 미진행."

    dec = get_10n(price)
    qty = floor_to_decimal(free_amt * 0.999, dec) if dec > 0 else floor_to_n(free_amt * 0.999, -dec)

    pair = ensure_pair_exists(coin)
    info = create_order(pair, "SELL", str(qty), api_key, api_secret)
    if "msg" in info:
        return f"거래 중 오류 발생 : {info['msg']}"
    return f"{pair} {qty}개 SELL 정상 진행"

################################################################
# ============================ 메인 =============================
################################################################

def get_buy_sell_days():
    base = datetime.now()
    days = []
    while base < datetime(2030, 12, 31):
        days.append(base.strftime("%Y-%m-%d"))
        base += timedelta(days=t)
    return days

def fetch_klines_df(symbol, client, end_date_str):
    # (#1) 최근 DAYS일만
    pair = ensure_pair_exists(symbol)
    kl = client.get_historical_klines(pair, Client.KLINE_INTERVAL_1DAY, f"{DAYS} day ago UTC", end_date_str)
    df = pd.DataFrame(
        kl,
        columns=['timestamp','open','high','low','close','volume','close_time',
                 'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
                 'taker_buy_quote_asset_volume','ignore']
    )
    if df.empty:
        df = pd.DataFrame(columns=['timestamp','open','high','low','close','volume','close_time',
                                   'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
                                   'taker_buy_quote_asset_volume','ignore'])
    df = compute_indicators_vectorized(df)  # (#3)
    return symbol, df

def jjobs():
    global coin_list_y  # 전역 사용

    loaded = load_data()
    slack_token = loaded['STOKEN']
    api_key = loaded['B_API']
    api_secret = loaded['B_SEC']
    client = Client(api_key, api_secret)

    today_date = today_str()
    yesterdate = yesterday_str()
    end_date_str = (datetime.now() - timedelta(days=1)).strftime("%d %b, %Y")

    lines = []  # (#6) Slack 묶음 전송 버퍼

    try:
        # S3에서 CSV 로드 (#7 제외)
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
        data = obj['Body'].read().decode('utf-8')
        top_df = pd.read_csv(StringIO(data))

        # 어제 날짜 행 + TOP m만 (#4)
        coin_list_for_date = top_df.loc[top_df['Date'] == yesterdate, ['Date', 'RANK', 'Symbol']].copy()
        if coin_list_for_date.empty:
            raise RuntimeError(f"{yesterdate} 심볼 목록이 비어 있습니다. CSV 확인 필요.")
        coin_list_for_date['Symbol'] = coin_list_for_date['Symbol'].apply(sanitize_symbol)
        coin_list_for_date = coin_list_for_date[coin_list_for_date['Symbol'] != ""]
        coin_list_for_date = coin_list_for_date.sort_values('RANK').head(m).copy()

        # 바이낸스 교집합
        exch = fetch_exchange_info()
        all_symbols = [s['symbol'] for s in exch['symbols']]
        base_clean = []
        for sym in all_symbols:
            if sym.endswith('USDT'):
                coin = sym[:-4]
                if coin.endswith('DOWN') or coin.endswith('BEAR') or coin.endswith('BULL'):
                    coin = coin[:-4]
                elif coin.endswith('UP') and coin != 'JUP':
                    coin = coin[:-2]
                base_clean.append(sanitize_symbol(coin))
        base_clean = sorted(set([x for x in base_clean if x]))
        valid_syms = sorted(set(base_clean) & set(coin_list_for_date['Symbol']))

        lines.append(f"{today_date} 코인 목록 가져오기 완료. ({len(valid_syms)}개)")

        # 심볼별 klines 병렬 수집 (#2) + 지표 벡터화(#3)
        symbol_dfs = {}
        if FAST_MODE:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futs = [ex.submit(fetch_klines_df, s, client, end_date_str) for s in valid_syms]
                for fut in as_completed(futs):
                    try:
                        sym, df = fut.result()
                        symbol_dfs[f'{sym}_DF'] = df
                    except Exception as e:
                        lines.append(f"{today_date} {sym} 데이터 수집 에러: {e}")
        else:
            for sym in valid_syms:
                try:
                    sym, df = fetch_klines_df(sym, client, end_date_str)
                    symbol_dfs[f'{sym}_DF'] = df
                except Exception as e:
                    lines.append(f"{today_date} {sym} 데이터 수집 에러: {e}")

        lines.append(f"{today_date} 각 코인 가격 업데이트 완료.")

        # 어제자 값 주입
        add_cols = ['open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades',
                    'taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore',
                    '3_day_ma','20_day_ma','30_day_ma','50_day_ma','60_day_ma','100_day_ma','112_day_ma','120_day_ma',
                    '1_day_price_change','2_day_price_change','3_day_price_change','4_day_price_change',
                    '5_day_price_change','6_day_price_change','7_day_price_change','RSI','listing_period']
        for c in add_cols:
            coin_list_for_date[c] = 0

        for idx, row in coin_list_for_date.iterrows():
            sym = row['Symbol']
            df2 = symbol_dfs.get(f'{sym}_DF')
            if df2 is None or df2.empty:
                continue
            mask = df2['timestamp'].dt.strftime("%Y-%m-%d") == yesterdate
            filtered = df2.loc[mask]
            if not filtered.empty:
                coin_list_for_date.loc[idx, add_cols] = filtered.iloc[0][add_cols].values

        # 종가 0 제외 (안전)
        before = len(coin_list_for_date)
        coin_list_for_date = coin_list_for_date[coin_list_for_date['close'] > 0].copy()
        after = len(coin_list_for_date)
        if after == 0:
            raise RuntimeError(f"{yesterdate} 기준 종가 0 심볼만 남아 랭킹 불가. CSV/klines 확인 필요.")
        elif after < before:
            lines.append(f"* 종가 0인 {before - after}개 심볼 제외 후 랭킹 진행 ({after}개 남음)")

        # n_Score 계산/랭킹
        coin_list_for_date['n_Score'] = coin_list_for_date.apply(
            lambda r: calculate_n_score(r, portion_values, d), axis=1)
        coin_list_for_date['n_Score_Rank'] = coin_list_for_date['n_Score'].rank(method='first', ascending=False)

        # 랭킹 상위 n 추출
        coin_list = [None]*n
        coin_close_list = [0]*n
        coin_chng_list  = [0]*n
        coin_1chng_list = [0]*n
        coin_7chng_list = [0]*n
        coin_RSI_list   = [0]*n

        for i in range(n):
            sel = coin_list_for_date.loc[coin_list_for_date['n_Score_Rank'] == i+1]
            if sel.empty: continue
            coin_list[i]       = str(sel['Symbol'].values[0])
            coin_close_list[i] = round(float(sel['close'].values[0]), 1)
            coin_chng_list[i]  = round(float(sel[f'{d}_day_price_change'].values[0]), 1)
            coin_1chng_list[i] = round(float(sel['1_day_price_change'].values[0]), 1)
            coin_7chng_list[i] = round(float(sel['7_day_price_change'].values[0]), 1)
            coin_RSI_list[i]   = round(float(sel['RSI'].values[0]), 1)

        # BTC 지표 표시용(마진은 간단 계산)
        btc_df = symbol_dfs.get('BTC_DF', pd.DataFrame())
        btc_1chng=btc_3chng=btc_7chng=btc_20MA=btc_50MA=btc_60MA=btc_112MA=btc_120MA=0.0
        if not btc_df.empty:
            msk = btc_df['timestamp'].dt.strftime("%Y-%m-%d") == yesterdate
            row = btc_df.loc[msk]
            if not row.empty:
                r = row.iloc[0]
                btc_1chng = round(float(r.get('1_day_price_change', 0)), 1)
                btc_3chng = round(float(r.get('3_day_price_change', 0)), 1)
                btc_7chng = round(float(r.get('7_day_price_change', 0)), 1)
                btc_20MA  = round(float(r.get('close', 0) - r.get('20_day_ma', 0)), 1)
                btc_50MA  = round(float(r.get('close', 0) - r.get('50_day_ma', 0)), 1)
                btc_60MA  = round(float(r.get('close', 0) - r.get('60_day_ma', 0)), 1)
                btc_112MA = round(float(r.get('close', 0) - r.get('112_day_ma', 0)), 1)
                btc_120MA = round(float(r.get('close', 0) - r.get('120_day_ma', 0)), 1)

        lines.append(f"{today_date} 모든 코인 부가 정보 업데이트 완료.")

        # (#5) 가격/잔고 캐시
        all_prices_list = fetch_all_prices()
        price_map = {x['symbol']: float(x['price']) for x in all_prices_list}
        balances = get_account_balances(api_key, api_secret)

        # ====== 매도/매수 로직 (매도 여부와 무관하게 매수 진행) ======
        sell_logs, buy_logs = [], []

        if btc_112MA > 0:
            if today_date in get_buy_sell_days():
                # ---- SELL 단계 ----
                for i in range(n):
                    prev = coin_list_y[i] if i < len(coin_list_y) else None
                    curr = coin_list[i]
                    if prev and curr and prev != curr:
                        msg = sell_coin(prev, api_key, api_secret, balances, price_map)
                        sell_logs.append(msg)
                    else:
                        label = prev if prev else "—"
                        sell_logs.append(f"* {label} 어제와 동일 또는 보유 없음. 매도 미진행.")
            else:
                lines.append(f"* 매도/매수 날짜에 해당하지 않아 매도는 미진행.")
        else:
            # 112MA 이하: 전량 매도 시도(전략 유지)
            for i in range(n):
                prev = coin_list_y[i] if i < len(coin_list_y) else None
                if prev:
                    msg = sell_coin(prev, api_key, api_secret, balances, price_map)
                    sell_logs.append(msg)
            lines.append(f"* 112MA 미만으로, (전략상) 매수는 미진행. ")

        # ---- BUY 단계 (매도 여부와 무관하게 진행) ----
        buy_targets = []
        for i in range(n):
            curr = coin_list[i]
            if not curr:
                continue
            prev = coin_list_y[i] if i < len(coin_list_y) else None

            # 현재 보유량/가치 확인 (수동 매도/첫날 상황 포함)
            curr_qty = float(balances.get(curr, 0.0))
            curr_price = get_coin_price_from_cache(curr, price_map)
            hold_value = curr_qty * curr_price

            need_buy = (prev != curr) or (hold_value <= 10)  # 다른 코인으로 갈아타거나 사실상 미보유면 매수
            if need_buy:
                buy_targets.append(curr)

        if buy_targets:
            ratio = 1 / len(buy_targets)  # 보유 USDT를 균등 분배
            for sym in buy_targets:
                time.sleep(2)  # 잔고 반영 대기
                msg = buy_coin(sym, ratio, api_key, api_secret, balances, price_map)
                buy_logs.append(msg)
        else:
            buy_logs.append("* 새로 매수할 대상이 없습니다.")

        if sell_logs:
            lines.append(":outbox_tray: SELL 결과")
            lines.extend(sell_logs)
        if buy_logs:
            lines.append(":inbox_tray: BUY 결과")
            lines.extend(buy_logs)

        # 자산 요약 (캐시 활용, 안전 인덱싱)
        total_coin_value = 0.0
        for i in range(n):
            curr = coin_list[i]
            prev = coin_list_y[i] if i < len(coin_list_y) else None
            sym = curr or prev
            if not sym:
                continue
            price = get_coin_price_from_cache(sym, price_map)
            qty = float(balances.get(sym, 0.0))  # balances 키는 자산 티커(BTC 등)
            total_coin_value += round(price * qty, 2)

        usdt_bal = float(balances.get("USDT", 0.0))
        total_asset = round(total_coin_value + usdt_bal, 2)
        multiple = 1.00  # 분모 저장 안 하므로 1배 표시

        lines.append(f"거래 요약 자산 : {total_asset} $")
        lines.append(f"{today_date} 기준 {multiple}배")

        # 다음 실행을 위해 오늘 선정 리스트 저장 (길이 n 보장)
        coin_list_y = coin_list[:]

        # Slack 한 번에 전송 (#6)
        post_message(slack_token, SLACK_CHANNEL, "\n".join(lines))

    except Exception as e:
        err = f"An error occurred : {e}"
        print(err)
        try:
            post_message(slack_token, SLACK_CHANNEL, err)
        except Exception:
            pass

# 스케줄 모드 유지 (#9 제외)
schedule.every().day.at("09:01").do(jjobs)

if __name__ == "__main__":
    while True:
        schedule.run_pending()
        time.sleep(1)
