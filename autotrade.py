import os
from dotenv import load_dotenv
import pyupbit
from openai import OpenAI
import json
import time
import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volatility import BollingerBands
import requests
from datetime import datetime, timedelta
from youtube_transcript_api import YouTubeTranscriptApi
from pydantic import BaseModel
import sqlite3

load_dotenv()

# 과거 데이터로 반성
def generate_reflection(client, past_trades, current_market_data):
    # 과거 거래 데이터와 현재 시장 데이터를 문자열로 변환
    past_trades_str = json.dumps(past_trades[-3:], default=str)  # 최근 3개의 거래만 사용
    current_market_data_str = json.dumps({
        "ohlcv_data": {
            "daily": current_market_data["ohlcv_data"]["daily"][-5:],  # 최근 5일 데이터만 사용
            "hourly": current_market_data["ohlcv_data"]["hourly"][-12:]  # 최근 12시간 데이터만 사용
        },
        "fear_greed_index": current_market_data["fear_greed_index"],
        "news_headlines": current_market_data["news_headlines"][:3]  # 최근 3개의 뉴스 헤드라인만 사용
    }, default=str)

    response = client.chat.completions.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": "You are an AI trading assistant tasked with analyzing past trading decisions and current market conditions to provide insights and improvements for future trading strategies."},
            {"role": "user", "content": f"Based on these past trades: {past_trades_str} and current market conditions: {current_market_data_str}, provide a brief reflection on past decisions and suggest improvements for future trading. Focus on alignment with Wonyotti's principles and any deviations from them."}
        ],
        max_tokens=500
    )

    return response.choices[0].message.content

def update_reflection_in_db(reflection):
    conn = sqlite3.connect('trading_data.db')
    cursor = conn.cursor()
    cursor.execute('UPDATE trades SET reflection_text = ? WHERE id = (SELECT MAX(id) FROM trades)', (reflection,))
    conn.commit()
    conn.close()

def create_database():
    conn = sqlite3.connect('trading_data.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        decision TEXT,
        percentage INTEGER,
        reason TEXT,
        btc_balance REAL,
        krw_balance REAL,
        btc_avg_buy_price REAL,
        btc_krw_price REAL
    )
    ''')
    conn.commit()
    conn.close()

def insert_trade_data(timestamp, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price):
    conn = sqlite3.connect('trading_data.db')
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO trades (timestamp, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price))
    conn.commit()
    conn.close()


class TradingDecision(BaseModel):
    decision: str
    percentage: int
    reason: str

def get_transcript_text(video_id, source_lang='ko', target_lang='en'):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript([source_lang])
        translated_transcript = transcript.translate(target_lang)
        transcript_data = translated_transcript.fetch()
        text_only = [segment['text'] for segment in transcript_data]
        full_text = ' '.join(text_only)
        return full_text
    except Exception as e:
        print(f"자막 가져오기 실패: {e}")
        return None

def get_youtube_data():
    # 여기에 분석하고자 하는 투자 관련 유튜브 영상 ID 리스트를 추가하세요
    video_ids = ['J-7tPXNz30A']
    transcripts = []
    for video_id in video_ids:
        transcript = get_transcript_text(video_id)
        if transcript:
            transcripts.append(transcript)
    return transcripts

# btc 관련 뉴스 검색해서 가져오기
def get_news_headlines():
    api_key = os.getenv("SERPAPI_API_KEY")
    params = {
        "engine": "google",
        "q": "btc",
        "tbm": "nws",
        "num": 3,
        "api_key": api_key
    }
    
    response = requests.get("https://serpapi.com/search", params=params)
    if response.status_code == 200:
        data = response.json()
        
        headlines = []
        for news_item in data.get('news_results', []):
            title = news_item['title']
            date_str = news_item.get('date')
            
            # 날짜 형식 변환
            if date_str:
                try:
                    # 상대적 시간 표현을 절대적 시간으로 변환
                    if 'ago' in date_str:
                        time_units = {'minute': 60, 'hour': 3600, 'day': 86400}
                        value, unit = date_str.split()[:2]
                        seconds = int(value) * time_units.get(unit.rstrip('s'), 0)
                        date = datetime.now() - timedelta(seconds=seconds)
                    else:
                        # 이미 절대적 시간 형식인 경우
                        date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                    
                    formatted_date = date.strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    formatted_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            else:
                formatted_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            headlines.append({"title": title, "date": formatted_date})
        
        return headlines
    else:
        print(f"Error fetching news: {response.status_code}")
        return []

# Fear and Greed Index API 함수
def get_fear_greed_index():
    try:
        url = "https://api.alternative.me/fng/"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return {
                'value': int(data['data'][0]['value']),
                'value_classification': data['data'][0]['value_classification'],
                'timestamp': int(data['data'][0]['timestamp'])
            }
        else:
            print("Fear and Greed Index API 호출 실패")
            return None
    except Exception as e:
        print(f"Fear and Greed Index API 호출 에러: {e}")
        return None    

# 현재 자산 정보 가져오기
def get_investment_status(upbit):
    try:
      balances = upbit.get_balances()
      krw_balance = next((balance['balance'] for balance in balances if balance['currency'] == 'KRW'), '0')
      btc_balance = next((balance['balance'] for balance in balances if balance['currency'] == 'BTC'), '0')
      return {
          'KRW_balance': float(krw_balance),
          'BTC_balance': float(btc_balance)
      }
    except pyupbit.errors.RemainingReqParsingError:
        print("API 요청 제한 오류 발생.")
        return None
    except Exception as e:
        print(f"OHLCV 데이터 조회 중 오류 발생: {e}")
        return None    

def get_orderbook(ticker="KRW-BTC"):
    try:
      return pyupbit.get_orderbook(ticker)
    except pyupbit.errors.RemainingReqParsingError:
        print("API 요청 제한 오류 발생.")
        return None
    except Exception as e:
        print(f"OHLCV 데이터 조회 중 오류 발생: {e}")
        return None
    
# ohlcv 데이터 가져오기(pyupbit 사용)
def get_ohlcv_data(ticker="KRW-BTC"):
    try:
        daily_data = pyupbit.get_ohlcv(ticker, interval="day", count=10)  # 30일에서 10일로 변경
        hourly_data = pyupbit.get_ohlcv(ticker, interval="minute60", count=12)  # 24시간에서 12시간으로 변경

        # 일별 데이터에 기술적 지표 추가
        daily_data = add_selected_indicators(daily_data)
        # 시간별 데이터에 기술적 지표 추가
        hourly_data = add_selected_indicators(hourly_data)

        # DataFrame을 JSON 직렬화 가능한 형식으로 변환
        daily_data_dict = daily_data.reset_index().to_dict(orient='records')
        hourly_data_dict = hourly_data.reset_index().to_dict(orient='records')

        return {
            'daily': daily_data_dict,
            'hourly': hourly_data_dict
        }
    except pyupbit.errors.RemainingReqParsingError:
        print("API 요청 제한 오류 발생. 잠시 후 재시도합니다.")
        return get_ohlcv_data(ticker)
    except Exception as e:
        print(f"OHLCV 데이터 조회 중 오류 발생: {e}")
        return None

def add_selected_indicators(df):
    # SMA (Simple Moving Average)
    sma_indicator = SMAIndicator(close=df['close'], window=20)
    df['sma_20'] = sma_indicator.sma_indicator()
    
    # EMA (Exponential Moving Average)
    ema_indicator = EMAIndicator(close=df['close'], window=20)
    df['ema_20'] = ema_indicator.ema_indicator()
    
    # MACD (Moving Average Convergence Divergence)
    macd = MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # RSI (Relative Strength Index)
    rsi_indicator = RSIIndicator(close=df['close'])
    df['rsi'] = rsi_indicator.rsi()
    
    # Bollinger Bands
    bollinger = BollingerBands(close=df['close'])
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    df['bb_mid'] = bollinger.bollinger_mavg()
    
    return df
    
### 메인 함수 ###
def ai_trading():
    # 데이터베이스 생성 (처음 실행 시에만 필요)
    create_database()

    # 1. 업비트 연결 및 데이터 가져오기
    access = os.getenv("UPBIT_ACCESS_TOKEN")
    secret = os.getenv("UPBIT_SECRET_KEY")
    upbit = pyupbit.Upbit(access, secret)
    
    fear_greed_data = get_fear_greed_index()

    investment_status = get_investment_status(upbit)
    orderbook = get_orderbook()
    ohlcv_data = get_ohlcv_data()
    news_headlines = get_news_headlines()
    # youtube_transcripts = get_youtube_data()

    if not all([investment_status, orderbook, ohlcv_data]):
      print("데이터 조회 실패. 다음 실행을 기다립니다.")
      return
    
    # 최근 반성 내용 가져오기
    conn = sqlite3.connect('trading_data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT reflection_text FROM trades WHERE reflection_text IS NOT NULL ORDER BY id DESC LIMIT 1')
    recent_reflection = cursor.fetchone()
    conn.close()

    recent_reflection_text = recent_reflection[0] if recent_reflection else "No previous reflection available."


    # 2. 오픈AI API 호출하여 예측 결과 받기
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": """You are a Bitcoin expert following the trading strategy of the legendary Korean crypto trader 'Wonyotti'. Always consider his method when analyzing the current situation and making trading decisions. Here's a summary of Wonyotti's key trading principles:

    1. Focus on long-term trends rather than short-term fluctuations.
    2. Use the 20-day moving average as a key indicator for trend direction.
    3. Buy when the price is above the 20-day moving average and sell when it's below.
    4. Consider market sentiment and external factors that might affect Bitcoin price.
    5. Be patient and don't make hasty decisions based on temporary market movements.
    6. Always manage your risk and never invest more than you can afford to lose.

    Analyze the provided data including technical indicators, the Fear and Greed Index, and news headlines. Based on this analysis and Wonyotti's principles, respond with:

    1. A decision (buy, sell, or hold)
    2. If the decision is 'buy', provide a percentage (1-100) of available KRW to use for buying.
    If the decision is 'sell', provide a percentage (1-100) of held BTC to sell.
    If the decision is 'hold', set the percentage to 0.
    3. A reason for your decision, explaining how it aligns with Wonyotti's strategy.

    Ensure that the percentage is an integer between 1 and 100 for buy/sell decisions, and exactly 0 for hold decisions.
    Your percentage should reflect the strength of your conviction in the decision based on the analyzed data and Wonyotti's principles."""
            },
             {"role": "user", "content": f"Consider this recent reflection on past trades: {recent_reflection_text}"},
            {
                "role": "user",
                "content": json.dumps({
                    "investment_status": investment_status,
                    "orderbook": orderbook,
                    "ohlcv_data": ohlcv_data,
                    "fear_greed_index": fear_greed_data,
                    "news_headlines": news_headlines
                }, default=str)
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "trading_decision",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "decision": {"type": "string", "enum": ["buy", "sell", "hold"]},
                        "percentage": {"type": "integer"},
                        "reason": {"type": "string"}
                    },
                    "required": ["decision", "percentage", "reason"],
                    "additionalProperties": False
                }
            }
        },
        max_tokens=4095
    )      
  # 최신 pydantic 메서드 사용
    result = TradingDecision.model_validate_json(response.choices[0].message.content)
    print(f"### AI Decision: {result.decision.upper()} ###")
    print(f"### Reason: {result.reason} ###")


    order_executed = False
    # 3. 판단에 따라 자동매매 실행하기
    if result.decision == "buy":
        krw_balance = investment_status['KRW_balance']
      
        buy_amount = krw_balance * (result.percentage / 100) * 0.9995  # 수수료 고려
        if buy_amount > 5000:
            print(f"### Buy Order Executed: {result.percentage}% of available KRW ###")
            order = upbit.buy_market_order("KRW-BTC", buy_amount)
            if order:
                order_executed = True
            print(order)
        else:
            print("### Buy Order Failed: Insufficient KRW (less than 5000 KRW) ###")
    elif result.decision == "sell":
        btc_balance = investment_status['BTC_balance']
        current_price = pyupbit.get_current_price("KRW-BTC")
        sell_amount = btc_balance * (result.percentage / 100)
        if sell_amount * current_price > 5000:
            print(f"### Sell Order Executed: {result.percentage}% of held BTC ###")
            order = upbit.sell_market_order("KRW-BTC", sell_amount)
            if order:
                order_executed = True
            print(order)
    else:
        print("### Sell Order Failed: Insufficient BTC (less than 5000 KRW worth) ###")

    # 거래 실행 후
    if order_executed:
        # 과거 거래 데이터 가져오기
        conn = sqlite3.connect('trading_data.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM trades ORDER BY id DESC LIMIT 5')
        past_trades = cursor.fetchall()
        conn.close()

        # 현재 시장 데이터 준비
        current_market_data = {
            "ohlcv_data": ohlcv_data,
            "fear_greed_index": fear_greed_data,
            "news_headlines": news_headlines
        }

        # 반성 및 개선점 생성
        reflection = generate_reflection(client, past_trades, current_market_data)

        # 반성 내용을 DB에 저장
        update_reflection_in_db(reflection)

        print("반성 및 개선점이 생성되어 데이터베이스에 저장되었습니다.")
        print(f"반성 내용: {reflection}")
    
    # 투자 상태 조회
    investment_status = get_investment_status(upbit)
    if investment_status is None:
        print("투자 상태 조회 실패. 다음 실행을 기다립니다.")
        return

    # 현재 BTC 가격 조회
    btc_krw_price = pyupbit.get_current_price("KRW-BTC")
    if btc_krw_price is None:
        print("BTC 가격 조회 실패. 다음 실행을 기다립니다.")
        return

    # AI 결정 후 데이터베이스에 저장
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    btc_balance = investment_status['BTC_balance']
    krw_balance = investment_status['KRW_balance']
    btc_avg_buy_price = upbit.get_avg_buy_price("KRW-BTC")

    insert_trade_data(
        timestamp,
        result.decision,
        result.percentage,
        result.reason,
        btc_balance,
        krw_balance,
        btc_avg_buy_price,
        btc_krw_price
    )
    print(f"거래 데이터가 데이터베이스에 저장되었습니다.")




while True:
    ai_trading()
    time.sleep(3600*8)  # 8시간마다 실행