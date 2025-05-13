import streamlit as st
import openai
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import json
import warnings
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 경로 지정
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rcParams["font.family"] = font_name


warnings.filterwarnings('ignore')
/*
# 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['axes.unicode_minus'] = False
*/
# KRX 종목명-티커 매핑
try:
    with open('krx_ticker_map.json', 'r', encoding='utf-8') as f:
        kr_tickers = json.load(f)
except FileNotFoundError:
    st.warning("krx_ticker_map.json 파일을 찾을 수 없습니다.")
    kr_tickers = {}

# OpenAI 설정
openai.api_key = "3p1vX5a5zu1nTmEdd0lxhT1E0lpkNKq2vmUif4GrGv0eRa1jV7rHJQQJ99BCACHYHv6XJ3w3AAAAACOGR64o"
openai.api_base = "https://ai-jhs51470758ai014414829313.openai.azure.com/"
openai.api_type = "azure"
openai.api_version = "2023-03-15-preview"

def get_ticker_from_name(stock_name):
    if stock_name in kr_tickers:
        return kr_tickers[stock_name]
    us_tickers = {
        '애플': 'AAPL', '테슬라': 'TSLA', '마이크로소프트': 'MSFT',
        '알파벳': 'GOOGL', '아마존': 'AMZN', '메타': 'META',
        '엔비디아': 'NVDA', '페이팔': 'PYPL', '넷플릭스': 'NFLX', '팔란티어': 'PLTR',
        'AMD': 'AMD', '인텔': 'INTC', 'IBM': 'IBM', '퀄컴': 'QCOM',
    }
    return us_tickers.get(stock_name, None)

def calculate_technical_indicators(stock_symbol):
    data = yf.download(stock_symbol, period="1y", progress=False)
    close = data['Close']
    ma_5 = close.rolling(5).mean()
    ma_20 = close.rolling(20).mean()
    ma_60 = close.rolling(60).mean()
    ma_120 = close.rolling(120).mean()

    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return ma_5.iloc[-1], ma_20.iloc[-1], ma_60.iloc[-1], ma_120.iloc[-1], rsi.iloc[-1]

def get_stock_info(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    info = stock.info
    history = stock.history(period="1y")
    current_price = history['Close'].iloc[-1]
    prev_close = history['Close'].iloc[-2] if len(history) > 1 else current_price
    change_pct = (current_price - prev_close) / prev_close * 100 if prev_close else 0
    ma_5, ma_20, ma_60, ma_120, rsi = calculate_technical_indicators(stock_symbol)

    return {
        'symbol': stock_symbol,
        'name': info.get('shortName', stock_symbol),
        'price': current_price,
        'change_pct': change_pct,
        'market_cap': info.get('marketCap', 0) / 1e12,
        'high_52w': info.get('fiftyTwoWeekHigh'),
        'low_52w': info.get('fiftyTwoWeekLow'),
        'sector': info.get('sector', 'N/A'),
        'industry': info.get('industry', 'N/A'),
        'ma_5': float(ma_5),
        'ma_20': float(ma_20),
        'ma_60': float(ma_60),
        'ma_120': float(ma_120),
        'rsi': float(rsi),
        'history': history
    }

def get_ai_analysis(stock_data):
    prompt = f"""
    {stock_data['name']} ({stock_data['symbol']}) 분석 요청:
    - 현재가: {stock_data['price']:,.0f}원 ({stock_data['change_pct']:+.1f}%)
    - 시가총액: {stock_data['market_cap']:,.1f}조원
    - 52주 범위: {stock_data['low_52w']:,.0f}~{stock_data['high_52w']:,.0f}원
    - 업종: {stock_data['sector']} > {stock_data['industry']}
    - 이동평균: 5일 {stock_data['ma_5']:,.0f}, 20일 {stock_data['ma_20']:,.0f}, 60일 {stock_data['ma_60']:,.0f}, 120일 {stock_data['ma_120']:,.0f}
    - RSI: {stock_data['rsi']:.1f}
    AI 분석 요청:
    - 현재 주가 평가
    - 업종 내 경쟁력
    - 다중 이동평균 분석
    - 종합 투자 의견 (300자 내외)
    """

    try:
        response = openai.ChatCompletion.create(
            engine="gpt-35-turbo",
            messages=[
                {"role": "system", "content": "주식 분석 전문가"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=600
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"AI 분석 실패: {str(e)}"

def plot_stock_chart(stock_data, stock_name):
    history = stock_data['history']
    close = history['Close']
    ma_5 = close.rolling(5).mean()
    ma_20 = close.rolling(20).mean()
    ma_60 = close.rolling(60).mean()
    ma_120 = close.rolling(120).mean()

    fig, ax = plt.subplots()
    ax.plot(close.index, close, label="종가", color="blue", linewidth=2)
    ax.plot(ma_5.index, ma_5, label="5일", color="red")
    ax.plot(ma_20.index, ma_20, label="20일", color="green")
    ax.plot(ma_60.index, ma_60, label="60일", color="orange")
    ax.plot(ma_120.index, ma_120, label="120일", color="purple")
    ax.set_title(f"{stock_name} 주가 차트", fontproperties=font_prop)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# ✅ Streamlit 앱 시작
st.title("📈 ChatJOY AI 주식 분석")

stock_name = st.text_input("분석할 종목명을 입력하세요 (예: 삼성전자)")
if st.button("분석 시작"):
    ticker = get_ticker_from_name(stock_name)
    if not ticker:
        st.error("❌ 종목명을 찾을 수 없습니다.")
    else:
        with st.spinner("데이터 조회 중..."):
            data = get_stock_info(ticker)
        st.subheader("📊 기본 정보")
        st.text(f"{data['name']} ({ticker})\n"
                f"현재가: {data['price']:,.0f}원 ({data['change_pct']:+.1f}%)\n"
                f"시가총액: {data['market_cap']:,.1f}조원\n"
                f"52주 고가: {data['high_52w']:,.0f}원\n"
                f"52주 저가: {data['low_52w']:,.0f}원\n"
                f"RSI: {data['rsi']:.1f}")

        st.subheader("🤖 AI 분석")
        analysis = get_ai_analysis(data)
        st.text(analysis)

        st.subheader("📈 주가 차트")
        fig = plot_stock_chart(data, stock_name)
        st.pyplot(fig)
