import streamlit as st
import sqlite3
import pandas as pd

# 데이터베이스 연결
conn = sqlite3.connect('trading_data.db')
cursor = conn.cursor()

# Streamlit 앱 제목
st.title('거래 데이터 대시보드')

# 데이터베이스에서 데이터 가져오기
query = "SELECT * FROM trades"
df = pd.read_sql_query(query, conn)

# 데이터프레임 표시
st.dataframe(df)

# 기본 통계 정보 표시
st.subheader('기본 통계 정보')
st.write(df.describe())

# 결정별 거래 수 차트
st.subheader('결정별 거래 수')
decision_counts = df['decision'].value_counts()
st.bar_chart(decision_counts)

# KRW 잔액과 BTC 평가금액의 합 계산
df['btc_value_in_krw'] = df['btc_balance'] * df['btc_krw_price']
df['total_value'] = df['krw_balance'] + df['btc_value_in_krw']

# 총 자산 가치 변화 라인 차트
st.subheader('총 자산 가치 변화 (KRW 잔액 + BTC 평가금액)')
chart_data = df[['timestamp', 'total_value']].set_index('timestamp')
st.line_chart(chart_data)

# 데이터베이스 연결 종료
conn.close()