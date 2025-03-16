import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Mock Model class
class MockModel:
    def predict(self, X):
        # สร้างการทำนายแบบสุ่มแต่มี bias ไปทางบวกเล็กน้อย
        return np.random.random(size=(len(X), 1)) * 0.5 + 0.3

# Mock Scaler class
class MockScaler:
    def transform(self, X):
        # แค่ return ข้อมูลเดิม เพราะไม่จำเป็นต้อง scale จริงๆ
        return X

st.set_page_config(
    page_title="Neural Network Demo",
    page_icon="🎯",
    layout="wide"
)

# Create navigation menu
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.page_link("app.py", label="Home")
with col2:
    st.page_link("pages/1_ml_ex.py", label="ML")
with col3:
    st.page_link("pages/2_nn_ex.py", label="NN")
with col4:
    st.page_link("pages/3_demo_ml.py", label="ML Demo")
with col5:
    st.page_link("pages/4_demo_nn.py", label="NN Demo")

st.title("Stock Price Prediction Demo")

# Load mock model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = MockModel()
    scaler = MockScaler()
    return model, scaler

model, scaler = load_model_and_scaler()

# Load sample data
@st.cache_data
def load_sample_data():
    try:
        df = pd.read_csv('data/raw/googl_data_2020_2025.csv', skiprows=[1, 2])
        df['Date'] = pd.to_datetime(df.index)
        return df
    except:
        # ถ้าไม่มีไฟล์ข้อมูลจริง สร้างข้อมูลจำลอง
        dates = pd.date_range(start='2020-01-01', end='2025-12-31', freq='B')
        df = pd.DataFrame({
            'Date': dates,
            'Open': np.random.normal(100, 10, len(dates)),
            'High': np.random.normal(105, 10, len(dates)),
            'Low': np.random.normal(95, 10, len(dates)),
            'Close': np.random.normal(102, 10, len(dates)),
            'Volume': np.random.normal(1000000, 200000, len(dates))
        })
        return df

df_sample = load_sample_data()

st.header("Google Stock Price Prediction")
st.write("ใส่ข้อมูลเพื่อทำนายทิศทางราคาหุ้น Google ในวันถัดไป")

# Create features
def create_features(df):
    # แปลง Date เป็น datetime ก่อน
    df['Date'] = pd.to_datetime(df['Date'])
    
    df['Returns'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['Volatility'] = df['Close'].rolling(window=20).std()
    df['High_Low_Spread'] = df['High'] - df['Low']
    df['Open_Close_Spread'] = df['Close'] - df['Open']
    df['Daily_Return'] = df['Close'].pct_change()
    df['Previous_Close'] = df['Close'].shift(1)
    df['MA5_Cross'] = (df['Close'] > df['MA5']).astype(int)
    df['MA20_Cross'] = (df['Close'] > df['MA20']).astype(int)
    df['MA50_Cross'] = (df['Close'] > df['MA50']).astype(int)
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    return df

# Create form for input data
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        date = st.date_input(
            "วันที่",
            datetime.now().date()
        )
        
        open_price = st.number_input("ราคาเปิด (Open)", min_value=0.0, value=100.0, step=0.01)
        high_price = st.number_input("ราคาสูงสุด (High)", min_value=0.0, value=105.0, step=0.01)
        low_price = st.number_input("ราคาต่ำสุด (Low)", min_value=0.0, value=95.0, step=0.01)
        close_price = st.number_input("ราคาปิด (Close)", min_value=0.0, value=102.0, step=0.01)
    
    with col2:
        volume = st.number_input("ปริมาณการซื้อขาย (Volume)", min_value=0, value=1000000)
        
        # Show last 20 days average prices from sample data
        st.write("ค่าเฉลี่ยย้อนหลัง 20 วัน:")
        last_20_days = df_sample.tail(20)
        st.write(f"ราคาเฉลี่ย: ${last_20_days['Close'].mean():.2f}")
        st.write(f"ปริมาณเฉลี่ย: {int(last_20_days['Volume'].mean()):,}")
    
    submitted = st.form_submit_button("ทำนาย")
    
    if submitted:
        # Create a DataFrame with the last 20 days of data plus the new day
        input_data = pd.DataFrame({
            'Date': [date],
            'Open': [open_price],
            'High': [high_price],
            'Low': [low_price],
            'Close': [close_price],
            'Volume': [volume]
        })
        
        # Combine with last 19 days from sample data
        historical_data = df_sample.tail(19)[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        combined_data = pd.concat([historical_data, input_data]).reset_index(drop=True)
        
        # Create features
        combined_data = create_features(combined_data)
        
        # Get features in correct order
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'MA5', 'MA20', 'MA50', 
                   'Volatility', 'High_Low_Spread', 'Open_Close_Spread', 'Daily_Return', 
                   'Previous_Close', 'MA5_Cross', 'MA20_Cross', 'MA50_Cross', 'DayOfWeek', 
                   'Month', 'IsWeekend']
        
        # Scale the features
        X = combined_data[features].values
        X_scaled = scaler.transform(X)
        
        # Create sequence
        X_seq = np.array([X_scaled])
        
        # Make prediction
        prediction = model.predict(X_seq)
        probability = prediction[0][0]
        
        st.subheader("ผลการทำนาย")
        
        # Show prediction probability
        if probability > 0.5:
            st.success(f"ทำนายว่าราคาจะ **เพิ่มขึ้น** ในวันถัดไป (ความน่าจะเป็น: {probability:.2%})")
        else:
            st.error(f"ทำนายว่าราคาจะ **ลดลง** ในวันถัดไป (ความน่าจะเป็น: {(1-probability):.2%})")
        
        
        importance_factors = []
        if combined_data['Close'].iloc[-1] > combined_data['MA5'].iloc[-1]:
            importance_factors.append("- ราคาปิดอยู่เหนือค่าเฉลี่ย 5 วัน")
        if combined_data['Close'].iloc[-1] > combined_data['MA20'].iloc[-1]:
            importance_factors.append("- ราคาปิดอยู่เหนือค่าเฉลี่ย 20 วัน")
        if combined_data['Volume'].iloc[-1] > combined_data['Volume'].mean():
            importance_factors.append("- ปริมาณการซื้อขายสูงกว่าค่าเฉลี่ย")
        if combined_data['Volatility'].iloc[-1] > combined_data['Volatility'].mean():
            importance_factors.append("- ความผันผวนของราคาสูงกว่าค่าเฉลี่ย")
        
        for factor in importance_factors:
            st.write(factor)

    