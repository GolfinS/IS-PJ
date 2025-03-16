import streamlit as st
import pandas as pd
import numpy as np
# from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(
    page_title="Olympics 2028 Prediction",
    page_icon="🏅",
    layout="wide"
)

# Navigation menu
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

st.title("🏅 Olympics 2028 Medal Prediction")

# สร้างข้อมูลจำลอง
@st.cache_data
def create_mock_data():
    # รายชื่อประเทศตัวอย่าง
    countries = [
        'USA', 'CHN', 'GBR', 'JPN', 'AUS', 'ITA', 'GER', 'FRA', 'NED', 'KOR',
        'THA', 'VNM', 'MYS', 'SGP', 'IDN', 'PHL', 'LAO', 'MMR', 'KHM', 'BRN'
    ]
    
    # สร้างข้อมูลประวัติการแข่งขัน
    historical_data = []
    years = list(range(1896, 2025, 4))
    for country in countries:
        for year in years:
            if year >= 1950:  # เริ่มมีข้อมูลจากปี 1950
                gold = np.random.poisson(10 if country in ['USA', 'CHN', 'GBR'] else 3)
                silver = np.random.poisson(8 if country in ['USA', 'CHN', 'GBR'] else 2)
                bronze = np.random.poisson(6 if country in ['USA', 'CHN', 'GBR'] else 2)
                historical_data.append({
                    'NOC': country,
                    'Year': year,
                    'Gold': gold,
                    'Silver': silver,
                    'Bronze': bronze
                })
    
    # สร้างการทำนายปี 2028
    predictions = []
    for country in countries:
        gold_pred = np.random.poisson(15 if country in ['USA', 'CHN', 'GBR'] else 5)
        total_pred = gold_pred + np.random.poisson(10 if country in ['USA', 'CHN', 'GBR'] else 3)
        predictions.append({
            'NOC': country,
            'Predicted_Gold_2028': gold_pred,
            'Predicted_Total_2028': total_pred
        })
    
    return pd.DataFrame(historical_data), pd.DataFrame(predictions)

# โหลดข้อมูลจำลอง
historical_data, predictions_2028 = create_mock_data()

# Show top 10 predictions
st.header("การทำนายเหรียญรางวัลโอลิมปิก 2028")

col1, col2 = st.columns(2)

with col1:
    st.subheader("10 อันดับประเทศที่คาดว่าจะได้เหรียญทองมากที่สุด")
    top_gold = predictions_2028.sort_values('Predicted_Gold_2028', ascending=False).head(10)
    st.dataframe(
        top_gold[['NOC', 'Predicted_Gold_2028']].rename(
            columns={'NOC': 'ประเทศ', 'Predicted_Gold_2028': 'จำนวนเหรียญทอง'}
        ),
        hide_index=True
    )
    
with col2:
    st.subheader("10 อันดับประเทศที่คาดว่าจะได้เหรียญรวมมากที่สุด")
    top_total = predictions_2028.sort_values('Predicted_Total_2028', ascending=False).head(10)
    st.dataframe(
        top_total[['NOC', 'Predicted_Total_2028']].rename(
            columns={'NOC': 'ประเทศ', 'Predicted_Total_2028': 'จำนวนเหรียญรวม'}
        ),
        hide_index=True
    )

# Country specific analysis
st.header("วิเคราะห์รายประเทศ")
selected_country = st.selectbox(
    "เลือกประเทศที่ต้องการวิเคราะห์",
    options=predictions_2028['NOC'].unique()
)

if selected_country:
    col1, col2, col3 = st.columns(3)
    
    country_pred = predictions_2028[predictions_2028['NOC'] == selected_country].iloc[0]
    country_hist = historical_data[historical_data['NOC'] == selected_country]
    
    with col1:
        st.metric(
            "เหรียญทองที่ทำนาย (2028)", 
            f"{int(country_pred['Predicted_Gold_2028'])} 🥇"
        )
        
    with col2:
        st.metric(
            "เหรียญรวมที่ทำนาย (2028)", 
            f"{int(country_pred['Predicted_Total_2028'])} 🏅"
        )
        
    with col3:
        avg_gold = country_hist['Gold'].mean()
        st.metric(
            "ค่าเฉลี่ยเหรียญทองในอดีต",
            f"{avg_gold:.1f} 🥇"
        )

    # Show historical performance
    st.subheader(f"ผลงานในอดีตของ {selected_country}")
    
    # Display historical data as a table
    st.dataframe(
        country_hist[['Year', 'Gold', 'Silver', 'Bronze']].rename(
            columns={
                'Year': 'ปี',
                'Gold': 'เหรียญทอง',
                'Silver': 'เหรียญเงิน',
                'Bronze': 'เหรียญทองแดง'
            }
        ).sort_values('ปี', ascending=False),
        hide_index=True
    )
