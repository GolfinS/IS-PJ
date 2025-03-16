import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    layout="wide"
)

# สร้างเมนูนำทาง
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

st.title("Web Application for Machine Learning and Neural Network")

# st.subheader("เว็บแอพพลิเคชันนี้ถูกพัฒนาขึ้นเพื่อทดลองการทำงานของโมเดล Machine Learning และ Neural Network")
# st.markdown("""
# - Machine Learning (ML)
# - Neural Network (NN)
# """)
            


st.markdown("---")
