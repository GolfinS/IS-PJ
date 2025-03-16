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

st.markdown(
    """
    ยินดีต้อนรับสู่เว็บแอปพลิเคชันสำหรับการทดลองและเรียนรู้เกี่ยวกับ Machine Learning (ML) และ Neural Network (NN)  
    แอปพลิเคชันนี้ถูกออกแบบมาเพื่อให้ผู้ใช้สามารถศึกษาแนวทางการพัฒนาโมเดล AI และทดลองใช้งานโมเดลจริงผ่านระบบอินเทอร์เฟซที่ใช้งานง่าย  
      
    **คุณสมบัติของเว็บแอปพลิเคชันนี้**  
    - อธิบายแนวทางการพัฒนาโมเดล Machine Learning และ Neural Network  
    - ทดลองใช้งานโมเดล Machine Learning และ Neural Network ผ่านอินพุตที่ผู้ใช้กำหนด  
    - รองรับโมเดล Machine Learning อย่างน้อย 2 อัลกอริทึม และ Neural Network ที่สามารถทำงานกับข้อมูลที่กำหนด  
      
    โปรดเลือกเมนูด้านบนเพื่อเริ่มต้นการใช้งาน  
    """
)

st.markdown("---")

