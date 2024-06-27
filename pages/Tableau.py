import streamlit as st
import pandas as pd
import numpy as np
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.colored_header import colored_header

# 페이지 구성 설정
st.set_page_config(layout="wide")

# 세션 상태 초기화
if "page" not in st.session_state:
    st.session_state.page = "Home"

DATA_PATH = "./"
SEED = 42

# 데이터 로드
data = pd.read_csv(f"{DATA_PATH}test_data.csv")


# 타이틀
colored_header(
    label= '🔥화재안전 빅데이터 플랫폼',
    description=None,
    color_name="blue-70",
)



st.title("2023년 지역별 화재 및 기상데이터 시각화")

# Tableau Public에서 대시보드 임베딩
# 여기에 Tableau Public 대시보드 URL을 입력합니다.
tableau_public_url = "https://public.tableau.com/app/profile/jonghyeon.park/viz/_17194904158790/1_1"
tableau_embed_code = f"""
<iframe src="{tableau_public_url}" width="100%" height="800" frameborder="0"></iframe>
"""
st.markdown(tableau_embed_code, unsafe_allow_html=True)

