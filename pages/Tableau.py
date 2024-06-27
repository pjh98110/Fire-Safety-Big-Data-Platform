import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.colored_header import colored_header

# 페이지 구성 설정
st.set_page_config(layout="wide")

# 세션 상태 초기화
if "page" not in st.session_state:
    st.session_state.page = "Home"


# 타이틀
colored_header(
    label= '🔥화재안전 빅데이터 플랫폼',
    description=None,
    color_name="blue-70",
)


st.title("2023년 지역별 화재 및 기상데이터 시각화")
st.write("팝업 차단으로 보이지 않는다면 이 링크로 접속해 주세요. https://public.tableau.com/app/profile/jonghyeon.park/viz/_17194904158790/1_1")

# Tableau Public에서 대시보드 임베딩
# 여기에 Tableau Public 대시보드 URL을 입력합니다.
tableau_public_url = "https://public.tableau.com/app/profile/jonghyeon.park/viz/_17194904158790/1_1"
tableau_embed_code = f"""
<iframe src="{tableau_public_url}" width="100%" height="1000"></iframe>
"""
st.markdown(tableau_embed_code, unsafe_allow_html=True)

