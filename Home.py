import streamlit as st
import pandas as pd
import numpy as np
import random
import os
from streamlit_extras.switch_page_button import switch_page
from st_pages import Page, show_pages
from streamlit_extras.colored_header import colored_header
from datetime import datetime, timedelta
import requests

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import shap
import pickle

# Streamlit의 경우 로컬 환경에서 실행할 경우 터미널 --> (폴더 경로)Streamlit run Home.py로 실행 / 로컬 환경과 스트리밋 웹앱 환경에서 기능의 차이가 일부 있을 수 있음
# 파일 경로를 잘못 설정할 경우 오류가 발생하고 실행이 불가능하므로 파일 경로 수정 필수
# 데이터 파일의 경우 배포된 웹앱 깃허브에서 다운로드 가능함

# 페이지 구성 설정
st.set_page_config(layout="wide")

show_pages(
    [
        Page("Home.py", "기상 요인에 따른 화재위험등급 제공", "🔥"),
        Page("pages/Chatbot.py", "화재위험등급 안내 챗봇", "🤖"),
        Page("pages/Tableau.py", "Tableau", "🖥️"),
        Page("pages/Explainable_AI.py", "Explainable_AI", "📑"),
    ]
)

if "page" not in st.session_state:
    st.session_state.page = "Home"

DATA_PATH = "./"

X = pd.read_csv(f'{DATA_PATH}x_train.csv')
y = pd.read_csv(f'{DATA_PATH}y_train.csv')

# 데이터 샘플링
# X = X.sample(frac=0.2, random_state=42)
# y = y.sample(frac=0.2, random_state=42)

def reset_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

reset_seeds(42)

# 한글 폰트 설정 함수
def set_korean_font():
    font_path = f"{DATA_PATH}NanumGothic.ttf"  # 폰트 파일 경로

    from matplotlib import font_manager, rc
    font_manager.fontManager.addfont(font_path)
    rc('font', family='NanumGothic')

# 한글 폰트 설정 적용
set_korean_font()


# 세션 변수에 저장
if 'type_of_case' not in st.session_state:
    st.session_state.type_of_case = None

if 'selected_district' not in st.session_state:
    st.session_state.selected_district = "서울특별시"

if 'selected_day' not in st.session_state:
    st.session_state.selected_day = datetime.now()

if 'questions' not in st.session_state:
    st.session_state.questions = None

if 'gpt_input' not in st.session_state:
    st.session_state.gpt_input = None

if 'gemini_input' not in st.session_state:
    st.session_state.gemini_input = None   

if 'selected_survey' not in st.session_state:
    st.session_state.selected_survey = []



# 공공데이터 포털 API KEY
API_KEY = st.secrets["secrets"]["WEATHER_KEY"]

# 기상청 API 엔드포인트 URL을 정의
BASE_URL = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst'

# 날짜와 시도 정보를 매핑하는 함수
def weather_info(date, sido):
    # 시도별로 기상청 격자 좌표를 정의
    sido_coordinates = {
        '서울특별시': (60, 127),
        '부산광역시': (98, 76),
        '대구광역시': (89, 90),
        '인천광역시': (55, 124),
        '광주광역시': (58, 74),
        '대전광역시': (67, 100),
        '울산광역시': (102, 84),
        '세종특별자치시': (66, 103),
        '경기도': (60, 120),
        '강원특별자치도': (73, 134),
        '충청북도': (69, 107),
        '충청남도': (68, 100),
        '전북특별자치도': (63, 89),
        '전라남도': (51, 67),
        '경상북도': (91, 106),
        '경상남도': (91, 77),
        '제주특별자치도': (52, 38),
    }

    if sido not in sido_coordinates:
        raise ValueError(f"'{sido}'는 유효한 시도가 아닙니다.")
    
    nx, ny = sido_coordinates[sido]

    params = {
        'serviceKey': API_KEY,
        'pageNo': 1,
        'numOfRows': 1000,
        'dataType': 'JSON',
        'base_date': date,
        'base_time': '0500',  # 05:00 AM 기준
        'nx': nx,
        'ny': ny,
    }

    # 시간대별로 유효한 데이터를 찾기 위한 반복
    valid_times = ['0200', '0500', '0800', '1100', '1400', '1700', '2000', '2300']  # 기상청 단기예보 API 제공 시간
    response_data = None

    for time in valid_times:
        params['base_time'] = time
        response = requests.get(BASE_URL, params=params)
        
        # 응답 상태 코드 확인
        if response.status_code == 200:
            try:
                data = response.json()
                if 'response' in data and 'body' in data['response'] and 'items' in data['response']['body']:
                    response_data = data['response']['body']['items']['item']
                    break  # 유효한 데이터를 찾으면 루프 종료
            except ValueError as e:
                st.error(f"JSON 디코딩 오류: {e}")
                st.text(response.text)
                continue
        else:
            st.error(f"HTTP 오류: {response.status_code}")
            st.text(response.text)
            continue
    
    if response_data:
        df = pd.DataFrame(response_data)
        return df
    else:
        st.error("유효한 데이터를 찾을 수 없습니다.")
        return None

# 오늘 날짜와 1일 전 날짜 계산(기상청에서 최근 3일만 제공)
today = datetime.today()
three_days_ago = today - timedelta(days=1)





# 타이틀
colored_header(
    label= '🔥화재안전 빅데이터 플랫폼',
    description=None,
    color_name="orange-70",
)



# [사이드바]
st.sidebar.markdown(f"""
            <span style='font-size: 20px;'>
            <div style=" color: #000000;">
                <strong>지역 및 날짜 선택</strong>
            </div>
            """, unsafe_allow_html=True)


# 사이드바에서 지역 선택
selected_district = st.sidebar.selectbox(
    "(1) 당신의 지역을 선택하세요:",
    ('서울특별시', '경기도', '부산광역시', '인천광역시', '충청북도', '충청남도', 
     '세종특별자치시', '대전광역시', '전북특별자치도', '전라남도', '광주광역시', 
     '경상북도', '경상남도', '대구광역시', '울산광역시', '강원특별자치도', '제주특별자치도')
)
st.session_state.selected_district = selected_district

# 사이드바에서 날짜 선택
selected_day = st.sidebar.date_input(
    "(2) 오늘의 날짜를 선택하세요:", 
    today, 
    min_value=three_days_ago, 
    max_value=today
).strftime('%Y%m%d')
st.session_state.selected_day = selected_day


# 날짜와 시도의 기상 정보 가져오기
weather_data = weather_info(st.session_state.selected_day, st.session_state.selected_district)


# 특정 시간의 날씨 데이터를 필터링하는 함수
def get_weather_value(df, category, time="0600"):
    row = df[(df['category'] == category) & (df['fcstTime'] == time)]
    return row['fcstValue'].values[0] if not row.empty else None

# 특정 시간의 날씨 데이터 추출
temperature = get_weather_value(weather_data, "TMP")
wind_direction = get_weather_value(weather_data, "VEC")
wind_speed = get_weather_value(weather_data, "WSD")
precipitation_prob = get_weather_value(weather_data, "POP")
precipitation_amount = get_weather_value(weather_data, "PCP")
humidity = get_weather_value(weather_data, "REH")
sky_condition = get_weather_value(weather_data, "SKY")
snow_amount = get_weather_value(weather_data, "SNO")
wind_speed_uuu = get_weather_value(weather_data, "UUU")
wind_speed_vvv = get_weather_value(weather_data, "VVV")

# 범주에 따른 강수량 텍스트 변환 함수
def format_precipitation(pcp):
    try:
        pcp = float(pcp)
        if pcp == 0 or pcp == '-' or pcp is None:
            return "강수없음"
        elif 0.1 <= pcp < 1.0:
            return "1.0mm 미만"
        elif 1.0 <= pcp < 30.0:
            return f"{pcp}mm"
        elif 30.0 <= pcp < 50.0:
            return "30.0~50.0mm"
        else:
            return "50.0mm 이상"
    except:
        return "강수없음"

# 신적설 텍스트 변환 함수
def format_snow_amount(sno):
    try:
        sno = float(sno)
        if sno == 0 or sno == '-' or sno is None:
            return "적설없음"
        elif 0.1 <= sno < 1.0:
            return "1.0cm 미만"
        elif 1.0 <= sno < 5.0:
            return f"{sno}cm"
        else:
            return "5.0cm 이상"
    except:
        return "적설없음"

# 하늘 상태 코드값 변환 함수
def format_sky_condition(sky):
    mapping = {1: "맑음", 3: "구름많음", 4: "흐림"}
    return mapping.get(int(sky), "알 수 없음") if sky else "알 수 없음"

# 강수 형태 코드값 변환 함수
def format_precipitation_type(pty):
    mapping = {0: "없음", 1: "비", 2: "비/눈", 3: "눈", 4: "소나기", 5: "빗방울", 6: "빗방울/눈날림", 7: "눈날림"}
    return mapping.get(int(pty), "알 수 없음") if pty else "알 수 없음"

# 풍향 값에 따른 16방위 변환 함수
def wind_direction_to_16point(wind_deg):
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "N"]
    index = int((wind_deg + 22.5 * 0.5) / 22.5) % 16
    return directions[index]

# 풍속에 따른 바람 강도 텍스트 변환 함수
def wind_speed_category(wind_speed):
    try:
        wind_speed = float(wind_speed)
        if wind_speed < 4.0:
            return "바람이 약하다"
        elif 4.0 <= wind_speed < 9.0:
            return "바람이 약간 강하다"
        elif 9.0 <= wind_speed < 14.0:
            return "바람이 강하다"
        else:
            return "바람이 매우 강하다"
    except:
        return "알 수 없음"



selected_survey = st.selectbox(
    "사용할 모델을 선택하세요.",
    options=["XGBoost 기반 화재위험등급 제공", "GPT를 활용한 화재위험등급 제공", "Gemini를 활용한 화재위험등급 제공"],
    placeholder="하나를 선택하세요.",
    help="선택한 모델에 따라 다른 분석 결과를 제공합니다."
)

st.session_state.selected_survey = selected_survey


if selected_survey == "XGBoost 기반 화재위험등급 제공":

    # 데이터 크기 조정
    if len(X) != len(y):
        X = X[:min(len(X), len(y))]
        y = y[:min(len(X), len(y))]

    # 범주형 변수 레이블 인코딩
    categorical_columns = ['fire_firefighting_district_full_', 'season', 'lisa_category']
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # 목표 변수 레이블 인코딩
    le_y = LabelEncoder()
    y = le_y.fit_transform(y)

    # 학습 데이터와 테스트 데이터로 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 저장된 모델 로드
    with open(f'{DATA_PATH}xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)

    # 분석 실행 버튼
    if st.button("분석실행"):
        # 예측 및 성능 평가
        y_pred = xgb_model.predict(X_test)
        y_pred = np.round(y_pred).astype(int)  # 예측 값을 정수형으로 변환
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        # 색상 맵 설정
        class_colors = plt.get_cmap('tab10', len(np.unique(y_train)))

        # 각 클래스에 대해 SHAP 요약 플롯 생성
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_train)

        # SHAP 값 구조 확인
        st.write(f"SHAP values shape: {np.array(shap_values).shape}")
        st.write(f"X_train shape: {X_train.shape}")

        # 모델 성능 지표 설명 추가
        st.markdown("### 모델 성능 지표")
        st.markdown(f"**Accuracy**: {accuracy}")
        st.markdown("정확도(Accuracy)는 모델이 올바르게 예측한 비율을 나타냅니다. 높은 정확도는 모델이 대부분의 경우 올바르게 예측함을 의미합니다.")
        
        st.markdown("**Classification Report**:")
        st.text(report)
        st.markdown("""
        분류 보고서(Classification Report)는 Precision, Recall, F1-Score 등의 성능 지표를 제공합니다.
        - Precision: 모델이 예측한 양성 샘플 중 실제 양성 샘플의 비율
        - Recall: 실제 양성 샘플 중 모델이 양성으로 올바르게 예측한 비율
        - F1-Score: Precision과 Recall의 조화 평균
        """)

        st.markdown("### SHAP 값")
        st.markdown("""
        SHAP (SHapley Additive exPlanations) 값은 각 특성이 모델의 예측에 미치는 영향을 설명합니다.
        각 클래스에 대한 SHAP 요약 플롯을 통해 특성의 중요도를 시각화할 수 있습니다.
        - 막대 그래프: 특성의 평균 절대 SHAP 값을 나타내며, 값이 클수록 해당 특성이 모델 예측에 중요한 역할을 함을 의미합니다.
        - 도트 그래프: 각 샘플에 대한 특성 값과 SHAP 값을 시각화하여 특성 값이 예측에 미치는 영향을 보여줍니다.
        """)

        # SHAP 값 시각화
        shap_values = np.array(shap_values)  # SHAP 값을 numpy 배열로 변환
        if len(shap_values.shape) == 3:  # 다중 클래스인 경우
            for class_ind in range(shap_values.shape[0]):
                col1, col2 = st.columns(2)
                with col1:
                    shap.summary_plot(
                        shap_values[class_ind], 
                        X_train, 
                        feature_names=X_train.columns, 
                        plot_type="bar", 
                        max_display=10, 
                        show=False
                    )
                    plt.title(f"{class_ind}번 클래스 (막대 그래프)", fontsize=20)
                    st.pyplot(plt)
                    plt.clf()

                with col2:
                    shap.summary_plot(
                        shap_values[class_ind], 
                        X_train, 
                        feature_names=X_train.columns, 
                        plot_type="dot", 
                        max_display=10, 
                        show=False
                    )
                    plt.title(f"{class_ind}번 클래스 (도트 그래프)", fontsize=20)
                    st.pyplot(plt)
                    plt.clf()
                    
        else:  # 단일 클래스인 경우
            col1, col2 = st.columns(2)
            with col1:
                shap.summary_plot(
                    shap_values, 
                    X_train, 
                    feature_names=X_train.columns, 
                    plot_type="bar", 
                    max_display=10, 
                    show=False
                )
                plt.title(f"SHAP 값 (막대 그래프)", fontsize=20)
                st.pyplot(plt)
                plt.clf()

            with col2:
                shap.summary_plot(
                    shap_values, 
                    X_train, 
                    feature_names=X_train.columns, 
                    plot_type="dot", 
                    max_display=10, 
                    show=False
                )
                plt.title(f"SHAP 값 (도트 그래프)", fontsize=20)
                st.pyplot(plt)
                plt.clf()
                
        # 스트리밋 클라우드 서버의 데이터 크기 제한으로 인해, 현재 웹앱에서 모델을 전체적으로 
        # 실행하는 것이 불가능합니다. 이에 따라, 웹앱에서는 모델의 결과를 예시로 보여주는 샘플데이터(25mb 이하)로 분석을 제공하며, 
        # 실제로 정확한 모델 결과를 얻고자 한다면 제출된 모델의 코드를 자신의 로컬 환경에서 실행해야 합니다.
        # 전체적인 xgboost 모델은 제출한 코드에 있으며, 여기에는 샘플데이터 분석 결과만 있습니다.
    
    st.markdown(
        """
        <style>
        .stButton > button {
            background-color: #A7FFEB;
            width: 100%; /
            display: inline-block;
            margin: 0; /
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def page1():
        want_to_Chatbot = st.button("화재위험등급 안내 챗봇")
        if want_to_Chatbot:
            st.session_state.type_of_case = "Chatbot"
            switch_page("화재위험등급 안내 챗봇")
            
    def page2():
        want_to_Tableau = st.button("Tableau")
        if want_to_Tableau:
            st.session_state.type_of_case = "Tableau"
            switch_page("Tableau")

    def page3():
        want_to_Explainable_AI = st.button("Explainable_AI")
        if want_to_Explainable_AI:
            st.session_state.type_of_case = "Explainable_AI"
            switch_page("Explainable_AI")

    col1, col2, col3 = st.columns(3)
    with col1:
        page1()
    with col2:
        page2()
    with col3:
        page3()


if selected_survey == "GPT를 활용한 화재위험등급 제공":


    # 사용자의 기상 요인(날씨 정보) 수집
    gpt_input = {
    "기온(°C)": st.number_input("기온(°C)을 입력하세요.", value=float(temperature) if temperature is not None else 0.0, step=0.1, format="%.1f", key="p1"),
    "풍향(deg)": st.number_input("풍향(deg)을 입력하세요.", value=float(wind_direction) if wind_direction is not None else 0.0, step=1.0, format="%.1f", key="p2"),
    "풍속(m/s)": st.number_input("풍속(m/s)을 입력하세요.", value=float(wind_speed) if wind_speed is not None else 0.0, step=0.1, format="%.1f", key="p3"),
    "풍속(동서성분) UUU (m/s)": st.number_input("풍속(동서성분) UUU (m/s)을 입력하세요.", value=float(wind_speed_uuu) if wind_speed_uuu is not None else 0.0, step=0.1, format="%.1f", key="p4"),
    "풍속(남북성분) VVV (m/s)": st.number_input("풍속(남북성분) VVV (m/s)을 입력하세요.", value=float(wind_speed_vvv) if wind_speed_vvv is not None else 0.0, step=0.1, format="%.1f", key="p5"),
    "강수확률(%)": st.number_input("강수확률(%)을 입력하세요.", value=float(precipitation_prob) if precipitation_prob is not None else 0.0, step=1.0, format="%.1f", key="p6"),
    "강수형태(코드값)": st.selectbox("강수형태를 선택하세요.", options=[0, 1, 2, 3, 5, 6, 7], format_func=format_precipitation_type, key="p7"),
    "강수량(범주)": st.text_input("강수량(범주)을 입력하세요.", value=format_precipitation(precipitation_amount) if precipitation_amount is not None else "강수없음", key="p8"),
    "습도(%)": st.number_input("습도(%)를 입력하세요.", value=float(humidity) if humidity is not None else 0.0, step=1.0, format="%.1f", key="p9"),
    "1시간 신적설(범주(1 cm))": st.text_input("1시간 신적설(범주(1 cm))을 입력하세요.", value=snow_amount if snow_amount is not None else "적설없음", key="p10"),
    "하늘상태(코드값)": st.selectbox("하늘상태를 선택하세요.", options=[1, 3, 4], format_func=format_sky_condition, key="p11"),
    }
    st.session_state.gpt_input = gpt_input

    # 제출 버튼을 누를 경우
    if st.button("제출"):
    
        st.markdown(f"당신의 지역은 {selected_district}이며, 선택한 날짜는 {selected_day}입니다.")
        st.markdown(f"화재위험등급 안내 챗봇 버튼을 클릭하세요. 챗봇 페이지로 이동합니다.")


    st.markdown(
        """
        <style>
        .stButton > button {
            background-color: #A7FFEB;
            width: 100%; /
            display: inline-block;
            margin: 0; /
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


    def page1():
        want_to_Chatbot = st.button("화재위험등급 안내 챗봇")
        if want_to_Chatbot:
            st.session_state.type_of_case = "Chatbot"
            switch_page("화재위험등급 안내 챗봇")
            
    def page2():
        want_to_Tableau = st.button("Tableau")
        if want_to_Tableau:
            st.session_state.type_of_case = "Tableau"
            switch_page("Tableau")

    def page3():
        want_to_Explainable_AI = st.button("Explainable_AI")
        if want_to_Explainable_AI:
            st.session_state.type_of_case = "Explainable_AI"
            switch_page("Explainable_AI")


    col1, col2, col3 = st.columns(3)
    with col1:
        page1()
    with col2:
        page2()
    with col3:
        page3()


if selected_survey == "Gemini를 활용한 화재위험등급 제공":

    # 사용자의 기상 요인(날씨 정보) 수집
    gemini_input = {
    "기온(°C)": st.number_input("기온(°C)을 입력하세요.", value=float(temperature) if temperature is not None else 0.0, step=0.1, format="%.1f", key="p1"),
    "풍향(deg)": st.number_input("풍향(deg)을 입력하세요.", value=float(wind_direction) if wind_direction is not None else 0.0, step=1.0, format="%.1f", key="p2"),
    "풍속(m/s)": st.number_input("풍속(m/s)을 입력하세요.", value=float(wind_speed) if wind_speed is not None else 0.0, step=0.1, format="%.1f", key="p3"),
    "풍속(동서성분) UUU (m/s)": st.number_input("풍속(동서성분) UUU (m/s)을 입력하세요.", value=float(wind_speed_uuu) if wind_speed_uuu is not None else 0.0, step=0.1, format="%.1f", key="p4"),
    "풍속(남북성분) VVV (m/s)": st.number_input("풍속(남북성분) VVV (m/s)을 입력하세요.", value=float(wind_speed_vvv) if wind_speed_vvv is not None else 0.0, step=0.1, format="%.1f", key="p5"),
    "강수확률(%)": st.number_input("강수확률(%)을 입력하세요.", value=float(precipitation_prob) if precipitation_prob is not None else 0.0, step=1.0, format="%.1f", key="p6"),
    "강수형태(코드값)": st.selectbox("강수형태를 선택하세요.", options=[0, 1, 2, 3, 5, 6, 7], format_func=format_precipitation_type, key="p7"),
    "강수량(범주)": st.text_input("강수량(범주)을 입력하세요.", value=format_precipitation(precipitation_amount) if precipitation_amount is not None else "강수없음", key="p8"),
    "습도(%)": st.number_input("습도(%)를 입력하세요.", value=float(humidity) if humidity is not None else 0.0, step=1.0, format="%.1f", key="p9"),
    "1시간 신적설(범주(1 cm))": st.text_input("1시간 신적설(범주(1 cm))을 입력하세요.", value=snow_amount if snow_amount is not None else "적설없음", key="p10"),
    "하늘상태(코드값)": st.selectbox("하늘상태를 선택하세요.", options=[1, 3, 4], format_func=format_sky_condition, key="p11"),
    }
    st.session_state.gemini_input = gemini_input


    # 제출 버튼을 누를 경우
    if st.button("제출"):

        st.markdown(f"당신의 지역은 {selected_district}이며, 선택한 날짜는 {selected_day}입니다.")
        st.markdown(f"화재위험등급 안내 챗봇 버튼을 클릭하세요. 챗봇 페이지로 이동합니다.")


    st.markdown(
        """
        <style>
        .stButton > button {
            background-color: #A7FFEB;
            width: 100%; /
            display: inline-block;
            margin: 0; /
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


    def page1():
        want_to_Chatbot = st.button("화재위험등급 안내 챗봇")
        if want_to_Chatbot:
            st.session_state.type_of_case = "Chatbot"
            switch_page("화재위험등급 안내 챗봇")
            
    def page2():
        want_to_Tableau = st.button("Tableau")
        if want_to_Tableau:
            st.session_state.type_of_case = "Tableau"
            switch_page("Tableau")

    def page3():
        want_to_Explainable_AI = st.button("Explainable_AI")
        if want_to_Explainable_AI:
            st.session_state.type_of_case = "Explainable_AI"
            switch_page("Explainable_AI")


    col1, col2, col3 = st.columns(3)
    with col1:
        page1()
    with col2:
        page2()
    with col3:
        page3()
