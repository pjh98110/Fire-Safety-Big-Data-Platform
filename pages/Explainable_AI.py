import streamlit as st
import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import streamlit.components.v1 as components
import time

st.set_page_config(layout="wide")

# 데이터 로드
DATA_PATH = "./"
try:
    data = pd.read_csv(f"{DATA_PATH}test_data.csv")
except FileNotFoundError:
    st.error("데이터 파일을 찾을 수 없습니다. 데이터 파일의 경로를 확인하세요.")
    st.stop()

# 한글 폰트 설정 함수
def set_korean_font():
    font_path = f"{DATA_PATH}NanumGothic.ttf"  # 폰트 파일 경로

    from matplotlib import font_manager, rc
    font_manager.fontManager.addfont(font_path)
    rc('font', family='NanumGothic')

# 한글 폰트 설정 적용
set_korean_font()

# 데이터 샘플링 (속도 개선을 위해 샘플링 비율 조정)
sample_fraction = 0.1  # 10% 샘플링
data_sampled = data.sample(frac=sample_fraction, random_state=42)

# 페이지 상태 초기화
if "page" not in st.session_state:
    st.session_state.page = "Home"

# 페이지 헤더
st.markdown(f"""
<span style='font-size: 24px;'>
<div style=" color: #000000;">
<strong>Explainer Dashboard</strong>
</div>
""", unsafe_allow_html=True)

# XAI 선택 박스
selected_xai = st.selectbox(
    label="원하는 Explainable AI(XAI)를 선택하세요.",
    options=["SHAP", "LIME"],
    placeholder="하나를 선택하세요.",
    help="XAI는 사용자가 머신러닝 알고리즘으로 생성된 결과를 쉽게 이해할 수 있도록 도와주는 프로세스와 방법입니다.",
    key="xai_key"
)

# 분석 실행 버튼
if st.button("분석 실행"):
    # 데이터 준비
    # Label Encoding을 사용하여 범주형 데이터를 수치형으로 변환
    label_encoders = {}
    for column in data_sampled.columns:
        if data_sampled[column].dtype == 'object':
            le = LabelEncoder()
            data_sampled[column] = le.fit_transform(data_sampled[column])
            label_encoders[column] = le

    # 분석에 사용할 피처와 타겟 변수
    X = data_sampled.drop(columns=["target_sum"])
    y = data_sampled["target_sum"]
    data_sampled['target_sum'] = y  # target_sum을 다시 데이터프레임에 추가

    # 모델 훈련 (예시로 RandomForestRegressor 사용)
    model = RandomForestRegressor(random_state=42, n_jobs=-1) # xgboost로 변경하기
    model.fit(X, y)

    # 로딩바 생성
    progress_bar = st.progress(0)
    
    for percent_complete in range(100):
        time.sleep(0.01)
        progress_bar.progress(percent_complete + 1)

    # SHAP 분석 실행
    if selected_xai == "SHAP":
        st.write("SHAP 분석을 실행합니다...")

        # SHAP 분석
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # SHAP summary plot
        st.subheader("SHAP Summary Plot")
        fig1, ax1 = plt.subplots()
        shap.summary_plot(shap_values, X, plot_type="dot", show=False)
        st.pyplot(fig1)

        # SHAP 해석 설명
        st.markdown("""
        **SHAP (SHapley Additive exPlanations)**: 
        - SHAP는 모델 예측에 대한 각 피처의 기여도를 계산합니다.
        - Summary Plot은 전체 피처의 중요도를 시각화합니다.
        """)

    # LIME 분석 실행
    elif selected_xai == "LIME":
        st.write("LIME 분석을 실행합니다...")

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(X),
            feature_names=X.columns,
            mode='regression'
        )
        
        # 샘플 데이터 예측 및 설명
        i = np.random.randint(0, X.shape[0])
        exp = explainer.explain_instance(X.iloc[i], model.predict, num_features=10)
        st.write(f"Instance {i} 설명:")
        components.html(exp.as_html(), height=500)

        # LIME 해석 설명
        st.markdown("""
        **LIME (Local Interpretable Model-agnostic Explanations)**:
        - LIME은 모델 예측을 쉽게 해석할 수 있도록 하는 도구입니다.
        - 특정 데이터 포인트에 대해 모델이 내린 예측을 설명합니다.
        - 각 피처가 예측에 미친 영향을 보여줍니다.
        """)
