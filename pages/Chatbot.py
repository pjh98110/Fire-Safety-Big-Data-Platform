import streamlit as st
import openai
import google.generativeai as genai
from streamlit_chat import message
import os
import requests
from streamlit_extras.colored_header import colored_header
import pandas as pd

# 페이지 구성 설정
st.set_page_config(layout="wide")

openai.api_key = st.secrets["secrets"]["OPENAI_API_KEY"]

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "gpt_api_key" not in st.session_state:
    st.session_state.gpt_api_key = openai.api_key # gpt API Key

if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = st.secrets["secrets"]["GEMINI_API_KEY"]


DATA_PATH = "./"


# 데이터 불러오기
# data = pd.read_csv(f"{DATA_PATH}name.csv")



# GPT 프롬프트 엔지니어링 함수
def gpt_prompt(user_input):
    base_prompt = f"""
    너는 지금부터 '입력된 기상 측정 값에 따라 [화재위험등급]을 예측하는 프로그램'이야. 너는 기상 데이터를 분석하고, 화재 위험성을 평가하는 전문가야. 역할에 충실해줘.
    내가 채팅을 입력하면 아래의 <규칙>에 따라서 출력해줘.
    해당하는 날짜는 {st.session_state.selected_day}이며, 지역은 {st.session_state.selected_district}, 기상 요인은 다음과 같아:{st.session_state.gpt_input}

    <규칙>
    1) [화재위험등급]은 [위험], [보통], [안전]으로 구성되어 있어.
    2) 화재 발생에 영향을 미치는 주요 기상 요인은 ["습도", "강수량"]이야.
    3) 날짜, 지역, 모든 기상 요인을 종합적으로 고려해서 화재위험등급을 판단해줘.
    4) 보고서는 사용자가 쉽게 이해하고 납득할 수 있는 근거와 함께 작성해줘.
    5) 분석 결과를 바탕으로 명확한 조치를 제안해줘.

    예시:
    [위험]
    - 기상 요인이 화재 발생과 매우 높은 상관관계를 가지고 있어.
    - 오늘 기상 요인으로 화재위험성이 높다고 판단되면, 이 등급을 선택하고, 이유를 보고서 형식으로 자세히 설명해줘. 다음과 같은 구체적인 예방 조치를 포함해야 해:
        * **즉각적인 행동**: 야외에서의 불사용을 피하고, 화재 안전 장비가 작동하는지 확인하고, 불타기 쉬운 물질을 방치하지 마세요.
        * **긴급 대비**: 대피 경로를 준비하고, 통신 기기를 충전하고, 긴급 연락처를 준비해두세요.

    [보통]
    - 기상 요인이 화재 발생과 약간의 상관관계를 가지고 있어.
    - 오늘 기상 요인으로 화재위험성이 약간 있다고 판단되면, 이 등급을 선택하고, 이유를 보고서 형식으로 설명해줘. 다음과 같은 예방 조치를 포함해야 해:
        * **주의 조치**: 지역 뉴스에서 화재 위험 경고를 주시하고, 열린 불이나 바비큐를 제한하고, 집 주변의 마른 잎이나 쓰레기를 제거하세요.
        * **준비 사항**: 소화기가 사용 가능한 상태인지 확인하고, 가족과 비상 계획을 점검하고, 날씨 변화에 주의하세요.

    [안전]
    - 기상 요인이 화재 발생과 전혀 상관관계가 없어.
    - 오늘 기상 요인으로 화재위험성이 거의 없다고 판단되면, 이 등급을 선택하고, 이유를 보고서 형식으로 설명해줘. 다음과 같은 안전 팁을 포함해야 해:
        * **안전 팁**: 정기적으로 화재 안전 점검을 실시하고, 연기 감지기가 작동하는지 확인하고, 위험이 낮을 때에도 일반적인 화재 안전 조치를 취하세요.
        * **주의 사항**: 지역 날씨 변화를 지속적으로 확인하고, 예기치 않은 상황 변화에 대비하세요.

        
    예시 보고서:
    예시로, "2024년 6월 25일, 서울특별시, 기상 요인은 <기온: 25.5°C, 습도: 60%, 강수량: 0mm>일 경우"라는 조건을 입력하면:
    오늘 기상 요인은 다음과 같습니다: 기온 25.5°C, 습도 60%, 강수량 0mm.
    이로 인해 화재 위험 등급은 [보통]으로 판단됩니다.
    이유: 습도가 상대적으로 낮고 강수량이 없기 때문에 화재 발생 가능성이 중간 정도로 예측됩니다.
    예시 예방 조치:
    * 주의 조치: 지역 뉴스에서 화재 위험 경고를 주시하고, 열린 불이나 바비큐를 제한하고, 집 주변의 마른 잎이나 쓰레기를 제거하세요.
    * 준비 사항: 소화기가 사용 가능한 상태인지 확인하고, 가족과 비상 계획을 점검하고, 날씨 변화에 주의하세요.

    사용자 입력: {user_input}
    """
    return base_prompt

    


# Gemini 프롬프트 엔지니어링 함수
def gemini_prompt(user_input):
    # 프롬프트 엔지니어링 관련 로직
    base_prompt = f"""
    1. 너는 지금부터 '입력된 기상 측정 값에 따라 [화재위험등급]을 예측하는 프로그램'이야.
    2. 내가 채팅을 입력하면 아래의 <규칙>에 따라서 출력해줘.
    3. 해당하는 날짜는 {st.session_state.selected_day}이며, 지역은 {st.session_state.selected_district}, 기상 요인은 {st.session_state.gemini_input}이다.

    <규칙>
    1) [화재위험등급]은 [위험], [보통], [안전]으로 구성되어 있어.
    2) 화재 발생에 영향을 미치는 기상 요인은 ["습도", "강수량"]이야.
    3) 날짜, 지역, 기상 요인을 바탕으로 화재위험등급을 판단해줘.
    4) 보고서는 너가 출력할 수 있는 최대 글자 수까지 최대한 작성해줘.

    [위험]
    - 기상 요인이 화재 발생과 매우 높은 상관관계를 가지고 있어.
    - 오늘 기상 요인으로 화재위험성이 높다고 판단되면, 이 등급을 선택하고, 이유를 보고서 형식으로 자세히 설명해줘. 다음과 같은 구체적인 예방 조치를 포함해야 해:
        * **즉각적인 행동**: 야외에서의 불사용을 피하고, 화재 안전 장비가 작동하는지 확인하고, 불타기 쉬운 물질을 방치하지 마세요.
        * **긴급 대비**: 대피 경로를 준비하고, 통신 기기를 충전하고, 긴급 연락처를 준비해두세요.

    
    [보통]
    - 기상 요인이 화재 발생과 약간의 상관관계를 가지고 있어.
    - 오늘 기상 요인으로 화재위험성이 약간 있다고 판단되면, 이 등급을 선택하고, 이유를 보고서 형식으로 설명해줘. 다음과 같은 예방 조치를 포함해야 해:
        * **주의 조치**: 지역 뉴스에서 화재 위험 경고를 주시하고, 열린 불이나 바비큐를 제한하고, 집 주변의 마른 잎이나 쓰레기를 제거하세요.
        * **준비 사항**: 소화기가 사용 가능한 상태인지 확인하고, 가족과 비상 계획을 점검하고, 날씨 변화에 주의하세요.


    [안전]
    - 기상 요인이 화재 발생과 전혀 상관관계가 없어.
    - 오늘 기상 요인으로 화재위험성이 거의 없다고 판단되면, 이 등급을 선택하고, 이유를 보고서 형식으로 설명해줘. 다음과 같은 안전 팁을 포함해야 해:
        * **안전 팁**: 정기적으로 화재 안전 점검을 실시하고, 연기 감지기가 작동하는지 확인하고, 위험이 낮을 때에도 일반적인 화재 안전 조치를 취하세요.
        * **주의 사항**: 지역 날씨 변화를 지속적으로 확인하고, 예기치 않은 상황 변화에 대비하세요.
    """
    return f"{base_prompt}\n사용자 입력: {user_input}"


# # 스트림 표시 함수
# def stream_display(response, placeholder):
#     text = ''
#     for chunk in response:
#         if parts := chunk.parts:
#             if parts_text := parts[0].text:
#                 text += parts_text
#                 placeholder.write(text + "▌")
#     return text


# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = [
#         {"role": "system", "content": "날씨 정보로 GPT가 예측한 현재 지역의 화재위험성을 알려드립니다."}
#     ]

# # # 세션 변수 체크
# # def check_session_vars():
# #     if 'selected_day' not in st.session_state or 'selected_district' not in st.session_state:
# #         st.warning("처음으로 돌아가서 정보를 다시 입력해주세요.")
# #         st.stop()

# # 세션 변수 체크
# def check_session_vars():
#     required_vars = ['selected_day', 'selected_district']
#     for var in required_vars:
#         if var not in st.session_state:
#             st.warning("필요한 정보가 없습니다. 처음으로 돌아가서 정보를 입력해 주세요.")
#             st.stop()



# selected_chatbot = st.selectbox(
#     "원하는 챗봇을 선택하세요.",
#     options=["GPT를 활용한 화재위험등급 제공", "Gemini를 활용한 화재위험등급 제공"],
#     placeholder="챗봇을 선택하세요.",
#     help="선택한 LLM 모델에 따라 다른 챗봇을 제공합니다."
#     )



# if selected_chatbot == "GPT를 활용한 화재위험등급 제공":
#     colored_header(
#         label='GPT를 활용한 화재위험등급 제공',
#         description=None,
#         color_name="orange-70",)

#     # 세션 변수 체크
#     check_session_vars()

#     # 대화 초기화 버튼
#     def on_clear_chat_gpt():
#         st.session_state.messages = [
#             {"role": "system", "content": "날씨 정보로 GPT가 예측한 현재 지역의 화재위험성을 알려드립니다."}
#         ]

#     st.button("대화 초기화", on_click=on_clear_chat_gpt)

#     # 이전 메시지 표시
#     for msg in st.session_state.messages:
#         role = 'user' if msg['role'] == 'user' else 'assistant'
#         with st.chat_message(role):
#             st.write(msg['content'])

#     # 사용자 입력 처리
#     if prompt := st.chat_input("챗봇과 대화하기:"):
#         # 사용자 메시지 추가
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message('user'):
#             st.write(prompt)

#         # 프롬프트 엔지니어링 적용
#         enhanced_prompt = gpt_prompt(prompt)

#         # 모델 호출 및 응답 처리
#         try:
#             response = openai.ChatCompletion.create(
#                 model="gpt-4o",
#                 messages=[
#                     {"role": "system", "content": enhanced_prompt}
#                 ] + st.session_state.messages,
#                 max_tokens=1500,
#                 temperature=0.8,
#                 top_p=1.0,
#                 frequency_penalty=0.0,
#                 presence_penalty=0.0
#             )
#             text = response.choices[0]['message']['content']

#             # 응답 메시지 표시 및 저장
#             st.session_state.messages.append({"role": "assistant", "content": text})
#             with st.chat_message("assistant"):
#                 st.write(text)
#         except Exception as e:
#             st.error(f"OpenAI API 요청 중 오류가 발생했습니다: {str(e)}")





# elif selected_chatbot == "Gemini를 활용한 화재위험등급 제공":
#     # 세션 변수 체크
#     check_session_vars()

#     # 사이드바에서 모델의 파라미터 설정
#     with st.sidebar:
#         st.header("모델 설정")
#         model_name = st.selectbox(
#             "모델 선택",
#             ['gemini-1.5-flash', "gemini-1.5-pro"]
#         )
#         temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, help="생성 결과의 다양성을 조절합니다.")
#         max_output_tokens = st.number_input("Max Tokens", min_value=1, value=2048, help="생성되는 텍스트의 최대 길이를 제한합니다.")
#         top_k = st.slider("Top K", min_value=1, value=40, help="다음 단어를 선택할 때 고려할 후보 단어의 최대 개수를 설정합니다.")
#         top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.95, help="다음 단어를 선택할 때 고려할 후보 단어의 누적 확률을 설정합니다.")

#     st.button("대화 초기화", on_click=lambda: st.session_state.update({
#         "messages": [{"role": "model", "parts": [{"text": "날씨 정보로 Gemini가 예측한 현재 지역의 화재위험성을 알려드립니다."}]}]
#     }))

#     # 이전 메시지 표시
#     for msg in st.session_state.messages:
#         role = 'human' if msg['role'] == 'user' else 'ai'
#         with st.chat_message(role):
#             st.write(msg['parts'][0]['text'] if 'parts' in msg and 'text' in msg['parts'][0] else '')

#     # 사용자 입력 처리
#     if prompt := st.chat_input("챗봇과 대화하기:"):
#         # 사용자 메시지 추가
#         st.session_state.messages.append({"role": "user", "parts": [{"text": prompt}]})
#         with st.chat_message('human'):
#             st.write(prompt)

#     # # 이전 메시지 표시
#     # for msg in st.session_state.messages:
#     #     role = 'human' if msg['role'] == 'user' else 'ai'
#     #     with st.chat_message(role):
#     #         st.write(msg.get('content', '')) 

#         # 프롬프트 엔지니어링 적용
#         enhanced_prompt = gemini_prompt(prompt)

#         # 모델 호출 및 응답 처리
#         try:
#             genai.configure(api_key=st.session_state.gemini_api_key)
#             generation_config = {
#                 "temperature": temperature,
#                 "max_output_tokens": max_output_tokens,
#                 "top_k": top_k,
#                 "top_p": top_p
#             }
#             model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
#             chat = model.start_chat(history=st.session_state.messages)
#             response = chat.send_message(enhanced_prompt, stream=True)

#             with st.chat_message("ai"):
#                 placeholder = st.empty()
                
#             text = stream_display(response, placeholder)
#             if not text:
#                 if (content := response.parts) is not None:
#                     text = "Wait for function calling response..."
#                     placeholder.write(text + "▌")
#                     response = chat.send_message(content, stream=True)
#                     text = stream_display(response, placeholder)
#             placeholder.write(text)

#             # 응답 메시지 표시 및 저장
#             st.session_state.messages.append({"role": "model", "parts": [{"text": text}]})
#         except Exception as e:
#             st.error(f"Gemini API 요청 중 오류가 발생했습니다: {str(e)}")










# 스트림 표시 함수
def stream_display(response, placeholder):
    text = ''
    for chunk in response:
        if parts := chunk.parts:
            if parts_text := parts[0].text:
                text += parts_text
                placeholder.write(text + "▌")
    return text

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = {
        "gpt": [
            {"role": "system", "content": "날씨 정보로 GPT가 예측한 현재 지역의 화재위험성을 알려드립니다."}
        ],
        "gemini": [
            {"role": "model", "parts": [{"text": "날씨 정보로 Gemini가 예측한 현재 지역의 화재위험성을 알려드립니다."}]}
        ]
    }

# 세션 변수 체크
def check_session_vars():
    required_vars = ['selected_day', 'selected_district']
    for var in required_vars:
        if var not in st.session_state:
            st.warning("필요한 정보가 없습니다. 처음으로 돌아가서 정보를 입력해 주세요.")
            st.stop()

selected_chatbot = st.selectbox(
    "원하는 챗봇을 선택하세요.",
    options=["GPT를 활용한 화재위험등급 제공", "Gemini를 활용한 화재위험등급 제공"],
    placeholder="챗봇을 선택하세요.",
    help="선택한 LLM 모델에 따라 다른 챗봇을 제공합니다."
)

if selected_chatbot == "GPT를 활용한 화재위험등급 제공":
    colored_header(
        label='GPT를 활용한 화재위험등급 제공',
        description=None,
        color_name="orange-70",
    )

    # 세션 변수 체크
    check_session_vars()

    # 대화 초기화 버튼
    def on_clear_chat_gpt():
        st.session_state.messages["gpt"] = [
            {"role": "system", "content": "날씨 정보로 GPT가 예측한 현재 지역의 화재위험성을 알려드립니다."}
        ]

    st.button("대화 초기화", on_click=on_clear_chat_gpt)

    # 이전 메시지 표시
    if "gpt" not in st.session_state.messages:
        st.session_state.messages["gpt"] = [
            {"role": "system", "content": "날씨 정보로 GPT가 예측한 현재 지역의 화재위험성을 알려드립니다."}
        ]
        
    for msg in st.session_state.messages["gpt"]:
        role = 'user' if msg['role'] == 'user' else 'assistant'
        with st.chat_message(role):
            st.write(msg['content'])

    # 사용자 입력 처리
    if prompt := st.chat_input("챗봇과 대화하기:"):
        # 사용자 메시지 추가
        st.session_state.messages["gpt"].append({"role": "user", "content": prompt})
        with st.chat_message('user'):
            st.write(prompt)

        # 프롬프트 엔지니어링 적용
        enhanced_prompt = gpt_prompt(prompt)

        # 모델 호출 및 응답 처리
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": enhanced_prompt}
                ] + st.session_state.messages["gpt"],
                max_tokens=1500,
                temperature=0.8,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            text = response.choices[0]['message']['content']

            # 응답 메시지 표시 및 저장
            st.session_state.messages["gpt"].append({"role": "assistant", "content": text})
            with st.chat_message("assistant"):
                st.write(text)
        except Exception as e:
            st.error(f"OpenAI API 요청 중 오류가 발생했습니다: {str(e)}")

elif selected_chatbot == "Gemini를 활용한 화재위험등급 제공":
    # 세션 변수 체크
    check_session_vars()

    # 사이드바에서 모델의 파라미터 설정
    with st.sidebar:
        st.header("모델 설정")
        model_name = st.selectbox(
            "모델 선택",
            ['gemini-1.5-flash', "gemini-1.5-pro"]
        )
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, help="생성 결과의 다양성을 조절합니다.")
        max_output_tokens = st.number_input("Max Tokens", min_value=1, value=2048, help="생성되는 텍스트의 최대 길이를 제한합니다.")
        top_k = st.slider("Top K", min_value=1, value=40, help="다음 단어를 선택할 때 고려할 후보 단어의 최대 개수를 설정합니다.")
        top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.95, help="다음 단어를 선택할 때 고려할 후보 단어의 누적 확률을 설정합니다.")

    st.button("대화 초기화", on_click=lambda: st.session_state.update({
        "messages": {"gemini": [{"role": "model", "parts": [{"text": "날씨 정보로 Gemini가 예측한 현재 지역의 화재위험성을 알려드립니다."}]}]}
    }))

    # 이전 메시지 표시
    if "gemini" not in st.session_state.messages:
        st.session_state.messages["gemini"] = [
            {"role": "model", "parts": [{"text": "날씨 정보로 Gemini가 예측한 현재 지역의 화재위험성을 알려드립니다."}]}
        ]
        
    for msg in st.session_state.messages["gemini"]:
        role = 'human' if msg['role'] == 'user' else 'ai'
        with st.chat_message(role):
            st.write(msg['parts'][0]['text'] if 'parts' in msg and 'text' in msg['parts'][0] else '')

    # 사용자 입력 처리
    if prompt := st.chat_input("챗봇과 대화하기:"):
        # 사용자 메시지 추가
        st.session_state.messages["gemini"].append({"role": "user", "parts": [{"text": prompt}]})
        with st.chat_message('human'):
            st.write(prompt)

        # 프롬프트 엔지니어링 적용
        enhanced_prompt = gemini_prompt(prompt)

        # 모델 호출 및 응답 처리
        try:
            genai.configure(api_key=st.session_state.gemini_api_key)
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "top_k": top_k,
                "top_p": top_p
            }
            model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
            chat = model.start_chat(history=st.session_state.messages["gemini"])
            response = chat.send_message(enhanced_prompt, stream=True)

            with st.chat_message("ai"):
                placeholder = st.empty()
                
            text = stream_display(response, placeholder)
            if not text:
                if (content := response.parts) is not None:
                    text = "Wait for function calling response..."
                    placeholder.write(text + "▌")
                    response = chat.send_message(content, stream=True)
                    text = stream_display(response, placeholder)
            placeholder.write(text)

            # 응답 메시지 표시 및 저장
            st.session_state.messages["gemini"].append({"role": "model", "parts": [{"text": text}]})
        except Exception as e:
            st.error(f"Gemini API 요청 중 오류가 발생했습니다: {str(e)}")