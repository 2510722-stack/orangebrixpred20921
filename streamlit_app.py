import warnings

import joblib
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)


@st.cache_resource
def load_model():
    return joblib.load("brix_model.joblib")


model = load_model()
feature_names = list(getattr(model, "feature_names_in_", ["평균기온", "최저기온", "가조시간", "최저초상온도"]))

st.set_page_config(page_title="제주도 감귤 당도 예측", page_icon="🍊", layout="centered")
st.title("제주도 성산지역 감귤 당도 예측")
st.caption("회귀 모델 기반: 평균기온, 최저기온, 가조시간, 최저초상온도 값을 입력하면 당도를 예측합니다.")

with st.form("brix_form"):
    col1, col2 = st.columns(2)

    with col1:
        avg_temp = st.number_input("평균기온 (°C)", min_value=-30.0, max_value=60.0, value=20.0, step=0.1)
        min_temp = st.number_input("최저기온 (°C)", min_value=-30.0, max_value=60.0, value=12.0, step=0.1)

    with col2:
        sunshine = st.number_input("가조시간 (시간)", min_value=0.0, max_value=24.0, value=7.5, step=0.1)
        lowest_ground_temp = st.number_input("최저초상온도 (°C)", min_value=-30.0, max_value=60.0, value=8.0, step=0.1)

    submitted = st.form_submit_button("당도 예측하기")

if submitted:
    input_df = pd.DataFrame(
        [{
            "평균기온": avg_temp,
            "최저기온": min_temp,
            "가조시간": sunshine,
            "최저초상온도": lowest_ground_temp,
        }]
    )

    if len(feature_names) == 4 and all(name in input_df.columns for name in feature_names):
        input_df = input_df[feature_names]
    else:
        input_df = input_df[["평균기온", "최저기온", "가조시간", "최저초상온도"]]

    predicted_brix = model.predict(input_df)[0]

    st.success(f"예측된 감귤 당도: {predicted_brix:.2f} °Brix")
    st.metric("예상 당도", f"{predicted_brix:.2f} °Brix")

    if predicted_brix >= 12:
        st.info("당도가 높아 고품질 감귤로 평가될 가능성이 큽니다.")
    elif predicted_brix >= 9:
        st.warning("적정 수준의 당도로, 출하 시기와 품질 관리가 중요합니다.")
    else:
        st.warning("당도 수준이 낮아 추가 관리 또는 적기 출하를 고려해 보세요.")

st.markdown("---")
st.write("입력 변수: 평균기온, 최저기온, 가조시간, 최저초상온도")
