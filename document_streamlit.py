import streamlit as st
from openai import OpenAI
import base64
import json
import tempfile
import os

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(page_title="홈스캐너 문서 AI", layout="centered")
st.title("🏠 홈스캐너 문서 AI (Vision AI 기반)")
st.write("문서 이미지를 업로드하면 AI가 내용을 이해하고 핵심 정보를 추출합니다.")

client = OpenAI()

# -----------------------------
# 유틸 함수
# -----------------------------
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_document_info(image_path):
    img_base64 = encode_image(image_path)

    prompt = prompt = """
너는 API 서버다.
반드시 JSON만 반환해야 한다.
설명, 주석, 문장, ``` 절대 포함하지 마라.
이미지 분석해서 텍스트 추출에 힘을 줘라.
순위 번호별로 다 반환해라

형식:
{
  "등기목적" : "...",
  "등기원인" : "...",
}

문서 분석을 수행하라.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=800
    )

    return response.choices[0].message.content

# -----------------------------
# UI 영역
# -----------------------------
uploaded_file = st.file_uploader(
    "📄 문서 이미지 업로드 (jpg / png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    st.image(uploaded_file, caption="업로드한 문서", use_column_width=True)

    if st.button("🔍 문서 분석하기"):
        with st.spinner("Vision AI가 문서를 분석 중입니다..."):
            # 임시 파일 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                result = extract_document_info(tmp_path)

                st.subheader("📌 분석 결과")

                # JSON 파싱 시도
                try:
                    parsed = json.loads(result)
                    st.json(parsed)
                except json.JSONDecodeError:
                    st.warning("JSON 파싱 실패 – 원본 응답을 표시합니다")
                    st.text(result)

            finally:
                os.remove(tmp_path)

# -----------------------------
# 하단 안내
# -----------------------------
st.markdown("---")
st.markdown("""
### ℹ️ 사용 방법
1. 계약서 / 고지서 / 공문 이미지를 업로드
2. **문서 분석하기** 버튼 클릭
3. AI가 문서를 이해하고 핵심 정보를 JSON으로 추출

> OpenCV, OCR 설치 없이 Vision AI만 사용합니다.
""")
