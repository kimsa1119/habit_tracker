```python
# app.py
# Streamlit "AI 습관 트래커" - single file app
# Requirements: streamlit, requests, pandas, openai (>=1.0)

from __future__ import annotations

import os
import re
import json
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd
import streamlit as st

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="AI 습관 트래커", page_icon="📊", layout="wide")

st.title("📊 AI 습관 트래커")
st.caption("오늘의 습관 체크인 → 날씨/강아지 → AI 코치 리포트까지 한 번에!")

# -----------------------------
# Sidebar: API Keys
# -----------------------------
with st.sidebar:
    st.header("🔑 API 설정")
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        placeholder="sk-... (또는 환경변수 OPENAI_API_KEY)",
        help="OpenAI 리포트 생성에 사용됩니다.",
    )
    owm_api_key = st.text_input(
        "OpenWeatherMap API Key",
        type="password",
        value=os.getenv("OPENWEATHERMAP_API_KEY", ""),
        placeholder="OpenWeatherMap API Key",
        help="날씨 정보 조회에 사용됩니다.",
    )
    st.divider()
    st.caption("✅ 키는 세션에만 사용되며 저장되지 않습니다(단, 앱 로그/배포 환경 설정은 별도).")

# -----------------------------
# Helpers: Session State
# -----------------------------
HABITS = [
    ("wake", "🌅", "기상 미션"),
    ("water", "💧", "물 마시기"),
    ("study", "📚", "공부/독서"),
    ("workout", "🏃", "운동하기"),
    ("sleep", "😴", "수면"),
]

CITIES = [
    "Seoul",
    "Busan",
    "Incheon",
    "Daegu",
    "Daejeon",
    "Gwangju",
    "Ulsan",
    "Suwon",
    "Jeju",
    "Sejong",
]

COACH_STYLES = {
    "스파르타 코치": "당신은 엄격하고 직설적인 코치입니다. 변명은 허용하지 않되, 구체적이고 실행 가능한 피드백을 제공합니다.",
    "따뜻한 멘토": "당신은 따뜻하고 공감적인 멘토입니다. 사용자의 감정과 맥락을 존중하며, 부드럽고 현실적인 격려와 제안을 합니다.",
    "게임 마스터": "당신은 RPG 게임 마스터입니다. 사용자의 하루를 퀘스트/레벨업/보상처럼 재미있게 해석해 동기부여합니다.",
}


def _today_str() -> str:
    return date.today().isoformat()


def _calc_rate(habits_checked: Dict[str, bool]) -> Tuple[int, int, int]:
    done = sum(1 for k, _, _ in HABITS if habits_checked.get(k, False))
    total = len(HABITS)
    rate = int(round((done / total) * 100))
    return done, total, rate


def _ensure_demo_history():
    """Initialize 6-day demo data in session_state, plus empty container for records."""
    if "records" not in st.session_state:
        st.session_state.records: List[Dict[str, Any]] = []

    if "initialized_demo" in st.session_state:
        return

    # 6 days sample (yesterday-6 ~ yesterday-1): simple pattern
    sample: List[Dict[str, Any]] = []
    base = date.today()
    for i in range(6, 0, -1):
        d = (base - timedelta(days=i)).isoformat()
        # pseudo pattern: increasing completion, varying mood
        completed = min(5, max(1, (7 - i) // 1))  # 1..5
        habits_checked = {k: (idx < completed) for idx, (k, _, _) in enumerate(HABITS)}
        mood = max(1, min(10, 4 + (6 - i)))  # 4..9
        done, total, rate = _calc_rate(habits_checked)
        sample.append(
            {
                "date": d,
                "city": "Seoul",
                "coach_style": "따뜻한 멘토",
                "mood": mood,
                "habits": habits_checked,
                "done": done,
                "rate": rate,
            }
        )

    st.session_state.records = sample
    st.session_state.initialized_demo = True


def upsert_today_record(record: Dict[str, Any]) -> None:
    """Insert or update today's record in session_state.records."""
    d = record["date"]
    records = st.session_state.records
    for i, r in enumerate(records):
        if r.get("date") == d:
            records[i] = record
            return
    records.append(record)


def get_last_n_days_df(n: int = 7) -> pd.DataFrame:
    """Return last n days including today if present; fill missing dates with 0 rate."""
    end = date.today()
    days = [(end - timedelta(days=i)).isoformat() for i in range(n - 1, -1, -1)]
    by_date = {r["date"]: r for r in st.session_state.records}

    rows = []
    for d in days:
        r = by_date.get(d)
        rows.append(
            {
                "date": d,
                "달성률(%)": int(r["rate"]) if r else 0,
                "기분": int(r["mood"]) if r else 0,
                "달성습관": int(r["done"]) if r else 0,
            }
        )
    return pd.DataFrame(rows)


# -----------------------------
# API: Weather (OpenWeatherMap)
# -----------------------------
def get_weather(city: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    OpenWeatherMap current weather.
    - Korean language (lang=kr)
    - Celsius (units=metric)
    - timeout=10
    Returns dict or None on failure.
    """
    if not api_key:
        return None
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": api_key, "lang": "kr", "units": "metric"}
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()

        weather0 = (data.get("weather") or [{}])[0]
        main = data.get("main") or {}
        name = data.get("name") or city

        return {
            "city": name,
            "temp_c": main.get("temp"),
            "feels_like_c": main.get("feels_like"),
            "humidity": main.get("humidity"),
            "description": weather0.get("description"),
            "icon": weather0.get("icon"),
        }
    except Exception:
        return None


# -----------------------------
# API: Dog CEO
# -----------------------------
def _breed_from_dog_url(url: str) -> Optional[str]:
    # Typical: https://images.dog.ceo/breeds/hound-afghan/n02088094_1003.jpg
    m = re.search(r"/breeds/([^/]+)/", url)
    if not m:
        return None
    token = m.group(1)  # e.g., hound-afghan
    parts = token.split("-")
    # make it a nicer label
    return " ".join(p.capitalize() for p in parts if p)


def get_dog_image() -> Optional[Dict[str, Any]]:
    """
    Dog CEO random dog image.
    Returns {url, breed} or None on failure.
    timeout=10
    """
    try:
        url = "https://dog.ceo/api/breeds/image/random"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if data.get("status") != "success":
            return None
        img_url = data.get("message")
        if not img_url:
            return None
        breed = _breed_from_dog_url(img_url) or "Unknown"
        return {"url": img_url, "breed": breed}
    except Exception:
        return None


# -----------------------------
# AI Report (OpenAI)
# -----------------------------
def _build_system_prompt(style_name: str) -> str:
    style_desc = COACH_STYLES.get(style_name, COACH_STYLES["따뜻한 멘토"])
    rules = """
출력은 반드시 아래 형식(섹션 헤더 포함)으로 작성하세요. 한국어로 답변하세요.

[컨디션 등급] S/A/B/C/D 중 하나
[습관 분석] 오늘 습관 체크 결과를 구체적으로(좋은 점/빈틈/원인 추정) 3~6줄
[날씨 코멘트] 날씨가 습관/컨디션에 줄 수 있는 영향 + 대응 팁 2~4줄
[내일 미션] 실행 가능한 미션 3개(체크리스트 형태)
[오늘의 한마디] 1~2줄, 스타일에 맞게

주의:
- 과장된 의학/건강 진단 금지(일반적 조언만).
- 사용자가 체크하지 않은 습관을 비난 대신 '다음 시도'로 전환.
- 길게 늘어지지 말고 밀도 있게.
"""
    return f"{style_desc}\n{rules}".strip()


def generate_report(
    openai_key: str,
    coach_style: str,
    habits_checked: Dict[str, bool],
    mood: int,
    weather: Optional[Dict[str, Any]],
    dog: Optional[Dict[str, Any]],
) -> Optional[str]:
    """
    Sends context to OpenAI and returns report text.
    Model: gpt-5-mini
    Returns None on failure.
    """
    if not openai_key:
        return None

    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        st.error("openai 패키지가 설치되어 있지 않습니다. requirements.txt에 openai를 추가해주세요.")
        return None

    client = OpenAI(api_key=openai_key)

    habits_payload = []
    for k, emoji, label in HABITS:
        habits_payload.append({"habit": label, "key": k, "done": bool(habits_checked.get(k, False)), "emoji": emoji})

    context = {
        "date": _today_str(),
        "mood_1_to_10": mood,
        "habits": habits_payload,
        "weather": weather or None,
        "dog": dog or None,
    }

    sys_prompt = _build_system_prompt(coach_style)
    user_prompt = (
        "아래 JSON 컨텍스트를 기반으로 오늘의 컨디션 리포트를 작성해줘.\n"
        "JSON:\n"
        f"{json.dumps(context, ensure_ascii=False)}"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text if text else None
    except Exception:
        return None


def _make_share_text(
    coach_style: str,
    done: int,
    rate: int,
    mood: int,
    city: str,
    weather: Optional[Dict[str, Any]],
    dog: Optional[Dict[str, Any]],
    report: str,
) -> str:
    w_line = "날씨: (조회 실패)"
    if weather:
        w_line = f"날씨: {weather.get('description','-')} / {weather.get('temp_c','-')}°C (체감 {weather.get('feels_like_c','-')}°C)"

    d_line = "강아지: (조회 실패)"
    if dog:
        d_line = f"강아지: {dog.get('breed','Unknown')}"

    return f"""AI 습관 트래커 - 오늘 기록 ({_today_str()})
- 도시: {city}
- 코치 스타일: {coach_style}
- 달성률: {rate}% ({done}/5)
- 기분: {mood}/10
- {w_line}
- {d_line}

{report}
""".strip()


# -----------------------------
# Init demo data
# -----------------------------
_ensure_demo_history()

# -----------------------------
# Check-in UI
# -----------------------------
st.subheader("✅ 오늘의 습관 체크인")

# Default values
default_city = "Seoul"
default_style = "따뜻한 멘토"

colA, colB = st.columns([1.2, 1.0], gap="large")

with colA:
    st.markdown("**습관 체크(2열 배치)**")
    c1, c2 = st.columns(2, gap="medium")

    # Keep checkbox state stable across reruns
    for idx, (k, emoji, label) in enumerate(HABITS):
        target_col = c1 if idx % 2 == 0 else c2
        with target_col:
            st.checkbox(f"{emoji} {label}", key=f"habit_{k}")

    mood = st.slider("🙂 오늘 기분(1~10)", min_value=1, max_value=10, value=6, step=1)

with colB:
    st.markdown("**환경 설정**")
    city = st.selectbox("도시 선택", options=CITIES, index=CITIES.index(default_city))
    coach_style = st.radio("코치 스타일", options=list(COACH_STYLES.keys()), index=list(COACH_STYLES.keys()).index(default_style))
    st.markdown("---")

    # Compute today stats from current UI
    habits_checked = {k: bool(st.session_state.get(f"habit_{k}", False)) for k, _, _ in HABITS}
    done, total, rate = _calc_rate(habits_checked)

    m1, m2, m3 = st.columns(3, gap="medium")
    m1.metric("달성률", f"{rate}%")
    m2.metric("달성 습관", f"{done}/{total}")
    m3.metric("기분", f"{mood}/10")

    save_col1, save_col2 = st.columns([1, 1])
    with save_col1:
        if st.button("💾 오늘 기록 저장", use_container_width=True):
            record = {
                "date": _today_str(),
                "city": city,
                "coach_style": coach_style,
                "mood": mood,
                "habits": habits_checked,
                "done": done,
                "rate": rate,
            }
            upsert_today_record(record)
            st.success("오늘 기록을 저장했어요! (session_state)")

    with save_col2:
        if st.button("🧹 오늘 체크 초기화", use_container_width=True):
            for k, _, _ in HABITS:
                st.session_state[f"habit_{k}"] = False
            st.experimental_rerun()

# -----------------------------
# Chart Section
# -----------------------------
st.subheader("📈 달성률 추이 (7일)")

df7 = get_last_n_days_df(7)
# Streamlit default chart (bar)
chart_df = df7.set_index("date")[["달성률(%)"]]
st.bar_chart(chart_df)

with st.expander("원본 데이터 보기"):
    st.dataframe(df7, use_container_width=True)

st.divider()

# -----------------------------
# Result: Weather + Dog + AI Report
# -----------------------------
st.subheader("🧠 AI 코치 컨디션 리포트")

gen_btn = st.button("📝 컨디션 리포트 생성", type="primary")

weather_data: Optional[Dict[str, Any]] = None
dog_data: Optional[Dict[str, Any]] = None
report_text: Optional[str] = None

if gen_btn:
    # Fetch weather & dog first (even if OpenAI key missing, show cards)
    with st.spinner("날씨/강아지 정보를 불러오는 중..."):
        weather_data = get_weather(city, owm_api_key)
        dog_data = get_dog_image()

    # Generate report
    with st.spinner("AI 코치가 리포트를 작성하는 중..."):
        report_text = generate_report(
            openai_key=openai_api_key,
            coach_style=coach_style,
            habits_checked=habits_checked,
            mood=mood,
            weather=weather_data,
            dog=dog_data,
        )

    # Save today's record automatically (so chart includes it)
    record = {
        "date": _today_str(),
        "city": city,
        "coach_style": coach_style,
        "mood": mood,
        "habits": habits_checked,
        "done": done,
        "rate": rate,
    }
    upsert_today_record(record)

    # Layout: 2-column cards + report
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("### 🌤️ 오늘의 날씨")
        if weather_data:
            st.write(f"**도시:** {weather_data.get('city', city)}")
            st.write(f"**상태:** {weather_data.get('description', '-')}")
            st.write(f"**기온:** {weather_data.get('temp_c', '-') }°C (체감 {weather_data.get('feels_like_c','-')}°C)")
            st.write(f"**습도:** {weather_data.get('humidity', '-') }%")
            icon = weather_data.get("icon")
            if icon:
                st.image(f"https://openweathermap.org/img/wn/{icon}@2x.png", width=80)
        else:
            st.info("날씨를 불러오지 못했습니다. (API Key/도시/네트워크를 확인해주세요)")

    with right:
        st.markdown("### 🐶 오늘의 강아지")
        if dog_data:
            st.write(f"**품종:** {dog_data.get('breed', 'Unknown')}")
            st.image(dog_data["url"], use_container_width=True)
        else:
            st.info("강아지 이미지를 불러오지 못했습니다. (네트워크를 확인해주세요)")

    st.markdown("### 📄 AI 리포트")
    if report_text:
        st.markdown(report_text)
        share_text = _make_share_text(
            coach_style=coach_style,
            done=done,
            rate=rate,
            mood=mood,
            city=city,
            weather=weather_data,
            dog=dog_data,
            report=report_text,
        )
        st.markdown("### 🔗 공유용 텍스트")
        st.code(share_text, language="markdown")
    else:
        if not openai_api_key:
            st.warning("OpenAI API Key가 없어 리포트를 생성할 수 없습니다. 사이드바에 키를 입력해주세요.")
        else:
            st.error("리포트 생성에 실패했습니다. (키/모델/네트워크/패키지 상태를 확인해주세요.)")

st.divider()

# -----------------------------
# API 안내 (Expander)
# -----------------------------
with st.expander("📌 API 안내 / 문제 해결"):
    st.markdown(
        """
**1) OpenAI API Key**
- OpenAI 리포트 생성에 사용됩니다.
- 배포 환경(Streamlit Community Cloud 등)에서는 **Secrets** 또는 환경변수로 설정하는 것을 권장합니다.
  - 예: `OPENAI_API_KEY`

**2) OpenWeatherMap API Key**
- 현재 날씨 조회에 사용됩니다.
- 키 발급 후 **Current Weather Data** API를 사용합니다.
  - 본 앱은 `lang=kr`, `units=metric(섭씨)`로 요청합니다.

**3) Dog CEO API**
- 무료/키 불필요 랜덤 강아지 이미지 API입니다.
- 네트워크 오류가 나면 None을 반환하도록 설계되어 있습니다.

**4) 자주 발생하는 오류**
- 날씨가 안 나와요: OpenWeatherMap 키가 비었거나, 무료 플랜 호출 제한/도시명이 인식되지 않을 수 있습니다.
- 리포트가 안 나와요: OpenAI 키 누락, `openai` 패키지 미설치, 네트워크/권한 이슈일 수 있습니다.

**5) 참고**
- 기록은 `st.session_state`에 저장됩니다. 새로고침/세션 종료 시 초기화될 수 있습니다.
"""
    )
```
