import streamlit as st
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from PIL import Image, ImageEnhance
import os  # 📂 파일 경로 처리를 위해 추가

# ==========================================
# 1. 페이지 설정 및 경로 정의
# ==========================================
st.set_page_config(page_title="세굴방지공 단면제원 계산", layout="wide", page_icon="🌊")

# GitHub/서버 환경에서도 파일을 정확히 찾기 위한 절대 경로 설정
base_path = os.path.dirname(__file__)
csv_path = os.path.join(base_path, "tav_data_all.csv")
img_path = os.path.join(base_path, "image_efd977.png")

# --- [안전 함수] 3제곱근 계산 (복소수 에러 방지) ---
def safe_cbrt(x):
    return np.sign(x) * (abs(x)**(1.0/3.0))

# --- [함수] 항만설계기준 분산관계식 시산법 파장(L) 산출 ---
def calc_wave_length(T, h):
    T = max(abs(T), 0.1)
    h = max(abs(h), 0.1)
    g = 9.81
    L0 = (g * (T**2)) / (2 * math.pi)
    L_curr = L0
    for _ in range(100):
        L_new = L0 * math.tanh(2 * math.pi * h / L_curr)
        if abs(L_new - L_curr) < 0.0001:
            break
        L_curr = L_new
    return max(L_curr, 0.001)

st.title("🌊 항외측 세굴방지공 단면제원 자동 계산")
st.markdown("### 산정 결과값(-) 표시 및 직립제/경사제 로직 완벽 분리")

summary_placeholder = st.empty()

# ==========================================
# 2. 입력부 (사이드바)
# ==========================================
st.sidebar.header("설계파랑 및 지반 제원 입력")
raw_H = st.sidebar.number_input("유의파고 H_s (m)", value=4.10, format="%.2f")
raw_T = st.sidebar.number_input("유의주기 T_s (sec)", value=10.83, format="%.2f")
raw_h = st.sidebar.number_input("현재 설계수심 h (m)", value=22.51, format="%.2f")
ds_input = st.sidebar.number_input("저질 평균입경 d_s (m)", value=0.00006, format="%.6f")

H_input = max(abs(raw_H), 0.01)
T_input = max(abs(raw_T), 0.1)
h_bed = max(abs(raw_h), 0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("구조물 및 보호공 조건")

structure_type = st.sidebar.radio("구조물 형식", ["직립제 (Vertical)", "경사제 (Rubble Mound)"])
location_type = st.sidebar.radio("적용 구간 (C.E.M 세굴심 산정용)", ["제두부 (Head)", "제간부 (Trunk)"])

Cu_input = 1.0
if structure_type == "직립제 (Vertical)":
    if location_type == "제두부 (Head)":
        head_shape = st.sidebar.radio("제두부 형상", ["사각형 (Square)", "원형 (Circular)"])
        wave_condition = "비쇄파 규칙파 (Sumer & Fredsoe)"
    else:
        head_shape = "N/A"
        wave_condition = st.sidebar.radio("파랑 조건", ["비쇄파 규칙파 (Xie)", "비쇄파 불규칙파 (Hughes & Fowler)"])
else:
    head_shape = "N/A"
    wave_condition = "N/A"
    st.sidebar.markdown("---")
    st.sidebar.subheader("경험계수 입력 (경사제)")
    Cu_input = st.sidebar.number_input("경험계수 C_u (표 참조)", value=1.00, step=0.1, format="%.2f")

protection_type = st.sidebar.radio("보호공 형식", ["매설형 (Buried Type)", "사석마운드형 (Berm Type)"])
r_stone = st.sidebar.number_input("피복재 공칭직경 r (d_n50, m)", value=1.5, step=0.1)
B_width = st.sidebar.number_input("구조물 폭 또는 직경 B (m)", value=15.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("수립자/조류속 및 Isbash 공식 제원")
gamma_r = st.sidebar.number_input("사석 단위중량 gamma_r (kN/m^3)", value=26.0, step=0.1)
gamma_w = st.sidebar.number_input("해수 단위중량 gamma_w (kN/m^3)", value=10.10, step=0.01)
isbash_y = st.sidebar.number_input("Isbash 계수 y (매설: 0.86 / 돌출: 1.2)", value=0.86, step=0.01)
theta_angle = st.sidebar.number_input("사면경사 theta (도)", value=33.69, step=0.01)
z_depth = st.sidebar.number_input("속도 산정 수심 z (m, 해수면=0)", value=-5.0, step=0.1)
v_tidal = st.sidebar.number_input("설계 조류속 V_c (m/s)", value=1.50, step=0.1)

# ==========================================
# 3. 기본 수리 제원 선계산
# ==========================================
g_val = 9.81
L0_val = (g_val * (T_input**2)) / (2 * math.pi)
L_init = calc_wave_length(T_input, h_bed)
kh_init = 2 * math.pi * h_bed / L_init
sinh_kh = math.sinh(kh_init) if math.sinh(kh_init) != 0 else 0.001
tanh_kh = math.tanh(kh_init) if math.tanh(kh_init) != 0 else 0.001
n_val = 0.5 * (1 + (2 * kh_init) / sinh_kh)
Ks_val = math.sqrt(abs(1 / (tanh_kh * 2 * n_val)))
H0_prime = H_input / Ks_val
u_bottom = (math.pi * H_input) / (T_input * sinh_kh)
term_z = 2 * math.pi * (z_depth + h_bed) / L_init
u_z = (math.pi * H_input / T_input) * (math.cosh(term_z) / sinh_kh)

# --- 원지반 세굴여부 판정 함수 (코드 구조 유지를 위해 정의) ---
def run_sato_tanaka_details(alpha):
    h_curr = 15.0 
    rows = []
    for i in range(1, 11):
        L = calc_wave_length(T_input, h_curr)
        constant = alpha * ((ds_input / L0_val)**(1/3))
        term = (H0_prime / L0_val) / constant * (H_input / H0_prime)
        h_next = L * math.asinh(term) / (2 * math.pi)
        diff = abs(h_curr - h_next)
        rows.append({"회차": i, "가정수심(m)": h_curr, "산정수심(m)": h_next, "오차": diff})
        if diff < 0.001: break
        h_curr = h_next
    return h_curr, pd.DataFrame(rows)

# ==========================================
# 4. 1. 원지반 세굴여부 판정
# ==========================================
st.header("1. 원지반 세굴여부 판정")
h_surf, _ = run_sato_tanaka_details(1.35)
scour_status = "필요" if h_bed <= h_surf else "불필요"
st.write(f"판정 결과: 보강 {scour_status}")

# ==========================================
# 5. 2. 세굴방지공 계획
# ==========================================
st.markdown("---")
st.header("2. 세굴방지공 계획")

Sm_val = 0.0 
d_final = 0.0
W_final_ton = 0.0
B_sp = 0.0
thickness = 0.0
control_factor = "-"

if scour_status == "필요":
    st.subheader("가. 세굴방지공 규격검토 (Isbash 공식 적용)")
    # ... Isbash 계산 로직 (기존 코드와 동일) ...
    S_r = gamma_r / gamma_w
    theta_rad = math.radians(theta_angle)
    denom_W = 48 * (g_val**3) * (isbash_y**6) * ((S_r - 1.0)**3) * ((math.cos(theta_rad)-math.sin(theta_rad))**3)
    W_wave_kN = (math.pi * gamma_r * (u_z**6)) / max(denom_W, 0.001)
    d_wave = safe_cbrt((6.0 * W_wave_kN) / (math.pi * gamma_r))
    W_current_kN = (math.pi * gamma_r * (v_tidal**6)) / max(denom_W, 0.001)
    d_current = safe_cbrt((6.0 * W_current_kN) / (math.pi * gamma_r))
    d_final = max(d_wave, d_current)
    W_final_ton = max(W_wave_kN, W_current_kN) / g_val
    control_factor = "파랑" if d_wave >= d_current else "조류"

    st.subheader("나. 세굴심도($S_m$) 산정 상세")
    if structure_type == "직립제 (Vertical)" and location_type == "제간부 (Trunk)" and "Hughes" in wave_condition:
        st.markdown("#### Hughes and Fowler (1991) 산정 과정")
        Tp = 1.05 * T_input
        Lp = calc_wave_length(Tp, h_bed)
        kp = 2 * math.pi / Lp
        kph = kp * h_bed
        d_bar = h_bed / (g_val * (Tp**2))

        # 📊 Hughes and Fowler 그래프 데이터 연동 (수정 핵심)
        x_user = np.array([0.0013, 0.06])  # 기본값 선언
        y_user = np.array([1.475, 1.003])
        
        if os.path.exists(csv_path):
            try:
                df_tav = pd.read_csv(csv_path, skiprows=2, header=None)
                x_raw = df_tav.iloc[:, 2].dropna().values
                y_raw = df_tav.iloc[:, 3].dropna().values
                x_unique, idx = np.unique(x_raw, return_index=True)
                x_user, y_user = x_unique, y_raw[idx]
            except Exception as e:
                st.warning(f"데이터 로드 오류: {e}")

        pchip = PchipInterpolator(x_user, y_user)
        Hs_ratio = float(pchip(d_bar))
        Hmo = H_input / Hs_ratio
        
        # 그래프 출력
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(x_user, y_user, 'k-', label='Average Curve')
        ax.plot(d_bar, Hs_ratio, 'bo', markersize=8)
        ax.set_xscale('log')
        ax.set_xlabel(r'$d / g T_p^2$'); ax.set_ylabel(r'$H_s / H_{mo}$')
        st.pyplot(fig)

        term1 = math.sqrt(2) / (4 * math.pi * math.cosh(kph))
        term2 = 0.54 * math.cosh((1.5 - kph) / 2.8)
        Urms_m = (g_val * kp * Tp * Hmo) * term1 * term2
        Sm_val = (Urms_m * Tp * 0.05) / (math.sinh(kph)**0.35)
        st.latex(rf"S_m = {Sm_val:.2f} \, m")
    
    # ... 보강폭 산정 및 이미지 로직 (기존 코드 유지) ...
    B_sp = (2.0 if "매설형" in protection_type else 3.0) * max(0, Sm_val)
    thickness = 2.0 * r_stone
    
    if os.path.exists(img_path):
        img = Image.open(img_path)
        st.image(img, caption=f"{protection_type} 상세도")

# 최종 결과 요약
with summary_placeholder.container():
    st.header("📋 전체 산정 결과 요약")
    st.write(f"지배 요소: {control_factor} / 세굴심: {Sm_val:.2f}m / 보강폭: {B_sp:.2f}m")
