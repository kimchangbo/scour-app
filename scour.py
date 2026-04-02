import streamlit as st
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from PIL import Image, ImageEnhance
import os

# ==========================================
# 1. 페이지 설정 및 경로 정의
# ==========================================
st.set_page_config(page_title="세굴방지공 단면제원 계산", layout="wide", page_icon="🌊")

# 파일 경로를 안전하게 설정 (GitHub/Streamlit Cloud 대응)
base_path = os.path.dirname(__file__)
csv_path = os.path.join(base_path, "tav_data_all.csv")

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
summary_placeholder = st.empty()

# ==========================================
# 2. 입력부 (사이드바) - 기존과 동일
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
structure_type = st.sidebar.radio("구조물 형식", ["직립제 (Vertical)", "경사제 (Rubble Mound)"])
location_type = st.sidebar.radio("적용 구간", ["제두부 (Head)", "제간부 (Trunk)"])

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
    Cu_input = st.sidebar.number_input("경험계수 C_u", value=1.00, step=0.1)

protection_type = st.sidebar.radio("보호공 형식", ["매설형 (Buried Type)", "사석마운드형 (Berm Type)"])
r_stone = st.sidebar.number_input("피복재 공칭직경 r (m)", value=1.5, step=0.1)
B_width = st.sidebar.number_input("구조물 폭 또는 직경 B (m)", value=15.0, step=0.1)

# 고정 상수들
gamma_r, gamma_w, isbash_y = 26.0, 10.10, 0.86
theta_angle, z_depth, v_tidal = 33.69, -5.0, 1.50

# ==========================================
# 3. 기본 수리 제원 선계산 - 기존과 동일
# ==========================================
g_val = 9.81
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

# [1. 원지반 판정 생략 - 기존 로직 유지]
scour_status = "필요" 

# ==========================================
# 4. 세굴방지공 계획
# ==========================================
st.header("2. 세굴방지공 계획")

if scour_status == "필요":
    # 가. 규격검토 (Isbash) - 기존 동일
    st.subheader("가. 세굴방지공 규격검토 (Isbash 공식 적용)")
    S_r = gamma_r / gamma_w
    theta_rad = math.radians(theta_angle)
    cos_sin = math.cos(theta_rad) - math.sin(theta_rad)
    denom_W = 48 * (g_val**3) * (isbash_y**6) * ((S_r - 1.0)**3) * (max(cos_sin, 0.01)**3)
    W_wave_kN = (math.pi * gamma_r * (u_z**6)) / denom_W
    W_current_kN = (math.pi * gamma_r * (v_tidal**6)) / denom_W
    d_final = max(safe_cbrt((6.0 * W_wave_kN) / (math.pi * gamma_r)), safe_cbrt((6.0 * W_current_kN) / (math.pi * gamma_r)))
    W_final_ton = max(W_wave_kN, W_current_kN) / g_val

    # 나. 세굴심도(Sm) 산정 - 수정 핵심 구간
    st.subheader("나. 세굴심도($S_m$) 산정 상세")
    Sm_val = 0.0
    
    if structure_type == "직립제 (Vertical)":
        if location_type == "제두부 (Head)":
            KC = (u_bottom * T_input) / B_width
            Sm_ratio = (-0.09 + 0.123 * KC) if head_shape == "사각형 (Square)" else (-0.02 + 0.04 * KC)
            Sm_val = B_width * Sm_ratio
        else:
            if "Xie" in wave_condition:
                Sm_val = (0.4 * H_input) / (math.sinh(kh_init)**1.35)
            else: # Hughes and Fowler (1991) 그래프 수정 구간
                Tp = 1.05 * T_input
                Lp = calc_wave_length(Tp, h_bed)
                kp = 2 * math.pi / Lp
                kph = kp * h_bed
                d_bar = h_bed / (g_val * (Tp**2))
                
                # 🌟 변수 초기화 (에러 방지용 기본 데이터)
                x_user = np.array([0.0013, 0.06])
                y_user = np.array([1.475, 1.003])
                
                # CSV 로드 시도
                if os.path.exists(csv_path):
                    try:
                        df_tav = pd.read_csv(csv_path, skiprows=2, header=None)
                        x_user = df_tav.iloc[:, 2].dropna().values # 3번째 열
                        y_user = df_tav.iloc[:, 3].dropna().values # 4번째 열
                    except:
                        st.warning("데이터 로드 실패 - 기본 데이터를 사용합니다.")

                # 보간 및 계산
                x_unique, idx = np.unique(x_user, return_index=True)
                pchip = PchipInterpolator(x_unique, y_user[idx])
                Hs_ratio = float(pchip(d_bar))
                Hmo = H_input / Hs_ratio
                Urms_m = (g_val * kp * Tp * Hmo) * (math.sqrt(2)/(4*math.pi*math.cosh(kph))) * (0.54*math.cosh((1.5-kph)/2.8))
                Sm_val = (Urms_m * Tp * 0.05) / (math.sinh(kph)**0.35)

                # 그래프 출력
                fig, ax = plt.subplots(figsize=(7, 6))
                ax.plot(x_unique, y_user[idx], 'k-', linewidth=1.5)
                ax.plot(d_bar, Hs_ratio, 'bo', markersize=8)
                ax.set_xscale('log')
                ax.set_xlabel(r'$d / g T_p^2$'); ax.set_ylabel(r'$H_s / H_{mo}$')
                st.pyplot(fig)
    else:
        Tp = 1.05 * T_input
        Sm_val = H_input * (0.01 * Cu_input * ((Tp * math.sqrt(g_val * H_input)) / h_bed)**1.5)

    final_sm_for_design = max(0.0, Sm_val)
    st.success(f"최종 산정 세굴심도: {final_sm_for_design:.2f} m")

    # 다. 보강폭/두께 및 이미지 - 기존 동일
    st.subheader("다. 세굴방지 보강폭 및 두께 산정")
    B_sp = (2.0 if "매설형" in protection_type else 3.0) * final_sm_for_design
    thickness = 2.0 * r_stone
    st.info(f"보강폭: {B_sp:.2f} m / 두께: {thickness:.2f} m")
