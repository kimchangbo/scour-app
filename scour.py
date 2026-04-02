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

# 파일 경로 설정 (GitHub/Streamlit Cloud 대응)
base_path = os.path.dirname(__file__)
csv_path = os.path.join(base_path, "tav_data_all.csv")
# 이미지 파일명 (업로드하신 파일명에 맞춰 수정 필요할 수 있음)
img_path = os.path.join(base_path, "image_efd977.png") 

# --- [함수] 안전 계산 및 파장 산출 ---
def safe_cbrt(x):
    return np.sign(x) * (abs(x)**(1.0/3.0))

def calc_wave_length(T, h):
    T, h = max(abs(T), 0.1), max(abs(h), 0.1)
    g, L0 = 9.81, (9.81 * (T**2)) / (2 * math.pi)
    L_curr = L0
    for _ in range(100):
        L_new = L0 * math.tanh(2 * math.pi * h / L_curr)
        if abs(L_new - L_curr) < 0.0001: break
        L_curr = L_new
    return max(L_curr, 0.001)

st.title("🌊 항외측 세굴방지공 단면제원 자동 계산")
summary_placeholder = st.empty()

# ==========================================
# 2. 입력부 (사이드바)
# ==========================================
st.sidebar.header("설계파랑 및 지반 제원 입력")
raw_H = st.sidebar.number_input("유의파고 H_s (m)", value=4.10, format="%.2f")
raw_T = st.sidebar.number_input("유의주기 T_s (sec)", value=10.83, format="%.2f")
raw_h = st.sidebar.number_input("현재 설계수심 h (m)", value=22.51, format="%.2f")
ds_input = st.sidebar.number_input("저질 평균입경 d_s (m)", value=0.00006, format="%.6f")

H_input, T_input, h_bed = max(abs(raw_H), 0.01), max(abs(raw_T), 0.1), max(abs(raw_h), 0.1)

st.sidebar.markdown("---")
structure_type = st.sidebar.radio("구조물 형식", ["직립제 (Vertical)", "경사제 (Rubble Mound)"])
location_type = st.sidebar.radio("적용 구간", ["제두부 (Head)", "제간부 (Trunk)"])

# 조건별 변수 초기화
head_shape, wave_condition, Cu_input = "N/A", "N/A", 1.0
if structure_type == "직립제 (Vertical)":
    if location_type == "제두부 (Head)":
        head_shape = st.sidebar.radio("제두부 형상", ["사각형 (Square)", "원형 (Circular)"])
        wave_condition = "비쇄파 규칙파 (Sumer & Fredsoe)"
    else:
        wave_condition = st.sidebar.radio("파랑 조건", ["비쇄파 규칙파 (Xie)", "비쇄파 불규칙파 (Hughes & Fowler)"])
else:
    Cu_input = st.sidebar.number_input("경험계수 C_u", value=1.00, step=0.1)

protection_type = st.sidebar.radio("보호공 형식", ["매설형 (Buried Type)", "사석마운드형 (Berm Type)"])
r_stone = st.sidebar.number_input("피복재 공칭직경 r (m)", value=1.5, step=0.1)
B_width = st.sidebar.number_input("구조물 폭 또는 직경 B (m)", value=15.0, step=0.1)

st.sidebar.markdown("---")
gamma_r, gamma_w, isbash_y = 26.0, 10.10, 0.86
theta_angle, z_depth, v_tidal = 33.69, -5.0, 1.50

# ==========================================
# 3. 기본 수리 제원 계산 및 판정
# ==========================================
g_val = 9.81
L_init = calc_wave_length(T_input, h_bed)
kh_init = 2 * math.pi * h_bed / L_init
sinh_kh = math.sinh(kh_init) if math.sinh(kh_init) != 0 else 0.001
u_bottom = (math.pi * H_input) / (T_input * sinh_kh)
term_z = 2 * math.pi * (z_depth + h_bed) / L_init
u_z = (math.pi * H_input / T_input) * (math.cosh(term_z) / sinh_kh)

# [1. 원지반 판정 생략 - 기존 로직 유지]
scour_status = "필요" # 테스트를 위해 필요로 고정 (실제 코드에선 판정 로직 사용)

# ==========================================
# 4. 세굴방지공 계획
# ==========================================
st.header("2. 세굴방지공 계획")

if scour_status == "필요":
    # 가. 규격검토 (Isbash)
    st.subheader("가. 세굴방지공 규격검토 (Isbash 공식 적용)")
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

    # 나. 세굴심도(Sm) 산정
    st.subheader("나. 세굴심도($S_m$) 산정 상세")
    Sm_val = 0.0
    
    if structure_type == "직립제 (Vertical)":
        if location_type == "제두부 (Head)":
            KC = (u_bottom * T_input) / B_width
            Sm_ratio = (-0.09 + 0.123 * KC) if head_shape == "사각형 (Square)" else (-0.02 + 0.04 * KC)
            Sm_val = B_width * Sm_ratio
            st.latex(rf"KC = {KC:.3f}, \quad S_m = {Sm_val:.2f}m")
        else:
            if "Xie" in wave_condition:
                Sm_val = (0.4 * H_input) / (math.sinh(kh_init)**1.35)
                st.latex(rf"S_m = {Sm_val:.2f}m")
            else: # Hughes & Fowler (에러 발생 지점)
                Tp = 1.05 * T_input
                Lp = calc_wave_length(Tp, h_bed)
                kp = 2 * math.pi / Lp
                kph = kp * h_bed
                d_bar = h_bed / (g_val * (Tp**2))
                
                # 🌟 변수 초기화 보장 (NameError 방지)
                x_user = np.array([0.0013, 0.06])
                y_user = np.array([1.475, 1.003])
                
                if os.path.exists(csv_path):
                    try:
                        df_tav = pd.read_csv(csv_path, skiprows=2, header=None)
                        x_user = df_tav.iloc[:, 2].dropna().values
                        y_user = df_tav.iloc[:, 3].dropna().values
                    except: st.warning("CSV 로드 실패 - 기본 데이터 사용")
                
                pchip = PchipInterpolator(x_user, y_user)
                Hs_ratio = float(pchip(d_bar))
                Hmo = H_input / Hs_ratio
                Urms_m = (g_val * kp * Tp * Hmo) * (math.sqrt(2)/(4*math.pi*math.cosh(kph))) * (0.54*math.cosh((1.5-kph)/2.8))
                Sm_val = (Urms_m * Tp * 0.05) / (math.sinh(kph)**0.35)
                
                # 그래프 출력
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(x_user, y_user, 'k-', label='Average Curve')
                ax.scatter([d_bar], [Hs_ratio], color='red', zorder=5)
                ax.set_xscale('log')
                ax.set_xlabel('d_bar'); ax.set_ylabel('Hs/Hmo')
                st.pyplot(fig)

    else: # 경사제
        Tp = 1.05 * T_input
        Sm_val = H_input * (0.01 * Cu_input * ((Tp * math.sqrt(g_val * H_input)) / h_bed)**1.5)
        st.latex(rf"S_m = {Sm_val:.2f}m")

    # 다. 보강폭(Bsp) 및 두께(t)
    st.subheader("다. 세굴방지 보강폭($B_{sp}$) 및 두께($t$) 산정")
    B_sp = (2.0 if "매설형" in protection_type else 3.0) * max(0, Sm_val)
    thickness = 2.0 * r_stone
    
    st.success(f"**최종 보강폭 (B_sp): {B_sp:.2f} m** / **설계두께 (t): {thickness:.2f} m**")

    # 🖼️ 이미지 출력 (보정 로직 포함)
    if os.path.exists(img_path):
        img = Image.open(img_path)
        w, h = img.size
        # 매설형/마운드형에 따라 크롭
        crop_box = (0, 0, w//2, h) if "매설형" in protection_type else (w//2, 0, w, h)
        st.image(img.crop(crop_box), caption=f"{protection_type} 상세도")
    else:
        st.warning("이미지 파일(image_efd977.png)이 없습니다.")

# 요약표 렌더링 (코드 최상단 empty 컨테이너)
with summary_placeholder.container():
    st.info(f"결과 요약: {control_factor} 지배 / B_sp={B_sp:.2f}m / t={thickness:.2f}m")
