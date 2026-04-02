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

# GitHub/서버 환경에서 파일 경로를 안전하게 가져오기 위한 설정
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

# ==========================================
# 4. 1. 원지반 세굴여부 판정
# ==========================================
st.header("1. 원지반 세굴여부 판정")

st.subheader("가. 환산심해파고($H_0'$) 산정 과정")
col_a1, col_a2 = st.columns(2)
with col_a1:
    st.latex(rf"L_0 = \frac{{g T_s^2}}{{2\pi}} = {L0_val:.2f} \, m")
    st.latex(rf"L = L_0 \tanh(2\pi h / L) \approx {L_init:.2f} \, m")
with col_a2:
    st.latex(rf"K_s = \sqrt{{1 / (\tanh kh \cdot 2n)}} \approx {Ks_val:.4f}")
    st.latex(rf"H_0' = H_s / K_s = {H0_prime:.2f} \, m")

st.subheader("나. 이동한계 수심($h_i$) 산정 상세과정")

def run_sato_tanaka_details(alpha):
    h_curr = 15.0 
    rows = []
    for i in range(1, 11):
        L = calc_wave_length(T_input, h_curr)
        constant = alpha * ((ds_input / L0_val)**(1/3))
        term = (H0_prime / L0_val) / constant * (H_input / H0_prime)
        h_next = L * math.asinh(term) / (2 * math.pi)
        
        diff = abs(h_curr - h_next)
        rows.append({
            "회차": i,
            "가정수심(m)": round(h_curr, 3),
            "파장(m)": round(L, 3),
            "산정수심(m)": round(h_next, 3),
            "오차": round(diff, 5)
        })
        if diff < 0.001: break
        h_curr = h_next
    return h_curr, pd.DataFrame(rows)

h_surf, df_surf = run_sato_tanaka_details(1.35)
h_full, df_full = run_sato_tanaka_details(2.40)

tab1, tab2 = st.tabs(["표층 이동한계 (\\alpha=1.35)", "완전 이동한계 (\\alpha=2.40)"])
with tab1:
    st.table(df_surf)
    st.success(f"최종 표층이동 한계수심 ($h_s$): **{h_surf:.2f} m**")
with tab2:
    st.table(df_full)
    st.success(f"최종 완전이동 한계수심 ($h_c$): **{h_full:.2f} m**")

st.subheader("다. 원지반 세굴 여부 최종 판정")

col_h1, col_h2, col_h3 = st.columns(3)
col_h1.metric("현재 설계수심 ($h$)", f"{h_bed:.2f} m")
col_h2.metric("표층이동 한계수심 ($h_s$)", f"{h_surf:.2f} m")
col_h3.metric("완전이동 한계수심 ($h_c$)", f"{h_full:.2f} m")

if h_bed <= h_surf:
    st.error("🚨 **세굴방지공 설치 필요**")
    scour_status = "필요"
else:
    st.success("✅ **원지반 안정 / 보강 불필요**")
    scour_status = "불필요"

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
    S_r = gamma_r / gamma_w
    if S_r <= 1.0: S_r = 1.01 
    theta_rad = math.radians(theta_angle)
    cos_sin = math.cos(theta_rad) - math.sin(theta_rad)
    if cos_sin <= 0.01: cos_sin = 0.01 
    denom_W = 48 * (g_val**3) * (isbash_y**6) * ((S_r - 1.0)**3) * (cos_sin**3)
    
    # 파랑 Isbash
    W_wave_kN = (math.pi * gamma_r * (u_z**6)) / denom_W
    W_wave_ton = W_wave_kN / g_val
    d_wave = safe_cbrt((6.0 * W_wave_kN) / (math.pi * gamma_r))
    
    # 조류 Isbash
    W_current_kN = (math.pi * gamma_r * (v_tidal**6)) / denom_W
    W_current_ton = W_current_kN / g_val
    d_current = safe_cbrt((6.0 * W_current_kN) / (math.pi * gamma_r))
    
    d_final = max(d_wave, d_current)
    W_final_ton = max(W_wave_ton, W_current_ton)
    control_factor = "파랑 (Wave)" if d_wave >= d_current else "조류 (Tidal Current)"

    st.subheader("나. 세굴심도($S_m$) 산정 상세")
    if structure_type == "직립제 (Vertical)":
        if location_type == "제두부 (Head)":
            KC = (u_bottom * T_input) / B_width
            if head_shape == "사각형 (Square)":
                Sm_ratio = -0.09 + 0.123 * KC
            else:
                Sm_ratio = -0.02 + 0.04 * KC
            Sm_val = round(B_width * Sm_ratio, 2)
        else: # 제간부 (Trunk)
            if "Xie" in wave_condition:
                Sm_val = round((0.4 * H_input) / (math.sinh(kh_init)**1.35), 2)
            else: # Hughes and Fowler (1991) - CSV 연동
                Tp = 1.05 * T_input
                Lp = calc_wave_length(Tp, h_bed) 
                kp = 2 * math.pi / Lp
                kph = kp * h_bed
                d_bar = h_bed / (g_val * (Tp**2))

                # CSV 파일 로드 로직 보완
                load_success = False
                try:
                    if os.path.exists(csv_path):
                        df_tav = pd.read_csv(csv_path, skiprows=2, header=None)
                        x_raw = df_tav.iloc[:, 2].dropna().values
                        y_raw = df_tav.iloc[:, 3].dropna().values
                        x_unique, unique_idx = np.unique(x_raw, return_index=True)
                        y_user = y_raw[unique_idx]
                        x_user = x_unique
                        load_success = True
                except:
                    x_user = np.array([0.0013, 0.06])
                    y_user = np.array([1.475, 1.003])
                
                pchip = PchipInterpolator(x_user, y_user)
                Hs_ratio = round(float(pchip(d_bar)), 2)
                Hmo = H_input / Hs_ratio
                
                term1 = math.sqrt(2) / (4 * math.pi * math.cosh(kph))
                term2 = 0.54 * math.cosh((1.5 - kph) / 2.8)
                Urms_m = (g_val * kp * Tp * Hmo) * term1 * term2
                Sm_val = round((Urms_m * Tp * 0.05) / (math.sinh(kph)**0.35), 2)

                # 그래프 출력
                fig, ax = plt.subplots()
                ax.plot(x_user, y_user, 'k-')
                ax.plot(d_bar, Hs_ratio, 'bo')
                ax.set_xscale('log')
                st.pyplot(fig)
    else: # 경사제
        Tp = 1.05 * T_input
        Sm_ratio = 0.01 * Cu_input * ((Tp * math.sqrt(g_val * H_input)) / h_bed)**1.5
        Sm_val = round(H_input * Sm_ratio, 2)

    final_sm_for_design = max(0.0, Sm_val)
    st.success(f"**최종 최대 세굴심도 ($S_m$): {Sm_val:.2f} m**")

    # 보강폭 및 두께
    width_coeff = 2.0 if "매설형" in protection_type else 3.0
    B_sp = width_coeff * final_sm_for_design
    thickness = 2.0 * r_stone
    
    st.info(f"**보강폭 ($B_{{sp}}$): {B_sp:.2f} m / 두께 ($t$): {thickness:.2f} m**")

# ==========================================
# 6. 요약표 출력
# ==========================================
with summary_placeholder.container():
    st.header("📋 전체 산정 결과 요약")
    if scour_status == "필요":
        sum_data = {
            "구 분": ["구조물 형식", "원지반 세굴여부", "지배 외력", "소요 직경 (d)", "최종 세굴심도 (S_m)", "보강폭 (B_sp)"],
            "결 과": [structure_type, "보강 필요 🚨", control_factor, f"{d_final:.3f} m", f"{Sm_val:.2f} m", f"{B_sp:.2f} m"]
        }
        st.table(pd.DataFrame(sum_data).set_index("구 분"))
    else:
        st.success("✅ **보강 불필요**")
