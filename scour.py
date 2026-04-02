import streamlit as st
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from PIL import Image, ImageEnhance
import os # ★ 추가: 파일 경로 문제 해결을 위한 모듈

# ★ 추가: 현재 실행 중인 스크립트 파일(scour.py)이 위치한 폴더의 절대 경로를 가져옵니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. 페이지 설정
# ==========================================
st.set_page_config(page_title="세굴방지공 단면제원 계산", layout="wide", page_icon="🌊")

# --- [안전 함수] 3제곱근 계산 (복소수 에러 방지) ---
def safe_cbrt(x):
    return np.sign(x) * (abs(x)**(1.0/3.0))

# --- [함수] 항만설계기준 분산관계식 시산법 파장(L) 산출 ---
def calc_wave_length(T, h):
    T = max(abs(T), 0.1) # 0초 이하 방지
    h = max(abs(h), 0.1) # 0m 이하 방지
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

# ==========================================
# ★ 추가: 전체결과 요약표가 들어갈 자리 확보 (st.empty)
# ==========================================
summary_placeholder = st.empty()

# ==========================================
# 2. 입력부 (사이드바)
# ==========================================
st.sidebar.header("설계파랑 및 지반 제원 입력")
raw_H = st.sidebar.number_input("유의파고 H_s (m)", value=4.10, format="%.2f")
raw_T = st.sidebar.number_input("유의주기 T_s (sec)", value=10.83, format="%.2f")
raw_h = st.sidebar.number_input("현재 설계수심 h (m)", value=22.51, format="%.2f")
ds_input = st.sidebar.number_input("저질 평균입경 d_s (m)", value=0.00006, format="%.6f")

# 🌟 에러 방지: 사용자가 음수를 넣어도 계산은 절대값(양수)으로 처리하도록 보정
H_input = max(abs(raw_H), 0.01)
T_input = max(abs(raw_T), 0.1)
h_bed = max(abs(raw_h), 0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("구조물 및 보호공 조건")

structure_type = st.sidebar.radio("구조물 형식", ["직립제 (Vertical)", "경사제 (Rubble Mound)"])
location_type = st.sidebar.radio("적용 구간 (C.E.M 세굴심 산정용)", ["제두부 (Head)", "제간부 (Trunk)"])

Cu_input = 1.0 # 변수 초기화
if structure_type == "직립제 (Vertical)":
    if location_type == "제두부 (Head)":
        head_shape = st.sidebar.radio("제두부 형상", ["사각형 (Square)", "원형 (Circular)"])
        wave_condition = "비쇄파 규칙파 (Sumer & Fredsoe)"
    else:
        head_shape = "N/A"
        wave_condition = st.sidebar.radio("파랑 조건", ["비쇄파 규칙파 (Xie)", "비쇄파 불규칙파 (Hughes & Fowler)"])
else: # 경사제일 경우
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

# 🌟 에러 방지: 0으로 나누기 방지
kh_init = 2 * math.pi * h_bed / L_init
sinh_kh = math.sinh(kh_init) if math.sinh(kh_init) != 0 else 0.001
tanh_kh = math.tanh(kh_init) if math.tanh(kh_init) != 0 else 0.001

n_val = 0.5 * (1 + (2 * kh_init) / sinh_kh)
# 🌟 에러 방지: math.sqrt 안에는 무조건 양수(abs)만 들어가도록 방어
Ks_val = math.sqrt(abs(1 / (tanh_kh * 2 * n_val)))
H0_prime = H_input / Ks_val

# 전역(Global) 수립자 속도 산정
u_bottom = (math.pi * H_input) / (T_input * sinh_kh)
term_z = 2 * math.pi * (z_depth + h_bed) / L_init
u_z = (math.pi * H_input / T_input) * (math.cosh(term_z) / sinh_kh)

# ==========================================
# 4. 1. 원지반 세굴여부 판정 (상세 시산 과정 포함)
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

st.markdown("#### [판정 결과]")
if h_bed <= h_surf:
    st.latex(rf"h \, ({h_bed:.2f} \, \text{{m}}) \le h_s \, ({h_surf:.2f} \, \text{{m}})")
    st.error("🚨 **세굴방지공 설치 필요** (현재 수심이 표층이동 한계수심보다 얕거나 같습니다.)")
    scour_status = "필요"
else:
    st.latex(rf"h \, ({h_bed:.2f} \, \text{{m}}) > h_s \, ({h_surf:.2f} \, \text{{m}})")
    st.success("✅ **원지반 안정 / 보강 불필요** (현재 수심이 표층이동 한계수심보다 깊어 세굴이 발생하지 않습니다.)")
    scour_status = "불필요"

# ==========================================
# 5. 2. 세굴방지공 계획
# ==========================================
st.markdown("---")
st.header("2. 세굴방지공 계획")

# 초기화 (요약표에서 에러 방지용)
Sm_val = 0.0 
d_final = 0.0
W_final_ton = 0.0
B_sp = 0.0
thickness = 0.0
control_factor = "-"

if scour_status == "필요":
    
    st.subheader("가. 세굴방지공 규격검토 (Isbash 공식 적용)")
    
    # 🌟 에러 방지: Isbash 계수 0 분모 방어
    S_r = gamma_r / gamma_w
    if S_r <= 1.0: S_r = 1.01 
    
    theta_rad = math.radians(theta_angle)
    cos_sin = math.cos(theta_rad) - math.sin(theta_rad)
    if cos_sin <= 0.01: cos_sin = 0.01 
    
    denom_W = 48 * (g_val**3) * (isbash_y**6) * ((S_r - 1.0)**3) * (cos_sin**3)
    
    st.markdown("#### 1) 파랑에 의한 규격 검토")
    st.markdown(f"**가) 파랑에 의한 수립자 속도($U_z$) 산정** (수심 z = {z_depth:.2f}m)")
    st.latex(rf"U_z = \frac{{\pi H}}{{T}} \frac{{\cosh[2\pi(z+h)/L]}}{{\sinh(2\pi h/L)}}")
    st.latex(rf"U_z = \frac{{\pi \times {H_input:.2f}}}{{{T_input:.2f}}} \times \frac{{\cosh[2\pi({z_depth:.2f} + {h_bed:.2f})/{L_init:.2f}]}}{{\sinh(2\pi \times {h_bed:.2f} / {L_init:.2f})}} = {u_z:.4f} \, m/s")
    
    # 파랑 Isbash 산정
    W_wave_kN = (math.pi * gamma_r * (u_z**6)) / denom_W
    W_wave_ton = W_wave_kN / g_val
    V_wave_m3 = W_wave_kN / gamma_r
    d_wave = safe_cbrt((6.0 * W_wave_kN) / (math.pi * gamma_r))
    
    st.markdown("**나) 피복석 소요 중량($W$) 및 규격($d$) 산정**")
    st.latex(rf"S_r = \frac{{\gamma_r}}{{\gamma_w}} = \frac{{{gamma_r:.3f}}}{{{gamma_w:.3f}}} = {S_r:.4f}")
    st.latex(r"W = \frac{\pi \gamma_r U_z^6}{48 g^3 y^6 (S_r - 1)^3 (\cos\theta - \sin\theta)^3}")
    st.latex(rf"W = \frac{{\pi \times {gamma_r:.2f} \times ({u_z:.4f})^6}}{{48 \times ({g_val})^3 \times ({isbash_y})^6 \times ({S_r:.4f} - 1)^3 \times (\cos{theta_angle}^\circ - \sin{theta_angle}^\circ)^3}} = {W_wave_kN:.4f} \, kN")
    st.latex(rf"d = \left( \frac{{6W}}{{\pi \gamma_r}} \right)^{{1/3}} = \left( \frac{{6 \times {W_wave_kN:.4f}}}{{\pi \times {gamma_r:.2f}}} \right)^{{1/3}} = {d_wave:.3f} \, m")
    
    st.markdown("#### 2) 조류에 의한 규격 검토")
    st.markdown("**가) 설계 조류속($V_c$) 적용**")
    st.latex(rf"V_c = {v_tidal:.2f} \, m/s \quad \text{{(설계 적용 조류속)}}")
    
    # 조류 Isbash 산정
    W_current_kN = (math.pi * gamma_r * (v_tidal**6)) / denom_W
    W_current_ton = W_current_kN / g_val
    V_current_m3 = W_current_kN / gamma_r
    d_current = safe_cbrt((6.0 * W_current_kN) / (math.pi * gamma_r))
    
    st.markdown("**나) 피복석 소요 중량($W$) 및 규격($d$) 산정**")
    st.latex(r"W = \frac{\pi \gamma_r V_c^6}{48 g^3 y^6 (S_r - 1)^3 (\cos\theta - \sin\theta)^3}")
    st.latex(rf"W = \frac{{\pi \times {gamma_r:.2f} \times ({v_tidal:.2f})^6}}{{48 \times ({g_val})^3 \times ({isbash_y})^6 \times ({S_r:.4f} - 1)^3 \times (\cos{theta_angle}^\circ - \sin{theta_angle}^\circ)^3}} = {W_current_kN:.4f} \, kN")
    st.latex(rf"d = \left( \frac{{6W}}{{\pi \gamma_r}} \right)^{{1/3}} = \left( \frac{{6 \times {W_current_kN:.4f}}}{{\pi \times {gamma_r:.2f}}} \right)^{{1/3}} = {d_current:.3f} \, m")
    
    st.markdown("#### 3) 최종 규격 결정 (파랑 vs 조류 비교)")
    
    comp_data = {
        "구분": ["파랑 (Wave)", "조류 (Tidal Current)"],
        "적용 유속 (m/s)": [f"{u_z:.4f}", f"{v_tidal:.2f}"],
        "소요 직경 d (m)": [f"{d_wave:.3f}", f"{d_current:.3f}"],
        "소요 중량 W (kN)": [f"{W_wave_kN:.3f}", f"{W_current_kN:.3f}"],
        "소요 중량 W (ton)": [f"{W_wave_ton:.3f}", f"{W_current_ton:.3f}"],
        "소요 부피 V (m³(루베))": [f"{V_wave_m3:.3f}", f"{V_current_m3:.3f}"]
    }
    df_comp = pd.DataFrame(comp_data).set_index("구분")
    st.table(df_comp)
    
    d_final = max(d_wave, d_current)
    W_final_kN = max(W_wave_kN, W_current_kN)
    W_final_ton = max(W_wave_ton, W_current_ton)
    V_final_m3 = max(V_wave_m3, V_current_m3)
    control_factor = "파랑 (Wave)" if d_wave >= d_current else "조류 (Tidal Current)"
    
    st.info(f"**💡 결정 지배 요소:** {control_factor}\n\n**최종 필요 소요 직경 (d):** {d_final:.3f} m  /  **최종 필요 소요 중량 (W):** {W_final_kN:.3f} kN ({W_final_ton:.3f} ton, **{V_final_m3:.3f} m³**)\n\n*(설계 적용 피복재 공칭직경 r = {r_stone:.2f} m)*")

    # 🌟 나. 세굴심도 상세
    st.subheader("나. 세굴심도($S_m$) 산정 상세")
    
    if structure_type == "직립제 (Vertical)":
        if location_type == "제두부 (Head)":
            KC = (u_bottom * T_input) / B_width
            
            st.markdown("#### 1) Keulegan-Carpenter 수 (KC) 산정")
            st.latex(rf"KC = \frac{{u_{{bottom}} T_s}}{{B}} = \frac{{{u_bottom:.3f} \times {T_input:.2f}}}{{{B_width}}} = {KC:.3f}")
            
            st.markdown("#### 2) 구간별 세굴깊이($S_m$) 산정 과정")
            if head_shape == "사각형 (Square)":
                Sm_ratio = -0.09 + 0.123 * KC
                Sm_val_raw = B_width * Sm_ratio
                Sm_val = round(Sm_val_raw, 2)
                st.latex(r"\frac{S_m}{B} = -0.09 + 0.123 \cdot KC \quad \text{(식 VI-5-258)}")
            else: # 원형 (Circular)
                Sm_ratio = -0.02 + 0.04 * KC
                Sm_val_raw = B_width * Sm_ratio
                Sm_val = round(Sm_val_raw, 2)
                st.latex(r"\frac{S_m}{B} = -0.02 + 0.04 \cdot KC \quad \text{(식 VI-5-257)}")
                
            st.latex(rf"S_m = {B_width} \times ({Sm_ratio:.4f}) = {Sm_val:.2f} \, m")
            if Sm_val < 0:
                st.warning(f"계산된 세굴심($S_m$)이 {Sm_val:.2f}m로 음수이므로, 물리적으로 세굴이 발생하지 않는 것으로 간주합니다.")
                
        else: # 제간부 (Trunk)
            if "Xie" in wave_condition:
                st.markdown("#### Xie (1981, 1985) 산정 과정")
                Sm_val_raw = (0.4 * H_input) / (math.sinh(kh_init)**1.35)
                Sm_val = round(Sm_val_raw, 2)
                st.latex(r"S_m = \frac{0.4 \cdot H_s}{[\sinh(kh)]^{1.35}} = " + f"{Sm_val:.2f} \, m")
            else:
                st.markdown("#### Hughes and Fowler (1991) 산정 과정")
                Tp = 1.05 * T_input
                Lp = calc_wave_length(Tp, h_bed) 
                kp = 2 * math.pi / Lp
                kph = kp * h_bed
                
                st.latex(r"T_p = 1.05 T_s = " + f"{Tp:.2f} \, s, \quad k_p = 2\pi/L_p = {kp:.5f}")
                validity_str = "O.K" if 0.05 < kph < 3.0 else "N.G"
                st.latex(r"(U_{rms})_m \text{ 적용 판별 } (k_p h) = " + f"{kph:.4f} \quad (0.05 < k_p h < 3.0) \rightarrow \mathbf{{{validity_str}}}")

                d_bar = h_bed / (g_val * (Tp**2))
                
                # =====================================================================
                # ★ 데이터 누락 방지 및 원본 완벽 재현을 위한 삽도 그리기 강제 로직 추가 ★
                # =====================================================================
                load_success = False
                try:
                    # ★ 수정: 경로를 절대 경로(BASE_DIR)로 안전하게 지정
                    csv_path = os.path.join(BASE_DIR, "tav_data_all.csv")
                    df_tav = pd.read_csv(csv_path, skiprows=2, header=None)
                    for col in df_tav.columns:
                        df_tav[col] = pd.to_numeric(df_tav[col], errors='coerce')
                    
                    x_raw = df_tav.iloc[:, 2].dropna().values
                    y_raw = df_tav.iloc[:, 3].dropna().values
                    
                    x_unique, unique_idx = np.unique(x_raw, return_index=True)
                    y_unique = y_raw[unique_idx]
                    
                    x_user = x_unique
                    y_user = y_unique
                    load_success = True
                except Exception as e:
                    # CSV가 없더라도 앱이 멈추지 않고, 평균선 계산은 되도록 기본 데이터 주입
                    x_user = np.array([0.0013, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.012, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06])
                    y_user = np.array([1.475, 1.340, 1.245, 1.185, 1.145, 1.118, 1.097, 1.082, 1.071, 1.062, 1.047, 1.033, 1.022, 1.016, 1.012, 1.008, 1.005, 1.003])
                    st.warning(f"⚠️ 'tav_data_all.csv' 연동 실패. 내장 데이터로 도표를 자동 복구합니다. ({e})")
                
                # 독취값 보간
                if x_user.min() <= d_bar <= x_user.max():
                    pchip = PchipInterpolator(x_user, y_user)
                    Hs_ratio_raw = float(pchip(d_bar))
                else:
                    Hs_ratio_raw = float(np.interp(d_bar, x_user, y_user))
                
                Hs_ratio = round(Hs_ratio_raw, 2)
                Hmo = H_input / Hs_ratio
                
                st.markdown(f"**$H_{{mo}}$ 산정 (Thompson and Vincent 1985 도표 적용)**")
                st.latex(r"\bar{d} = \frac{d}{g T_p^2} = " + f"{d_bar:.3e}")
                st.latex(r"H_s / H_{mo} = " + f"{Hs_ratio:.2f} \quad \text{{(도표 적용)}}")
                st.latex(r"H_{mo} = \frac{H_s}{H_s / H_{mo}} = \frac{" + f"{H_input:.2f}" + r"}{" + f"{Hs_ratio:.2f}" + r"} = " + f"{Hmo:.2f} \, m")
                
                # ----------------------- 그래프 그리기 -----------------------
                fig, ax = plt.subplots(figsize=(7, 6.5))
                
                if load_success:
                    # 1. MAXIMUM 곡선 (실선)
                    mx = df_tav.iloc[:, 0].dropna().values
                    my = df_tav.iloc[:, 1].dropna().values
                    if len(mx) > 1:
                        s_idx = np.argsort(mx)
                        mx_u, my_u = mx[s_idx], my[s_idx]
                        ax.plot(mx_u, my_u, 'k--', linewidth=1.5, zorder=2)
                        
                        x_target_max = 0.002
                        y_pointer_max = np.interp(x_target_max, mx_u, my_u)
                        ax.annotate('MAXIMUM\n$H_s/H_{mo}$', xy=(x_target_max, y_pointer_max), xytext=(0.0003, 1.62),
                                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.05", color='black', lw=1.2),
                                    fontsize=11, ha='center', va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))

                    # 2. 모든 Epsilon 곡선군 (ε)
                    eps_col_map = {'0.01': (4,5), '0.008': (6,7), '0.007': (8,9), '0.006': (10,11), '0.005': (12,13), '0.004': (14,15), '0.003': (16,17), '0.002': (18,19)}
                    for eps, (x_col, y_col) in eps_col_map.items():
                        if x_col < len(df_tav.columns):
                            ex = df_tav.iloc[:, x_col].dropna().values
                            ey = df_tav.iloc[:, y_col].dropna().values
                            if len(ex) > 1:
                                s_idx = np.argsort(ex)
                                ex_u, ey_u = ex[s_idx], ey[s_idx]
                                ax.plot(ex_u, ey_u, 'k-', linewidth=0.8, alpha=0.8, zorder=1)
                                
                                x_lab = 0.0015 if eps in ['0.01','0.008','0.007','0.006','0.005'] else (0.003 if eps=='0.004' else 0.006)
                                if ex_u.min() < x_lab < ex_u.max():
                                    y_lab = np.interp(x_lab, ex_u, ey_u)
                                    ax.text(x_lab, y_lab, eps.replace("0.", "."), fontsize=9, rotation=45, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', pad=0.1, alpha=0.8))
                else:
                    # CSV 파일 누락시에도 완벽한 그림 출력을 위한 강제 내장 데이터 삽입
                    fb_max_x = np.array([0.001, 0.002, 0.004, 0.008, 0.015, 0.03, 0.06])
                    fb_max_y = np.array([1.68, 1.48, 1.32, 1.18, 1.09, 1.03, 1.01])
                    p_max = PchipInterpolator(fb_max_x, fb_max_y)
                    xs = np.logspace(np.log10(0.001), np.log10(0.06), 50)
                    ax.plot(xs, p_max(xs), 'k--', linewidth=1.5, zorder=2)
                    ax.annotate('MAXIMUM\n$H_s/H_{mo}$', xy=(0.002, float(p_max(0.002))), xytext=(0.0003, 1.62), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.05", color='black', lw=1.2), fontsize=11, ha='center', va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))

                    fb_eps = {
                        '0.01': ([0.003, 0.006, 0.01, 0.02, 0.04], [0.98, 0.94, 0.89, 0.83, 0.78]),
                        '0.008': ([0.0025, 0.005, 0.008, 0.015, 0.03], [0.99, 0.95, 0.90, 0.84, 0.80]),
                        '0.007': ([0.0022, 0.004, 0.007, 0.012, 0.025], [1.00, 0.96, 0.91, 0.85, 0.81]),
                        '0.006': ([0.002, 0.0035, 0.006, 0.01, 0.02], [1.02, 0.97, 0.92, 0.86, 0.82]),
                        '0.005': ([0.0015, 0.0025, 0.005, 0.009, 0.015], [1.05, 1.00, 0.93, 0.87, 0.84]),
                        '0.004': ([0.0012, 0.002, 0.004, 0.007, 0.012], [1.08, 1.02, 0.95, 0.89, 0.86]),
                        '0.003': ([0.0008, 0.0015, 0.003, 0.005, 0.01], [1.13, 1.05, 0.97, 0.91, 0.88]),
                        '0.002': ([0.0005, 0.001, 0.002, 0.004, 0.008], [1.18, 1.09, 1.01, 0.94, 0.90])
                    }
                    for eps, (fx, fy) in fb_eps.items():
                        pf_eps = PchipInterpolator(fx, fy)
                        xs = np.logspace(np.log10(fx[0]), np.log10(fx[-1]), 50)
                        ax.plot(xs, pf_eps(xs), 'k-', linewidth=0.8, alpha=0.8, zorder=1)
                        x_lab = 0.0015 if eps in ['0.01','0.008','0.007','0.006','0.005'] else (0.003 if eps=='0.004' else 0.006)
                        if fx[0] < x_lab < fx[-1]:
                            ax.text(x_lab, float(pf_eps(x_lab)), eps.replace("0.", "."), fontsize=9, rotation=45, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', pad=0.1, alpha=0.8))
                
                # 3. AVERAGE 곡선 플롯 (CSV 존재 여부와 상관없이 무조건 그림)
                s_idx = np.argsort(x_user)
                xu, yu = x_user[s_idx], y_user[s_idx]
                p_avg = PchipInterpolator(xu, yu)
                x_avg_smooth = np.logspace(np.log10(xu.min()), np.log10(xu.max()), 100)
                ax.plot(x_avg_smooth, p_avg(x_avg_smooth), 'k-', linewidth=1.8, label='AVERAGE Curve', zorder=3)
                
                x_target_avg = 0.002
                y_pointer_avg = float(p_avg(x_target_avg))
                ax.annotate('AVERAGE\n$H_s/H_{mo}$', xy=(x_target_avg, y_pointer_avg), xytext=(0.0003, 1.40),
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.1", color='black', lw=1.2),
                            fontsize=11, ha='center', va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
                
                # 4. PRE-BREAKING 텍스트 및 지시 화살표
                ax.text(0.012, 1.25, "PRE-BREAKING", fontsize=11, ha='left', va='center')
                ax.annotate('', xy=(0.004, 1.20), xytext=(0.011, 1.25), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.15", color='black', lw=1.0, alpha=0.9))
                ax.annotate('', xy=(0.008, 1.05), xytext=(0.011, 1.25), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color='black', lw=1.0, alpha=0.9))
                
                # 5. 현재 계산된 독취 위치 마킹 (파란 점 및 선)
                ax.plot(d_bar, Hs_ratio, 'bo', markersize=6, zorder=5)
                ax.axvline(x=d_bar, color='b', linestyle='--', alpha=0.8, linewidth=1.5, zorder=4)
                ax.axhline(y=Hs_ratio, color='b', linestyle='--', alpha=0.8, linewidth=1.5, zorder=4)
                
                # 독취값 박스
                ax.text(d_bar * 1.05, Hs_ratio + 0.015, f"({d_bar:.2e}, {Hs_ratio:.2f})", 
                        color='blue', fontsize=11, fontweight='bold', ha='left', va='bottom',
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9), zorder=6)
                
                # 축 및 눈금 세팅
                ax.set_xscale('log')
                ax.set_xlim(1e-4, 1e-1)
                ax.set_ylim(0.9, 1.7)
                
                ax.tick_params(axis='both', which='major', direction='in', length=8, width=1.5, labelsize=11)
                ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1)
                for spine in ax.spines.values():
                    spine.set_linewidth(1.5)
                
                ax.set_xlabel(r'$\bar{d} = d / g T_p^2$', fontsize=13)
                ax.set_ylabel(r'$H_s / H_{mo}$', fontsize=13)
                ax.grid(False)
                
                col_graph, _ = st.columns([1, 1]) 
                with col_graph:
                    st.pyplot(fig)
                # -----------------------------------------------------------

                term1 = math.sqrt(2) / (4 * math.pi * math.cosh(kph))
                term2 = 0.54 * math.cosh((1.5 - kph) / 2.8)
                Urms_m = (g_val * kp * Tp * Hmo) * term1 * term2
                
                st.markdown("**수평바닥유속의 rms 및 세굴심도 산정**")
                st.latex(r"(U_{rms})_m = \frac{\sqrt{2}}{4\pi \cosh(k_p h)} \times \left[ 0.54 \cosh\left(\frac{1.5 - k_p h}{2.8}\right) \right] \times g k_p T_p H_{mo}")
                st.latex(r"(U_{rms})_m = " + f"{Urms_m:.4f} \, m/s")
                
                Sm_val_raw = (Urms_m * Tp * 0.05) / (math.sinh(kph)**0.35)
                Sm_val = round(Sm_val_raw, 2)
                st.latex(r"S_m = (U_{rms})_m T_p \frac{0.05}{[\sinh(k_p h)]^{0.35}} = " + f"{Sm_val:.2f} \, m")
                
    else: 
        if location_type == "제두부 (Head)":
            st.markdown("#### 경사제 제두부 세굴심도 산정 과정")
        else:
            st.markdown("#### 경사제 제간부 세굴심도 산정 과정 (제두부와 동일 수식 반영)")
            
        Tp = 1.05 * T_input
        st.latex(rf"T_p = 1.05 T_s = 1.05 \times {T_input:.2f} = {Tp:.2f} \, s")
        
        # 🌟 에러 방지: math.sqrt 안에는 무조건 양수(abs)만 들어가도록 방어
        term = (Tp * math.sqrt(abs(g_val * H_input))) / h_bed
        Sm_ratio = 0.01 * Cu_input * (term**1.5)
        Sm_val_raw = H_input * Sm_ratio
        Sm_val = round(Sm_val_raw, 2)
        
        st.latex(r"\frac{S_m}{H_s} = 0.01 C_u \left( \frac{T_p \sqrt{g H_s}}{h} \right)^{3/2}")
        st.latex(rf"\frac{{S_m}}{{H_s}} = 0.01 \times {Cu_input:.2f} \times \left( \frac{{{Tp:.2f} \times \sqrt{{{g_val} \times {H_input:.2f}}}}}{{{h_bed:.2f}}} \right)^{{1.5}} = {Sm_ratio:.4f}")
        st.latex(rf"S_m = {H_input:.2f} \times {Sm_ratio:.4f} = {Sm_val:.2f} \, m")

    final_sm_for_design = max(0.0, Sm_val)
    st.success(f"**최종 산정 최대 세굴심도 ($S_m$): {Sm_val:.2f} m**")

    # 🌟 다. 세굴방지 보강폭 및 두께 산정
    st.subheader("다. 세굴방지 보강폭($B_{sp}$) 및 두께($t$) 산정")
    width_coeff = 2.0 if "매설형" in protection_type else 3.0
    
    B_sp = width_coeff * final_sm_for_design
    thickness = 2.0 * r_stone
    
    st.markdown(f"**적용 보호공 형식:** {protection_type}")
    st.latex(rf"B_{{sp}} = {int(width_coeff)} \times \max(0, S_m) = {B_sp:.2f} \, m")
    
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.info(f"**최종 보강폭 ($B_{{sp}}$): {B_sp:.2f} m**")
    with col_res2:
        st.info(f"**최종 설계두께 ($t = 2r$): {thickness:.2f} m**")

    # ==========================================
    # ★ 수정: 이미지 Crop 시 짤림 현상 완전 방지 (정확히 절반(50%)만 자르기)
    # ==========================================
    try:
        # ★ 수정: 경로를 절대 경로(BASE_DIR)로 지정
        img_path = os.path.join(BASE_DIR, "image_efd977.png")
        img = Image.open(img_path)
        w, h = img.size
        
        # 이미지 선명도 및 대비 보정 로직
        enhancer_contrast = ImageEnhance.Contrast(img)
        img = enhancer_contrast.enhance(1.2)  
        
        enhancer_sharpness = ImageEnhance.Sharpness(img)
        img = enhancer_sharpness.enhance(1.8)  
        
        # 단 1픽셀도 잘려나가지 않도록 정확히 절반을 가르는 기준선(50%) 지정
        mid_x = w // 2
        
        if "매설형" in protection_type:
            # 매설형(Buried Type): 왼쪽 50%를 100% 온전하게 표시
            cropped_img = img.crop((0, 0, mid_x, h)) 
        else:
            # 사석마운드형(Berm Type): 오른쪽 50%를 100% 온전하게 표시
            cropped_img = img.crop((mid_x, 0, w, h))
            
        st.markdown(f"**[{protection_type.split(' ')[0]} 산정 기준 삽도 (보정됨)]**")
        
        # 요청하신 [1.2, 1.5, 1.2] 코딩 크기를 그대로 유지
        col_img1, col_img2, col_img3 = st.columns([1.2, 1.5, 1.2])
        with col_img2:
            st.image(cropped_img, use_container_width=True)
            
    except FileNotFoundError as e:
        st.warning(f"⚠️ 설계 삽도 이미지 파일('image_efd977.png')을 찾을 수 없습니다. ({e})")

else:
    st.write("원지반이 안정하여 추가적인 보강 계획이 필요하지 않습니다.")

# ==========================================
# ★ 추가: 전체결과 요약표 렌더링 (맨 위 st.empty 컨테이너에 삽입)
# ==========================================
with summary_placeholder.container():
    st.header("📋 전체 산정 결과 요약")
    
    if scour_status == "필요":
        sum_data = {
            "구 분": [
                "구조물 / 보호공 형식", 
                "원지반 세굴여부", 
                "지배 외력 (파랑 vs 조류)", 
                "최종 필요 소요 직경 (d)", 
                "최종 필요 소요 중량 (W)", 
                "최대 세굴심도 (S_m)", 
                "세굴방지공 최종 보강폭 (B_sp)", 
                "세굴방지공 설계두께 (t)"
            ],
            "산 정 결 과": [
                f"{structure_type.split(' ')[0]} / {protection_type.split(' ')[0]}",
                "보강 필요 🚨",
                "파랑 (Wave)" if "파랑" in control_factor else "조류 (Tidal Current)",
                f"{d_final:.3f} m",
                f"{W_final_ton:.3f} ton",
                f"{Sm_val:.2f} m",
                f"{B_sp:.2f} m",
                f"{thickness:.2f} m"
            ]
        }
        df_summary = pd.DataFrame(sum_data).set_index("구 분")
        st.table(df_summary)
    else:
        st.success("✅ **원지반 안정 / 보강 불필요** (현재 수심이 표층이동 한계수심보다 깊어 세굴방지공이 불필요합니다.)")
    st.markdown("---")
