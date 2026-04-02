import streamlit as st
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from PIL import Image, ImageEnhance
import os

# ★ 현재 실행 중인 파일의 절대 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. 페이지 설정 및 공통 함수
# ==========================================
st.set_page_config(page_title="세굴방지공 단면제원 계산", layout="wide", page_icon="🌊")

def safe_cbrt(x):
    return np.sign(x) * (abs(x)**(1.0/3.0))

def calc_wave_length(T, h):
    T = max(abs(T), 0.1)
    h = max(abs(h), 0.1)
    g = 9.81
    L0 = (g * (T**2)) / (2 * math.pi)
    L_curr = L0
    for _ in range(100):
        L_new = L0 * math.tanh(2 * math.pi * h / L_curr)
        if abs(L_new - L_curr) < 0.0001: break
        L_curr = L_new
    return max(L_curr, 0.001)

st.title("🌊 항외측 세굴방지공 단면제원 자동 계산")
st.markdown("### Thompson and Vincent 도표 및 보호공 삽도 완벽 재현")

# 요약표 위치 홀더
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

if structure_type == "직립제 (Vertical)":
    if location_type == "제두부 (Head)":
        head_shape = st.sidebar.radio("제두부 형상", ["사각형 (Square)", "원형 (Circular)"])
        wave_condition = "비쇄파 규칙파"
    else:
        wave_condition = st.sidebar.radio("파랑 조건", ["비쇄파 규칙파 (Xie)", "비쇄파 불규칙파 (Hughes & Fowler)"])
else:
    Cu_input = st.sidebar.number_input("경험계수 C_u (경사제)", value=1.00, step=0.1)

protection_type = st.sidebar.radio("보호공 형식", ["매설형 (Buried Type)", "사석마운드형 (Berm Type)"])
r_stone = st.sidebar.number_input("피복재 공칭직경 r (m)", value=1.5, step=0.1)
B_width = st.sidebar.number_input("구조물 폭 B (m)", value=15.0, step=0.1)

gamma_r, gamma_w = 26.0, 10.10
isbash_y, theta_angle = 0.86, 33.69
z_depth, v_tidal = -5.0, 1.50

# 기본 수리 제원 선계산
L_init = calc_wave_length(T_input, h_bed)
kh_init = 2 * math.pi * h_bed / L_init
u_bottom = (math.pi * H_input) / (T_input * math.sinh(kh_init))
u_z = (math.pi * H_input / T_input) * (math.cosh(2 * math.pi * (z_depth + h_bed) / L_init) / math.sinh(kh_init))

# ==========================================
# 3. 1. 원지반 세굴여부 판정 (사토-다나카 시산법)
# ==========================================
st.header("1. 원지반 세굴여부 판정")
def run_sato_tanaka_details(alpha):
    h_curr = 15.0 
    rows = []
    for i in range(1, 11):
        L = calc_wave_length(T_input, h_curr)
        constant = alpha * ((ds_input / ((9.81 * T_input**2)/(2*math.pi)))**(1/3))
        h_next = L * math.asinh((H_input/L) / constant) / (2 * math.pi)
        diff = abs(h_curr - h_next)
        rows.append({"회차": i, "가정수심(m)": round(h_curr, 3), "산정수심(m)": round(h_next, 3)})
        if diff < 0.001: break
        h_curr = h_next
    return h_curr, pd.DataFrame(rows)

h_surf_limit, df_st = run_sato_tanaka_details(1.35)
scour_status = "필요" if h_bed <= h_surf_limit else "불필요"

if scour_status == "필요":
    st.error(f"🚨 세굴방지공 설치 필요 (현재수심 {h_bed:.2f}m <= 한계수심 {h_surf_limit:.2f}m)")
else:
    st.success(f"✅ 원지반 안정 (현재수심 {h_bed:.2f}m > 한계수심 {h_surf_limit:.2f}m)")

# ==========================================
# 4. 2. 세굴방지공 계획
# ==========================================
st.markdown("---")
st.header("2. 세굴방지공 계획")

Sm_val, d_final, W_final_ton, B_sp, thickness = 0.0, 0.0, 0.0, 0.0, 0.0
control_factor = "파랑 (Wave)"

if scour_status == "필요":
    # 가. Isbash 규격 산정
    S_r = gamma_r / gamma_w
    theta_rad = math.radians(theta_angle)
    denom = 48 * (9.81**3) * (isbash_y**6) * ((S_r - 1.0)**3) * ((math.cos(theta_rad)-math.sin(theta_rad))**3)
    W_wave = (math.pi * gamma_r * (u_z**6)) / denom
    W_tidal = (math.pi * gamma_r * (v_tidal**6)) / denom
    W_max_kN = max(W_wave, W_tidal)
    d_final = safe_cbrt((6.0 * W_max_kN) / (math.pi * gamma_r))
    W_final_ton = W_max_kN / 9.81
    control_factor = "파랑 (Wave)" if W_wave >= W_tidal else "조류 (Tidal)"

    # 나. 세굴심도 상세 산정
    if structure_type == "직립제 (Vertical)" and location_type == "제간부 (Trunk)" and "Hughes" in wave_condition:
        Tp = 1.05 * T_input
        Lp = calc_wave_length(Tp, h_bed)
        kp = 2 * math.pi / Lp
        kph = kp * h_bed
        d_bar = h_bed / (9.81 * (Tp**2))

        try:
            csv_path = os.path.join(BASE_DIR, "tav_data_all.csv")
            df_tav = pd.read_csv(csv_path, skiprows=2, header=None)
            for col in df_tav.columns: df_tav[col] = pd.to_numeric(df_tav[col], errors='coerce')

            fig, ax = plt.subplots(figsize=(8.5, 7.5))
            
            # ε 곡선 데이터 설정 (원본 좌표 및 라벨 위치 최적화)
            eps_data = [
                ('0.01', 4, 5, 0.0013), ('0.008', 6, 7, 0.0016), ('0.007', 8, 9, 0.0020),
                ('0.006', 10, 11, 0.0026), ('0.005', 12, 13, 0.0035), ('0.004', 14, 15, 0.0050),
                ('0.003', 16, 17, 0.0075), ('0.002', 18, 19, 0.012)
            ]

            ax.text(0.0006, 1.05, r'$\epsilon =$', fontsize=11, fontweight='bold', zorder=4)

            for name, xc, yc, tx in eps_data:
                ex, ey = df_tav.iloc[:, xc].dropna().values, df_tav.iloc[:, yc].dropna().values
                if len(ex) > 1:
                    s_idx = np.argsort(ex)
                    p_eps = PchipInterpolator(ex[s_idx], ey[s_idx])
                    xs = np.logspace(np.log10(ex.min()), np.log10(ex.max()), 100)
                    ax.plot(xs, p_eps(xs), 'k-', linewidth=0.8, alpha=0.6, zorder=1)
                    
                    # 기울기를 수학적으로 계산하여 텍스트 각도 조정
                    lab_x = max(ex.min()*1.05, min(tx, ex.max()*0.95))
                    lab_y = float(p_eps(lab_x))
                    dx = lab_x * 0.05
                    dy = float(p_eps(lab_x + dx)) - lab_y
                    slope_angle = np.degrees(np.arctan2(dy, (np.log10(lab_x+dx)-np.log10(lab_x))*5))
                    
                    ax.text(lab_x, lab_y, rf"$\epsilon={name}$", fontsize=8.5, rotation=slope_angle, 
                            ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', pad=0.1), zorder=2)

            # MAXIMUM & AVERAGE (실선)
            for col_idx, label, text_y in [(0, 'MAXIMUM', 1.76), (2, 'AVERAGE', 1.70)]:
                rx, ry = df_tav.iloc[:, col_idx].dropna().values, df_tav.iloc[:, col_idx+1].dropna().values
                idx = np.argsort(rx)
                p_curve = PchipInterpolator(rx[idx], ry[idx])
                ax.plot(np.logspace(np.log10(rx.min()), np.log10(rx.max()), 100), p_curve(np.logspace(np.log10(rx.min()), np.log10(rx.max()), 100)), 'k-', linewidth=2.0, zorder=3)
                ax.text(0.00015, text_y, f"{label}\n$H_s/H_{{mo}}$", fontsize=10, bbox=dict(facecolor='white', edgecolor='none', pad=0.2), zorder=4)
                if label == 'AVERAGE': Hs_ratio = float(p_curve(d_bar))

            # 지시선 및 가이드 포인트
            ax.annotate('MAXIMUM\n$H_s/H_{mo}$', xy=(0.002, 1.48), xytext=(0.0003, 1.62), arrowprops=dict(arrowstyle="->"), fontsize=10)
            ax.annotate('AVERAGE\n$H_s/H_{mo}$', xy=(0.002, 1.34), xytext=(0.0003, 1.40), arrowprops=dict(arrowstyle="->"), fontsize=10)
            ax.text(0.012, 1.25, "PRE-BREAKING", fontsize=10)
            ax.annotate('', xy=(0.005, 1.15), xytext=(0.011, 1.25), arrowprops=dict(arrowstyle="->"))
            
            ax.axvline(d_bar, color='b', linestyle='--'); ax.axhline(Hs_ratio, color='b', linestyle='--')
            ax.plot(d_bar, Hs_ratio, 'bo', markersize=6, zorder=6)
            ax.text(d_bar*1.1, Hs_ratio+0.02, f"({d_bar:.2e}, {Hs_ratio:.2f})", color='b', fontweight='bold')

            ax.set_xscale('log'); ax.set_xlim(1e-4, 1e-1); ax.set_ylim(0.9, 1.8)
            ax.tick_params(direction='in', which='both', length=6)
            
            col_plt, _ = st.columns([1, 1])
            with col_plt: st.pyplot(fig)

            # 세굴심 최종 계산 (Hughes & Fowler)
            Hmo = H_input / Hs_ratio
            Urms = (9.81 * kp * Tp * Hmo) * (math.sqrt(2)/(4*math.pi*math.cosh(kph))) * (0.54 * math.cosh((1.5-kph)/2.8))
            Sm_val = round((Urms * Tp * 0.05) / (math.sinh(kph)**0.35), 2)
        except Exception as e: st.error(f"도표 생성 에러: {e}")

    elif structure_type == "경사제 (Rubble Mound)":
        Tp = 1.05 * T_input
        Sm_val = round(H_input * 0.01 * Cu_input * ((Tp * math.sqrt(9.81*H_input))/h_bed)**1.5, 2)
    
    # 보강폭 및 두께
    B_sp = (2.0 if "매설형" in protection_type else 3.0) * max(0.0, Sm_val)
    thickness = 2.0 * r_stone
    st.info(f"**최종 최대 세굴심도 ($S_m$): {Sm_val:.2f} m**")

    # 삽도 이미지 표시 (여백 4% 적용하여 지저분한 글씨 완전 제거)
    try:
        img = Image.open(os.path.join(BASE_DIR, "image_efd977.png"))
        img = ImageEnhance.Contrast(img).enhance(1.2); img = ImageEnhance.Sharpness(img).enhance(2.0)
        w, h = img.size
        if "매설형" in protection_type:
            cropped = img.crop((0, 0, int(w*0.46), h)) 
        else:
            cropped = img.crop((int(w*0.54), 0, w, h))
        st.markdown(f"**[{protection_type}] 기준 삽도**")
        c1, c2, c3 = st.columns([1.2, 1.5, 1.2])
        with c2: st.image(cropped, use_column_width=True)
    except: pass

# 5. 전체 결과 요약표 출력
with summary_placeholder.container():
    st.header("📋 전체 산정 결과 요약")
    if scour_status == "필요":
        sum_df = pd.DataFrame({
            "항목": ["구조물/보호공", "지배외력", "소요 직경 (d)", "소요 중량 (W)", "최대 세굴심 (Sm)", "보강폭 (Bsp)", "설계두께 (t)"],
            "결과": [f"{structure_type}/{protection_type}", control_factor, f"{d_final:.3f} m", f"{W_final_ton:.3f} ton", f"{Sm_val:.2f} m", f"{B_sp:.2f} m", f"{thickness:.2f} m"]
        }).set_index("항목")
        st.table(sum_df)
    else: st.success("✅ 원지반 안정으로 추가 보강이 불필요합니다.")
