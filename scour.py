import streamlit as st
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 페이지 설정
# ==========================================
st.set_page_config(page_title="세굴방지공 단면제원 계산", layout="wide", page_icon="🌊")

# --- [함수] 항만설계기준 분산관계식 시산법 파장(L) 산출 ---
def calc_wave_length(T, h):
    g = 9.81
    L0 = (g * (T**2)) / (2 * math.pi)
    L_curr = L0
    for _ in range(100):
        L_new = L0 * math.tanh(2 * math.pi * h / L_curr)
        if abs(L_new - L_curr) < 0.0001:
            break
        L_curr = L_new
    return L_curr

st.title("🌊 항외측 세굴방지공 단면제원 자동 계산")
st.markdown("### 산정 결과값(-) 표시 및 직립제/경사제 로직 완벽 분리")

# ==========================================
# 2. 입력부 (사이드바)
# ==========================================
st.sidebar.header("설계파랑 및 지반 제원 입력")
H_input = st.sidebar.number_input("유의파고 $H_s$ (m)", value=4.10, format="%.2f")
T_input = st.sidebar.number_input("유의주기 $T_s$ (sec)", value=10.83, format="%.2f")
h_bed = st.sidebar.number_input("현재 설계수심 $h$ (m)", value=22.51, format="%.2f")
ds_input = st.sidebar.number_input("저질 평균입경 $d_s$ (m)", value=0.00006, format="%.6f")

st.sidebar.markdown("---")
st.sidebar.subheader("구조물 및 보호공 조건")

structure_type = st.sidebar.radio("구조물 형식", ["직립제 (Vertical)", "경사제 (Rubble Mound)"])
location_type = st.sidebar.radio("적용 구간 (C.E.M 세굴심 산정용)", ["제두부 (Head)", "제간부 (Trunk)"])

Cu_input = 1.0 # 변수 초기화
# 선택된 조건에 따라 하위 메뉴 분기
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
    Cu_input = st.sidebar.number_input("경험계수 $C_u$ (표 참조)", value=1.00, step=0.1, format="%.2f")

protection_type = st.sidebar.radio("보호공 형식", ["매설형 (Buried Type)", "사석마운드형 (Berm Type)"])
r_stone = st.sidebar.number_input("피복재 공칭직경 $r$ ($d_{n50}$, m)", value=1.5, step=0.1)
B_width = st.sidebar.number_input("구조물 폭 또는 직경 $B$ (m)", value=15.0, step=0.1)

# 🌟 수립자/조류속 및 Isbash 공식 제원 입력
st.sidebar.markdown("---")
st.sidebar.subheader("수립자/조류속 및 Isbash 공식 제원")
gamma_r = st.sidebar.number_input("사석 단위중량 $\gamma_r$ ($kN/m^3$)", value=26.0, step=0.1)
gamma_w = st.sidebar.number_input("해수 단위중량 $\gamma_w$ ($kN/m^3$)", value=10.10, step=0.01)
isbash_y = st.sidebar.number_input("Isbash 계수 $y$ (매설: 0.86 / 돌출: 1.2)", value=0.86, step=0.01)
theta_angle = st.sidebar.number_input("사면경사 $\\theta$ (도)", value=33.69, step=0.01)
z_depth = st.sidebar.number_input("속도 산정 수심 $z$ (m, 해수면=0)", value=-5.0, step=0.1)
v_tidal = st.sidebar.number_input("설계 조류속 $V_c$ (m/s)", value=1.50, step=0.1)

# ==========================================
# 3. 기본 수리 제원 선계산 (NameError 방지)
# ==========================================
g_val = 9.81
L0_val = (g_val * (T_input**2)) / (2 * math.pi)
L_init = calc_wave_length(T_input, h_bed)
kh_init = 2 * math.pi * h_bed / L_init
n_val = 0.5 * (1 + (2 * kh_init) / math.sinh(2 * kh_init))
Ks_val = math.sqrt(1 / (math.tanh(kh_init) * 2 * n_val))
H0_prime = H_input / Ks_val

# 전역(Global) 수립자 속도 산정
# 1) 세굴심도(C.E.M) 산정용 바닥 최대유속 (z = -h)
u_bottom = (math.pi * H_input) / (T_input * math.sinh(kh_init))
# 2) 이쉬바쉬 규격검토용 임의수심(z) 최대유속 (엑셀 수식 반영)
term_z = 2 * math.pi * (z_depth + h_bed) / L_init
u_z = (math.pi * H_input / T_input) * (math.cosh(term_z) / math.sinh(kh_init))

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

tab1, tab2 = st.tabs(["표층 이동한계 ($\alpha=1.35$)", "완전 이동한계 ($\alpha=2.40$)"])
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

Sm_val = 0.0 

if scour_status == "필요":
    
    # 🌟 가. 세굴방지공 규격검토 (파랑 및 조류속 비교)
    st.subheader("가. 세굴방지공 규격검토 (Isbash 공식 적용)")
    
    # 공통 계수 계산
    S_r = gamma_r / gamma_w
    theta_rad = math.radians(theta_angle)
    denom_W = 48 * (g_val**3) * (isbash_y**6) * ((S_r - 1.0)**3) * ((math.cos(theta_rad) - math.sin(theta_rad))**3)
    
    st.markdown("#### 1) 파랑에 의한 규격 검토")
    st.markdown(f"**가) 파랑에 의한 수립자 속도($U_z$) 산정** (수심 $z = {z_depth:.2f}m$)")
    st.latex(rf"U_z = \frac{{\pi H}}{{T}} \frac{{\cosh[2\pi(z+h)/L]}}{{\sinh(2\pi h/L)}}")
    st.latex(rf"U_z = \frac{{\pi \times {H_input:.2f}}}{{{T_input:.2f}}} \times \frac{{\cosh[2\pi({z_depth:.2f} + {h_bed:.2f})/{L_init:.2f}]}}{{\sinh(2\pi \times {h_bed:.2f} / {L_init:.2f})}} = {u_z:.4f} \, m/s")
    
    # 파랑에 의한 Isbash 산정
    W_wave_kN = (math.pi * gamma_r * (u_z**6)) / denom_W
    W_wave_ton = W_wave_kN / g_val
    V_wave_m3 = W_wave_kN / gamma_r
    d_wave = ((6.0 * W_wave_kN) / (math.pi * gamma_r))**(1.0/3.0)
    
    st.markdown("**나) 피복석 소요 중량($W$) 및 규격($d$) 산정**")
    st.latex(rf"S_r = \frac{{\gamma_r}}{{\gamma_w}} = \frac{{{gamma_r:.3f}}}{{{gamma_w:.3f}}} = {S_r:.4f}")
    st.latex(r"W = \frac{\pi \gamma_r U_z^6}{48 g^3 y^6 (S_r - 1)^3 (\cos\theta - \sin\theta)^3}")
    st.latex(rf"W = \frac{{\pi \times {gamma_r:.2f} \times ({u_z:.4f})^6}}{{48 \times ({g_val})^3 \times ({isbash_y})^6 \times ({S_r:.4f} - 1)^3 \times (\cos{theta_angle}^\circ - \sin{theta_angle}^\circ)^3}} = {W_wave_kN:.4f} \, kN")
    st.latex(rf"d = \left( \frac{{6W}}{{\pi \gamma_r}} \right)^{{1/3}} = \left( \frac{{6 \times {W_wave_kN:.4f}}}{{\pi \times {gamma_r:.2f}}} \right)^{{1/3}} = {d_wave:.3f} \, m")
    
    st.markdown("#### 2) 조류에 의한 규격 검토")
    st.markdown("**가) 설계 조류속($V_c$) 적용**")
    st.latex(rf"V_c = {v_tidal:.2f} \, m/s \quad \text{{(설계 적용 조류속)}}")
    
    # 조류속에 의한 Isbash 산정
    W_current_kN = (math.pi * gamma_r * (v_tidal**6)) / denom_W
    W_current_ton = W_current_kN / g_val
    V_current_m3 = W_current_kN / gamma_r
    d_current = ((6.0 * W_current_kN) / (math.pi * gamma_r))**(1.0/3.0)
    
    st.markdown("**나) 피복석 소요 중량($W$) 및 규격($d$) 산정**")
    st.latex(r"W = \frac{\pi \gamma_r V_c^6}{48 g^3 y^6 (S_r - 1)^3 (\cos\theta - \sin\theta)^3}")
    st.latex(rf"W = \frac{{\pi \times {gamma_r:.2f} \times ({v_tidal:.2f})^6}}{{48 \times ({g_val})^3 \times ({isbash_y})^6 \times ({S_r:.4f} - 1)^3 \times (\cos{theta_angle}^\circ - \sin{theta_angle}^\circ)^3}} = {W_current_kN:.4f} \, kN")
    st.latex(rf"d = \left( \frac{{6W}}{{\pi \gamma_r}} \right)^{{1/3}} = \left( \frac{{6 \times {W_current_kN:.4f}}}{{\pi \times {gamma_r:.2f}}} \right)^{{1/3}} = {d_current:.3f} \, m")
    
    st.markdown("#### 3) 최종 규격 결정 (파랑 vs 조류 비교)")
    
    # 비교표 데이터 프레임 생성
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
    
    # 지배 조건 결정 및 출력
    d_final = max(d_wave, d_current)
    W_final_kN = max(W_wave_kN, W_current_kN)
    W_final_ton = max(W_wave_ton, W_current_ton)
    V_final_m3 = max(V_wave_m3, V_current_m3)
    control_factor = "파랑 (Wave)" if d_wave >= d_current else "조류 (Tidal Current)"
    
    st.info(f"**💡 결정 지배 요소:** {control_factor}\n\n**최종 필요 소요 직경 ($d$):** {d_final:.3f} m  /  **최종 필요 소요 중량 ($W$):** {W_final_kN:.3f} kN ({W_final_ton:.3f} ton, **{V_final_m3:.3f} m³**)\n\n*(설계 적용 피복재 공칭직경 $r = {r_stone:.2f} m$)*")

    # 🌟 나. 세굴심도 상세
    st.subheader("나. 세굴심도($S_m$) 산정 상세")
    
    # 분기 1: 직립제일 경우
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
                
                x_vals = np.array([0.0001, 0.001, 0.002, 0.003, 0.005, 0.01, 0.016, 0.02, 0.05, 0.1, 0.2])
                y_vals = np.array([1.50, 1.46, 1.33, 1.25, 1.16, 1.09, 1.04, 1.03, 1.01, 1.00, 1.00])
                
                Hs_ratio_raw = np.interp(d_bar, x_vals, y_vals)
                Hs_ratio = round(Hs_ratio_raw, 2)
                Hmo = H_input / Hs_ratio
                
                st.markdown(f"**$H_{{mo}}$ 산정 (Thompson and Vincent 1985 그래프 독취 자동화)**")
                st.latex(r"\bar{d} = \frac{d}{g T_p^2} = " + f"{d_bar:.3e}")
                st.latex(r"H_s / H_{mo} = " + f"{Hs_ratio:.2f} \quad \text{{(그래프 독취값)}}")
                st.latex(r"H_{mo} = \frac{H_s}{H_s / H_{mo}} = \frac{" + f"{H_input:.2f}" + r"}{" + f"{Hs_ratio:.2f}" + r"} = " + f"{Hmo:.2f} \, m")
                
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.plot(x_vals, y_vals, 'k-', linewidth=1.5, label='AVERAGE Curve ($H_s/H_{mo}$)')
                ax.plot(d_bar, Hs_ratio, 'ro', markersize=6, label=f'Current\n(Ratio: {Hs_ratio:.2f})')
                ax.axvline(x=d_bar, color='r', linestyle='--', alpha=0.6, linewidth=1)
                ax.axhline(y=Hs_ratio, color='r', linestyle='--', alpha=0.6, linewidth=1)
                
                ax.set_xscale('log')
                ax.set_xlim(1e-4, 2e-1)
                ax.set_ylim(0.9, 1.7)
                ax.set_xlabel(r'$\bar{d} = d / g T_p^2$', fontsize=10)
                ax.set_ylabel(r'$H_s / H_{mo}$', fontsize=10)
                ax.set_title("Variation of $H_s/H_{mo}$", fontsize=11)
                ax.grid(True, which="both", ls="--", alpha=0.5)
                ax.legend(loc="upper right", fontsize=8)
                
                col_graph, _ = st.columns([1, 1]) 
                with col_graph:
                    st.pyplot(fig) 
                
                term1 = math.sqrt(2) / (4 * math.pi * math.cosh(kph))
                term2 = 0.54 * math.cosh((1.5 - kph) / 2.8)
                Urms_m = (g_val * kp * Tp * Hmo) * term1 * term2
                
                st.markdown("**수평바닥유속의 rms 및 세굴심도 산정**")
                st.latex(r"(U_{rms})_m = \frac{\sqrt{2}}{4\pi \cosh(k_p h)} \times \left[ 0.54 \cosh\left(\frac{1.5 - k_p h}{2.8}\right) \right] \times g k_p T_p H_{mo}")
                st.latex(r"(U_{rms})_m = " + f"{Urms_m:.4f} \, m/s")
                
                Sm_val_raw = (Urms_m * Tp * 0.05) / (math.sinh(kph)**0.35)
                Sm_val = round(Sm_val_raw, 2)
                st.latex(r"S_m = (U_{rms})_m T_p \frac{0.05}{[\sinh(k_p h)]^{0.35}} = " + f"{Sm_val:.2f} \, m")
                
    # 분기 2: 경사제일 경우
    else: 
        if location_type == "제두부 (Head)":
            st.markdown("#### 경사제 제두부 세굴심도 산정 과정")
        else:
            st.markdown("#### 경사제 제간부 세굴심도 산정 과정 (제두부와 동일 수식 반영)")
            
        Tp = 1.05 * T_input
        st.latex(rf"T_p = 1.05 T_s = 1.05 \times {T_input:.2f} = {Tp:.2f} \, s")
        
        term = (Tp * math.sqrt(g_val * H_input)) / h_bed
        Sm_ratio = 0.01 * Cu_input * (term**1.5)
        Sm_val_raw = H_input * Sm_ratio
        Sm_val = round(Sm_val_raw, 2)
        
        st.latex(r"\frac{S_m}{H_s} = 0.01 C_u \left( \frac{T_p \sqrt{g H_s}}{h} \right)^{3/2}")
        st.latex(rf"\frac{{S_m}}{{H_s}} = 0.01 \times {Cu_input:.2f} \times \left( \frac{{{Tp:.2f} \times \sqrt{{{g_val} \times {H_input:.2f}}}}}{{{h_bed:.2f}}} \right)^{{1.5}} = {Sm_ratio:.4f}")
        st.latex(rf"S_m = {H_input:.2f} \times {Sm_ratio:.4f} = {Sm_val:.2f} \, m")

    final_sm_for_design = max(0.0, Sm_val)
    st.success(f"**최종 산정 최대 세굴심도 ($S_m$): {Sm_val:.2f} m**")

    # 🌟 다. 세굴방지 보강폭 및 두께 산정 (원복된 계수 방식)
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

else:
    st.write("원지반이 안정하여 추가적인 보강 계획이 필요하지 않습니다.")