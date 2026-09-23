%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import io

# -----------------------------------------------------------------------------
# 0. Streamlit 기본 페이지 설정
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ELISA Data Analysis",
    page_icon="🐢",
    layout="wide"
)

st.title("ELISA Automated Data Analysis Pipeline")
st.caption("SoftMax Pro Exact Precision Parity | Unweighted 4PL Fit | Automated Excel/CSV Reporting")
st.markdown("---")

# -----------------------------------------------------------------------------
# 1. 공통 연산 및 파싱 함수
# -----------------------------------------------------------------------------
def parse_softmax_wavelengths(text):
    rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    cols = [str(i) for i in range(1, 13)]

    lines = text.strip().split('\n')
    mat_450, mat_650, mat_reduced = [], [], []

    for line in lines:
        tokens = line.replace('\t', ' ').split()
        row_vals = []
        for t in tokens:
            try:
                row_vals.append(float(t))
            except ValueError:
                pass

        if len(row_vals) == 25:
            row_vals = row_vals[1:]

        if len(row_vals) == 24 and row_vals[:12] != list(range(1, 13)):
            mat_450.append(row_vals[:12])
            mat_650.append(row_vals[12:])
        elif len(row_vals) == 12 and row_vals != list(range(1, 13)):
            mat_reduced.append(row_vals)

    df_450 = pd.DataFrame(mat_450, index=rows, columns=cols) if len(mat_450) == 8 else None
    df_650 = pd.DataFrame(mat_650, index=rows, columns=cols) if len(mat_650) == 8 else None
    df_reduced = pd.DataFrame(mat_reduced, index=rows, columns=cols) if len(mat_reduced) == 8 else None

    return df_450, df_650, df_reduced

def render_combined_well_table(df_450, df_650, df_reduced):
    rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    cols = [str(i) for i in range(1, 13)]

    html = ["<table style='border-collapse: collapse; width: 100%; font-family: sans-serif; font-size: 11px; text-align: center;'>"]
    html.append("<tr style='background-color: #f2f2f2; font-weight: bold;'>")
    html.append("<th style='border: 1px solid #ccc; padding: 6px;'>Well</th>")
    for c in cols:
        html.append(f"<th style='border: 1px solid #ccc; padding: 6px;'>{c}</th>")
    html.append("</tr>")

    for r in rows:
        html.append("<tr>")
        html.append(f"<td style='border: 1px solid #ccc; background-color: #f9f9f9; font-weight: bold;'>{r}</td>")
        for c in cols:
            v450 = f"{df_450.loc[r, c]:.4f}" if (df_450 is not None and r in df_450.index and c in df_450.columns) else "-"
            v650 = f"{df_650.loc[r, c]:.4f}" if (df_650 is not None and r in df_650.index and c in df_650.columns) else "-"
            vred = f"{df_reduced.loc[r, c]:.4f}" if (df_reduced is not None and r in df_reduced.index and c in df_reduced.columns) else "-"

            cell_html = f"<td style='border: 1px solid #ddd; padding: 3px 1px; background-color: #ffffff; vertical-align: middle;'>"
            cell_html += f"<div style='color: #0d6efd; font-weight: 600;' title='450nm Raw'>{v450}</div>"
            cell_html += f"<div style='color: #6f42c1; font-weight: 600;' title='650nm Ref'>{v650}</div>"
            cell_html += f"<div style='color: #198754; font-weight: bold; border-top: 1px dashed #e0e0e0; margin-top: 2px; padding-top: 2px;' title='450nm-650nm Reduced'>{vred}</div>"
            cell_html += "</td>"
            html.append(cell_html)
        html.append("</tr>")
    html.append("</table>")
    return "".join(html)

def expand_well_selection(selection_str, df_index, df_columns):
    if not selection_str: return []
    rows, cols = list(df_index), list(df_columns)
    tokens = selection_str.replace(',', ' ').split()
    expanded_wells = []
    for t in tokens:
        t = t.strip().upper()
        if not t: continue
        if ':' in t or '-' in t:
            sep = ':' if ':' in t else '-'
            parts = t.split(sep)
            if len(parts) == 2:
                start, end = parts[0].strip(), parts[1].strip()
                if len(start) >= 2 and len(end) >= 2 and start[0] in rows and end[0] in rows:
                    try:
                        r_start, c_start = start[0], int(start[1:])
                        r_end, c_end = end[0], int(end[1:])
                        r_idx1, r_idx2 = min(rows.index(r_start), rows.index(r_end)), max(rows.index(r_start), rows.index(r_end))
                        c_min, c_max = min(c_start, c_end), max(c_start, c_end)
                        for r_i in range(r_idx1, r_idx2 + 1):
                            for c_i in range(c_min, c_max + 1):
                                w = f"{rows[r_i]}{c_i}"
                                if w not in expanded_wells and str(c_i) in cols:
                                    expanded_wells.append(w)
                        continue
                    except ValueError: pass
        if t.isdigit() and t in cols:
            for r in rows:
                w = f"{r}{t}"
                if w not in expanded_wells: expanded_wells.append(w)
            continue
        if len(t) == 1 and t in rows:
            for c in cols:
                w = f"{t}{c}"
                if w not in expanded_wells: expanded_wells.append(w)
            continue
        if len(t) >= 2 and t[0] in rows and t[1:] in cols:
            if t not in expanded_wells: expanded_wells.append(t)
    return expanded_wells

def process_blank_deduction(df_raw, apply_blank, blank_str):
    if df_raw is None: return None, 0.0, [], None
    valid_blanks = expand_well_selection(blank_str, df_raw.index, df_raw.columns)
    if not valid_blanks or not apply_blank: return df_raw.copy(), 0.0, [], None
    blank_vals = [df_raw.loc[w[0], w[1:]] for w in valid_blanks if w[0] in df_raw.index and w[1:] in df_raw.columns]
    if not blank_vals: return df_raw.copy(), 0.0, [], None
    blank_mean = np.mean(blank_vals)
    df_deducted = df_raw - blank_mean
    blank_summary_df = pd.DataFrame({
        '항목 (Item)': ['지정된 Blank 웰', '개별 측정 OD 값', 'Blank 평균 OD (차감액)'],
        '수치 (Value)': [", ".join(valid_blanks), ", ".join([f"{v:.4f}" for v in blank_vals]), f"{blank_mean:.6f}"]
    })
    return df_deducted, blank_mean, valid_blanks, blank_summary_df

def calculate_qc_metrics(df_source, qc_wells_str, divisor=10.0, source_tag=""):
    qc_wells = expand_well_selection(qc_wells_str, df_source.index, df_source.columns)
    qc_values = [df_source.loc[w[0], w[1:]] for w in qc_wells if w[0] in df_source.index and w[1:] in df_source.columns]
    if not qc_values: return None, [], 0.0
    qc_mean = np.mean(qc_values)
    qc_std = np.std(qc_values, ddof=1) if len(qc_values) > 1 else 0.0
    qc_cv = (qc_std / qc_mean) * 100 if qc_mean != 0 else 0.0
    adjusted_qc = qc_mean / divisor if divisor != 0 else 0.0
    qc_summary_df = pd.DataFrame({
        '항목 (Item)': [
            'QC 연산 적용 데이터 기준', '지정된 QC 웰', '개별 OD 수치',
            '평균 (Mean OD)', '표본표준편차 (Std)', '변동계수 (CV %)', f'보정 평균 Target OD (/{divisor:.1f})'
        ],
        '수치 (Value)': [
            source_tag, ", ".join(qc_wells), ", ".join([f"{v:.4f}" for v in qc_values]),
            f"{qc_mean:.4f}", f"{qc_std:.6f}", f"{qc_cv:.2f}%", f"{adjusted_qc:.6f}"
        ]
    })
    return qc_summary_df, qc_wells, adjusted_qc

def parse_sample_cols(cols_str):
    if not cols_str: return [str(i) for i in range(1, 11)]
    tokens = cols_str.replace(',', ' ').split()
    parsed_cols = []
    for t in tokens:
        t = t.strip()
        if ':' in t or '-' in t:
            sep = ':' if ':' in t else '-'
            parts = t.split(sep)
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                start_c, end_c = int(parts[0]), int(parts[1])
                for c in range(min(start_c, end_c), max(start_c, end_c) + 1):
                    if 1 <= c <= 12 and str(c) not in parsed_cols:
                        parsed_cols.append(str(c))
        elif t.isdigit() and 1 <= int(t) <= 12:
            if t not in parsed_cols: parsed_cols.append(t)
    return sorted(parsed_cols, key=lambda x: int(x))

def four_param_logistic(x, A, B, C, D):
    return D + (A - D) / (1 + (x / C)**B)

def interp_x(y, A, B, C, D):
    try:
        val = (A - y) / (y - D)
        if val <= 0: return np.nan
        return C * (val ** (1 / B))
    except: return np.nan

def fit_4pl_curve(x_data, y_data):
    x_data = np.array(x_data, dtype=float)
    y_data = np.array(y_data, dtype=float)
    p0 = [max(y_data), 1.0, np.median(x_data), min(y_data)]
    try:
        popt, _ = curve_fit(four_param_logistic, x_data, y_data, p0=p0, maxfev=10000)
        residuals = y_data - four_param_logistic(x_data, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        return popt, r2
    except Exception: return None, 0.0

# -----------------------------------------------------------------------------
# STEP 1: Multi-Wavelength Raw Data Upload
# -----------------------------------------------------------------------------
st.header("1. Multi-Wavelength Raw Data Upload")

raw_text = ""
file_col, text_col = st.columns([1, 1])

with file_col:
    uploaded_file = st.file_uploader("SoftMax Pro Raw Text (.txt) 파일 선택", type=["txt"])
    if uploaded_file is not None:
        raw_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")

with text_col:
    raw_text = st.text_area("또는 텍스트 직접 붙여넣기", value=raw_text, height=130)

if not raw_text.strip():
    st.info("💡 분석을 시작하려면 SoftMax Pro .txt 파일을 업로드하거나 텍스트를 입력하세요.")
    st.stop()

df_450, df_650, df_reduced = parse_softmax_wavelengths(raw_text)
df_plate = df_reduced if df_reduced is not None else df_450

if df_plate is None:
    st.error("⚠️ 파싱 가능한 ELISA OD 데이터 형식(8x12 Plate)을 찾을 수 없습니다.")
    st.stop()

st.success("Raw Data 파싱 완료!")

view_mode = st.radio(
    "시각화 방식 선택:",
    options=[
        '웰별 3개 수치 한눈에 보기 (450nm / 650nm / Reduced)',
        '450-650nm Reduced OD 단독 보기',
        '450nm Raw OD 단독 보기',
        '650nm Ref OD 단독 보기'
    ],
    horizontal=True
)

if '3개 수치 한눈에 보기' in view_mode:
    st.markdown("""
    <div style='margin-bottom: 8px; font-size: 13px;'>
        <span style='color: #0d6efd; font-weight: bold;'>■ 450nm Raw (Blue)</span> &nbsp;|&nbsp; 
        <span style='color: #6f42c1; font-weight: bold;'>■ 650nm Reference (Purple)</span> &nbsp;|&nbsp; 
        <span style='color: #198754; font-weight: bold;'>■ 450nm - 650nm Reduced (Green)</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(render_combined_well_table(df_450, df_650, df_reduced), unsafe_allow_html=True)
elif 'Reduced' in view_mode and df_reduced is not None:
    st.dataframe(df_reduced.style.format("{:.4f}").background_gradient(cmap="Greens"), use_container_width=True)
elif '450nm Raw' in view_mode and df_450 is not None:
    st.dataframe(df_450.style.format("{:.4f}").background_gradient(cmap="Blues"), use_container_width=True)
elif '650nm Ref' in view_mode and df_650 is not None:
    st.dataframe(df_650.style.format("{:.4f}").background_gradient(cmap="Purples"), use_container_width=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# STEP 2: Blank Deduction & QC Analysis
# -----------------------------------------------------------------------------
st.header("2. Blank & QC Analysis Module")

col1, col2, col3 = st.columns([1, 1.2, 1])
with col1:
    use_blank = st.checkbox("Blank Deduction 적용", value=True)
    blank_wells_str = st.text_input("Blank Wells", value="12", help="예: 12, H12, A12:H12")
with col2:
    qc_base_option = st.radio("QC Target OD 계산 기준", options=['Blank 차감 후 최종 OD 기준', '450-650nm 원본 OD 기준 (Blank 차감 전)'], index=0)
with col3:
    qc_wells_str = st.text_input("QC Wells", value="A11, B11", help="예: A11, B11")
    divisor = st.number_input("Target OD Divisor", value=10.0, step=1.0)

df_final_od, blank_val, active_blanks, blank_summary_df = process_blank_deduction(df_plate, use_blank, blank_wells_str)
if 'Blank 차감 전' in qc_base_option:
    df_qc_source = df_plate.copy()
    tag = "450-650nm 원본 OD (Blank 차감 전)"
else:
    df_qc_source = df_final_od.copy()
    tag = "Blank 차감 후 최종 OD"

qc_summary_df, active_qc_wells, target_od = calculate_qc_metrics(df_qc_source, qc_wells_str, divisor=divisor, source_tag=tag)
c_left, c_right = st.columns([1, 1])
with c_left:
    st.subheader("Final Plate Map (4PL 분석용)")
    def highlight_plate(df):
        style_df = pd.DataFrame('', index=df.index, columns=df.columns)
        for bw in active_blanks:
            if bw[0] in df.index and bw[1:] in df.columns:
                style_df.loc[bw[0], bw[1:]] = 'background-color: #ffffcc; font-weight: bold; color: #8a6d3b;'
        for qw in active_qc_wells:
            if qw[0] in df.index and qw[1:] in df.columns:
                style_df.loc[qw[0], qw[1:]] = 'background-color: #ffaaaa; font-weight: bold; color: #990000;'
        return style_df
    st.dataframe(df_final_od.style.format("{:.4f}").background_gradient(cmap="Blues").apply(highlight_plate, axis=None), use_container_width=True)
with c_right:
    st.subheader(f"⚙️ QC 요약 (Target OD: :red[{target_od:.6f}])")
    if blank_summary_df is not None and use_blank:
        st.markdown("**[Blank Summary]**")
        st.dataframe(blank_summary_df, hide_index=True, use_container_width=True)
    if qc_summary_df is not None:
        st.markdown("**[QC Target Summary]**")
        st.dataframe(qc_summary_df, hide_index=True, use_container_width=True)
st.markdown("---")

# -----------------------------------------------------------------------------
# STEP 3: Multi-Sample 4PL & InterpX Pipeline
# -----------------------------------------------------------------------------
st.header("3. 4PL Regression & InterpX Titer Analysis")

# 세션 상태 세팅 초기화
if 'col_dilution_config' not in st.session_state:
    st.session_state.col_dilution_config = {str(i): {'start': 300.0, 'fold': 3.0} for i in range(1, 13)}
    for i in range(1, 13):
        st.session_state[f"start_{i}"] = 300.0
        st.session_state[f"fold_{i}"] = 3.0

# 💡 일괄 적용 콜백 함수 (개별 위젯 상태까지 한번에 초기화)
def apply_batch_dilution():
    b_start = st.session_state.get('base_start', 300.0)
    b_fold = st.session_state.get('base_fold', 3.0)
    for c in range(1, 13):
        st.session_state.col_dilution_config[str(c)] = {'start': b_start, 'fold': b_fold}
        st.session_state[f"start_{c}"] = b_start
        st.session_state[f"fold_{c}"] = b_fold

d_col1, d_col2, d_col3, d_col4 = st.columns([1.2, 1, 1, 1])
with d_col1:
    sample_cols_str = st.text_input("분석 샘플 열 범위", value="1-10", help="예: 1-10, 1,2,5")
with d_col2:
    st.number_input("일괄 시작 희석비", value=300.0, step=10.0, key="base_start")
with d_col3:
    st.number_input("일괄 Fold 배수", value=3.0, step=0.5, key="base_fold")
with d_col4:
    st.write(" ")
    st.write(" ")
    st.button("전체 열 희석 일괄 적용", on_click=apply_batch_dilution)

with st.expander("개별 열 희석 조건 변경 (선택 사항)"):
    sub_c1, sub_c2, sub_c3 = st.columns([1, 1, 1])
    with sub_c1:
        sel_col = st.selectbox("수정할 샘플 열", options=[str(i) for i in range(1, 13)])

    def update_single_dilution():
        st.session_state.col_dilution_config[sel_col] = {
            'start': st.session_state[f"start_{sel_col}"],
            'fold': st.session_state[f"fold_{sel_col}"]
        }

    with sub_c2:
        st.number_input("시작 희석비", key=f"start_{sel_col}", on_change=update_single_dilution)
    with sub_c3:
        st.number_input("Fold 배수", key=f"fold_{sel_col}", on_change=update_single_dilution)

sample_cols = parse_sample_cols(sample_cols_str)
summary_list = []
fit_results = {}
rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

for col in sample_cols:
    cfg = st.session_state.col_dilution_config[col]
    s_dil, f_val = cfg['start'], cfg['fold']
    x_series = [s_dil * (f_val ** i) for i in range(8)]
    y_series = [df_final_od.loc[r, col] for r in rows if r in df_final_od.index]
    popt, r2 = fit_4pl_curve(x_series, y_series)
    if popt is not None:
        A, B, C, D = popt
        calc_x = interp_x(target_od, A, B, C, D)
        fit_results[col] = {'x': x_series, 'y': y_series, 'popt': popt, 'r2': r2, 'interp_x': calc_x, 'start': s_dil, 'fold': f_val}
        summary_list.append({
            'Column': f"Column {col}", 'Start Dilution': s_dil, 'Fold Factor': f_val,
            'Bottom (A)': round(A, 6), 'HillSlope (B)': round(B, 6), 'EC50 (C)': round(C, 6), 'Top (D)': round(D, 6),
            'R2': round(r2, 5), f'InterpX Titer ({target_od:.4f} OD)': round(calc_x, 2) if not np.isnan(calc_x) else "NaN"
        })
    else:
        summary_list.append({
            'Column': f"Column {col}", 'Start Dilution': s_dil, 'Fold Factor': f_val,
            'Bottom (A)': "-", 'HillSlope (B)': "-", 'EC50 (C)': "-", 'Top (D)': "-",
            'R2': "Fit Failed", f'InterpX Titer ({target_od:.4f} OD)': "-"
        })

df_summary = pd.DataFrame(summary_list)
st.subheader("샘플 열별 4PL & InterpX 요약 표")
st.dataframe(df_summary, hide_index=True, use_container_width=True)

st.subheader("📈 Combined 4PL Fit Overlay Plot")

# 💡 50% 수준으로 아담하게 축소된 그래프 (강제 확대 방지)
fig, ax = plt.subplots(figsize=(6.0, 3.5))
cmap = plt.cm.get_cmap('tab10', max(len(fit_results), 1))
ax.axhline(target_od, color='#d9534f', linestyle='--', linewidth=1.5, label=f'Target OD ({target_od:.4f})', zorder=2)

for idx, (col, res) in enumerate(fit_results.items()):
    x_data, y_data = res['x'], res['y']
    popt, r2, calc_x = res['popt'], res['r2'], res['interp_x']
    x_smooth = np.logspace(np.log10(min(x_data)), np.log10(max(x_data)), 300)
    y_smooth = four_param_logistic(x_smooth, *popt)
    color = cmap(idx)
    ax.plot(x_smooth, y_smooth, color=color, linewidth=1.8, label=f'Col {col} ({res["start"]:.0f}/{res["fold"]:.0f}x, R²={r2:.3f})')
    ax.scatter(x_data, y_data, color=color, s=25, alpha=0.7)
    if not np.isnan(calc_x):
        ax.scatter([calc_x], [target_od], color=color, edgecolor='black', marker='*', s=130, zorder=6)

ax.set_xscale('log')
ax.set_xlabel('Dilution Factor (X axis)')
ax.set_ylabel('Absorbance OD (Y axis)')
ax.set_title('4PL Fit Curves & InterpX Titers (SoftMax Pro Parity)')
ax.grid(True, which="both", ls="--", alpha=0.4)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
plt.tight_layout()

# 45% 폭 레이아웃에 배치하고 use_container_width=False 적용
graph_col, empty_col = st.columns([0.45, 0.55])
with graph_col:
    st.pyplot(fig, use_container_width=False)

st.markdown("---")

# -----------------------------------------------------------------------------
# STEP 4: Export Reports
# -----------------------------------------------------------------------------
st.header("4. Export Analysis Reports")
exp_col1, exp_col2, exp_col3 = st.columns([1, 1, 1])
with exp_col1:
    export_format = st.radio("파일 형식", options=['Excel 파일 (.xlsx - 전체 시트 통합)', 'CSV 파일 (.csv - InterpX 결과 전용)'])
with exp_col2:
    file_name = st.text_input("저장 파일명 (확장자 제외)", value="ELISA_Analysis_Report")
with exp_col3:
    st.write(" ")
    st.write(" ")
    base_filename = file_name.strip() or "ELISA_Analysis_Report"
    if 'Excel' in export_format:
        output_buffer = io.BytesIO()
        with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
            df_summary.to_excel(writer, sheet_name='InterpX_Titer_Results', index=False)
            df_final_od.to_excel(writer, sheet_name='Plate_Map_Final_OD')
            start_row = 0
            if blank_summary_df is not None:
                blank_summary_df.to_excel(writer, sheet_name='QC_and_Blank_Summary', index=False, startrow=start_row)
                start_row += len(blank_summary_df) + 3
            if qc_summary_df is not None:
                qc_summary_df.to_excel(writer, sheet_name='QC_and_Blank_Summary', index=False, startrow=start_row)
        excel_data = output_buffer.getvalue()
        st.download_button(
            label="Excel 보고서 다운로드",
            data=excel_data,
            file_name=f"{base_filename}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )
    else:
        csv_data = df_summary.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="CSV 보고서 다운로드",
            data=csv_data,
            file_name=f"{base_filename}_InterpX_Results.csv",
            mime="text/csv",
            type="primary"
        )
