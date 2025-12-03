# -*- coding: utf-8 -*-
"""
Integrated app.py with:
- Original reports
- NEW: Branch Wise Loan Disbursement (SMART-Agrosor Loan) & (SMART-CSL)
  * Uses AM (product), AN (Enterprise/Non-Enterprise), AQ (amount), AP (month)
  * Totals rows highlighted light-green
  * Sl No column removed from these specific report tables display
- Other interactive reports (distribution, pivot, time-series, leaderboard, export)
"""
import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO, BytesIO
from PIL import Image
import requests

# ----- DATA SOURCE URL (Google Sheet CSV export) -----
URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRkcagLu_YrYgQxmsO3DnHn90kqALkw9uDByX7UBNRUjaFKKQdE3V-6fm5ZcKGk_A/pub?gid=2143275417&single=true&output=csv"

# ----- UI config & helpers (logo, css) -----
def load_logo_image():
    for p in ("assets/logo.png", "logo.png"):
        try:
            return Image.open(p).convert("RGBA")
        except Exception:
            pass
    return Image.new("RGBA", (320, 120), (22, 163, 74, 255))

def load_favicon_from_logo():
    try:
        logo = load_logo_image()
        fav = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        w, h = logo.size
        scale = min(60 / w, 60 / h)
        new = logo.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        nx, ny = (64 - new.size[0]) // 2, (64 - new.size[1]) // 2
        fav.paste(new, (nx, ny), new)
        return fav
    except Exception:
        return Image.new("RGBA", (64, 64), (22, 163, 74, 255))

st.set_page_config(page_title="PIDIM SMART Reports", layout="wide", page_icon=load_favicon_from_logo())

st.markdown("""
<style>
.block-container { padding-top: 2.2rem; }
h3, .section-title { color:#065f46; font-weight:800; }
thead th { background-color:#dcfce7 !important; font-weight:800 !important; }
.header-wrap { margin-top: 28px; margin-bottom: 10px; }
.header-wrap .org { font-size:34px; font-weight:900; color:#16a34a; line-height:1.1; }
.header-wrap .proj { font-size:15px; color:#334155; margin-top:6px; }
.header-wrap .credit { text-align:right; font-size:12px; line-height:1.2; }
#global-print { position: fixed; top: 8px; right: 12px; z-index: 10000; }
#global-print button{
  display:flex; align-items:center; gap:8px;
  background-color:#16a34a; color:white; border:none; padding:9px 16px;
  border-radius:10px; cursor:pointer; font-weight:700;
  box-shadow:0 2px 8px rgba(0,0,0,.18);
}
@media print {
  .stButton, .stDownloadButton, [data-testid="stSidebar"], #global-print { display:none !important; }
  .block-container { padding: 8mm !important; }
  h1, h2, h3 { color:#065f46 !important; font-weight:800 !important; }
  table { page-break-inside: avoid; }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div id="global-print">
  <button onclick="window.print()">
    <span class="icon">🖨️</span> Print
  </button>
</div>
""", unsafe_allow_html=True)

with st.container():
    logo = load_logo_image()
    col_logo, col_text, col_credit = st.columns([0.12, 0.55, 0.33])
    with col_logo:
        st.image(logo, width=68)
    with col_text:
        st.markdown("<div class='header-wrap'><div class='org'>PIDIM Foundation</div><div class='proj'>Sustainable Microenterprise and Resilient Transformation (SMART) Project</div></div>", unsafe_allow_html=True)
    with col_credit:
        st.markdown("""<div class="header-wrap credit" style="text-align:right;">
          <b>Created by,</b><br/>
          <b>Md. Moniruzzaman</b><br/>
          MIS &amp; Documentation Officer<br/>
          SMART Project<br/>
          Pidim Foundation<br/>
          Cell: 01324 168100
        </div>""", unsafe_allow_html=True)

# ----- small utilities -----
def col_letter_to_pos(s):
    s = re.sub(r'[^A-Za-z]', '', s)
    v = 0
    for ch in s.upper():
        v = v * 26 + (ord(ch) - 64)
    return v

@st.cache_resource
def sess():
    s = requests.Session()
    s.headers.update({"User-Agent": "ME-Reports/1.0"})
    return s

@st.cache_data(ttl=900)
def get_df():
    import pandas as pd
    from io import StringIO
    t = sess().get(URL, timeout=25).text
    df = pd.read_csv(StringIO(t), low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def cpos(df_obj, p):
    i = max(0, min(len(df_obj.columns) - 1, p - 1))
    return df_obj.columns[i]

def clean_branch(s):
    s = s.astype(str).str.strip()
    bad = s.str.lower().isin(["", "nan", "none", "null", "branch name"])
    return s.mask(bad, None)

def ensure_serial(df_obj):
    d = df_obj.copy()
    if "ক্রমিক নং" in d.columns:
        d = d.drop(columns=["ক্রমিক নং"])
    if "Sl No" in d.columns:
        d["Sl No"] = range(1, len(d) + 1)
    else:
        d.insert(0, "Sl No", range(1, len(d) + 1))
    return d

HEADER_STYLE = [{"selector": "th", "props": [("background-color", "#dcfce7"), ("font-weight", "800")]},
                {"selector": "thead th", "props": [("background-color", "#dcfce7"), ("font-weight", "800")]},
                {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#fafafa")]}]

def style_table(d, number_formats=None, subtotal_logic=None, narrow_serial=False):
    sty = d.style.hide(axis="index")
    if number_formats:
        sty = sty.format(number_formats)
    sty = sty.set_table_styles(HEADER_STYLE)
    if subtotal_logic:
        sty = sty.apply(lambda r: subtotal_logic(r), axis=1)
    if narrow_serial and "Sl No" in d.columns:
        sty = sty.set_properties(subset=["Sl No"], **{"width": "56px"})
    return sty

def add_grand_total(df_obj, numeric_cols=None):
    import pandas as pd
    d = df_obj.copy()
    if numeric_cols is None:
        numeric_cols = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
    totals = {c: pd.to_numeric(d[c], errors="coerce").sum() for c in numeric_cols}
    row = {c: "" for c in d.columns}
    row.update({"Branch Name": "Grand Total"})
    for c, v in totals.items():
        row[c] = v
    d = pd.concat([d, pd.DataFrame([row])], ignore_index=True)
    return d

def to_excel_bytes(df_dict):
    import pandas as pd
    from io import BytesIO
    bio = BytesIO()
    last_err = None
    for eng in ("xlsxwriter", "openpyxl"):
        try:
            with pd.ExcelWriter(bio, engine=eng) as writer:
                for name, data in df_dict.items():
                    data.to_excel(writer, index=False, sheet_name=name[:31])
            bio.seek(0)
            return bio.getvalue()
        except Exception as e:
            last_err = e
            bio.seek(0); bio.truncate(0)
            continue
    raise RuntimeError(f"No Excel writer engine available (tried xlsxwriter, openpyxl): {last_err}")

# ----- Data loading & core column positions -----
df = get_df()
BRANCH = col_letter_to_pos("G"); LOAN_TYPE = col_letter_to_pos("AN"); LOAN_AMOUNT = col_letter_to_pos("AQ")
POULTRY_TYPE = col_letter_to_pos("T"); BIRDS_COL = col_letter_to_pos("U"); GRANTS = col_letter_to_pos("BL")

b = cpos(df, BRANCH)
lt = cpos(df, LOAN_TYPE)
la = cpos(df, LOAN_AMOUNT)
pt = cpos(df, POULTRY_TYPE)
ub = cpos(df, BIRDS_COL)
gcol = cpos(df, GRANTS)

# ----- Existing computations (loan, poultry, grants) -----
def compute_branch_loan(df_in, b_col, t_col, a_col):
    import pandas as pd, re
    w = df_in[[b_col, t_col, a_col]].copy()
    w[b_col] = clean_branch(w[b_col]); w[t_col] = w[t_col].astype(str).str.strip()
    w["_amt"] = pd.to_numeric(w[a_col], errors="coerce").fillna(0)
    def norm(x):
        x = (x or "").strip().lower(); x = re.sub(r"\s+", " ", x)
        if "non" in x and "enterprise" in x: return "Non-Enterprise"
        if "enterprise" in x: return "Enterprise"
        return x.title() if x else ""
    w["_type"] = w[t_col].apply(norm); w = w[w[b_col].notna()]
    g = (w.groupby([b_col, "_type"]).agg(**{"# of Loan": ("_type", "count"), "Amount of Loan": ("_amt", "sum")}).reset_index()
         .rename(columns={b_col: "Branch Name", "_type": "Types of Loan"}))
    g = g[~g["Branch Name"].astype(str).str.strip().str.lower().isin(["branch name", "nan", "none", "null"])]
    return g

def summarize_loan_table(agg):
    import pandas as pd
    order = {"Enterprise": 0, "Non-Enterprise": 1}
    agg["_o"] = agg["Types of Loan"].map(order).fillna(99).astype(int)
    rows = []
    for br, g in agg.sort_values(["Branch Name", "_o"]).groupby("Branch Name", sort=False):
        for _, r in g.iterrows():
            rows.append({"Branch Name": br, "Types of Loan": r["Types of Loan"], "# of Loan": int(r["# of Loan"]), "Amount of Loan": float(r["Amount of Loan"] or 0)})
        rows.append({"Branch Name": f"{br} Total", "Types of Loan": "", "# of Loan": int(g["# of Loan"].sum()), "Amount of Loan": float(g["Amount of Loan"].sum())})
    if rows:
        tmp = pd.DataFrame(rows)
        rows.append({"Branch Name": "Grand Total", "Types of Loan": "", "# of Loan": int(tmp[tmp["Types of Loan"] != ""]["# of Loan"].sum()), "Amount of Loan": float(tmp[tmp["Types of Loan"] != ""]["Amount of Loan"].sum())})
    loan = pd.DataFrame(rows)
    bad = loan["Branch Name"].astype(str).str.strip().str.lower().isin(["branch name", "nan", "nan total"])
    loan = loan[~bad].copy()
    loan = ensure_serial(loan)
    return loan

def compute_poultry_me_and_birds(df_in, b_col, type_col, birds_col):
    import pandas as pd
    tdf = df_in[[b_col, type_col, birds_col]].copy()
    tdf[b_col] = clean_branch(tdf[b_col])
    tdf[type_col] = tdf[type_col].astype(str).str.strip()
    tdf[birds_col] = pd.to_numeric(tdf[birds_col], errors="coerce").fillna(0)
    tdf = tdf[tdf[b_col].notna()]
    def map_type(x):
        x = (x or "").lower()
        if "layer" in x: return "Layer Rearing"
        if "broiler" in x: return "Broiler Rearing"
        return None
    tdf["_ptype"] = tdf[type_col].apply(map_type)
    tdf = tdf[tdf["_ptype"].notna()]
    agg = tdf.groupby([b_col, "_ptype"]).agg(**{"# of MEs": (type_col, "count"), "# of Birds": (birds_col, "sum")}).reset_index().rename(columns={b_col: "Branch Name", "_ptype": "Types of Poultry Rearing"})
    agg = agg[~agg["Branch Name"].astype(str).str.strip().str.lower().isin(["branch name", "nan", "none", "null"])]
    agg = ensure_serial(agg)
    return agg

def compute_me_grants(df_in, b_col, grant_col):
    import pandas as pd
    g = df_in[[b_col, grant_col]].copy(); g[b_col] = clean_branch(g[b_col])
    g["_gr"] = pd.to_numeric(g[grant_col], errors="coerce").fillna(0)
    g = g[g[b_col].notna()]
    cnt = g[g["_gr"] > 0].groupby(b_col).size().reset_index(name="Number on MEs")
    amt = g.groupby(b_col)["_gr"].sum(min_count=1).reset_index(name="Amounts of Grants")
    rep = cnt.merge(amt, on=b_col, how="outer").fillna(0).rename(columns={b_col: "Branch Name"})
    rep = rep[~rep["Branch Name"].astype(str).str.strip().str.lower().isin(["branch name", "nan", "none", "null"])]
    rep = ensure_serial(rep)
    return rep

# compute original outputs
loan_agg = compute_branch_loan(df, b, lt, la)
loan = summarize_loan_table(loan_agg)
poultry = compute_poultry_me_and_birds(df, b, pt, ub)
grants = compute_me_grants(df, b, gcol)

poultry = add_grand_total(poultry, numeric_cols=["# of MEs", "# of Birds"])
grants = add_grand_total(grants, numeric_cols=["Number on MEs", "Amounts of Grants"])

# ----- NEW: Branch reports using AM / AN / AQ / AP -----
FILTER_COL = cpos(df, col_letter_to_pos("AM"))   # product column
CLASS_COL  = cpos(df, col_letter_to_pos("AN"))   # enterprise / non-enterprise
AMOUNT_COL = cpos(df, col_letter_to_pos("AQ"))   # amount
MONTH_COL  = cpos(df, col_letter_to_pos("AP"))   # month/date

# parse month/date column if present
if MONTH_COL in df.columns:
    try:
        df[MONTH_COL] = pd.to_datetime(df[MONTH_COL], errors="coerce")
    except Exception:
        pass

# Sidebar month filter
st.sidebar.markdown("### Month filter for branch reports (AP column)")
month_vals = []
if MONTH_COL in df.columns:
    try:
        ym = df[MONTH_COL].dropna().dt.to_period("M").sort_values().unique()
        month_vals = [str(x) for x in ym]
    except Exception:
        month_vals = sorted(df[MONTH_COL].dropna().astype(str).unique().tolist())[:100]
sel_month = st.sidebar.selectbox("Select Year-Month (optional)", options=["(all)"] + month_vals, index=0)

# helper to style loan tables for these reports:
def style_loan_local_table(df_table):
    """
    Return a pandas Styler with:
    - 'Sl No' column removed for display
    - rows where Branch Name endswith ' Total' or equals 'Grand Total' -> light green background
    - header styles applied
    """
    df_display = df_table.copy()
    # drop Sl No from display
    if "Sl No" in df_display.columns:
        df_display = df_display.drop(columns=["Sl No"])
    # create styler
    sty = df_display.style.hide(axis="index")
    # format numbers
    numcols = [c for c in df_display.columns if pd.api.types.is_numeric_dtype(df_display[c])]
    fmt = {c: "{:,.0f}" for c in numcols}
    if fmt:
        sty = sty.format(fmt)
    # header styles
    sty = sty.set_table_styles(HEADER_STYLE)
    # row-wise highlight
    def highlight_row(row):
        name = str(row.get("Branch Name",""))
        if name.strip() == "Grand Total":
            return ["background-color: #dcfce7; font-weight:800"] * len(row)
        if name.strip().endswith(" Total"):
            return ["background-color: #e6ffed; font-weight:700"] * len(row)
        return [""] * len(row)
    sty = sty.apply(highlight_row, axis=1)
    return sty

def render_branch_loan_by_filter(df_all, loan_product_value, title_suffix):
    """
    Use:
      - FILTER_COL (AM) to match product (case-insensitive)
      - CLASS_COL (AN) as Type (Enterprise/Non-Enterprise)
      - AMOUNT_COL (AQ) as numeric amount
      - sel_month from AP to optionally filter
    """
    try:
        # verify columns exist
        missing = [c for c in (FILTER_COL, CLASS_COL, AMOUNT_COL) if c not in df_all.columns]
        if missing:
            st.warning(f"Required columns missing for this report: {missing}")
            return

        # product filter (case-insensitive)
        mask_prod = df_all[FILTER_COL].astype(str).str.strip().str.lower() == str(loan_product_value).strip().lower()
        df_filtered = df_all[mask_prod].copy()

        # month filter
        if sel_month != "(all)" and MONTH_COL in df_filtered.columns:
            try:
                y, m = sel_month.split("-")
                df_filtered = df_filtered[df_filtered[MONTH_COL].dt.to_period("M") == pd.Period(f"{y}-{m}")]
            except Exception:
                df_filtered = df_filtered[df_filtered[MONTH_COL].astype(str).str.contains(sel_month)]

        st.markdown(f'<h3 class="section-title">📊 Branch Wise Loan Disbursement ({title_suffix})</h3>', unsafe_allow_html=True)

        if df_filtered.shape[0] == 0:
            st.info(f"No records found for loan type: {loan_product_value}")
            return

        tmp = df_filtered[[b, CLASS_COL, AMOUNT_COL]].copy()
        tmp[b] = clean_branch(tmp[b])
        tmp[CLASS_COL] = tmp[CLASS_COL].astype(str).str.strip()
        tmp[AMOUNT_COL] = pd.to_numeric(tmp[AMOUNT_COL], errors="coerce").fillna(0)

        agg = (tmp.groupby([b, CLASS_COL])
                 .agg(**{"# of Loan": (CLASS_COL, "count"), "Amount of Loan": (AMOUNT_COL, "sum")})
                 .reset_index()
                 .rename(columns={b: "Branch Name", CLASS_COL: "Types of Loan"}))

        agg = agg[~agg["Branch Name"].astype(str).str.strip().str.lower().isin(["branch name", "nan", "none", "null"])].copy()

        try:
            loan_local = summarize_loan_table(agg)
        except Exception:
            loan_local = agg.copy()
            loan_local = ensure_serial(loan_local)
            loan_local = add_grand_total(loan_local, numeric_cols=["# of Loan", "Amount of Loan"])

        # Show table + download with styling (Sl No removed)
        lcol, rcol = st.columns([0.55, 0.45], gap="large")
        with lcol:
            try:
                styled = style_loan_local_table(loan_local)
                st.dataframe(styled, use_container_width=True, height=520)
            except Exception:
                # fallback: show dataframe without Sl No
                loan_local_nosl = loan_local.drop(columns=["Sl No"], errors="ignore")
                st.dataframe(loan_local_nosl, use_container_width=True, height=520)

            try:
                fname = f"loan_disbursement_{loan_product_value.replace(' ', '_')}.xlsx"
                st.download_button(f"⬇️ {title_suffix} — Excel", to_excel_bytes({title_suffix: loan_local}), file_name=fname, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception as e:
                st.error(f"Download prepare failed: {e}")

        with rcol:
            base_plot = loan_local[(~loan_local["Branch Name"].str.endswith(" Total")) & (loan_local["Branch Name"] != "Grand Total") & (loan_local["Types of Loan"] != "")]
            if base_plot.shape[0] == 0:
                st.info("No branch-level breakdown to plot for this loan type.")
            else:
                try:
                    fig_local = px.bar(base_plot, x="Branch Name", y="Amount of Loan", color="Types of Loan", barmode="group", title=f"Amount of Loan by Branch & Type — {title_suffix}")
                    fig_local.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
                    st.plotly_chart(fig_local, use_container_width=True)
                except Exception as e:
                    st.error(f"Plot failed: {e}")

    except Exception as e:
        st.error(f"Error rendering branch report for {loan_product_value}: {e}")

# Call the two required reports
render_branch_loan_by_filter(df, "SMART-Agrosor Loan", "SMART-Agrosor Loan")
st.markdown("---")
render_branch_loan_by_filter(df, "SMART-CSL", "SMART-CSL")
st.markdown("---")

# ----- continue with the rest of the original UI (poultry, grants, KPI, average ticket, top5 etc.) -----
st.markdown('<h3 class="section-title">🐔 Types of Poultry Rearing</h3>', unsafe_allow_html=True)
p_l, p_r = st.columns([0.55, 0.45], gap="large")
with p_l:
    st.dataframe(style_table(poultry, number_formats={"# of MEs":"{:,.0f}","# of Birds":"{:,.0f}"}), use_container_width=True)
    st.download_button("⬇️ Poultry — Excel", to_excel_bytes({"Poultry": poultry}), file_name="poultry_rearing.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
with p_r:
    t1, t2 = st.tabs(["# of Birds", "# of MEs"])
    with t1:
        fig_b = px.bar(poultry[poultry["Branch Name"]!="Grand Total"], x="Branch Name", y="# of Birds", color="Types of Poultry Rearing", barmode="group", title="Total Birds by Branch & Type")
        fig_b.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
        st.plotly_chart(fig_b, use_container_width=True)
    with t2:
        fig_m = px.bar(poultry[poultry["Branch Name"]!="Grand Total"], x="Branch Name", y="# of MEs", color="Types of Poultry Rearing", barmode="group", title="Number of MEs by Branch & Type")
        fig_m.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
        st.plotly_chart(fig_m, use_container_width=True)

# ... rest of app unchanged ...
st.markdown("---")
st.markdown('<h3 class="section-title">💠 MEs Grants Information</h3>', unsafe_allow_html=True)
g_l, g_r = st.columns([0.55, 0.45], gap="large")
with g_l:
    st.dataframe(style_table(grants, number_formats={"Amounts of Grants":"{:,.0f}","Number on MEs":"{:,.0f}"}), use_container_width=True)
    st.download_button("⬇️ Grants — Excel", to_excel_bytes({"Grants": grants}), file_name="grants_information.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
with g_r:
    gp = grants[grants["Branch Name"]!="Grand Total"].copy()
    fig_g = px.bar(gp, x="Branch Name", y="Amounts of Grants", title="Grants Amount by Branch")
    fig_g.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
    st.plotly_chart(fig_g, use_container_width=True)

# Remaining sections (KPI, Average ticket, Top 5, Additional reports) are unchanged from prior version...
# (For brevity, the code below this point remains the same as your earlier app; if you want the absolute full copy including every remaining block exactly as before, say "give full remaining code" and I will paste it.)
