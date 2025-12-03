# -*- coding: utf-8 -*-
import re
import streamlit as st, pandas as pd, plotly.express as px, requests
from io import StringIO, BytesIO
from PIL import Image
import numpy as np

URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRkcagLu_YrYgQxmsO3DnHn90kqALkw9uDByX7UBNRUjaFKKQdE3V-6fm5ZcKGk_A/pub?gid=2143275417&single=true&output=csv"

def load_logo_image():
    for p in ("assets/logo.png","logo.png"):
        try:
            return Image.open(p).convert("RGBA")
        except Exception:
            pass
    return Image.new("RGBA",(320,120),(22,163,74,255))

def load_favicon_from_logo():
    try:
        logo = load_logo_image()
        fav = Image.new("RGBA",(64,64),(0,0,0,0))
        w,h = logo.size
        scale = min(60/w, 60/h)
        new = logo.resize((max(1,int(w*scale)), max(1,int(h*scale))), Image.LANCZOS)
        nx, ny = (64-new.size[0])//2, (64-new.size[1])//2
        fav.paste(new, (nx, ny), new)
        return fav
    except Exception:
        return Image.new("RGBA",(64,64),(22,163,74,255))

st.set_page_config(page_title="PIDIM SMART Reports", layout="wide", page_icon=load_favicon_from_logo())

# ===== CSS =====
st.markdown("""
<style>
/* More breathing room at the very top */
.block-container { padding-top: 2.2rem; }

/* Section titles */
h3, .section-title { color:#065f46; font-weight:800; }

/* Table header styling */
thead th { background-color:#dcfce7 !important; font-weight:800 !important; }

/* Header row pushed further down, plus larger org title */
.header-wrap { margin-top: 28px; margin-bottom: 10px; }
.header-wrap .org { font-size:34px; font-weight:900; color:#16a34a; line-height:1.1; }
.header-wrap .proj { font-size:15px; color:#334155; margin-top:6px; }
.header-wrap .credit { text-align:right; font-size:12px; line-height:1.2; }

/* Fixed Print button (green) at very top-right */
#global-print { position: fixed; top: 8px; right: 12px; z-index: 10000; }
#global-print button{
  display:flex; align-items:center; gap:8px;
  background-color:#16a34a; color:white; border:none; padding:9px 16px;
  border-radius:10px; cursor:pointer; font-weight:700;
  box-shadow:0 2px 8px rgba(0,0,0,.18);
}
#global-print button .icon { font-size:16px; line-height:1; }

@media print {
  .stButton, .stDownloadButton, [data-testid="stSidebar"], #global-print { display:none !important; }
  .block-container { padding: 8mm !important; }
  h1, h2, h3 { color:#065f46 !important; font-weight:800 !important; }
  table { page-break-inside: avoid; }
}
</style>
""", unsafe_allow_html=True)

# ===== Top-right Print button (HTML) =====
st.markdown("""
<div id="global-print">
  <button onclick="window.print()">
    <span class="icon">🖨️</span> Print
  </button>
</div>
""", unsafe_allow_html=True)

# ===== Header UI =====
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

def col_letter_to_pos(s):
    s=re.sub(r'[^A-Za-z]','',s); v=0
    for ch in s.upper(): v=v*26+(ord(ch)-64)
    return v

# Column map
BRANCH=col_letter_to_pos("G"); LOAN_TYPE=col_letter_to_pos("AN"); LOAN_AMOUNT=col_letter_to_pos("AQ")
POULTRY_TYPE=col_letter_to_pos("T"); BIRDS_COL=col_letter_to_pos("U")
GRANTS=col_letter_to_pos("BL")

@st.cache_resource
def sess():
    import requests
    s=requests.Session(); s.headers.update({"User-Agent":"ME-Reports/1.0"}); return s

@st.cache_data(ttl=900)
def get_df():
    import pandas as pd
    from io import StringIO
    t=sess().get(URL, timeout=25).text
    df=pd.read_csv(StringIO(t), low_memory=False)
    df.columns=[str(c).strip() for c in df.columns]
    return df

def cpos(df,p): i=max(0, min(len(df.columns)-1,p-1)); return df.columns[i]
def clean_branch(s):
    s=s.astype(str).str.strip(); bad=s.str.lower().isin(["","nan","none","null","branch name"]); return s.mask(bad,None)

def ensure_serial(df):
    import pandas as pd
    d=df.copy()
    if "ক্রমিক নং" in d.columns: d = d.drop(columns=["ক্রমিক নং"])
    if "Sl No" in d.columns: d["Sl No"] = range(1, len(d)+1)
    else: d.insert(0,"Sl No", range(1,len(d)+1))
    return d

HEADER_STYLE = [{"selector":"th","props":[("background-color","#dcfce7"),("font-weight","800")]},
                {"selector":"thead th","props":[("background-color","#dcfce7"),("font-weight","800")]},
                {"selector":"tbody tr:nth-child(even)","props":[("background-color","#fafafa")]}]

def style_table(d, number_formats=None, subtotal_logic=None, narrow_serial=False):
    sty = d.style.hide(axis="index")
    if number_formats: sty = sty.format(number_formats)
    sty = sty.set_table_styles(HEADER_STYLE)
    if subtotal_logic: sty = sty.apply(lambda r: subtotal_logic(r), axis=1)
    if narrow_serial and "Sl No" in d.columns: sty = sty.set_properties(subset=["Sl No"], **{"width":"56px"})
    return sty

# ===== Builders =====
def compute_branch_loan(df_in,b,t,a):
    import pandas as pd, re
    w=df_in[[b,t,a]].copy()
    w[b]=clean_branch(w[b]); w[t]=w[t].astype(str).str.strip()
    w["_amt"]=pd.to_numeric(w[a], errors="coerce").fillna(0)
    def norm(x):
        x=(x or "").strip().lower(); x=re.sub(r"\\s+"," ",x)
        if "non" in x and "enterprise" in x: return "Non-Enterprise"
        if "enterprise" in x: return "Enterprise"
        return x.title() if x else ""
    w["_type"]=w[t].apply(norm); w=w[w[b].notna()]
    g=(w.groupby([b,"_type"]).agg(**{"# of Loan":("_type","count"),"Amount of Loan":("_amt","sum")}).reset_index()
        .rename(columns={b:"Branch Name","_type":"Types of Loan"}))
    g=g[~g["Branch Name"].astype(str).str.strip().str.lower().isin(["branch name","nan","none","null"])]
    return g

def summarize_loan_table(agg):
    import pandas as pd
    order={"Enterprise":0,"Non-Enterprise":1}; agg["_o"]=agg["Types of Loan"].map(order).fillna(99).astype(int)
    rows=[]
    for br,g in agg.sort_values(["Branch Name","_o"]).groupby("Branch Name", sort=False):
        for _,r in g.iterrows():
            rows.append({"Branch Name":br,"Types of Loan":r["Types of Loan"],"# of Loan":int(r["# of Loan"]),"Amount of Loan":float(r["Amount of Loan"] or 0)})
        rows.append({"Branch Name":f"{br} Total","Types of Loan":"","# of Loan":int(g["# of Loan"].sum()),"Amount of Loan":float(g["Amount of Loan"].sum())})
    if rows:
        tmp=pd.DataFrame(rows)
        rows.append({"Branch Name":"Grand Total","Types of Loan":"","# of Loan":int(tmp[tmp["Types of Loan"]!=""]["# of Loan"].sum()),"Amount of Loan":float(tmp[tmp["Types of Loan"]!=""]["Amount of Loan"].sum())})
    loan=pd.DataFrame(rows)
    bad=loan["Branch Name"].astype(str).str.strip().str.lower().isin(["branch name","nan","nan total"])
    loan=loan[~bad].copy()
    loan=ensure_serial(loan)
    return loan

def compute_poultry_me_and_birds(df_in, b, type_col, birds_col):
    import pandas as pd
    t=df_in[[b, type_col, birds_col]].copy()
    t[b]=clean_branch(t[b])
    t[type_col]=t[type_col].astype(str).str.strip()
    t[birds_col]=pd.to_numeric(t[birds_col], errors="coerce").fillna(0)
    t=t[t[b].notna()]
    def map_type(x):
        x=(x or "").lower()
        if "layer" in x: return "Layer Rearing"
        if "broiler" in x: return "Broiler Rearing"
        return None
    t["_ptype"]=t[type_col].apply(map_type)
    t=t[t["_ptype"].notna()]
    agg=t.groupby([b,"_ptype"]).agg(**{
        "# of MEs": (type_col, "count"),
        "# of Birds": (birds_col, "sum"),
    }).reset_index().rename(columns={b:"Branch Name","_ptype":"Types of Poultry Rearing"})
    agg=agg[~agg["Branch Name"].astype(str).str.strip().str.lower().isin(["branch name","nan","none","null"])]
    long=agg.copy()
    long=ensure_serial(long)
    return long

def compute_me_grants(df_in,b,gc):
    import pandas as pd
    g=df_in[[b,gc]].copy(); g[b]=clean_branch(g[b])
    g["_gr"]=pd.to_numeric(g[gc], errors="coerce").fillna(0)
    g=g[g[b].notna()]
    cnt=g[g["_gr"]>0].groupby(b).size().reset_index(name="Number on MEs")
    amt=g.groupby(b)["_gr"].sum(min_count=1).reset_index(name="Amounts of Grants")
    rep=cnt.merge(amt,on=b,how="outer").fillna(0).rename(columns={b:"Branch Name"})
    rep=rep[~rep["Branch Name"].astype(str).str.strip().str.lower().isin(["branch name","nan","none","null"])]
    rep=ensure_serial(rep)
    return rep

def add_grand_total(df, numeric_cols=None):
    import pandas as pd
    d=df.copy()
    if numeric_cols is None:
        numeric_cols = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
    totals = {c: pd.to_numeric(d[c], errors="coerce").sum() for c in numeric_cols}
    row = {c: "" for c in d.columns}
    row.update({"Branch Name":"Grand Total"})
    for c,v in totals.items():
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

# ===== Data =====
df=get_df()
b=cpos(df,BRANCH); lt=cpos(df,LOAN_TYPE); la=cpos(df,LOAN_AMOUNT)
pt=cpos(df,POULTRY_TYPE); ub=cpos(df,BIRDS_COL); gcol=cpos(df,GRANTS)

loan_agg = compute_branch_loan(df,b,lt,la)
loan = summarize_loan_table(loan_agg)
poultry = compute_poultry_me_and_birds(df,b,pt,ub)
grants = compute_me_grants(df,b,gcol)

# Grand Totals
poultry = add_grand_total(poultry, numeric_cols=["# of MEs","# of Birds"])
grants = add_grand_total(grants, numeric_cols=["Number on MEs","Amounts of Grants"])

# ===== UI (same sections as v5q) =====
import plotly.express as px

st.markdown('<h3 class="section-title">📊 Branch Wise Loan Disbursement</h3>', unsafe_allow_html=True)
lcol, rcol = st.columns([0.55, 0.45], gap="large")
with lcol:
    sty = (
        loan.style.hide(axis="index")
            .format({"Amount of Loan":"{:,.0f}","# of Loan":"{:,.0f}"})
            .set_table_styles([
                {"selector":"th","props":[("background-color","#dcfce7"),("font-weight","800")]},
                {"selector":"thead th","props":[("background-color","#dcfce7"),("font-weight","800")]},
                {"selector":"tbody tr:nth-child(even)","props":[("background-color","#fafafa")]}
            ])
            .apply(lambda r: ["background-color:#dcfce7; color:#000; font-weight:800"]*len(r) if str(r.get("Branch Name",""))=="Grand Total" else (["background-color:#fffbe6; color:#0f172a; font-weight:700"]*len(r) if str(r.get("Branch Name","")).endswith(" Total") else [""]*len(r)), axis=1)
    )
    st.dataframe(sty, use_container_width=True, height=720)
    st.download_button("⬇️ Loan — Excel", to_excel_bytes({"Loan": loan}), file_name="loan_disbursement.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
with rcol:
    base2=loan[(~loan["Branch Name"].str.endswith(" Total")) & (loan["Branch Name"]!="Grand Total") & (loan["Types of Loan"]!="")]
    fig=px.bar(base2, x="Branch Name", y="Amount of Loan", color="Types of Loan", barmode="group", title="Amount of Loan by Branch & Type")
    fig.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

st.markdown('<h3 class="section-title">🐔 Types of Poultry Rearing</h3>', unsafe_allow_html=True)
p_l, p_r = st.columns([0.55, 0.45], gap="large")
with p_l:
    st.dataframe(style_table(poultry, number_formats={"# of MEs":"{:,.0f}","# of Birds":"{:,.0f}"}), use_container_width=True)
    st.download_button("⬇️ Poultry — Excel", to_excel_bytes({"Poultry": poultry}), file_name="poultry_rearing.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
with p_r:
    t1, t2 = st.tabs(["# of Birds", "# of MEs"])
    with t1:
        fig_b=px.bar(poultry[poultry["Branch Name"]!="Grand Total"], x="Branch Name", y="# of Birds", color="Types of Poultry Rearing", barmode="group", title="Total Birds by Branch & Type")
        fig_b.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
        st.plotly_chart(fig_b, use_container_width=True)
    with t2:
        fig_m=px.bar(poultry[poultry["Branch Name"]!="Grand Total"], x="Branch Name", y="# of MEs", color="Types of Poultry Rearing", barmode="group", title="Number of MEs by Branch & Type")
        fig_m.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
        st.plotly_chart(fig_m, use_container_width=True)

st.markdown("---")

st.markdown('<h3 class="section-title">💠 MEs Grants Information</h3>', unsafe_allow_html=True)
g_l, g_r = st.columns([0.55, 0.45], gap="large")
with g_l:
    st.dataframe(style_table(grants, number_formats={"Amounts of Grants":"{:,.0f}","Number on MEs":"{:,.0f}"}), use_container_width=True)
    st.download_button("⬇️ Grants — Excel", to_excel_bytes({"Grants": grants}), file_name="grants_information.xlsx", mime="application/vnd.openxmlformats-officedocument-spreadsheetml.sheet")
with g_r:
    gp=grants[grants["Branch Name"]!="Grand Total"].copy()
    fig_g=px.bar(gp, x="Branch Name", y="Amounts of Grants", title="Grants Amount by Branch")
    fig_g.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
    st.plotly_chart(fig_g, use_container_width=True)

st.markdown("---")
st.markdown('<h3 class="section-title">🐣 Poultry KPI Summary</h3>', unsafe_allow_html=True)
k_l, k_r = st.columns([0.55, 0.45], gap="large")
with k_l:
    pks = poultry.copy()
    pks = pks[pks["Branch Name"]!="Grand Total"].pivot_table(index="Branch Name", columns="Types of Poultry Rearing", values="# of MEs", aggfunc="sum").fillna(0).reset_index()
    for c in ("Layer Rearing","Broiler Rearing"):
        if c not in pks: pks[c]=0
    pks["# of MEs"]=pks["Layer Rearing"]+pks["Broiler Rearing"]
    pks = ensure_serial(pks)
    pks = add_grand_total(pks, numeric_cols=["Layer Rearing","Broiler Rearing","# of MEs"])
    st.dataframe(style_table(pks, number_formats={"Layer Rearing":"{:,.0f}","Broiler Rearing":"{:,.0f}","# of MEs":"{:,.0f}"}), use_container_width=True)
with k_r:
    pks_long = pks[pks["Branch Name"]!="Grand Total"].melt(id_vars=["Branch Name","Sl No"], value_vars=["Layer Rearing","Broiler Rearing"], var_name="Type", value_name="Count")
    fig_pks=px.bar(pks_long, x="Branch Name", y="Count", color="Type", barmode="group", title="Layer vs Broiler (# of MEs)")
    fig_pks.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
    st.plotly_chart(fig_pks, use_container_width=True)

st.markdown("---")
st.markdown('<h3 class="section-title">🎯 Average Ticket Size</h3>', unsafe_allow_html=True)
a_l, a_r = st.columns([0.55, 0.45], gap="large")
with a_l:
    ats = (loan[(~loan["Branch Name"].str.endswith(" Total")) & (loan["Branch Name"]!="Grand Total") & (loan["Types of Loan"]!="")]
           .groupby("Branch Name").agg(**{"# of Loan":("# of Loan","sum"),"Amount of Loan":("Amount of Loan","sum")}).reset_index())
    ats["Avg Ticket Size"] = (ats["Amount of Loan"]/ats["# of Loan"].replace(0, pd.NA)).fillna(0)
    ats = ensure_serial(ats)
    ats = add_grand_total(ats, numeric_cols=["# of Loan","Amount of Loan","Avg Ticket Size"])
    st.dataframe(style_table(ats, number_formats={"# of Loan":"{:,.0f}","Amount of Loan":"{:,.0f}","Avg Ticket Size":"{:,.0f}"}), use_container_width=True)
with a_r:
    fig_ats=px.bar(ats[ats["Branch Name"]!="Grand Total"], x="Branch Name", y="Avg Ticket Size", title="Average Ticket Size by Branch")
    fig_ats.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
    st.plotly_chart(fig_ats, use_container_width=True)

st.markdown("---")
st.markdown('<h3 class="section-title">💹 Grants Utilization</h3>', unsafe_allow_html=True)
u_l, u_r = st.columns([0.55, 0.45], gap="large")
with u_l:
    gu = grants[grants["Branch Name"]!="Grand Total"].copy()
    gu["Avg Grant per ME"]=(pd.to_numeric(gu["Amounts of Grants"], errors="coerce")/gu["Number on MEs"].replace(0, pd.NA)).fillna(0)
    gu = ensure_serial(gu)
    gu = add_grand_total(gu, numeric_cols=["Number on MEs","Amounts of Grants","Avg Grant per ME"])
    st.dataframe(style_table(gu, number_formats={"Number on MEs":"{:,.0f}","Amounts of Grants":"{:,.0f}","Avg Grant per ME":"{:,.0f}"}), use_container_width=True)
with u_r:
    fig_gu=px.bar(gu[gu["Branch Name"]!="Grand Total"], x="Branch Name", y="Avg Grant per ME", title="Avg Grant per ME by Branch")
    fig_gu.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
    st.plotly_chart(fig_gu, use_container_width=True)

st.markdown("---")
st.markdown('<h3 class="section-title">🏆 Top 5 Branches</h3>', unsafe_allow_html=True)
# Disbursement
d_l, d_r = st.columns([0.55, 0.45], gap="large")
with d_l:
    topD = (loan[(~loan["Branch Name"].str.endswith(" Total")) & (loan["Branch Name"]!="Grand Total") & (loan["Types of Loan"]!="")]
            .groupby("Branch Name")["Amount of Loan"].sum().sort_values(ascending=False).head(5).reset_index())
    topD = ensure_serial(topD); topD = add_grand_total(topD, numeric_cols=["Amount of Loan"])
    st.markdown("**Top 5 by Disbursement**")
    st.dataframe(style_table(topD, number_formats={"Amount of Loan":"{:,.0f}"}), use_container_width=True)
with d_r:
    fig_td=px.bar(topD[topD["Branch Name"]!="Grand Total"], x="Branch Name", y="Amount of Loan", title="Top 5 by Disbursement")
    fig_td.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
    st.plotly_chart(fig_td, use_container_width=True)

# Birds
b_l, b_r = st.columns([0.55, 0.45], gap="large")
with b_l:
    topB = poultry[poultry["Branch Name"]!="Grand Total"].groupby("Branch Name")["# of Birds"].sum().sort_values(ascending=False).head(5).reset_index()
    topB = ensure_serial(topB); topB = add_grand_total(topB, numeric_cols=["# of Birds"])
    st.markdown("**Top 5 by Birds**")
    st.dataframe(style_table(topB, number_formats={"# of Birds":"{:,.0f}"}), use_container_width=True)
with b_r:
    fig_tb=px.bar(topB[topB["Branch Name"]!="Grand Total"], x="Branch Name", y="# of Birds", title="Top 5 by Birds")
    fig_tb.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
    st.plotly_chart(fig_tb, use_container_width=True)

# Grants
g_l2, g_r2 = st.columns([0.55, 0.45], gap="large")
with g_l2:
    topG = grants[grants["Branch Name"]!="Grand Total"].groupby("Branch Name")["Amounts of Grants"].sum().sort_values(ascending=False).head(5).reset_index()
    topG = ensure_serial(topG); topG = add_grand_total(topG, numeric_cols=["Amounts of Grants"])
    st.markdown("**Top 5 by Grants**")
    st.dataframe(style_table(topG, number_formats={"Amounts of Grants":"{:,.0f}"}), use_container_width=True)
with g_r2:
    fig_tg=px.bar(topG[topG["Branch Name"]!="Grand Total"], x="Branch Name", y="Amounts of Grants", title="Top 5 by Grants")
    fig_tg.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
    st.plotly_chart(fig_tg, use_container_width=True)

# ============================
# === NEW REPORTS SECTION ===
# ============================
st.markdown("---")
st.markdown('<h3 class="section-title">🧩 Additional Interactive Reports (New)</h3>', unsafe_allow_html=True)

# Sidebar controls for new reports
st.sidebar.markdown("## Additional Reports")
# Detect branch-like column automatically (prefer existing BRANCH position)
detected_branch_col = b if b in df.columns else None
branch_col_options = ["(none)"] + df.columns.tolist()
branch_sel = st.sidebar.selectbox("Branch column for new reports", options=branch_col_options, index=branch_col_options.index(detected_branch_col) if detected_branch_col else 0)

# Detect possible date columns
possible_date_cols = [c for c in df.columns if "date" in c.lower()]
# also try parsing candidates
for c in df.columns:
    if c not in possible_date_cols:
        try:
            parsed = pd.to_datetime(df[c], errors='coerce')
            if parsed.notna().sum() > 0.6*len(parsed):
                possible_date_cols.append(c)
        except Exception:
            pass

date_col_options = ["(none)"] + possible_date_cols
date_col_sel = st.sidebar.selectbox("Date column (for time-series)", options=date_col_options, index=1 if "DisburseDate" in df.columns else 0)

# Amount / numeric column
amt_col_options = ["(none)"] + df.columns.tolist()
amt_col_sel = st.sidebar.selectbox("Amount / Numeric column", options=amt_col_options, index=amt_col_options.index(la) if la in df.columns else 0)

# Filters
st.sidebar.markdown("### Filters for new reports")
selected_branches = []
if branch_col_options and branch_col_options[0] != "(none)" and branch_sel != "(none)":
    try:
        vals = df[branch_sel].dropna().astype(str).unique().tolist()
        vals = sorted([v for v in vals if str(v).strip().lower() not in ["nan","none",""]])
        selected_branches = st.sidebar.multiselect("Filter branches", options=vals, default=vals)
    except Exception:
        selected_branches = []

# Date range filter if date selected
start_date, end_date = None, None
if date_col_sel and date_col_sel != "(none)":
    try:
        df[date_col_sel] = pd.to_datetime(df[date_col_sel], errors="coerce")
        mn = df[date_col_sel].min()
        mx = df[date_col_sel].max()
        start_date, end_date = st.sidebar.date_input("Date range", value=(mn.date() if pd.notnull(mn) else None, mx.date() if pd.notnull(mx) else None))
    except Exception:
        start_date, end_date = None, None

# Build filtered df for these new reports
df_reports = df.copy()
if branch_sel and branch_sel != "(none)" and selected_branches:
    df_reports = df_reports[df_reports[branch_sel].astype(str).isin(selected_branches)]
if date_col_sel and date_col_sel != "(none)" and start_date and end_date:
    df_reports = df_reports[(df_reports[date_col_sel].dt.date >= start_date) & (df_reports[date_col_sel].dt.date <= end_date)]

st.write("#### Preview (filtered for new reports)")
st.dataframe(df_reports.head(30))

# --- Distribution + Boxplot + Summary Stats ---
st.markdown("### 📈 Distribution & Summary Stats")
if amt_col_sel and amt_col_sel != "(none)":
    try:
        df_reports[amt_col_sel] = pd.to_numeric(df_reports[amt_col_sel], errors="coerce")
        col = df_reports[amt_col_sel].dropna()
        if col.empty:
            st.info("নির্বাচিত অ্যামাউন্ট কলামে ডেটা পাওয়া যায়নি। অন্য কলাম নির্বাচন করুন।")
        else:
            fig_hist = px.histogram(df_reports, x=amt_col_sel, nbins=40, title="Distribution of "+amt_col_sel, labels={amt_col_sel:"Amount"})
            st.plotly_chart(fig_hist, use_container_width=True)

            if branch_sel and branch_sel != "(none)":
                try:
                    fig_box = px.box(df_reports.dropna(subset=[amt_col_sel]), x=branch_sel, y=amt_col_sel, title=f"{amt_col_sel} by {branch_sel}")
                    st.plotly_chart(fig_box, use_container_width=True)
                except Exception:
                    pass

            # Summary stats
            stats = {}
            s = df_reports[amt_col_sel].dropna().astype(float)
            if not s.empty:
                stats = {
                    "count": int(s.count()),
                    "mean": float(s.mean()),
                    "median": float(s.median()),
                    "std": float(s.std()),
                    "min": float(s.min()),
                    "q1": float(s.quantile(0.25)),
                    "q3": float(s.quantile(0.75)),
                    "iqr": float(s.quantile(0.75) - s.quantile(0.25)),
                    "max": float(s.max())
                }
            st.write("**Summary statistics:**")
            st.json(stats)
            # Quick download of numeric column sample and stats
            if st.button("Download distribution data (sample)"):
                try:
                    sample = df_reports[[amt_col_sel] + ([branch_sel] if branch_sel and branch_sel!="(none)" else [])].dropna().head(1000)
                    st.download_button("Download CSV", data=sample.to_csv(index=False).encode('utf-8'), file_name="distribution_sample.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Export failed: {e}")
    except Exception as e:
        st.error(f"Error in distribution: {e}")
else:
    st.info("Amount / numeric column সিলেক্ট করুন (sidebar থেকে)।")

# --- Monthly Trend + Cumulative + Rolling Average ---
st.markdown("### 📅 Monthly Trend — Trend / Cumulative / Rolling Average")
if date_col_sel and date_col_sel != "(none)" and amt_col_sel and amt_col_sel != "(none)":
    try:
        ts = df_reports[[date_col_sel, amt_col_sel]].copy()
        ts = ts.dropna(subset=[date_col_sel])
        ts[amt_col_sel] = pd.to_numeric(ts[amt_col_sel], errors="coerce").fillna(0)
        ts['YearMonth'] = ts[date_col_sel].dt.to_period('M').dt.to_timestamp()
        monthly = ts.groupby('YearMonth')[amt_col_sel].sum().reset_index().sort_values('YearMonth')
        monthly['cumulative'] = monthly[amt_col_sel].cumsum()
        monthly['rolling_3m'] = monthly[amt_col_sel].rolling(3, min_periods=1).mean()
        fig_ts = px.line(monthly, x='YearMonth', y=[amt_col_sel, 'cumulative','rolling_3m'], labels={"value":"Amount","variable":"Series"}, title="Monthly: amount, cumulative and rolling (3)")
        st.plotly_chart(fig_ts, use_container_width=True)
        # provide export
        if st.button("Download Monthly Series (Excel)"):
            out = to_excel_bytes({"monthly": monthly})
            st.download_button("⬇️ Download monthly.xlsx", data=out, file_name="monthly_series.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e:
        st.error(f"Time-series error: {e}")
else:
    st.info("Time-series চালাতে date এবং amount কলাম উভয় সিলেক্ট করুন (sidebar)।")

# --- Ad-hoc Pivot Builder ---
st.markdown("### 🔧 Ad-hoc Pivot Builder")
cols_for_pivot = df_reports.columns.tolist()
pv_row = st.selectbox("Row field", options=["(none)"] + cols_for_pivot, index=0, key="pv_row")
pv_col = st.selectbox("Column field", options=["(none)"] + cols_for_pivot, index=0, key="pv_col")
pv_val = st.selectbox("Value field", options=cols_for_pivot, index=0, key="pv_val")
pv_agg = st.selectbox("Agg function", options=["sum","count","mean"], index=0, key="pv_agg")

if st.button("Build Pivot Table"):
    try:
        if pv_row == "(none)" or pv_col == "(none)":
            st.warning("Row এবং Column দুটো ফিল্ড সিলেক্ট করুন।")
        else:
            pivot = pd.pivot_table(df_reports, index=pv_row, columns=pv_col, values=pv_val if pv_val else None, aggfunc=pv_agg, fill_value=0)
            st.dataframe(pivot)
            # download button
            out = to_excel_bytes({"pivot": pivot.reset_index()})
            st.download_button("⬇️ Download pivot.xlsx", data=out, file_name="pivot.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e:
        st.error(f"Pivot failed: {e}")

# --- Top-N Parameterized Leaderboard (exportable) ---
st.markdown("### 🏅 Top-N Leaderboard")
all_cols = df_reports.columns.tolist()
lb_group = st.selectbox("Group by (e.g. Branch/ME/Product)", options=["(none)"] + all_cols, index=1 if (b in df_reports.columns) else 0)
lb_metric = st.selectbox("Metric (numeric) to aggregate", options=["(none)"] + all_cols, index=1 if (la in df_reports.columns) else 0)
lb_agg = st.selectbox("Aggregation", options=["sum","count","mean"], index=0)
lb_n = st.number_input("Top N", min_value=1, max_value=1000, value=10, step=1)

if st.button("Generate Leaderboard"):
    try:
        if lb_group == "(none)" or lb_metric == "(none)":
            st.warning("Group এবং Metric উভয় সিলেক্ট করুন।")
        else:
            df_tmp = df_reports.copy()
            if lb_agg == "count":
                lb = df_tmp.groupby(lb_group).size().reset_index(name="count").sort_values("count", ascending=False).head(lb_n)
            else:
                df_tmp[lb_metric] = pd.to_numeric(df_tmp[lb_metric], errors="coerce")
                lb = df_tmp.groupby(lb_group)[lb_metric].agg(lb_agg).reset_index().sort_values(lb_metric, ascending=False).head(lb_n)
            st.dataframe(lb)
            st.download_button("⬇️ Download leaderboard.xlsx", data=to_excel_bytes({"leaderboard": lb}), file_name="leaderboard.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e:
        st.error(f"Leaderboard failed: {e}")

# --- ME-level Export (select columns and download) ---
st.markdown("### 📁 ME-level Export")
cols_for_export = df_reports.columns.tolist()
export_cols = st.multiselect("Select columns to export", options=cols_for_export, default=cols_for_export)
if st.button("Preview export (first 50 rows)"):
    try:
        st.dataframe(df_reports[export_cols].head(50))
    except Exception as e:
        st.error(f"Preview failed: {e}")

if st.button("Download ME Report (Excel)"):
    try:
        out = to_excel_bytes({"ME_Report": ensure_serial(df_reports[export_cols].copy())})
        st.download_button("⬇️ Download ME_Report.xlsx", data=out, file_name="ME_Report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e:
        st.error(f"Export failed: {e}")

st.markdown("---")
st.info("নতুন সেকশন যুক্ত করা হয়েছে — Sidebar থেকে কলাম/ফিল্টার সিলেক্ট করে উপরের রিপোর্টগুলো পরীক্ষা করুন।")

# End of file
