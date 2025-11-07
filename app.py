# -*- coding: utf-8 -*-
import os, re, base64
import streamlit as st, pandas as pd, plotly.express as px, requests
from io import StringIO, BytesIO

URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRkcagLu_YrYgQxmsO3DnHn90kqALkw9uDByX7UBNRUjaFKKQdE3V-6fm5ZcKGk_A/pub?gid=2143275417&single=true&output=csv"

st.set_page_config(page_title="PIDIM SMART Reports", layout="wide")

# ====== Minimal CSS (no green header), table header bold + light green, print fix ======
st.markdown("""
<style>
/* Light theme polish */
.block-container { padding-top: 0.6rem; }
[data-testid="stSidebar"] { background-color: #f0fdf4; }

/* Table header styling */
thead th { background-color:#dcfce7 !important; font-weight:800 !important; }

/* Global print button styling */
#global-print { text-align:right; margin-top:6px; }
#global-print button{
  background-color:#16a34a; color:white; border:none; padding:8px 16px;
  border-radius:8px; cursor:pointer;
}
/* Print media */
@media print {
  .stApp { visibility: visible; }
  .stButton, .stDownloadButton { display:none !important; }
  .stSidebar { display:none !important; }
  .block-container { padding: 6mm !important; }
  h1, h2, h3 { color:#065f46 !important; font-weight:800 !important; }
  table { page-break-inside: avoid; }
}
</style>
""", unsafe_allow_html=True)

# Single global print button (top-right)
st.markdown("""<div id="global-print">
<button onclick="window.print()">Print</button>
</div>""", unsafe_allow_html=True)

def col_letter_to_pos(s):
    s=re.sub(r'[^A-Za-z]','',s); v=0
    for ch in s.upper(): v=v*26+(ord(ch)-64)
    return v

# Column positions per mapping
BRANCH=col_letter_to_pos("G"); LOAN_TYPE=col_letter_to_pos("AN"); LOAN_AMOUNT=col_letter_to_pos("AQ")
POULTRY_TYPE=col_letter_to_pos("T"); BIRDS_COL=col_letter_to_pos("U")
GRANTS=col_letter_to_pos("BL")

@st.cache_resource
def sess():
    s=requests.Session(); s.headers.update({"User-Agent":"ME-Reports/1.0"}); return s

@st.cache_data(ttl=900)
def get_df():
    t=sess().get(URL, timeout=20).text
    df=pd.read_csv(StringIO(t), low_memory=False)
    df.columns=[str(c).strip() for c in df.columns]
    return df

def cpos(df,p): i=max(0, min(len(df.columns)-1,p-1)); return df.columns[i]
def clean_branch(s):
    s=s.astype(str).str.strip(); bad=s.str.lower().isin(["","nan","none","null","branch name"]); return s.mask(bad,None)

def ensure_serial(df):
    d=df.copy()
    # Rename or insert "Sl No"
    if "ক্রমিক নং" in d.columns:
        d = d.drop(columns=["ক্রমিক নং"])
    if "Sl No" in d.columns:
        d["Sl No"] = range(1, len(d)+1)
    else:
        d.insert(0, "Sl No", range(1, len(d)+1))
    return d

def add_grand_total(df, numeric_cols=None):
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

HEADER_STYLE = [{"selector":"th","props":[("background-color","#dcfce7"),("font-weight","800")]},
                {"selector":"thead th","props":[("background-color","#dcfce7"),("font-weight","800")]},
                {"selector":"tbody tr:nth-child(even)","props":[("background-color","#fafafa")]}]

def style_table(d: pd.DataFrame, number_formats=None, subtotal_logic=None, narrow_serial=False):
    sty = d.style.hide(axis="index")
    if number_formats:
        sty = sty.format(number_formats)
    sty = sty.set_table_styles(HEADER_STYLE)
    if subtotal_logic:
        sty = sty.apply(lambda r: subtotal_logic(r), axis=1)
    if narrow_serial and "Sl No" in d.columns:
        sty = sty.set_properties(subset=["Sl No"], **{"width":"56px"})
    return sty

# ===== Builders =====
def compute_branch_loan(df_in,b,t,a):
    w=df_in[[b,t,a]].copy()
    w[b]=clean_branch(w[b]); w[t]=w[t].astype(str).str.strip()
    w["_amt"]=pd.to_numeric(w[a], errors="coerce").fillna(0)
    def norm(x):
        x=(x or "").strip().lower(); x=re.sub(r"\s+"," ",x)
        if "non" in x and "enterprise" in x: return "Non-Enterprise"
        if "enterprise" in x: return "Enterprise"
        return x.title() if x else ""
    w["_type"]=w[t].apply(norm); w=w[w[b].notna()]
    g=(w.groupby([b,"_type"]).agg(**{"# of Loan":("_type","count"),"Amount of Loan":("_amt","sum")}).reset_index()
        .rename(columns={b:"Branch Name","_type":"Types of Loan"}))
    g=g[~g["Branch Name"].astype(str).str.strip().str.lower().isin(["branch name","nan","none","null"])]
    return g

def summarize_loan_table(agg):
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
    # Ensure we display only up to Grand Total (no extra elements after)
    return loan

def compute_poultry_me_and_birds(df_in, b, type_col, birds_col):
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
    g=df_in[[b,gc]].copy(); g[b]=clean_branch(g[b])
    g["_gr"]=pd.to_numeric(g[gc], errors="coerce").fillna(0)
    g=g[g[b].notna()]
    cnt=g[g["_gr"]>0].groupby(b).size().reset_index(name="Number on MEs")
    amt=g.groupby(b)["_gr"].sum(min_count=1).reset_index(name="Amounts of Grants")
    rep=cnt.merge(amt,on=b,how="outer").fillna(0).rename(columns={b:"Branch Name"})
    rep=rep[~rep["Branch Name"].astype(str).str.strip().str.lower().isin(["branch name","nan","none","null"])]
    rep=ensure_serial(rep)
    return rep

def average_ticket_size(loan):
    base=loan[(~loan["Branch Name"].str.endswith(" Total")) & (loan["Branch Name"]!="Grand Total") & (loan["Types of Loan"]!="")]
    agg=(base.groupby("Branch Name").agg(total_amount=("Amount of Loan","sum"), total_count=("# of Loan","sum")).reset_index())
    agg["Avg Ticket Size"]=agg["total_amount"]/agg["total_count"].replace(0, pd.NA)
    agg["Avg Ticket Size"]=agg["Avg Ticket Size"].fillna(0)
    out=agg[["Branch Name","total_count","total_amount","Avg Ticket Size"]].rename(columns={"total_count":"# of Loan","total_amount":"Amount of Loan"})
    out=ensure_serial(out)
    return out

def poultry_kpi_summary_counts(poultry_long):
    wide = poultry_long.pivot_table(index="Branch Name", columns="Types of Poultry Rearing", values="# of MEs", aggfunc="sum").fillna(0).reset_index()
    wide.columns=[c if isinstance(c,str) else c[1] for c in wide.columns]
    if "Layer Rearing" not in wide: wide["Layer Rearing"]=0
    if "Broiler Rearing" not in wide: wide["Broiler Rearing"]=0
    wide["# of MEs"] = wide["Layer Rearing"] + wide["Broiler Rearing"]
    wide=ensure_serial(wide)
    return wide[["Sl No","Branch Name","Layer Rearing","Broiler Rearing","# of MEs"]]

def to_excel_bytes(df_dict):
    bio=BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        for name, data in df_dict.items():
            data.to_excel(writer, index=False, sheet_name=name[:31])
        writer.book.close()
    bio.seek(0)
    return bio.getvalue()

# ===== Data =====
df=get_df()
b=cpos(df,BRANCH); lt=cpos(df,LOAN_TYPE); la=cpos(df,LOAN_AMOUNT)
pt=cpos(df,POULTRY_TYPE); ub=cpos(df,BIRDS_COL); gcol=cpos(df,GRANTS)

loan_agg = compute_branch_loan(df,b,lt,la)
loan = summarize_loan_table(loan_agg)
poultry = compute_poultry_me_and_birds(df,b,pt,ub)
grants = compute_me_grants(df,b,gcol)

# Add Grand Total rows where requested
poultry = add_grand_total(poultry, numeric_cols=["# of MEs","# of Birds"])
grants = add_grand_total(grants, numeric_cols=["Number on MEs","Amounts of Grants"])

# KPI Summary totals
pks = poultry_kpi_summary_counts(poultry[poultry["Branch Name"]!="Grand Total"])
pks = add_grand_total(pks, numeric_cols=["Layer Rearing","Broiler Rearing","# of MEs"])

# Average Ticket Size totals
ats = average_ticket_size(loan)
ats = add_grand_total(ats, numeric_cols=["# of Loan","Amount of Loan","Avg Ticket Size"])

# Grants Utilization totals
def grants_utilization_full(gdf):
    rep = gdf.copy()
    rep = rep[rep["Branch Name"]!="Grand Total"]
    rep["Avg Grant per ME"] = (pd.to_numeric(rep["Amounts of Grants"], errors="coerce")/rep["Number on MEs"].replace(0, pd.NA)).fillna(0)
    rep = ensure_serial(rep)
    rep = add_grand_total(rep, numeric_cols=["Number on MEs","Amounts of Grants","Avg Grant per ME"])
    return rep
gu = grants_utilization_full(grants)

# Top 5 tables (each with its own Grand Total for the shown rows)
base=loan[(~loan["Branch Name"].str.endswith(" Total")) & (loan["Branch Name"]!="Grand Total") & (loan["Types of Loan"]!="")]
topD = base.groupby("Branch Name")["Amount of Loan"].sum().sort_values(ascending=False).head(5).reset_index(name="Amount of Loan")
topD = ensure_serial(topD); topD = add_grand_total(topD, numeric_cols=["Amount of Loan"])

topB = poultry[poultry["Branch Name"]!="Grand Total"].groupby("Branch Name")["# of Birds"].sum().sort_values(ascending=False).head(5).reset_index(name="# of Birds")
topB = ensure_serial(topB); topB = add_grand_total(topB, numeric_cols=["# of Birds"])

topG = grants[grants["Branch Name"]!="Grand Total"].groupby("Branch Name")["Amounts of Grants"].sum().sort_values(ascending=False).head(5).reset_index(name="Amounts of Grants")
topG = ensure_serial(topG); topG = add_grand_total(topG, numeric_cols=["Amounts of Grants"])

# ===== UI =====
st.markdown("### 📊 Branch Wise Loan Disbursement")
lcol, rcol = st.columns([0.55, 0.45], gap="large")
with lcol:
    sty = style_table(
        loan,
        number_formats={"Amount of Loan":"{:,.0f}","# of Loan":"{:,.0f}"},
        subtotal_logic=lambda r: ["background-color:#dcfce7; color:#000; font-weight:800"]*len(r) if str(r.get("Branch Name",""))=="Grand Total" else (["background-color:#fffbe6; color:#0f172a; font-weight:700"]*len(r) if str(r.get("Branch Name","")).endswith(" Total") else [""]*len(r)),
    )
    st.dataframe(sty, use_container_width=True, height=720)
    st.download_button("⬇️ Loan — Excel", to_excel_bytes({"Loan": loan}), file_name="loan_disbursement.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
with rcol:
    base2=loan[(~loan["Branch Name"].str.endswith(" Total")) & (loan["Branch Name"]!="Grand Total") & (loan["Types of Loan"]!="")]
    fig=px.bar(base2, x="Branch Name", y="Amount of Loan", color="Types of Loan", barmode="group", title="Amount of Loan by Branch & Type")
    fig.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

st.markdown("### 🐔 Types of Poultry Rearing")
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

st.markdown("### 💠 MEs Grants Information")
g_l, g_r = st.columns([0.55, 0.45], gap="large")
with g_l:
    st.dataframe(style_table(grants, number_formats={"Amounts of Grants":"{:,.0f}","Number on MEs":"{:,.0f}"}), use_container_width=True)
    st.download_button("⬇️ Grants — Excel", to_excel_bytes({"Grants": grants}), file_name="grants_information.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
with g_r:
    gp=grants[grants["Branch Name"]!="Grand Total"].copy()
    fig_g=px.bar(gp, x="Branch Name", y="Amounts of Grants", title="Grants Amount by Branch")
    fig_g.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
    st.plotly_chart(fig_g, use_container_width=True)

st.markdown("---")

st.markdown("## 📚 Additional Analytics Reports")

st.markdown("### 🐣 Poultry KPI Summary")
k_l, k_r = st.columns([0.55, 0.45], gap="large")
with k_l:
    st.dataframe(style_table(pks, number_formats={"Layer Rearing":"{:,.0f}","Broiler Rearing":"{:,.0f}","# of MEs":"{:,.0f}"}), use_container_width=True)
    st.download_button("⬇️ Poultry KPI Summary — Excel", to_excel_bytes({"Poultry KPI Summary": pks}), file_name="poultry_kpi_summary.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
with k_r:
    pks_long = pks[pks["Branch Name"]!="Grand Total"].melt(id_vars=["Branch Name","Sl No"], value_vars=["Layer Rearing","Broiler Rearing"], var_name="Type", value_name="Count")
    fig_pks=px.bar(pks_long, x="Branch Name", y="Count", color="Type", barmode="group", title="Layer vs Broiler (# of MEs)")
    fig_pks.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
    st.plotly_chart(fig_pks, use_container_width=True)

st.markdown("### 🎯 Average Ticket Size")
a_l, a_r = st.columns([0.55, 0.45], gap="large")
with a_l:
    st.dataframe(style_table(ats, number_formats={"# of Loan":"{:,.0f}","Amount of Loan":"{:,.0f}","Avg Ticket Size":"{:,.0f}"}), use_container_width=True)
    st.download_button("⬇️ Average Ticket Size — Excel", to_excel_bytes({"Average Ticket Size": ats}), file_name="average_ticket_size.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
with a_r:
    fig_ats=px.bar(ats[ats["Branch Name"]!="Grand Total"], x="Branch Name", y="Avg Ticket Size", title="Average Ticket Size by Branch")
    fig_ats.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
    st.plotly_chart(fig_ats, use_container_width=True)

st.markdown("### 💹 Grants Utilization")
u_l, u_r = st.columns([0.55, 0.45], gap="large")
with u_l:
    st.dataframe(style_table(gu, number_formats={"Number on MEs":"{:,.0f}","Amounts of Grants":"{:,.0f}","Avg Grant per ME":"{:,.0f}"}), use_container_width=True)
    st.download_button("⬇️ Grants Utilization — Excel", to_excel_bytes({"Grants Utilization": gu}), file_name="grants_utilization.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
with u_r:
    fig_gu=px.bar(gu[gu["Branch Name"]!="Grand Total"], x="Branch Name", y="Avg Grant per ME", title="Avg Grant per ME by Branch")
    fig_gu.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
    st.plotly_chart(fig_gu, use_container_width=True)

st.markdown("### 🏆 Top 5 Branches")
# Disbursement
d_l, d_r = st.columns([0.55, 0.45], gap="large")
with d_l:
    st.markdown("**Top 5 by Disbursement**")
    st.dataframe(style_table(topD, number_formats={"Amount of Loan":"{:,.0f}"}), use_container_width=True)
with d_r:
    fig_td=px.bar(topD[topD["Branch Name"]!="Grand Total"], x="Branch Name", y="Amount of Loan", title="Top 5 by Disbursement")
    fig_td.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
    st.plotly_chart(fig_td, use_container_width=True)

# Birds
b_l, b_r = st.columns([0.55, 0.45], gap="large")
with b_l:
    st.markdown("**Top 5 by Birds**")
    st.dataframe(style_table(topB, number_formats={"# of Birds":"{:,.0f}"}), use_container_width=True)
with b_r:
    fig_tb=px.bar(topB[topB["Branch Name"]!="Grand Total"], x="Branch Name", y="# of Birds", title="Top 5 by Birds")
    fig_tb.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
    st.plotly_chart(fig_tb, use_container_width=True)

# Grants
g_l2, g_r2 = st.columns([0.55, 0.45], gap="large")
with g_l2:
    st.markdown("**Top 5 by Grants**")
    st.dataframe(style_table(topG, number_formats={"Amounts of Grants":"{:,.0f}"}), use_container_width=True)
with g_r2:
    fig_tg=px.bar(topG[topG["Branch Name"]!="Grand Total"], x="Branch Name", y="Amounts of Grants", title="Top 5 by Grants")
    fig_tg.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
    st.plotly_chart(fig_tg, use_container_width=True)
