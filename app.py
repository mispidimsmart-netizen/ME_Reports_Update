# -*- coding: utf-8 -*-
import os, re
import streamlit as st, pandas as pd, plotly.express as px, requests
from io import StringIO, BytesIO

URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRkcagLu_YrYgQxmsO3DnHn90kqALkw9uDByX7UBNRUjaFKKQdE3V-6fm5ZcKGk_A/pub?gid=2143275417&single=true&output=csv"

st.set_page_config(page_title="PIDIM SMART Reports", layout="wide", page_icon="assets/favicon.png")

# ====== Minimal CSS: remove big green header, keep header block with logo+name, make table headers bold+light-green ======
st.markdown("""
<style>
/* Remove Streamlit default top padding slightly */
.block-container { padding-top: 0.6rem; }

/* Section titles */
h3, .section-title { color:#065f46; font-weight:800; }

/* Column/table headers */
thead th { background-color:#dcfce7 !important; font-weight:800 !important; }

/* Header bar style (custom, not the old big green) */
.app-header { display:flex; align-items:center; gap:14px; margin-bottom:10px; }
.app-header .title { display:flex; flex-direction:column; }
.app-header .title .org { font-size:28px; font-weight:800; color:#16a34a; }
.app-header .title .proj { font-size:14px; color:#334155; }
.app-header .credit { margin-left:auto; text-align:right; font-size:12px; line-height:1.2; }

/* Print controls */
#global-print { text-align:right; margin:6px 0 10px; }
#global-print button{
  background-color:#16a34a; color:white; border:none; padding:8px 16px;
  border-radius:8px; cursor:pointer;
}

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

# Header with logo + names + credit
with st.container():
    c1, c2 = st.columns([0.7, 0.3])
    with c1:
        st.markdown("""
        <div class="app-header">
          <img src="assets/logo.png" width="64" alt="PIDIM Logo"/>
          <div class="title">
            <div class="org">PIDIM Foundation</div>
            <div class="proj">Sustainable Microenterprise and Resilient Transformation (SMART) Project</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="credit">
          <b>Created by,</b><br/>
          <b>Md. Moniruzzaman</b><br/>
          MIS &amp; Documentation Officer<br/>
          SMART Project<br/>
          Pidim Foundation<br/>
          Cell: 01324 168100
        </div>
        """, unsafe_allow_html=True)

# global print button
st.markdown("""<div id="global-print"><button onclick="window.print()">Print</button></div>""", unsafe_allow_html=True)

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

def to_excel_bytes(df_dict):
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

# Add Grand Total to every main table except loan (loan already includes Grand Total and stops there)
poultry = add_grand_total(poultry, numeric_cols=["# of MEs","# of Birds"])
grants = add_grand_total(grants, numeric_cols=["Number on MEs","Amounts of Grants"])

# ===== UI =====
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
    # Only show the table; nothing extra after Grand Total inside this section
    st.dataframe(sty, use_container_width=True, height=720)
with rcol:
    base2=loan[(~loan["Branch Name"].str.endswith(" Total")) & (loan["Branch Name"]!="Grand Total") & (loan["Types of Loan"]!="")]
    fig=px.bar(base2, x="Branch Name", y="Amount of Loan", color="Types of Loan", barmode="group", title="Amount of Loan by Branch & Type")
    fig.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

st.markdown('<h3 class="section-title">🐔 Types of Poultry Rearing</h3>', unsafe_allow_html=True)
p_l, p_r = st.columns([0.55, 0.45], gap="large")
with p_l:
    st.dataframe((poultry.style.hide(axis="index").format({"# of MEs":"{:,.0f}","# of Birds":"{:,.0f}"})
                  .set_table_styles([
                    {"selector":"th","props":[("background-color","#dcfce7"),("font-weight","800")]},
                    {"selector":"thead th","props":[("background-color","#dcfce7"),("font-weight","800")]},
                    {"selector":"tbody tr:nth-child(even)","props":[("background-color","#fafafa")]}
                  ])),
                 use_container_width=True)
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
    st.dataframe((grants.style.hide(axis="index").format({"Amounts of Grants":"{:,.0f}","Number on MEs":"{:,.0f}"})
                  .set_table_styles([
                    {"selector":"th","props":[("background-color","#dcfce7"),("font-weight","800")]},
                    {"selector":"thead th","props":[("background-color","#dcfce7"),("font-weight","800")]},
                    {"selector":"tbody tr:nth-child(even)","props":[("background-color","#fafafa")]}
                  ])),
                 use_container_width=True)
with g_r:
    gp=grants[grants["Branch Name"]!="Grand Total"].copy()
    fig_g=px.bar(gp, x="Branch Name", y="Amounts of Grants", title="Grants Amount by Branch")
    fig_g.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
    st.plotly_chart(fig_g, use_container_width=True)
