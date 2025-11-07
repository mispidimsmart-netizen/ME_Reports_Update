
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import requests
from io import StringIO

PUBLISHED_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRkcagLu_YrYgQxmsO3DnHn90kqALkw9uDByX7UBNRUjaFKKQdE3V-6fm5ZcKGk_A/pub?gid=2143275417&single=true&output=csv"

def col_letter_to_pos(s: str) -> int:
    s = re.sub(r'[^A-Za-z]', '', s); v = 0
    for ch in s.upper(): v = v*26 + (ord(ch)-64)
    return v

BRANCH_COL_POS = col_letter_to_pos("G")
LOAN_TYPE_POS  = col_letter_to_pos("AN")
AMOUNT_POS     = col_letter_to_pos("AQ")
BIRDS_POS      = col_letter_to_pos("U")

st.set_page_config(page_title="PIDIM Foundation — SMART Project Reports", layout="wide")

l, r = st.columns([0.12, 0.88])
with l:
    st.image("logo.png", width=72)
with r:
    st.markdown("<div style='font-size:34px; font-weight:800; color:#16a34a;'>PIDIM Foundation</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:16px; color:#334155;'>Sustainable Microenterprise and Resilient Transformation (SMART) Project</div>", unsafe_allow_html=True)

b1, b2, _ = st.columns([0.2, 0.18, 0.62])
with b1: refresh = st.button("🔄 Refresh data")
with b2: fast_mode = st.toggle("⚡ Fast mode", value=True)

st.markdown("<style>.card{background:#fff;border:1px solid #eaecef;border-radius:14px;padding:14px 16px;margin-bottom:18px}</style>", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def _sess():
    s = requests.Session(); s.headers.update({"User-Agent":"ME-Reports/1.0"}); return s

@st.cache_data(ttl=900, show_spinner=False)
def fetch_csv(url:str)->str:
    r = _sess().get(url, timeout=20); r.raise_for_status(); return r.text

@st.cache_data(ttl=900, show_spinner=False)
def parse_df(text:str)->pd.DataFrame:
    df = pd.read_csv(StringIO(text), low_memory=False); df.columns=[str(c).strip() for c in df.columns]; return df

if refresh:
    fetch_csv.clear(); parse_df.clear(); st.experimental_rerun()

text = fetch_csv(PUBLISHED_CSV_URL)
df = parse_df(text)
if df.empty: st.error("Empty sheet"); st.stop()

with st.expander("📂 View Raw Data", expanded=False):
    st.dataframe(df, use_container_width=True)

def _get_by_pos(df, pos1):
    i = max(0, min(len(df.columns)-1, pos1-1)); return df.columns[i]

@st.cache_data(ttl=900, show_spinner=False)
def compute_branch_loan(df_in, b_col, t_col, a_col):
    w = df_in[[b_col,t_col,a_col]].copy()
    w[b_col]=w[b_col].astype(str).str.strip(); w[t_col]=w[t_col].astype(str).str.strip()
    w["_amt_"]=pd.to_numeric(w[a_col], errors="coerce")
    def norm(x):
        x=(x or "").strip().lower(); x=re.sub(r"\s+"," ",x)
        if "non" in x and "enterprise" in x: return "Non-Enterprise"
        if "enterprise" in x: return "Enterprise"
        return x.title() if x else x
    w["_type_"]=w[t_col].apply(norm)
    w=w[w[b_col].notna() & (w[b_col].astype(str).str.strip()!="")]
    g=(w.groupby([b_col,"_type_"], dropna=False)
        .agg(**{"# of Loan":("_type_","count"),"Amount of Loan":("_amt_","sum")})
        .reset_index().rename(columns={b_col:"Branch Name","_type_":"Types of Loan"}))
    return g

@st.cache_data(ttl=900, show_spinner=False)
def compute_poultry(df_in, b_col, act_col, birds_col):
    t=df_in[[b_col,act_col,birds_col]].copy()
    t[b_col]=t[b_col].astype(str).str.strip(); t[act_col]=t[act_col].astype(str).str.strip()
    t["_birds_"]=pd.to_numeric(t[birds_col], errors="coerce")
    t=t[t[b_col].notna() & (t[b_col].astype(str).str.strip()!="")]
    def agg(key):
        d=t[t[act_col].str.lower().str.contains(key, na=False)]
        c=d.groupby(b_col, dropna=False).size().reset_index(name="count")
        b=d.groupby(b_col, dropna=False)["_birds_"].sum(min_count=1).reset_index(name="birds")
        return c.merge(b, on=b_col, how="outer")
    layer=agg("layer"); broil=agg("broiler")
    base=pd.DataFrame({b_col: sorted(t[b_col].dropna().unique(), key=lambda x: str(x))})
    out=base.merge(layer.rename(columns={"count":"Layer Rearing","birds":"Layer Birds"}), on=b_col, how="left")
    out=out.merge(broil.rename(columns={"count":"Broiler Rearing","birds":"Broiler Birds"}), on=b_col, how="left")
    out=out.fillna(0).rename(columns={b_col:"Branch Name"})
    return out

@st.cache_data(ttl=900, show_spinner=False)
def compute_farm_counts(df_in, b_col, f_col):
    t=df_in[[b_col,f_col]].copy()
    t[b_col]=t[b_col].astype(str).str.strip(); t[f_col]=t[f_col].astype(str).str.strip()
    t=t[t[b_col].notna() & (t[b_col].astype(str).str.strip()!="")]
    def cnt(k):
        d=t[t[f_col].str.lower().str.contains(k, na=False)]
        return d.groupby(b_col, dropna=False).size().reset_index(name="count")
    gen=cnt("general"); mod=cnt("model")
    base=pd.DataFrame({b_col: sorted(t[b_col].dropna().unique(), key=lambda x: str(x))})
    out=base.merge(gen.rename(columns={"count":"General Farm"}), on=b_col, how="left")
    out=out.merge(mod.rename(columns={"count":"Model Farm"}), on=b_col, how="left")
    out=out.fillna(0).rename(columns={b_col:"Branch Name"})
    return out

# ---------------- Side-by-side ----------------
c1, c2 = st.columns(2, gap="large")

with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Branch Wise Loan Disbursement")
    bc=_get_by_pos(df, BRANCH_COL_POS); tc=_get_by_pos(df, LOAN_TYPE_POS); ac=_get_by_pos(df, AMOUNT_POS)
    agg=compute_branch_loan(df, bc, tc, ac)
    order={"Enterprise":0,"Non-Enterprise":1}; agg["_o_"]=agg["Types of Loan"].map(order).fillna(99).astype(int)
    rows=[]
    for b,g in agg.sort_values(["Branch Name","_o_"]).groupby("Branch Name", sort=False):
        for _,r in g.iterrows():
            rows.append({"Branch Name":b,"Types of Loan":r["Types of Loan"],"# of Loan":int(r["# of Loan"]),"Amount of Loan":float(r["Amount of Loan"] or 0)})
        rows.append({"Branch Name":f"{b} Total","Types of Loan":"","# of Loan":int(g["# of Loan"].sum()),"Amount of Loan":float(g["Amount of Loan"].sum())})
    if rows:
        tmp=pd.DataFrame(rows)
        rows.append({"Branch Name":"Grand Total","Types of Loan":"","# of Loan":int(tmp[tmp["Types of Loan"]!=""]["# of Loan"].sum()),"Amount of Loan":float(tmp[tmp["Types of Loan"]!=""]["Amount of Loan"].sum())})
    out=pd.DataFrame(rows)
    out=out[out["Branch Name"].astype(str).str.strip().str.lower().ne("nan")]
    out.insert(0,"ক্রমিক নং", range(1,len(out)+1))
    if fast_mode:
        s=out.copy(); s["Amount of Loan"]=pd.to_numeric(s["Amount of Loan"], errors="coerce").fillna(0).round(0).astype(int); st.dataframe(s, use_container_width=True)
    else:
        def sty(df_in):
            d=df_in.copy(); d["Amount of Loan"]=pd.to_numeric(d["Amount of Loan"], errors="coerce").fillna(0).round(0).astype(int).map(lambda x:f"{x:,}")
            stl=d.style.hide(axis="index")
            def row(r):
                name=str(r.get("Branch Name",""))
                if name.endswith(" Total") and name!="Grand Total": return ["background-color:#fffbe6; color:#000; font-weight:700"]*len(r)
                if name=="Grand Total": return ["background-color:#ffe59a; color:#000; font-weight:700"]*len(r)
                return [""]*len(r)
            return stl.apply(row, axis=1)
        st.dataframe(sty(out), use_container_width=True)
    try:
        cb=out[(~out["Branch Name"].str.endswith(" Total")) & (out["Branch Name"]!="Grand Total") & (out["Types of Loan"]!="")]
        fig=px.bar(cb, x="Branch Name", y="Amount of Loan", color="Types of Loan", barmode="group"); fig.update_layout(title_text="Amount of Loan by Branch & Type")
        st.plotly_chart(fig, use_container_width=True)
    except Exception: pass
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🐔 Types of Poultry Rearing")
    bc=_get_by_pos(df, BRANCH_COL_POS); act=(lambda d: [c for c in d.columns if re.search(r'(layer|broiler)', str(c), re.I)] or [d.columns[0]])(df)[0]; birds=_get_by_pos(df, BIRDS_POS)
    stats=compute_poultry(df, bc, act, birds)
    stats=stats[stats["Branch Name"].astype(str).str.strip().str.lower().ne("nan")]
    disp=stats.copy(); disp.columns=["Branch Name","Layer Rearing","# of Birds","Broiler Rearing","# of Birds"]
    gt=pd.DataFrame([{"Branch Name":"Grand Total","Layer Rearing":int(stats["Layer Rearing"].sum()),"Layer Birds":int(stats["Layer Birds"].sum()),"Broiler Rearing":int(stats["Broiler Rearing"].sum()),"Broiler Birds":int(stats["Broiler Birds"].sum())}])
    gt_disp=gt.copy(); gt_disp.columns=["Branch Name","Layer Rearing","# of Birds","Broiler Rearing","# of Birds"]
    disp_all=pd.concat([disp, gt_disp], ignore_index=True)
    st.dataframe(disp_all, use_container_width=True)
    st.download_button("⬇️ Download — Types of Poultry Rearing (CSV)", data=disp_all.to_csv(index=False).encode("utf-8"), file_name="types_of_poultry_rearing.csv", mime="text/csv")
    try:
        fig1=px.bar(stats, x="Branch Name", y=["Layer Rearing","Broiler Rearing"], barmode="group", title="Poultry Rearing (Count) by Branch")
        fig2=px.bar(stats, x="Branch Name", y=["Layer Birds","Broiler Birds"], barmode="group", title="# of Birds by Branch (Layer vs Broiler)")
        st.plotly_chart(fig1, use_container_width=True); st.plotly_chart(fig2, use_container_width=True)
    except Exception: pass
    st.markdown("</div>", unsafe_allow_html=True)

# Third card (farm counts) omitted here for brevity in this quick patch
