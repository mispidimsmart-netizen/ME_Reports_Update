# ---------------------------------------------------------
# === PLACE THIS BLOCK RIGHT AFTER the loan/poultry/grants tables are generated ===
# ---------------------------------------------------------

FILTER_COL = cpos(df, col_letter_to_pos("AM"))
CLASS_COL  = cpos(df, col_letter_to_pos("AN"))
AMOUNT_COL = cpos(df, col_letter_to_pos("AQ"))
MONTH_COL  = cpos(df, col_letter_to_pos("AP"))

if MONTH_COL in df.columns:
    try:
        df[MONTH_COL] = pd.to_datetime(df[MONTH_COL], errors="coerce")
    except Exception:
        pass

st.sidebar.markdown("### Month filter for branch reports")
month_vals = []
if MONTH_COL in df.columns:
    try:
        ym = df[MONTH_COL].dropna().dt.to_period("M").sort_values().unique()
        month_vals = [str(x) for x in ym]
    except:
        month_vals = sorted(df[MONTH_COL].dropna().astype(str).unique().tolist())[:100]

sel_month = st.sidebar.selectbox("Select Year-Month (optional)", ["(all)"] + month_vals, index=0)

def render_branch_loan_by_filter(df_all, loan_product_value, title_suffix):
    try:
        mask_prod = df_all[FILTER_COL].astype(str).str.strip().str.lower() == loan_product_value.strip().lower()
        df_filtered = df_all[mask_prod].copy()

        if sel_month != "(all)" and MONTH_COL in df_filtered.columns:
            try:
                y,m = sel_month.split("-")
                df_filtered = df_filtered[df_filtered[MONTH_COL].dt.to_period("M") == pd.Period(f"{y}-{m}")]
            except:
                df_filtered = df_filtered[df_filtered[MONTH_COL].astype(str).str.contains(sel_month)]

        st.markdown(f"<h3>📊 Branch Wise Loan Disbursement ({title_suffix})</h3>", unsafe_allow_html=True)

        if df_filtered.shape[0] == 0:
            st.info(f"No records found for loan type: {loan_product_value}")
            return

        tmp = df_filtered[[b, CLASS_COL, AMOUNT_COL]].copy()
        tmp[b] = clean_branch(tmp[b])
        tmp[CLASS_COL] = tmp[CLASS_COL].astype(str).str.strip()
        tmp[AMOUNT_COL] = pd.to_numeric(tmp[AMOUNT_COL], errors="coerce").fillna(0)

        agg = tmp.groupby([b, CLASS_COL]).agg(
            **{
                "# of Loan": (CLASS_COL, "count"),
                "Amount of Loan": (AMOUNT_COL, "sum"),
            }
        ).reset_index().rename(columns={b: "Branch Name", CLASS_COL: "Types of Loan"})

        agg = agg[~agg["Branch Name"].astype(str).str.strip().str.lower().isin(["branch name","nan","none","null"])]

        try:
            loan_local = summarize_loan_table(agg)
        except:
            loan_local = ensure_serial(agg)
            loan_local = add_grand_total(loan_local, ["# of Loan","Amount of Loan"])

        st.dataframe(loan_local, use_container_width=True)

        st.download_button(
            f"⬇️ {title_suffix} — Excel",
            to_excel_bytes({title_suffix: loan_local}),
            file_name=f"loan_{loan_product_value.replace(' ','_')}.xlsx"
        )

    except Exception as e:
        st.error(str(e))


render_branch_loan_by_filter(df, "SMART-Agrosor Loan", "SMART-Agrosor Loan")
st.markdown("---")
render_branch_loan_by_filter(df, "SMART-CSL", "SMART-CSL")
st.markdown("---")
