# ---------------------------------------------------------
# === PLACE THIS BLOCK AFTER: df=get_df() and after b, lt, la are defined ===
# (i.e. immediately after the "# ===== Data =====" section)
# ---------------------------------------------------------

# Column letters we will use: AM (product), AN (enterprise/non-enterprise), AQ (amount), AP (month/date)
FILTER_COL = cpos(df, col_letter_to_pos("AM"))   # product label column (e.g. SMART-Agrosor Loan)
CLASS_COL  = cpos(df, col_letter_to_pos("AN"))   # Enterprise / Non-Enterprise classification
AMOUNT_COL = cpos(df, col_letter_to_pos("AQ"))   # Loan amount
MONTH_COL  = cpos(df, col_letter_to_pos("AP"))   # month/date column for filtering

# make sure month column parsed as datetime if present
if MONTH_COL in df.columns:
    try:
        df[MONTH_COL] = pd.to_datetime(df[MONTH_COL], errors="coerce")
    except Exception:
        pass

# build month dropdown values (Year-Month strings)
st.sidebar.markdown("### Month filter for branch reports")
month_vals = []
if MONTH_COL in df.columns:
    try:
        ym = df[MONTH_COL].dropna().dt.to_period("M").sort_values().unique()
        month_vals = [str(x) for x in ym]
    except Exception:
        # fallback: unique string samples
        month_vals = sorted(df[MONTH_COL].dropna().astype(str).unique().tolist())[:100]
sel_month = st.sidebar.selectbox("Select Year-Month (optional)", options=["(all)"] + month_vals, index=0)

# small debug helper (uncomment if you need to inspect values)
# st.write("DEBUG: FILTER_COL name:", FILTER_COL)
# st.write("DEBUG sample AM values:", df[FILTER_COL].dropna().astype(str).unique()[:20])
# st.write("DEBUG sample AP values:", df[MONTH_COL].dropna().astype(str).unique()[:10])

def render_branch_loan_by_filter(df_all, loan_product_value, title_suffix):
    """
    Filters df_all where FILTER_COL matches loan_product_value (case-insensitive),
    optionally filters by selected sel_month (if not '(all)'),
    then groups by branch (b) and CLASS_COL and sums AMOUNT_COL, and renders table/chart.
    """
    try:
        # verify required columns exist
        missing = [c for c in (FILTER_COL, CLASS_COL, AMOUNT_COL) if c not in df_all.columns]
        if missing:
            st.warning(f"Required columns missing for this report: {missing}")
            return

        # filter by product (case-insensitive, strip)
        mask_prod = df_all[FILTER_COL].astype(str).str.strip().str.lower() == str(loan_product_value).strip().lower()
        df_filtered = df_all[mask_prod].copy()

        # apply month filter if requested
        if sel_month != "(all)" and MONTH_COL in df_filtered.columns:
            try:
                y,m = sel_month.split("-")
                df_filtered = df_filtered[df_filtered[MONTH_COL].dt.to_period("M") == pd.Period(f"{y}-{m}")]
            except Exception:
                # fallback substring filter
                df_filtered = df_filtered[df_filtered[MONTH_COL].astype(str).str.contains(sel_month)]

        st.markdown(f'<h3 class="section-title">📊 Branch Wise Loan Disbursement ({title_suffix})</h3>', unsafe_allow_html=True)

        if df_filtered.shape[0] == 0:
            st.info(f"No records found for loan type: {loan_product_value}")
            return

        # prepare aggregation: group by branch (b) and CLASS_COL
        tmp = df_filtered[[b, CLASS_COL, AMOUNT_COL]].copy()
        tmp[b] = clean_branch(tmp[b])
        tmp[CLASS_COL] = tmp[CLASS_COL].astype(str).str.strip()
        tmp[AMOUNT_COL] = pd.to_numeric(tmp[AMOUNT_COL], errors="coerce").fillna(0)

        agg = (tmp.groupby([b, CLASS_COL])
                 .agg(**{"# of Loan": (CLASS_COL, "count"), "Amount of Loan": (AMOUNT_COL, "sum")})
                 .reset_index()
                 .rename(columns={b: "Branch Name", CLASS_COL: "Types of Loan"}))

        # remove bad branch rows
        agg = agg[~agg["Branch Name"].astype(str).str.strip().str.lower().isin(["branch name","nan","none","null"])].copy()

        # summarize (use existing summarize_loan_table which expects same columns)
        try:
            loan_local = summarize_loan_table(agg)
        except Exception:
            loan_local = agg.copy()
            loan_local = ensure_serial(loan_local)
            loan_local = add_grand_total(loan_local, numeric_cols=["# of Loan","Amount of Loan"])

        # Render table + download + chart
        lcol, rcol = st.columns([0.55, 0.45], gap="large")
        with lcol:
            try:
                sty_local = (
                    loan_local.style.hide(axis="index")
                             .format({"Amount of Loan":"{:,.0f}","# of Loan":"{:,.0f}"})
                             .set_table_styles([
                                 {"selector":"th","props":[("background-color","#dcfce7"),("font-weight","800")]},
                                 {"selector":"thead th","props":[("background-color","#dcfce7"),("font-weight","800")]},
                                 {"selector":"tbody tr:nth-child(even)","props":[("background-color","#fafafa")]}
                             ])
                )
                st.dataframe(sty_local, use_container_width=True, height=520)
            except Exception:
                st.dataframe(loan_local, use_container_width=True, height=520)

            # excel download
            try:
                fname = f"loan_disbursement_{loan_product_value.replace(' ','_')}.xlsx"
                st.download_button(f"⬇️ {title_suffix} — Excel", to_excel_bytes({title_suffix: loan_local}), file_name=fname, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception as e:
                st.error(f"Download prepare failed: {e}")

        with rcol:
            base_plot = loan_local[(~loan_local["Branch Name"].str.endswith(" Total")) & (loan_local["Branch Name"]!="Grand Total") & (loan_local["Types of Loan"]!="")]
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

# Call for the two products
render_branch_loan_by_filter(df, "SMART-Agrosor Loan", "SMART-Agrosor Loan")
st.markdown("---")
render_branch_loan_by_filter(df, "SMART-CSL", "SMART-CSL")
st.markdown("---")
# ---------------------------------------------------------
