# ---- Replace previous render_branch_loan_by_type and its calls with this block ----

# define column names by letter positions (uses existing cpos & col_letter_to_pos utilities)
FILTER_COL = cpos(df, col_letter_to_pos("AM"))   # column that contains loan product labels (e.g. SMART-Agrosor Loan, SMART-CSL)
CLASS_COL  = cpos(df, col_letter_to_pos("AN"))   # Enterprise / Non-Enterprise
AMOUNT_COL = cpos(df, col_letter_to_pos("AQ"))   # Loan Amount
MONTH_COL  = cpos(df, col_letter_to_pos("AP"))   # Date/month column for filtering (AP)

# ensure MONTH_COL is parsed as datetime (safe)
try:
    df[MONTH_COL] = pd.to_datetime(df[MONTH_COL], errors="coerce")
except Exception:
    pass

# Sidebar month selector (Year-Month)
st.sidebar.markdown("### Month filter for branch reports")
month_vals = []
if MONTH_COL in df.columns:
    ym = df[MONTH_COL].dropna().dt.to_period("M").sort_values().unique()
    month_vals = [str(x) for x in ym]
if month_vals:
    sel_month = st.sidebar.selectbox("Select Year-Month (optional)", options=["(all)"] + month_vals, index=0)
else:
    sel_month = "(all)"

def render_branch_loan_by_filter(df_all, loan_product_value, title_suffix):
    """
    Filters df_all where FILTER_COL == loan_product_value (case-insensitive),
    optionally filters by selected sel_month (if not '(all)'),
    then groups by branch (b) and CLASS_COL and sums AMOUNT_COL, and renders table/chart.
    """
    try:
        # basic checks
        if FILTER_COL not in df_all.columns:
            st.warning(f"Filter column {FILTER_COL} not found.")
            return
        if CLASS_COL not in df_all.columns:
            st.warning(f"Class column {CLASS_COL} not found.")
            return
        if AMOUNT_COL not in df_all.columns:
            st.warning(f"Amount column {AMOUNT_COL} not found.")
            return

        # build mask for product
        mask_prod = df_all[FILTER_COL].astype(str).str.strip().str.lower() == str(loan_product_value).strip().lower()
        df_filtered = df_all[mask_prod].copy()

        # apply month filter if selected
        if sel_month != "(all)" and MONTH_COL in df_filtered.columns:
            # sel_month like '2024-05'
            try:
                year, mon = sel_month.split("-")
                df_filtered = df_filtered[df_filtered[MONTH_COL].dt.to_period("M") == pd.Period(f"{year}-{mon}")]
            except Exception:
                # fallback: filter by substring
                df_filtered = df_filtered[df_filtered[MONTH_COL].astype(str).str.contains(sel_month)]

        st.markdown(f'<h3 class="section-title">📊 Branch Wise Loan Disbursement ({title_suffix})</h3>', unsafe_allow_html=True)

        if df_filtered.shape[0] == 0:
            st.info(f"No records found for loan type: {loan_product_value}")
            return

        # prepare aggregation: group by branch (b) and CLASS_COL (Enterprise / Non-Enterprise)
        tmp = df_filtered[[b, CLASS_COL, AMOUNT_COL]].copy()
        tmp[b] = clean_branch(tmp[b])
        tmp[CLASS_COL] = tmp[CLASS_COL].astype(str).str.strip()
        tmp[AMOUNT_COL] = pd.to_numeric(tmp[AMOUNT_COL], errors="coerce").fillna(0)

        agg = (tmp.groupby([b, CLASS_COL])
                 .agg(**{"# of Loan": (CLASS_COL, "count"), "Amount of Loan": (AMOUNT_COL, "sum")})
                 .reset_index()
                 .rename(columns={b: "Branch Name", CLASS_COL: "Types of Loan"}))

        # clean branch names and remove empties
        agg = agg[~agg["Branch Name"].astype(str).str.strip().str.lower().isin(["branch name","nan","none","null"])].copy()

        # produce summarized table using existing summarize_loan_table logic expects same column names
        # But our agg already matches expected columns, so can call summarize_loan_table
        try:
            loan_local = summarize_loan_table(agg)
        except Exception:
            # fallback: create a simple table with grand totals
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


# Now call for the two desired products (Agrosor and CSL)
render_branch_loan_by_filter(df, "SMART-Agrosor Loan", "SMART-Agrosor Loan")
st.markdown("---")
render_branch_loan_by_filter(df, "SMART-CSL", "SMART-CSL")
st.markdown("---")

# ---- end replacement block ----
