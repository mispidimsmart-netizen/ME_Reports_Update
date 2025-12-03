# ---- place this function AFTER helper functions (compute_branch_loan, summarize_loan_table, to_excel_bytes) AND AFTER df,b,lt,la are defined ----
def render_branch_loan_by_type(df_all, loan_type_value, title_suffix):
    """
    Renders a branch-wise loan disbursement table and chart for rows where loan-type == loan_type_value.
    Comparison is case-insensitive and strips whitespace.
    Requires existing names: b (branch colname), lt (loan type colname), la (loan amount colname)
    and helper functions compute_branch_loan, summarize_loan_table, to_excel_bytes.
    """
    try:
        # ensure loan-type column exists
        if lt not in df_all.columns:
            st.warning(f"Loan-type column ({lt}) not found in data.")
            return

        # filter safely: coerce to str, strip, lower and compare
        mask = df_all[lt].astype(str).str.strip().str.lower() == str(loan_type_value).strip().lower()
        df_lt = df_all[mask].copy()

        st.markdown(f'<h3 class="section-title">📊 Branch Wise Loan Disbursement ({title_suffix})</h3>', unsafe_allow_html=True)

        if df_lt.shape[0] == 0:
            st.info(f"No records found for loan type: {loan_type_value}")
            return

        # compute using existing helper
        loan_agg_local = compute_branch_loan(df_lt, b, lt, la)
        loan_local = summarize_loan_table(loan_agg_local)

        # render table + download + chart
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
                        .apply(lambda r: ["background-color:#dcfce7; color:#000; font-weight:800"]*len(r) if str(r.get("Branch Name",""))=="Grand Total" else (["background-color:#fffbe6; color:#0f172a; font-weight:700"]*len(r) if str(r.get("Branch Name","")).endswith(" Total") else [""]*len(r)), axis=1)
                )
                st.dataframe(sty_local, use_container_width=True, height=520)
            except Exception:
                # fallback simple table if styling fails
                st.dataframe(loan_local, use_container_width=True, height=520)
            # download
            try:
                st.download_button(f"⬇️ {title_suffix} — Excel", to_excel_bytes({title_suffix: loan_local}), file_name=f"loan_disbursement_{str(loan_type_value).strip().replace(' ','_')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
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
        st.error(f"Error rendering branch report for {loan_type_value}: {e}")
