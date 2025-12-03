# ===== Branch-wise reports split by loan type =====
# Replace the original single "Branch Wise Loan Disbursement" block with this code.

# Helper to build and render branch report for a given loan type filter
def render_branch_loan_by_type(df_all, loan_type_value, title_suffix):
    try:
        # lt is the column name for loan type (already defined earlier)
        df_lt = df_all.copy()
        # safe selection: coerce to str and compare (strip for safety)
        df_lt = df_lt[df_lt[lt].astype(str).str.strip() == loan_type_value]
        # If no rows, show info
        st.markdown(f'<h3 class="section-title">📊 Branch Wise Loan Disbursement ({title_suffix})</h3>', unsafe_allow_html=True)
        if df_lt.shape[0] == 0:
            st.info(f"No records found for loan type: {loan_type_value}")
            return

        # compute aggregation using existing builder functions
        loan_agg_local = compute_branch_loan(df_lt, b, lt, la)
        loan_local = summarize_loan_table(loan_agg_local)

        # Left: styled table + download, Right: bar chart
        lcol, rcol = st.columns([0.55, 0.45], gap="large")
        with lcol:
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
            st.download_button(f"⬇️ {title_suffix} — Excel", to_excel_bytes({title_suffix: loan_local}), file_name=f"loan_disbursement_{loan_type_value.replace(' ','_')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with rcol:
            base_plot = loan_local[(~loan_local["Branch Name"].str.endswith(" Total")) & (loan_local["Branch Name"]!="Grand Total") & (loan_local["Types of Loan"]!="")]
            if base_plot.shape[0] == 0:
                st.info("No branch-level breakdown to plot for this loan type.")
            else:
                fig_local = px.bar(base_plot, x="Branch Name", y="Amount of Loan", color="Types of Loan", barmode="group", title=f"Amount of Loan by Branch & Type — {title_suffix}")
                fig_local.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
                st.plotly_chart(fig_local, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering branch report for {loan_type_value}: {e}")

# Render SMART-Agrosor Loan report (only those loans)
render_branch_loan_by_type(df, "SMART-Agrosor Loan", "SMART-Agrosor Loan")

st.markdown("---")

# Render SMART-CSL report (only those loans)
render_branch_loan_by_type(df, "SMART-CSL", "SMART-CSL")

st.markdown("---")
