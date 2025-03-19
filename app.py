from collections import defaultdict
import pandas as pd
import streamlit as st
import plotly.express as px
from cfg import cfg

st.set_page_config(layout="wide")

from src.get_data_with_a_decision import (
    evaluate_decision
)

from src.metrics import (
    calculate_error_metrics,
    aggregate_metrics,
    calculate_correlation_metrics,
    calculate_efficiency_metrics, 
    aggregate_investment_metrics, 
    calculate_top_profit_metrics
)

raw_df = pd.read_csv(f"datasets/ready_to_work/{cfg.data_to_test}", comment="#")

st.title("📈 Approach Comparison")

@st.cache_data(show_spinner=True, hash_funcs={list: lambda l: tuple(l)})
def compute_results(raw_df, _approaches):
    results = {}
    for approach in _approaches:
        result_df = evaluate_decision(
            raw_df,
            approach["decision_func"],
            approach["params"],
            approach["predicted_price_attr"],
            approach["second_predicted_price_attr"]
        )
        result_df["approach"] = approach["name"]
        results[approach["name"]] = result_df.copy()
    return results

# Compute results once (this is the expensive part that's now cached)
results = compute_results(raw_df, cfg.approaches)

# Display brief results for each approach
for approach in cfg.approaches:
    name = approach["name"]
    with st.expander(f"Description and full data for the approach {name}"):
        st.write(f"**Description:**")
        st.write(approach["description"])
        st.write(f"**Approach parameters:**")
        st.json(approach["params"])
        st.write(f"**Source data for approach:**")
        st.dataframe(results[name])

# %% Sidebar: global filtering and display settings
with st.sidebar:
    st.header("Analysis Settings")
    min_price, max_price = st.slider(
        "Price Range",
        min_value=int(raw_df['buy_price'].min()),
        max_value=int(raw_df['buy_price'].max()),
        value=(int(raw_df['buy_price'].min()), int(raw_df['buy_price'].max())),
        step=1
    )

    # Filter parameters
    decision_filter = st.checkbox("Only accepted decisions (decision == True)", value=False)
    filter_type = st.selectbox("filter type for data slicing (sorted by sale_profit_percent)", options=["Percentages", "Number of rows"], index=0)
    
    if filter_type == "Percentages":
        lower_percent = st.slider("Trim bottom (%)", 0.0, 50.0, 0.0, 0.5)
        upper_percent = st.slider("Trim top (%)", 0.0, 50.0, 0.0, 0.5)
    else:
        lower_rows = st.slider("Trim rows from bottom", 0, 1000, 1)
        upper_rows = st.slider("Trim rows from top", 0, 1000, 5)
    
    metric_type = st.selectbox(
        "Metric type",
        options=["base", "corr", "error_percent", "investment", "efficiency", "top_profits"],
        index=0,
        format_func=lambda x: {
            'base': 'Basic metrics',
            "corr": "Correlation",
            'error_percent': "Prediction errors",
            'investment': 'Investment metrics',
            'efficiency': 'Strategy efficiency',
            'top_profits': 'Top profits'
        }.get(x, "Unknown Metric")  # Fallback for unexpected values
    )

    with st.expander("📚 Metric Descriptions"):
        st.write("""
        **Used metrics:**
        - **Basic:** 
            - Total number of deals
            - Decision acceptance rate
            - Average holding duration
                 
        - **Correlation:**
            - Correlation analysis
            - Error correlation visualization vs buy_price/preds_diff/pred_profit_%
            - 3D Correlation for buy_price/preds_diff/pred_profit
        
        - **Prediction errors:** 
            - Error distribution
            - Percentiles (5 25 50 75 95)
            - Mean error value
            
        - **Investment:** 
            - Total profit (USD)
            - ROI (Return on Investment)
            - Return percentiles (10th, 25th, 50th, 75th, 90th)
            
        - **Efficiency:**
            - Profit density (profit/time unit)
            - Capital efficiency (profit/investment)
            
        - **Top profits:**
            - Analysis by target investment amounts
            - Profit category distribution
        """)

# %% Apply filtering to each approach's results
filtered_results = {}
for approach_name, df_approach in results.items():
    try:
        if filter_type == "Percentages":
            lower_q = lower_percent / 100
            upper_q = 1 - (upper_percent / 100)
            lower_profit = df_approach['sale_profit_percent'].quantile(lower_q)
            upper_profit = df_approach['sale_profit_percent'].quantile(upper_q)
            filtered_df = df_approach[
                (df_approach['sale_profit_percent'] >= lower_profit) & 
                (df_approach['sale_profit_percent'] <= upper_profit)
            ]
        else:
            sorted_df = df_approach.sort_values('sale_profit_percent', ascending=False)
            n_lower = min(lower_rows, len(sorted_df))
            n_upper = min(upper_rows, len(sorted_df) - n_lower)
            filtered_df = sorted_df.iloc[n_lower:-n_upper]
    except Exception as e:
        st.error(f"Filter error in {approach_name}: {e}")
        filtered_df = df_approach.copy()
    
    if decision_filter:
        filtered_df = filtered_df[filtered_df['decision']]

    filtered_df = filtered_df[(filtered_df['buy_price'] >= min_price) & (filtered_df['buy_price'] <= max_price)]

    filtered_results[approach_name] = filtered_df.copy()

# For global visualizations, combine filtered data
combined_filtered_df = pd.concat(list(filtered_results.values()), ignore_index=True)

# %% Cache calculation of all metrics for filtered DataFrames
@st.cache_data(show_spinner=False)
def compute_all_metrics(filtered_df):
    metrics_dict = {}
    figures_dict = defaultdict(list)
    
    corr_metrics, corr_plots = calculate_correlation_metrics(filtered_df)
    metrics_dict["corr"] = corr_metrics
    figures_dict["corr"].extend(corr_plots)

    error_percent_metrics, error_percent_plots = calculate_error_metrics(filtered_df)
    metrics_dict["error_percent"] = error_percent_metrics
    figures_dict["error_percent"].extend(error_percent_plots)

    # Investment metrics
    investment_metrics = aggregate_investment_metrics(filtered_df)
    metrics_dict["investment"] = investment_metrics

    base_metrics, base_plots = aggregate_metrics(filtered_df)
    metrics_dict["base"] = base_metrics
    figures_dict["base"].extend(base_plots)

    metrics_dict["efficiency"] = calculate_efficiency_metrics(filtered_df)

    top_profit_metrics, top_profit_plots = calculate_top_profit_metrics(filtered_df, [100, 250, 800, 2000, 5000])
    metrics_dict["top_profits"] = top_profit_metrics
    figures_dict["top_profits"].extend(top_profit_plots)

    return metrics_dict, figures_dict

# %% Main tabs: Metric Analysis and Deal Details
tab1, tab2, tab3 = st.tabs(["Metric Analysis", "Deal Details", "Visualization"])

# ----- Tab 1: Metric Analysis -----
with tab1:
    st.header("Key Metrics")
    
    metrics_list = []
    for approach_name, df_approach in filtered_results.items():
        all_metrics, _ = compute_all_metrics(df_approach)
        m = all_metrics[metric_type]
        m["approach"] = approach_name
        metrics_list.append(m)
        
    combined_metrics = pd.concat(metrics_list, ignore_index=True)
    
    st.dataframe(
        combined_metrics.style.format({
            'total_profit': '${:+,.2f}',
            'average_profit_percent': '{:.2f}%',
            'profit_percent_25th': '{:.2f}%',
            'profit_percent_90th': '{:.2f}%',
            'capital_efficiency': '{:.2f}'
        }).background_gradient(cmap='Blues'),
        use_container_width=True
        )

# ----- Tab 2: Deal Details -----
with tab2:
    st.header("Deal Analysis")
    for approach_name, df_approach in filtered_results.items():
        st.subheader(f"Approach: {approach_name}")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Max Profit % before filtering", f"{results[approach_name]['sale_profit_percent'].max():.2f}%")
            st.metric("Max Profit % after filtering", f"{df_approach['sale_profit_percent'].max():.2f}%")
        with col2:
            st.metric("Min Profit % before filtering", f"{results[approach_name]['sale_profit_percent'].min():.2f}%")
            st.metric("Min Profit % after filtering", f"{df_approach['sale_profit_percent'].min():.2f}%")
            
        st.subheader("Top 5 by Profit %")
        top_deals = df_approach.nlargest(5, 'sale_profit_percent')
        st.dataframe(
            top_deals[['name', 'sale_profit_percent', 'sale_profit_usd']]
            .style.format({
                'sale_profit_percent': '{:.2f}%',
                'sale_profit_usd': '${:+,.2f}'
            }).background_gradient(
                subset=['sale_profit_percent'], cmap='Greens'
            ),
            use_container_width=True
        )
        
        st.subheader("Worst 5 by Profit %")
        worst_deals = df_approach.nsmallest(5, 'sale_profit_percent')
        st.dataframe(
            worst_deals[['name', 'sale_profit_percent', 'sale_profit_usd']]
            .style.format({
                'sale_profit_percent': '{:.2f}%',
                'sale_profit_usd': '${:+,.2f}'
            }).background_gradient(
                subset=['sale_profit_percent'], cmap='Reds'
            ),
            use_container_width=True
        )
        
        st.subheader("Profit Percentage Distribution")
        fig = px.histogram(
            df_approach,
            x='sale_profit_percent',
            nbins=50,
            title=f"{approach_name} - Distribution Frequency",
            labels={'sale_profit_percent': 'Profit Percentage'},
            color_discrete_sequence=['#2ecc71']
        )
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)

# ======== VISUALIZATION TAB ========
with tab3:
    st.header("📊 Metric Visualization")
    
    # Collect all plots for approaches
    all_figures = defaultdict(lambda: defaultdict(list))
    
    for approach_name, df_approach in filtered_results.items():
        _, figures = compute_all_metrics(df_approach)
        for _metric_type, figs in figures.items():
            all_figures[_metric_type][approach_name] = figs
    
    # Use selected metric type from sidebar
    selected_metric = metric_type
    
    # Display plots for selected metric
    if selected_metric in all_figures:
        metric_figures = all_figures[selected_metric]
                
        if not metric_figures:
            st.warning(f"No plots for metric type: {selected_metric}")
        else:
            # Determine max number of plots per approach
            num_graphs = max(len(figs) for figs in metric_figures.values())
            
            # Display plots in order
            for graph_idx in range(num_graphs):
                cols = st.columns(2)
                has_content = False
                
                for idx, (approach, figures) in enumerate(metric_figures.items()):
                    if graph_idx < len(figures):
                        has_content = True
                        with cols[idx % 2]:
                            fig = figures[graph_idx]
                            fig.update_layout(
                                title=f"{approach} - {fig.layout.title.text}",
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                # Add separator between plot groups
                if has_content and graph_idx < num_graphs - 1:
                    st.markdown("---")
    else:
        st.warning(f"No data for visualization type: {selected_metric}")

with st.expander("Technical Details"):
    total_deals = sum(len(df) for df in filtered_results.values())
    original_deals = sum(len(results[approach]) for approach in results)
    st.write(f"Total filtered deals (all approaches): {total_deals}")
    st.write(f"Original number of deals (all approaches): {original_deals}")
    st.download_button("Download Filtered Data", combined_filtered_df.to_csv(), "investment_data.csv")

st.caption("Analytics System v2.0 | Data updates in real-time")