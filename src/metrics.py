import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.utils.get_profit import get_profit_percent
from scipy.stats import pearsonr, spearmanr


def calculate_efficiency_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    accepted = results_df[results_df['decision'] == True]
    
    if len(accepted) == 0:
        return pd.DataFrame()

    metrics = {
        # Profit distribution density
        'profit_density': accepted['sale_profit_percent'].mean() / accepted['sale_profit_percent'].std(),
        
        # Capital efficiency (closer to 1 is better)
        'capital_efficiency': accepted['sale_profit_usd'].sum() / accepted['buy_price'].sum(),
        
        # Profit concentration (top 5% deals' share of total profit)
        'top5_profit_share': accepted.nlargest(int(len(accepted)*0.05), 'sale_profit_usd')['sale_profit_usd'].sum() / accepted['sale_profit_usd'].sum(),
        
    }
    
    return pd.DataFrame(metrics, index=[0])


#efficiency_metrics = calculate_efficiency_metrics(results_df)

def aggregate_investment_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates overall investment metrics, average profit percentage and its percentiles 
    for accepted deals using ready-made sale_profit_percent column.
    """
    accepted = results_df[results_df['decision'] == True].copy()
    
    # Calculate total metrics
    total_invested = (accepted['buy_price']).sum()
    total_received = (accepted['simulated_sale_price']).sum()
    total_profit = total_received - total_invested
    overall_profit_percent = get_profit_percent(total_invested, total_received)
    
    # Calculate profit statistics
    if not accepted.empty:
        # Filter valid deals (excluding NaN/Inf and zero investments)
        valid_deals = (
            accepted['sale_profit_percent'].notna() & 
            accepted['buy_price'].gt(0))  # Ensure invested > 0
        
        profit_series = accepted.loc[valid_deals, 'sale_profit_percent']
        
        # Calculate metrics
        mean_profit = profit_series.mean()
        percentiles = profit_series.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        profit_10 = percentiles.get(0.1, np.nan)
        profit_25 = percentiles.get(0.25, np.nan)
        profit_50 = percentiles.get(0.5, np.nan)
        profit_75 = percentiles.get(0.75, np.nan)
        profit_90 = percentiles.get(0.9, np.nan)

    else:
        mean_profit = profit_25 = profit_50 = profit_75 = np.nan
    
    metrics = {
        'total_invested': total_invested,
        'total_received': total_received,
        'total_profit': total_profit,
        'overall_profit_percent': overall_profit_percent,
        'average_profit_percent': mean_profit,
        'profit_percent_10th': profit_10,
        'profit_percent_25th': profit_25,
        'profit_percent_50th': profit_50,
        'profit_percent_75th': profit_75,
        'profit_percent_90th': profit_90
    }
    
    return pd.DataFrame(metrics, index=[0])


def calculate_correlation_metrics(results_df: pd.DataFrame) -> tuple[pd.DataFrame, list[go.Figure]]:
    """
    Analyzes relationships between prediction error and other parameters.
    Returns:
    1. DataFrame with correlation metrics
    2. List of Plotly figures
    """
    # Initialize output
    metrics_df = pd.DataFrame()
    
    # Check for empty data
    if results_df.empty or 'error_percent' not in results_df.columns:
        return metrics_df, []
    
    # Select key numerical parameters for analysis
    numeric_cols = [
        'buy_price', 'simulated_sale_price', 
        'predicted_price', 'predicted_profit_percent',
        'sale_profit_percent', 'error_percent'
    ]
    
    # Filter existing columns
    numeric_cols = [col for col in numeric_cols if col in results_df.columns]
    
    # 1. Calculate correlations
    corr_data = []
    for col in numeric_cols:
        if col == 'error_percent':
            continue
            
        # Filter NaN
        clean_df = results_df[[col, 'error_percent']].dropna()
        
        if len(clean_df) < 2:
            continue
            
        # Pearson correlation
        pearson_corr, pearson_p = pearsonr(clean_df[col], clean_df['error_percent'])
        # Spearman correlation
        spearman_corr, spearman_p = spearmanr(clean_df[col], clean_df['error_percent'])
        
        corr_data.append({
            'feature': col,
            'pearson_corr': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_corr': spearman_corr,
            'spearman_p_value': spearman_p
        })
    
    metrics_df = pd.DataFrame(corr_data)
    
    def _visualize(results_df):
        figures = []

        if len(numeric_cols) > 1:
            corr_matrix = results_df[numeric_cols].corr(method='pearson')
            fig = px.imshow(
                corr_matrix,
                text_auto=".2f",
                aspect="auto",
                title="Numerical Parameters Correlation Matrix",
                labels=dict(color="Correlation")
            )
            fig.update_layout(width=800, height=600)
            figures.append(fig)
        
        for col in ['buy_price', 'predicted_profit_percent', 'preds_diff']:
            if col not in results_df.columns or results_df[col].isnull().all():
                continue
                
            fig = px.scatter(
                results_df,
                x=col,
                y='error_percent',
                trendline="ols",
                title=f"Prediction Error vs {col}",
                labels={'error_percent': 'Error (%)', col: col.replace('_', ' ').title()}
            )
            fig.update_traces(marker=dict(size=5, opacity=0.6))
            figures.append(fig)
        
        # 2.3. Error distribution by price quartiles
        if 'buy_price' in results_df.columns:
            try:
                results_df['price_quantile'] = pd.qcut(
                    results_df['buy_price'],
                    q=4,
                    labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']
                )
                
                fig = px.box(
                    results_df,
                    x='price_quantile',
                    y='error_percent',
                    title="Error Distribution by Purchase Price Quartiles"
                )
                figures.append(fig)
            except Exception as e:
                pass
    
        # 2.4. 3D chart for key parameters
        if all(col in results_df.columns for col in ['buy_price', 'predicted_profit_percent', 'error_percent']):
            fig = px.scatter_3d(
                results_df,
                x='buy_price',
                y='predicted_price',
                z='error_percent',
                title="3D Dependency",
                opacity=0.5,
                color='error_percent',  # Gradient by error
                color_continuous_scale='Viridis'
            ).update_traces(
                marker=dict(size=4, line=dict(width=0))
            )
            figures.append(fig)
        
        return figures
    
    plots = _visualize(results_df)
    
    return metrics_df, plots


def calculate_error_metrics(results_df: pd.DataFrame) -> tuple[pd.DataFrame, list[go.Figure]]:
    """
    Calculates prediction error distribution metrics:
    - mean
    - percentiles (5, 25, 50, 75, 95)
    
    Returns DataFrame with one row of metrics
    """
    error_series = results_df['error_percent'].dropna()
    
    if error_series.empty:
        return pd.DataFrame()

    metrics = {
        'error_percent_mean': error_series.mean(),
        'error_percent_p5': np.percentile(error_series, 5),
        'error_percent_p25': np.percentile(error_series, 25),
        'error_percent_p50': np.percentile(error_series, 50),
        'error_percent_p75': np.percentile(error_series, 75),
        'error_percent_p95': np.percentile(error_series, 95)
    }

    def _visualize(results_df: pd.DataFrame):
        """
        Creates error distribution visualizations:
        1. Histogram with density kernel
        2. Boxplot
        """
        error_series = results_df['error_percent'].dropna()
        figures = []

        if error_series.empty:
            return figures

        # Histogram
        fig1 = px.histogram(
            error_series, 
            nbins=50,
            title='Error Percent Distribution',
            labels={'value': 'Prediction Error (%)'},
            marginal='box'
        )
        fig1.update_traces(marker=dict(line=dict(width=1, color='gray')))
        figures.append(fig1)

        # Boxplot
        fig2 = px.box(
            error_series,
            title='Error Percent Boxplot',
            labels={'value': 'Prediction Error (%)'}
        )
        fig2.update_layout(showlegend=False)
        figures.append(fig2)

        return figures

    plots = _visualize(results_df)
    
    return pd.DataFrame([metrics]), plots


def calculate_top_profit_metrics(results_df: pd.DataFrame, target_amounts: list) -> tuple[pd.DataFrame, list[go.Figure]]:
    """
    Calculates average profit percentage for given amounts by selecting most profitable deals.
    
    Parameters:
        results_df (pd.DataFrame): DataFrame with deal results containing 'buy_price' and 'sale_profit_percent'
        target_amounts (list): List of target amounts (e.g. [100, 250, 800])
    
    Returns:
        pd.DataFrame: With columns:
            - target_amount: target amount
            - total_invested: total invested amount
            - average_profit_percent: average profit percentage
            - num_deals: number of deals
    """
    # Create copy to avoid modifying original
    df = results_df.copy()
    # Calculate investment amount for each deal
    df['investment'] = df['buy_price']
    
    # Sort deals by profit percentage descending
    df_sorted = df.sort_values(by='sale_profit_percent', ascending=False)
    
    metrics = []
    for target in target_amounts:
        accumulated = 0.0
        selected_indices = []
        
        # Iteratively add deals until reaching or exceeding target
        for idx, row in df_sorted.iterrows():
            if accumulated >= target:
                break
            accumulated += row['investment']
            selected_indices.append(idx)
        
        # Calculate metrics
        if selected_indices:
            selected = df_sorted.loc[selected_indices]
            avg_profit = selected['sale_profit_percent'].mean()
            num_deals = len(selected_indices)
        else:
            avg_profit = None
            num_deals = 0
        
        metrics.append({
            'target_amount': target,
            'total_invested': accumulated,
            'average_profit_percent': avg_profit,
            'num_deals': num_deals
        })

    metrics_df = pd.DataFrame(metrics)

    def visualize():
        figures = []
        
        # 1. Investment efficiency chart
        fig1 = px.line(
            metrics_df,
            x='target_amount',
            y='average_profit_percent',
            title='Average Profit vs Target Investment',
            markers=True,
            labels={'target_amount': 'Target Amount ($)', 'average_profit_percent': 'Avg Profit (%)'}
        )
        fig1.update_traces(line_color='#9467bd')
        figures.append(fig1)
        
        
        # 2. Bubble chart
        fig2 = px.scatter(
            metrics_df,
            x='target_amount',
            y='average_profit_percent',
            size='num_deals',
            title='Deals Efficiency Bubble Chart',
            labels={'target_amount': 'Target Amount ($)',
                    'average_profit_percent': 'Avg Profit (%)',
                    'num_deals': 'Number of Deals'}
        )
        figures.append(fig2)
        
        return figures
    
    return metrics_df, visualize()


def aggregate_metrics(results_df: pd.DataFrame) -> tuple[pd.DataFrame, list[go.Figure]]:
    decision_mask = results_df['decision'] == True
    no_decision_mask = ~decision_mask
    num_purchases = decision_mask.sum()
    total_rows = len(results_df)
    
    total_actual_profit = results_df.loc[decision_mask, 'actual_profit_usd'].sum()
    avg_actual_profit = results_df.loc[decision_mask, 'actual_profit_usd'].mean() if num_purchases > 0 else np.nan
    
    total_potential_profit = results_df.loc[no_decision_mask, 'potential_profit_usd'].sum()
    total_potential_loss = results_df.loc[decision_mask, 'potential_loss_usd'].sum()
    
    avg_error_percent = results_df['error_percent'].mean()
    
    metrics = {
        'num_purchases': num_purchases,
        'total_rows': total_rows,
        'purchase_rate': num_purchases / total_rows,
        'total_actual_profit_usd': total_actual_profit,
        'total_potential_profit_usd': total_potential_profit,
        'total_potential_loss_usd': total_potential_loss,
        'avg_prediction_error_percent': avg_error_percent
    }

    metrics_df = pd.DataFrame(metrics, index=[0])

    def visualize():
        figures = []
        
        # 1. Sankey Diagram
        accepted = metrics_df['num_purchases'].values[0]
        rejected = metrics_df['total_rows'].values[0] - accepted
        
        fig1 = go.Figure(go.Sankey(
            node=dict(
                label=["Total Deals", "Accepted", "Rejected"]
            ),
            link=dict(
                source=[0, 0],
                target=[1, 2],
                value=[accepted, rejected]
            )
        ))
        fig1.update_layout(title='Deal Decision Flow')
        figures.append(fig1)
        
        # 2. Error Heatmap
        fig2 = px.density_heatmap(
            results_df,
            x='error_percent',
            y='sale_profit_percent',
            title='Error Distribution Heatmap',
            marginal_x="histogram",
            marginal_y="histogram"
        )
        figures.append(fig2)
        
        # 3. Profit-Loss Matrix
        fig3 = px.scatter(
            metrics_df,
            x='total_potential_profit_usd',
            y='total_potential_loss_usd',
            size='purchase_rate',
            title='Profit-Loss Matrix',
            labels={
                'total_potential_profit_usd': 'Potential Profit ($)',
                'total_potential_loss_usd': 'Potential Loss ($)',
                'purchase_rate': 'Purchase Rate'
            }
        )
        figures.append(fig3)
        
        return figures

    return metrics_df, visualize()






















import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.utils.get_profit import get_profit_percent
from scipy.stats import pearsonr, spearmanr














