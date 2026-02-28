"""
n3c_rfm_analyzer.py - RFM Analysis and Segmentation

This module provides functions for analyzing RFM score distributions
and creating traditional RFM-based customer segments.

Key Features:
- RFM score distribution analysis
- Standard RFM segment creation (Champions, Loyal, At-Risk, etc.)
- Configuration-driven visualization
- Traditional RFM framework from marketing literature
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any

from n3a_utils import setup_logger, get_project_root, load_config, get_output_paths, get_colors

logger = setup_logger(__name__)


def analyze_rfm_distribution(rfm_df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """
    Analyze and visualize RFM score distributions.
    
    Args:
        rfm_df: Customer-level RFM DataFrame
        config: Configuration dictionary
    """
    # Logger removed - redundant with "STEP 2: RFM SCORE ANALYSIS" section header
    
    # Calculate basic statistics
    try:
        print("\nRFM Score Statistics:")
        print("=" * 80)
    
        rfm_scores = rfm_df[['recency_score', 'frequency_score', 'monetary_score']]
        stats = rfm_scores.describe()
        print(stats)
    
        # Score distribution
        print("\nScore Distribution (1=Worst, 5=Best):")
        print("=" * 80)
        for score_col in ['recency_score', 'frequency_score', 'monetary_score']:
            print(f"\n{score_col.replace('_', ' ').title()}:")
            counts = rfm_df[score_col].value_counts().sort_index()
            for score, count in counts.items():
                pct = (count / len(rfm_df)) * 100
                print(f"  {score}: {count:6,} ({pct:5.1f}%)")
    
        # Correlation between scores
        print("\nRFM Score Correlations:")
        print("=" * 80)
        correlations = rfm_scores.corr()
        print(correlations)
    
        # Create visualization with config-driven colors
        output_paths = get_output_paths(config)
        colors = get_colors(config)
    
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Recency Score', 'Frequency Score', 'Monetary Score', 'RFM Score Correlation'],
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'heatmap'}]]
        )
    
        # Recency distribution
        recency_counts = rfm_df['recency_score'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=recency_counts.index, y=recency_counts.values, name='Recency',
                   marker_color=colors.get('primary', '#2E86C1')),
            row=1, col=1
        )
    
        # Frequency distribution
        frequency_counts = rfm_df['frequency_score'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=frequency_counts.index, y=frequency_counts.values, name='Frequency',
                   marker_color=colors.get('secondary', '#EA731D')),
            row=1, col=2
        )
    
        # Monetary distribution
        monetary_counts = rfm_df['monetary_score'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=monetary_counts.index, y=monetary_counts.values, name='Monetary',
                   marker_color=colors.get('success', '#27AE60')),
            row=2, col=1
        )
    
        # Correlation heatmap
        fig.add_trace(
            go.Heatmap(
                z=correlations.values,
                x=['R', 'F', 'M'],
                y=['R', 'F', 'M'],
                colorscale='RdBu',
                zmid=0,
                text=correlations.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 12}
            ),
            row=2, col=2
        )
    
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="RFM Score Distribution Analysis",
            title_x=0.5
        )
    
        # Update axes
        fig.update_xaxes(title_text="Score", row=1, col=1)
        fig.update_xaxes(title_text="Score", row=1, col=2)
        fig.update_xaxes(title_text="Score", row=2, col=1)
    
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=1)
    
        # Save figure
        output_file = output_paths['figures'] / 'rfm_score_distribution.html'
        fig.write_html(str(output_file))
        logger.info(f"Saved RFM distribution plot: {output_file}")
    except Exception as e:
        logger.error(f"analyze_rfm_distribution failed: {e}")
        raise


def create_rfm_segments(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create traditional RFM segments based on score combinations.
    
    NOTE: These segment rules are STANDARD RFM DEFINITIONS from marketing literature.
    They represent well-established business logic and are intentionally hard-coded.
    For custom business rules, use the dynamic segment naming in n3g_segment_profiler.py
    
    Segments (Standard RFM Framework):
    - Champions: R=5, F=5, M=5 (best customers)
    - Loyal Customers: High F and M (frequent, high-value)
    - Potential Loyalists: Recent customers with good F (building relationship)
    - New Customers: High R, low F (need nurturing)
    - At Risk: Low R, previously good F/M (losing them)
    - Can't Lose Them: Low R but historically high F/M (critical to recover)
    - Hibernating: Low R, F, M (dormant)
    - Lost: Lowest R, F, M (churned)
    - Others: Middle of the road (mixed characteristics)
    
    Args:
        rfm_df: Customer-level RFM DataFrame
        
    Returns:
        RFM DataFrame with 'rfm_segment' column added
    """
    # Logger removed - redundant with "STEP 2: RFM SCORE ANALYSIS" section header
    
    try:
        rfm_df = rfm_df.copy()
    
        # Create RFM string for easier segmentation
        rfm_df['rfm_string'] = (
            rfm_df['recency_score'].astype(str) +
            rfm_df['frequency_score'].astype(str) +
            rfm_df['monetary_score'].astype(str)
        )
    
        # Define segment rules (STANDARD RFM FRAMEWORK)
        def assign_segment(row):
            """Assign RFM segment label based on R/F/M scores."""
            try:
                r, f, m = row['recency_score'], row['frequency_score'], row['monetary_score']
        
                # Champions: Best customers
                if r == 5 and f == 5 and m == 5:
                    return 'Champions'
        
                # Loyal Customers: Frequent and high spenders
                elif f >= 4 and m >= 4:
                    return 'Loyal Customers'
        
                # Potential Loyalists: Recent with decent frequency
                elif r >= 4 and f >= 3:
                    return 'Potential Loyalists'
        
                # New Customers: Very recent but low frequency
                elif r == 5 and f <= 2:
                    return 'New Customers'
        
                # At Risk: Low recency, previously engaged
                elif r <= 2 and (f >= 3 or m >= 3):
                    return 'At Risk'
        
                # Can't Lose Them: Churning high-value customers
                elif r <= 2 and f >= 4 and m >= 4:
                    return "Can't Lose Them"
        
                # Hibernating: Low across all dimensions
                elif r <= 2 and f <= 2 and m <= 2:
                    return 'Hibernating'
        
                # Lost: Lowest engagement
                elif r == 1 and f == 1 and m == 1:
                    return 'Lost'
        
                # Others: Middle of the road
                else:
                    return 'Others'
            except Exception as e:
                logger.error(f"assign_segment failed: {e}")
                raise
    
        rfm_df['rfm_segment'] = rfm_df.apply(assign_segment, axis=1)
    
        # Log segment distribution
        segment_counts = rfm_df['rfm_segment'].value_counts()
    
        print("\nTraditional RFM Segments:")
        print("=" * 80)
        for segment, count in segment_counts.items():
            pct = (count / len(rfm_df)) * 100
            print(f"{segment:20s}: {count:6,} ({pct:5.1f}%)")
    
        logger.info(f"Created {len(segment_counts)} RFM segments")
    
        # Create segment profile
        segment_profile = rfm_df.groupby('rfm_segment').agg({
            'recency_days': 'mean',
            'frequency': 'mean',
            'monetary': 'mean',
            'loyalty_score': 'mean',
            'customer_id': 'count'
        }).round(2)
    
        segment_profile.columns = ['Avg Recency (days)', 'Avg Frequency', 'Avg Monetary', 'Avg Loyalty Score', 'Count']
        segment_profile = segment_profile.sort_values('Count', ascending=False)
    
        print("\nSegment Profiles:")
        print("=" * 80)
        print(segment_profile)
    
        return rfm_df
    except Exception as e:
        logger.error(f"create_rfm_segments failed: {e}")
        raise


def visualize_rfm_segments(rfm_df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """
    Create visualizations of RFM segments.
    
    Args:
        rfm_df: RFM DataFrame with 'rfm_segment' column
        config: Configuration dictionary
    """
    try:
        logger.info("Creating RFM segment visualizations")
    
        output_paths = get_output_paths(config)
        colors = get_colors(config)
    
        # Segment size pie chart
        segment_counts = rfm_df['rfm_segment'].value_counts()
    
        fig = go.Figure(data=[go.Pie(
            labels=segment_counts.index,
            values=segment_counts.values,
            hole=0.3,
            textinfo='label+percent',
            textposition='auto'
        )])
    
        fig.update_layout(
            title='Customer Distribution by RFM Segment',
            height=600
        )
    
        output_file = output_paths['figures'] / 'rfm_segment_distribution.html'
        fig.write_html(str(output_file))
        logger.info(f"Saved segment distribution: {output_file}")
    
        # RFM metrics by segment
        segment_metrics = rfm_df.groupby('rfm_segment').agg({
            'recency_days': 'mean',
            'frequency': 'mean',
            'monetary': 'mean'
        }).reset_index()
    
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=['Avg Recency (days)', 'Avg Frequency', 'Avg Monetary ($)']
        )
    
        # Recency
        fig.add_trace(
            go.Bar(x=segment_metrics['rfm_segment'], y=segment_metrics['recency_days'],
                   name='Recency', marker_color=colors.get('primary', '#2E86C1')),
            row=1, col=1
        )
    
        # Frequency
        fig.add_trace(
            go.Bar(x=segment_metrics['rfm_segment'], y=segment_metrics['frequency'],
                   name='Frequency', marker_color=colors.get('secondary', '#EA731D')),
            row=1, col=2
        )
    
        # Monetary
        fig.add_trace(
            go.Bar(x=segment_metrics['rfm_segment'], y=segment_metrics['monetary'],
                   name='Monetary', marker_color=colors.get('success', '#27AE60')),
            row=1, col=3
        )
    
        fig.update_layout(
            height=500,
            showlegend=False,
            title_text="RFM Metrics by Segment"
        )
    
        fig.update_xaxes(tickangle=45)
    
        output_file = output_paths['figures'] / 'rfm_segment_metrics.html'
        fig.write_html(str(output_file))
        logger.info(f"Saved segment metrics: {output_file}")
    except Exception as e:
        logger.error(f"visualize_rfm_segments failed: {e}")
        raise


__all__ = [
    'analyze_rfm_distribution',
    'create_rfm_segments',
    'visualize_rfm_segments',
]

if __name__ == "__main__":
    print("Testing n3c_rfm_analyzer module...")
    
    from n3b_data_loader import load_data_for_segmentation
    
    try:
        config = load_config()
        df, rfm_df = load_data_for_segmentation()
        analyze_rfm_distribution(rfm_df, config)
        rfm_df = create_rfm_segments(rfm_df)
        visualize_rfm_segments(rfm_df, config)
        print("RFM analysis test successful!")
    except Exception as e:
        print(f"Error: {e}")
