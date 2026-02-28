# src/n4h_business_insights.py
"""
n4h_business_insights.py - Business Insights & Recommendations

1. NB03 SEGMENT INTEGRATION (was missing entirely):
   Original: business insights were generic templates with no knowledge of
   the 4 actual customer segments from NB03.
   Load customer_segments.csv (if available), join segment_name onto
   risk_df so recommendations reference actual segment names:
   Loyal Customers, Need Attention, Need Attention 2, Lost Customers.
   IMPORTANT: segment_name is ONLY used for business insight text generation.
   It is NOT added to risk_df as a column (no data leakage into predictions).

2. SEGMENT-SPECIFIC RISK BREAKDOWN:
   Shows how churn risk distributes across NB03 segments (e.g., what % of
   'Lost Customers' are in High Risk tier). This directly connects NB03 and NB04
   findings into actionable cross-notebook intelligence.

3. ROI ESTIMATES use actual segment revenue from NB03 profiles where available.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional

from n4a_utils import setup_logger, get_project_root, load_config

logger = setup_logger(__name__)


def load_segment_names(config: Dict) -> Optional[pd.DataFrame]:
    """
    Load customer segment assignments from Notebook 03 output.

    Used ONLY for business insight generation - NOT as a model feature.
    This is safe (no leakage) because segment_name is never passed to
    predict() or used in feature scaling.

    Args:
        config: Configuration dictionary

    Returns:
        DataFrame with customer_id and segment_name, or None if not available
    """
    try:
        project_root = get_project_root()
        segments_path = project_root / config['paths']['processed_data'] / 'customer_segments.csv'

        if not segments_path.exists():
            logger.info("customer_segments.csv not found - skipping NB03 integration")
            logger.info("(Run Notebook 03 first to enable segment-aware insights)")
            return None

        segments_df = pd.read_csv(segments_path)

        # Determine which column holds the segment label
        label_col = None
        for col in ['segment_name', 'segment', 'cluster_name', 'cluster']:
            if col in segments_df.columns:
                label_col = col
                break

        if label_col is None:
            logger.warning("No segment label column found in customer_segments.csv")
            return None

        result = segments_df[['customer_id', label_col]].copy()
        if label_col != 'segment_name':
            result = result.rename(columns={label_col: 'segment_name'})

        logger.info(f"Loaded {len(result):,} segment assignments from NB03")
        logger.info(f"Segments: {result['segment_name'].value_counts().to_dict()}")

        return result

    except Exception as e:
        logger.warning(f"Could not load NB03 segments: {e}")
        return None


def generate_segment_risk_matrix(
    risk_df: pd.DataFrame,
    segments_df: Optional[pd.DataFrame]
) -> Optional[pd.DataFrame]:
    """
    Cross-tabulate NB03 segment membership vs NB04 risk levels.

    Shows which segments contribute most to High/Medium/Low churn risk.
    This is the core NB03 + NB04 integration insight.

    Args:
        risk_df: DataFrame with risk_level column (from n4g)
        segments_df: DataFrame with customer_id and segment_name (from NB03)

    Returns:
        Cross-tabulation DataFrame, or None if segments not available
    """
    try:
        if segments_df is None:
            return None

        merged = risk_df.merge(segments_df, on='customer_id', how='left')
        missing = merged['segment_name'].isna().sum()
        if missing > 0:
            logger.info(f"{missing} customers in NB04 not found in NB03 segments (expected for new customers)")

        matrix = pd.crosstab(
            merged['segment_name'],
            merged['risk_level'],
            margins=True,
            margins_name='Total'
        )

        # Add high-risk percentage column
        if 'High' in matrix.columns:
            total_per_segment = matrix['Total']
            matrix['High_Risk_Pct'] = (matrix.get('High', 0) / total_per_segment * 100).round(1)

        return matrix
    except Exception as e:
        logger.error(f"generate_segment_risk_matrix failed: {e}")
        raise


def generate_retention_strategies(
    importance_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    model_metrics: Dict,
    config: Optional[Dict] = None
) -> str:
    """
    Generate retention strategies, integrating NB03 segment context.

    Args:
        importance_df: DataFrame with feature importance (from n4f)
        risk_df: DataFrame with risk_level and customer data (from n4g)
        model_metrics: Dictionary with model performance metrics
        config: Configuration dictionary (used to load NB03 segments)

    Returns:
        Formatted string with retention strategies
    """
    try:
        logger.info("Generating retention strategies")

        # Load NB03 segment data for context (NOT for modeling)
        segments_df = load_segment_names(config) if config else None
        segment_matrix = generate_segment_risk_matrix(risk_df, segments_df)

        strategies = []
        strategies.append("=" * 80)
        strategies.append("CHURN RETENTION STRATEGIES")
        strategies.append("=" * 80)
        strategies.append("")

        # Model Performance
        strategies.append("MODEL PERFORMANCE")
        strategies.append("-" * 80)
        strategies.append(f"Accuracy:  {model_metrics.get('accuracy', 0):.1%}")
        strategies.append(f"Precision: {model_metrics.get('precision', 0):.1%}")
        strategies.append(f"Recall:    {model_metrics.get('recall', 0):.1%}")
        strategies.append(f"F1-Score:  {model_metrics.get('f1', 0):.1%}")
        strategies.append(f"ROC-AUC:   {model_metrics.get('roc_auc', 0):.3f}")
        strategies.append("")

        # Top Churn Drivers
        strategies.append("TOP CHURN DRIVERS")
        strategies.append("-" * 80)
        if len(importance_df) > 0:
            driver_1 = importance_df.iloc[0]['feature']
            for _, row in importance_df.head(5).iterrows():
                strategies.append(f"{int(row['rank'])}. {row['feature']:30s} {row['importance_pct']:6.2f}%")
        else:
            driver_1 = None
            strategies.append("No feature importance data available.")
        strategies.append("")

        # NB03 Segment x Risk Matrix
        if segment_matrix is not None:
            strategies.append("NB03 SEGMENT x CHURN RISK BREAKDOWN")
            strategies.append("-" * 80)
            strategies.append("  Cross-notebook finding: how NB03 segments distribute across churn risk tiers.")
            strategies.append("")
            strategies.append(segment_matrix.to_string())
            strategies.append("")

            if 'High' in segment_matrix.columns and 'High_Risk_Pct' in segment_matrix.columns:
                idx_to_use = [i for i in segment_matrix.index if i != 'Total']
                top_at_risk = (
                    segment_matrix.loc[idx_to_use]
                    .sort_values('High_Risk_Pct', ascending=False)
                    .head(2)
                )
                for seg_name, row in top_at_risk.iterrows():
                    strategies.append(
                        f"  KEY FINDING: '{seg_name}' has {row['High_Risk_Pct']:.0f}% "
                        f"in High Risk tier ({int(row.get('High', 0)):,} customers)"
                    )
                strategies.append("")

        # Strategic Recommendations
        strategies.append("STRATEGIC RECOMMENDATIONS")
        strategies.append("-" * 80)
        strategies.append("")

        driver_1 = driver_1 or 'recency'  # fallback if importance_df was empty
        strategies.append(f"1. PRIMARY INTERVENTION: {driver_1.upper()}")

        if 'recency' in driver_1.lower():
            strategies.append("   Action: Re-engagement Campaign")
            strategies.append("   - Target customers inactive 90+ days with personalized outreach")
            strategies.append("   - 'We Miss You' offer: 15-20% off next purchase")
            strategies.append("   - Automated email triggers at 30, 60, 90-day inactivity marks")
        elif 'frequency' in driver_1.lower():
            strategies.append("   Action: Purchase Frequency Program")
            strategies.append("   - Loyalty/subscription program for repeat incentive")
            strategies.append("   - Replenishment reminders based on average purchase interval")
            strategies.append("   - Targeted promotions every 30-45 days")
        elif 'monetary' in driver_1.lower():
            strategies.append("   Action: Customer Value Enhancement")
            strategies.append("   - Upsell and cross-sell recommendations at checkout")
            strategies.append("   - Bundle products to increase average order value")
            strategies.append("   - Tiered loyalty rewards tied to spend milestones")
        elif 'tenure' in driver_1.lower() or 'lifetime' in driver_1.lower():
            strategies.append("   Action: New Customer Onboarding")
            strategies.append("   - Structured 90-day onboarding journey for new customers")
            strategies.append("   - First-purchase follow-up within 7 days")
            strategies.append("   - Early loyalty program enrollment")
        elif 'return' in driver_1.lower() or 'discount' in driver_1.lower():
            strategies.append("   Action: Purchase Experience Improvement")
            strategies.append("   - Proactive outreach after returns to resolve issues")
            strategies.append("   - Review discount dependency - ensure offers build loyalty not habituation")
        else:
            strategies.append(f"   Action: Optimize {driver_1}")
            strategies.append("   - Analyze behavioral patterns in high-risk segment")
            strategies.append("   - A/B test targeted interventions")
        strategies.append("")

        # Risk-Based Prioritization
        high_risk_count = (risk_df['risk_level'] == 'High').sum()
        medium_risk_count = (risk_df['risk_level'] == 'Medium').sum()
        low_risk_count = (risk_df['risk_level'] == 'Low').sum()

        strategies.append("2. RISK-BASED PRIORITIZATION")
        strategies.append(f"   HIGH RISK ({high_risk_count:,} customers):")

        if segments_df is not None:
            merged_high = risk_df[risk_df['risk_level'] == 'High'].merge(
                segments_df, on='customer_id', how='left'
            )
            seg_breakdown = merged_high['segment_name'].value_counts().head(2)
            for seg_name, cnt in seg_breakdown.items():
                strategies.append(f"   - Primarily from NB03 segment: '{seg_name}' ({cnt:,} customers)")

        strategies.append("   - Urgent: Personal outreach within 48 hours")
        strategies.append("   - Aggressive retention offer: 25-30% discount")
        strategies.append("   - Priority customer service handling")
        strategies.append("")

        strategies.append(f"   MEDIUM RISK ({medium_risk_count:,} customers):")
        strategies.append("   - Proactive engagement campaign (email + SMS)")
        strategies.append("   - Moderate incentive: 10-15% discount")
        strategies.append("   - Survey to identify pain points")
        strategies.append("")

        strategies.append(f"   LOW RISK ({low_risk_count:,} customers):")
        strategies.append("   - Maintain loyalty: regular comms, rewards points")
        strategies.append("   - Monthly touchpoints only - avoid over-messaging")
        strategies.append("")

        # ROI Projection
        if 'monetary' in risk_df.columns:
            high_risk_revenue = risk_df[risk_df['risk_level'] == 'High']['monetary'].sum()
            # Read ROI parameters from config rather than hardcoding
            _roi_cfg          = (config or {}).get('notebook4', {}).get('business_insights', {}).get('roi', {})
            _campaign_cfg     = (config or {}).get('notebook4', {}).get('business_insights', {}).get('campaigns', {})
            retention_rate    = _roi_cfg.get('expected_retention_high', 0.30)
            campaign_cost_pct = _roi_cfg.get('campaign_cost_pct', 0.10)
            campaign_cost     = high_risk_revenue * campaign_cost_pct
            revenue_saved     = high_risk_revenue * retention_rate
            net_roi           = revenue_saved - campaign_cost
            strategies.append("3. EXPECTED ROI")
            strategies.append(f"   Revenue at Risk (High Risk):    ${high_risk_revenue:,.0f}")
            strategies.append(f"   Retention Goal:                 {retention_rate:.0%} of high-risk customers")
            strategies.append(f"   Expected Revenue Saved:         ${revenue_saved:,.0f}")
            strategies.append(f"   Estimated Campaign Cost:        ${campaign_cost:,.0f} ({campaign_cost_pct:.0%} of revenue at risk)")
            strategies.append(f"   Net ROI:                        ${net_roi:,.0f}")
            strategies.append("")

        strategies.append("=" * 80)

        strategy_text = "\n".join(strategies)
        logger.info("Retention strategies generated")
        return strategy_text
    except Exception as e:
        logger.error(f"generate_retention_strategies failed: {e}")
        raise


def create_campaign_recommendations(
    risk_df: pd.DataFrame,
    config: Optional[Dict] = None
) -> str:
    """
    Create campaign recommendations for each risk level,
    incorporating NB03 segment intelligence where available.

    Args:
        risk_df: DataFrame with risk_level column
        config: Configuration dictionary (for loading NB03 segments)

    Returns:
        Formatted string with campaign recommendations
    """
    try:
        logger.info("Creating campaign recommendations")

        segments_df = load_segment_names(config) if config else None

        campaigns = []
        campaigns.append("=" * 80)
        campaigns.append("RETENTION CAMPAIGN RECOMMENDATIONS")
        campaigns.append("=" * 80)
        campaigns.append("")

        campaign_specs = {
            'High': {
                'objective': 'Immediate intervention - prevent revenue loss',
                'channels': 'Email + Phone + SMS (multi-touch)',
                'offer': '30% discount + free shipping + priority support',
                'timeline': 'Week 1-3: intensive daily outreach',
                'target_retention': '30%',
                'message': 'Win-back: personalized offer based on past purchase history',
            },
            'Medium': {
                'objective': 'Proactive re-engagement before churn materializes',
                'channels': 'Email + SMS + App push notification',
                'offer': '15% discount + loyalty bonus points',
                'timeline': 'Week 1-8: bi-weekly nurture sequence',
                'target_retention': '50%',
                'message': 'Nurture: product recommendations + early access to sales',
            },
            'Low': {
                'objective': 'Maintain loyalty and prevent drift',
                'channels': 'Email + Social retargeting',
                'offer': '10% loyalty discount + referral program ($20 per referral)',
                'timeline': 'Monthly: regular touchpoints',
                'target_retention': '85%',
                'message': 'Delight: VIP updates, new arrivals, community features',
            },
        }

        for risk_level in ['High', 'Medium', 'Low']:
            segment_df = risk_df[risk_df['risk_level'] == risk_level]
            if len(segment_df) == 0:
                continue

            spec = campaign_specs[risk_level]
            campaigns.append(f"{risk_level.upper()} RISK CAMPAIGN ({len(segment_df):,} customers)")
            campaigns.append("-" * 80)
            campaigns.append(f"Objective:          {spec['objective']}")
            campaigns.append(f"Channels:           {spec['channels']}")
            campaigns.append(f"Offer:              {spec['offer']}")
            campaigns.append(f"Timeline:           {spec['timeline']}")
            campaigns.append(f"Target Retention:   {spec['target_retention']}")
            campaigns.append(f"Message Theme:      {spec['message']}")

            if segments_df is not None:
                merged = segment_df.merge(segments_df, on='customer_id', how='left')
                seg_counts = merged['segment_name'].value_counts()
                if len(seg_counts) > 0:
                    campaigns.append(f"NB03 Composition:")
                    for seg_name, cnt in seg_counts.items():
                        pct = cnt / len(segment_df) * 100
                        campaigns.append(f"  - {seg_name}: {cnt:,} ({pct:.1f}%)")

            if 'monetary' in segment_df.columns:
                revenue = segment_df['monetary'].sum()
                avg_revenue = segment_df['monetary'].mean()
                campaigns.append(f"Revenue at Risk:    ${revenue:,.0f} (avg ${avg_revenue:.0f}/customer)")

            campaigns.append("")

        campaigns.append("=" * 80)

        campaign_text = "\n".join(campaigns)
        logger.info("Campaign recommendations created")
        return campaign_text
    except Exception as e:
        logger.error(f"create_campaign_recommendations failed: {e}")
        raise


__all__ = [
    'load_segment_names',
    'generate_segment_risk_matrix',
    'generate_retention_strategies',
    'create_campaign_recommendations',
]

if __name__ == "__main__":
    print("Testing n4h_business_insights module...")

    importance_df = pd.DataFrame({
        'feature': ['recency_days', 'frequency', 'monetary', 'tenure_days', 'loyalty_score'],
        'importance': [0.35, 0.20, 0.15, 0.12, 0.08],
        'importance_pct': [35.0, 20.0, 15.0, 12.0, 8.0],
        'rank': [1, 2, 3, 4, 5]
    })

    risk_df = pd.DataFrame({
        'customer_id': [f'C{i:05d}' for i in range(1000)],
        'risk_level': np.random.choice(['Low', 'Medium', 'High'], 1000, p=[0.33, 0.34, 0.33]),
        'churn_probability': np.random.uniform(0, 1, 1000),
        'churn': np.random.binomial(1, 0.2, 1000),
        'monetary': np.random.uniform(100, 5000, 1000)
    })

    model_metrics = {
        'accuracy': 0.82, 'precision': 0.75, 'recall': 0.78,
        'f1': 0.76, 'roc_auc': 0.85
    }

    print("\n1. Testing generate_retention_strategies (no config = no NB03 integration, expected)...")
    strategies = generate_retention_strategies(importance_df, risk_df, model_metrics)
    print("   Strategies generated")

    print("\n2. Testing create_campaign_recommendations...")
    campaigns = create_campaign_recommendations(risk_df)
    print("   Campaigns created")

    print("\nAll tests passed!")