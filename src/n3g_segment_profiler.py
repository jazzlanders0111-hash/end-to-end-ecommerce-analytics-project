"""
n3g_segment_profiler.py - Customer Segment Profiling

Creates detailed profiles for customer segments based on actual characteristics.
Dynamically assigns meaningful names without hardcoded mappings.

Now includes configurable churn risk assessment based on recency thresholds.


"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple  

from n3a_utils import setup_logger, load_config

logger = setup_logger(__name__)


def calculate_churn_risk_category(
    recency_days: float,
    config: Dict[str, Any]
) -> str:
    """
    Calculate churn risk category based on recency and configurable thresholds.
    
    Uses simple recency-based heuristic as a defensible proxy for churn risk.
    This is more interpretable and business-friendly than ML-based predictions.
    
    Args:
        recency_days: Days since last purchase
        config: Configuration dictionary with churn_risk thresholds
        
    Returns:
        Risk category label ('Active', 'At Risk', 'Inactive', or 'Churned')
    """
    try:
        churn_cfg = config.get('notebook3', {}).get('churn_risk', {})
    
        # Get thresholds
        thresholds = churn_cfg.get('thresholds', {
            'active': 60,
            'at_risk': 120,
            'inactive': 180,
            'churned': 180
        })
        labels = churn_cfg.get('labels', ['Active', 'At Risk', 'Inactive', 'Churned'])
    
        # Categorize based on thresholds
        if recency_days <= thresholds['active']:
            return labels[0]  # Active
        elif recency_days <= thresholds['at_risk']:
            return labels[1]  # At Risk
        elif recency_days <= thresholds['inactive']:
            return labels[2]  # Inactive
        else:
            return labels[3]  # Churned
    except Exception as e:
        logger.error(f"calculate_churn_risk_category failed: {e}")
        raise


def calculate_segment_churn_distribution(
    cluster_data: pd.DataFrame,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate churn risk distribution for a segment.
    
    Args:
        cluster_data: DataFrame with customer data for one segment
        config: Configuration dictionary
        
    Returns:
        Dictionary with churn risk metrics
    """
    try:
        if not config.get('notebook3', {}).get('churn_risk', {}).get('enabled', True):
            return None
    
        # Calculate risk category for each customer
        cluster_data = cluster_data.copy()
        cluster_data['churn_risk'] = cluster_data['recency_days'].apply(
            lambda x: calculate_churn_risk_category(x, config)
        )
    
        # Get distribution
        risk_counts = cluster_data['churn_risk'].value_counts().to_dict()
        total = len(cluster_data)
        risk_percentages = {k: (v / total) * 100 for k, v in risk_counts.items()}
    
        # Find dominant category
        dominant_category = max(risk_counts, key=risk_counts.get)
    
        return {
            'distribution': risk_counts,
            'percentages': risk_percentages,
            'dominant_category': dominant_category,
            'churned_count': risk_counts.get('Churned', 0),
            'at_risk_count': risk_counts.get('At Risk', 0),
            'active_count': risk_counts.get('Active', 0)
        }
    except Exception as e:
        logger.error(f"calculate_segment_churn_distribution failed: {e}")
        raise


def create_segment_profiles(
    clust_df: pd.DataFrame,
    rfm_df: pd.DataFrame
) -> Dict[int, Dict[str, Any]]:
    """
    Create detailed profiles for each customer segment.
    
    Now includes churn risk metrics based on recency thresholds.

    Args:
        clust_df: DataFrame with customer_id and cluster assignments
        rfm_df: RFM DataFrame with customer metrics

    Returns:
        Dictionary mapping cluster_id to profile metrics
    """
    # Logger removed - redundant with notebook section header
    
    try:
        config = load_config()
    
        # Clean merge to avoid duplicate cluster columns
        rfm_clean = rfm_df.copy()
        if 'cluster' in rfm_clean.columns:
            rfm_clean = rfm_clean.drop(columns=['cluster'])

        # Merge cluster assignments with RFM data
        rfm_with_clusters = rfm_clean.merge(
            clust_df[['customer_id', 'cluster']],
            on='customer_id',
            how='inner'
        )

        profiles: Dict[int, Dict[str, Any]] = {}

        for cluster_id in sorted(rfm_with_clusters['cluster'].unique()):
            cluster_data = rfm_with_clusters[rfm_with_clusters['cluster'] == cluster_id]

            profile = {
                'cluster_id': cluster_id,
                'count': len(cluster_data),
                'percentage': (len(cluster_data) / len(rfm_with_clusters)) * 100,
            
                # Recency metrics
                'avg_recency': cluster_data['recency_days'].mean(),
                'median_recency': cluster_data['recency_days'].median(),
            
                # Frequency metrics
                'avg_frequency': cluster_data['frequency'].mean(),
                'median_frequency': cluster_data['frequency'].median(),
            
                # Monetary metrics
                'avg_monetary': cluster_data['monetary'].mean(),
                'median_monetary': cluster_data['monetary'].median(),
                'total_revenue': cluster_data['monetary'].sum(),
            
                # RFM scores
                'avg_recency_score': cluster_data['recency_score'].mean(),
                'avg_frequency_score': cluster_data['frequency_score'].mean(),
                'avg_monetary_score': cluster_data['monetary_score'].mean(),
            
                # Additional metrics (if available)
                'avg_loyalty': cluster_data['loyalty_score'].mean() if 'loyalty_score' in cluster_data.columns else None,
                'avg_order_value': cluster_data['avg_order_value'].mean() if 'avg_order_value' in cluster_data.columns else None,
                'avg_return_rate': cluster_data['return_rate'].mean() * 100 if 'return_rate' in cluster_data.columns else None,
                'avg_discount_usage': cluster_data['discount_usage_rate'].mean() * 100 if 'discount_usage_rate' in cluster_data.columns else None,
            }
        
            # Calculate churn risk distribution
            churn_metrics = calculate_segment_churn_distribution(cluster_data, config)
            if churn_metrics:
                profile['churn_risk'] = churn_metrics
            
                # Add convenience fields for backward compatibility and easy access
                profile['churn_rate'] = churn_metrics['percentages'].get('Churned', 0)
                profile['churned_count'] = churn_metrics['churned_count']
                profile['at_risk_count'] = churn_metrics['at_risk_count']
                profile['active_count'] = churn_metrics['active_count']
                profile['inactive_count'] = churn_metrics['distribution'].get('Inactive', 0)
                profile['dominant_risk'] = churn_metrics['dominant_category']

            profiles[cluster_id] = profile

            # Enhanced logging with churn risk
            churn_info = f", {profile.get('dominant_risk', 'N/A')} dominant" if churn_metrics else ""
            logger.info(
                f"Cluster {cluster_id}: {profile['count']:,} customers "
                f"({profile['percentage']:.1f}%), ${profile['total_revenue']:,.0f} revenue{churn_info}"
            )

        return profiles
    except Exception as e:
        logger.error(f"create_segment_profiles failed: {e}")
        raise


def assign_segment_names(profiles: Dict[int, Dict[str, Any]]) -> Dict[int, str]:
    """
    Dynamically assign meaningful names to segments based on characteristics.
    
    FIXED: Prioritizes high-value segments and eliminates numeric suffixes.
    """
    try:
        segment_names: Dict[int, str] = {}
        used_names = set()
        total_revenue = sum(p['total_revenue'] for p in profiles.values())

        # Calculate priority scores and sort
        cluster_priority = []
        for cluster_id, profile in profiles.items():
            priority = (
                profile['avg_monetary'] * 0.4 +
                profile['avg_frequency'] * 200 +
                (5 - profile['avg_recency']/100) * 100
            )
            cluster_priority.append((cluster_id, priority, profile))
        cluster_priority.sort(key=lambda x: x[1], reverse=True)

        # Assign names based on characteristics
        for cluster_id, priority, profile in cluster_priority:
            # Extract metrics
            r_score = profile['avg_recency_score']
            f_score = profile['avg_frequency_score']
            m_score = profile['avg_monetary_score']
            recency_days = profile['avg_recency']
            revenue_pct = (profile['total_revenue'] / total_revenue) * 100
        
            # Get churn risk data
            churn_risk_data = profile.get('churn_risk', {})
            dominant_risk = churn_risk_data.get('dominant_category', 'Unknown')
            churned_pct = churn_risk_data.get('percentages', {}).get('Churned', 0)
            at_risk_pct = churn_risk_data.get('percentages', {}).get('At Risk', 0)
            inactive_pct = churn_risk_data.get('percentages', {}).get('Inactive', 0)
        
            # Determine activity status
            if churn_risk_data:
                inactive = dominant_risk in ['Inactive', 'Churned'] or (inactive_pct + churned_pct) > 50
                highly_inactive = dominant_risk == 'Churned' or churned_pct > 50
            else:
                inactive = recency_days > 120
                highly_inactive = recency_days > 180

            # ============================================================
            # NAMING LOGIC (Priority Order Matters!)
            # ============================================================
        
            # 1. HIGH-VALUE AT RISK (NEW - Must come FIRST)
            if m_score >= 4.5 and (churned_pct > 25 or at_risk_pct > 35 or (inactive and m_score >= 4.8)):
                name = "High-Value at Risk"
        
            # 2. Champions: Best across all dimensions
            elif r_score >= 4.0 and f_score >= 4.0 and m_score >= 4.0 and not inactive:
                name = "Champions"
        
            # 3. Loyal Customers: Frequent and high-value
            elif f_score >= 3.5 and m_score >= 2.5 and r_score >= 3.0 and not inactive:
                name = "Loyal Customers"
        
            # 4. At-Risk: Moderate value with declining engagement
            elif m_score >= 3.0 and m_score < 4.5 and r_score <= 2.5 and inactive:
                name = "At-Risk"
        
            # 5. Potential Loyalists: Recent with building frequency
            elif r_score >= 4.0 and f_score >= 2.5 and f_score < 4.0:
                name = "Potential Loyalists"
        
            # 6. New Customers: Very recent, low frequency
            elif r_score >= 4.5 and f_score <= 2.0:
                name = "New Customers"
        
            # 7. Lost Customers: Lowest across all dimensions
            elif r_score <= 2.0 and f_score <= 2.0 and m_score <= 2.0 and highly_inactive:
                name = "Lost Customers"
        
            # 8. Hibernating: Poor recency, low engagement
            elif r_score <= 2.5 and f_score <= 2.5 and inactive:
                name = "Hibernating"
        
            # 9. About to Sleep: Declining engagement
            elif r_score <= 3.0 and r_score > 2.0 and inactive:
                name = "About to Sleep"
        
            # 10. Promising: Good potential
            elif r_score >= 3.5 and m_score >= 2.5:
                name = "Promising"
        
            # 11. Needs Engagement: Moderate metrics (RENAMED from "Need Attention")
            elif r_score >= 2.5 and f_score >= 2.0:
                name = "Needs Engagement"
        
            # 12. Default: Regular Customers
            else:
                name = "Regular Customers"

            # ============================================================
            # HANDLE DUPLICATES (Better than numeric suffixes)
            # ============================================================
            if name not in used_names:
                segment_names[cluster_id] = name
                used_names.add(name)
            else:
                # Add meaningful qualifier based on key characteristic
                if m_score >= 4.5:
                    qualifier = "Premium"
                elif f_score >= 4.0:
                    qualifier = "Frequent"
                elif r_score >= 4.0:
                    qualifier = "Recent"
                elif r_score <= 2.0:
                    qualifier = "Dormant"
                else:
                    qualifier = "Standard"
            
                final_name = f"{name} ({qualifier})"
            
                # If still duplicate, add counter
                if final_name in used_names:
                    counter = 2
                    while f"{name} ({qualifier} {counter})" in used_names:
                        counter += 1
                    final_name = f"{name} ({qualifier} {counter})"
            
                segment_names[cluster_id] = final_name
                used_names.add(final_name)

            # Log assignment
            risk_info = f", {dominant_risk}" if churn_risk_data else f", {recency_days:.0f}d recency"
            logger.info(
                f"Cluster {cluster_id}: {segment_names[cluster_id]} "
                f"(R={r_score:.1f}, F={f_score:.1f}, M={m_score:.1f}{risk_info}, Revenue={revenue_pct:.1f}%)"
            )

        return segment_names
    except Exception as e:
        logger.error(f"assign_segment_names failed: {e}")
        raise


def print_segment_summary(
    profiles: Dict[int, Dict[str, Any]],
    segment_names: Dict[int, str]
) -> None:
    """
    Print formatted segment summary with churn risk metrics.

    Args:
        profiles: Dictionary of segment profiles
        segment_names: Dictionary of segment names
    """
    try:
        print("\nCustomer Segment Summary:")
        print("=" * 80)

        # Sort by size (largest first)
        sorted_clusters = sorted(
            profiles.keys(),
            key=lambda x: profiles[x]['count'],
            reverse=True
        )

        total_customers = sum(p['count'] for p in profiles.values())
        total_revenue = sum(p['total_revenue'] for p in profiles.values())

        for cluster_id in sorted_clusters:
            profile = profiles[cluster_id]
            name = segment_names[cluster_id]
            revenue_pct = (profile['total_revenue'] / total_revenue) * 100

            print(f"\n{name} (Cluster {cluster_id})")
            print("-" * 80)
            print(f"Size: {profile['count']:,} customers ({profile['percentage']:.1f}%)")
            print(f"Revenue: ${profile['total_revenue']:,.2f} ({revenue_pct:.1f}%)")
            print(f"RFM Metrics:")
            print(f"  Recency: {profile['avg_recency']:.0f} days (median: {profile['median_recency']:.0f})")
            print(f"  Frequency: {profile['avg_frequency']:.1f} orders (median: {profile['median_frequency']:.1f})")
            print(f"  Monetary: ${profile['avg_monetary']:,.2f} (median: ${profile['median_monetary']:,.2f})")
        
            print(f"RFM Scores:")
            print(f"  Recency Score: {profile['avg_recency_score']:.2f}/5")
            print(f"  Frequency Score: {profile['avg_frequency_score']:.2f}/5")
            print(f"  Monetary Score: {profile['avg_monetary_score']:.2f}/5")
        
            if profile.get('avg_loyalty') is not None:
                print(f"  Loyalty Score: {profile['avg_loyalty']:.2f}")
        
            if profile.get('avg_order_value') is not None:
                print(f"Avg Order Value: ${profile['avg_order_value']:.2f}")
        
            # Display churn risk metrics
            churn_risk_data = profile.get('churn_risk')
            if churn_risk_data:
                print(f"Churn Risk Distribution:")
                for risk_cat, pct in sorted(churn_risk_data['percentages'].items()):
                    count = churn_risk_data['distribution'].get(risk_cat, 0)
                    print(f"  {risk_cat}: {count:,} ({pct:.1f}%)")
            
                # FIX 3: Show dominant with percentage and qualifier
                dominant = churn_risk_data['dominant_category']
                dominant_pct = churn_risk_data['percentages'][dominant]
            
                # Add qualifier if weak dominance
                if dominant_pct < 40:
                    qualifier = " (weak majority)"
                elif dominant_pct < 50:
                    qualifier = " (plurality)"
                else:
                    qualifier = ""
            
                print(f"  Dominant: {dominant} ({dominant_pct:.1f}%{qualifier})")

        print("\n" + "=" * 80)
    
        # Summary statistics
        print("\nOVERALL SUMMARY:")
        print(f"Total Customers: {total_customers:,}")
        print(f"Total Revenue: ${total_revenue:,.2f}")
        print(f"Number of Segments: {len(profiles)}")
    
        # Concentration analysis
        largest_segment = max(profiles.values(), key=lambda x: x['count'])
        print(f"\nLargest Segment: {largest_segment['count']:,} customers ({largest_segment['percentage']:.1f}%)")
    
        highest_revenue_segment = max(profiles.values(), key=lambda x: x['total_revenue'])
        highest_revenue_pct = (highest_revenue_segment['total_revenue'] / total_revenue) * 100
        print(f"Highest Revenue Segment: ${highest_revenue_segment['total_revenue']:,.2f} ({highest_revenue_pct:.1f}%)")
    
        # Aggregate churn risk across all segments
        if any(p.get('churn_risk') for p in profiles.values()):
            total_active = sum(p.get('active_count', 0) for p in profiles.values())
            total_at_risk = sum(p.get('at_risk_count', 0) for p in profiles.values())
            total_churned = sum(p.get('churned_count', 0) for p in profiles.values())
        
            # Calculate Inactive category (was missing in original implementation)
            total_inactive = sum(
                p.get('churn_risk', {}).get('distribution', {}).get('Inactive', 0) 
                for p in profiles.values()
            )
        
            print(f"\nChurn Risk Summary (All Customers):")
            print(f"  Active: {total_active:,} ({total_active/total_customers*100:.1f}%)")
            print(f"  At Risk: {total_at_risk:,} ({total_at_risk/total_customers*100:.1f}%)")
            print(f"  Inactive: {total_inactive:,} ({total_inactive/total_customers*100:.1f}%)")
            print(f"  Churned: {total_churned:,} ({total_churned/total_customers*100:.1f}%)")
    
        print("=" * 80)
    except Exception as e:
        logger.error(f"print_segment_summary failed: {e}")
        raise


__all__ = [
    'calculate_churn_risk_category',
    'calculate_segment_churn_distribution',
    'create_segment_profiles',
    'assign_segment_names',
    'print_segment_summary',
]

if __name__ == "__main__":
    print("Testing n3g_segment_profiler module...")
    print("Module loaded successfully")