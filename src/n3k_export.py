"""
n3k_export.py - Export Segment Data

Exports segment assignments and profiles for downstream CRM and marketing use.

UPDATED: Now includes churn_risk categories in all export formats for better
customer targeting and campaign optimization.
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from n3a_utils import setup_logger, get_project_root, load_config

logger = setup_logger(__name__)


def _convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(item) for item in obj]
    else:
        return obj


def export_segment_data(
    rfm_df: pd.DataFrame,
    segment_profiles: Dict[int, Dict[str, Any]],
    segment_names: Dict[int, str],
    recommendations: Dict[int, List[str]]
) -> Dict[str, Path]:
    """
    Export segment data in multiple formats.

    Exports customer assignments, segment profiles, and recommendations
    for use in CRM systems and marketing campaigns.

    Args:
        rfm_df: RFM DataFrame with cluster assignments
        segment_profiles: Dictionary of segment profiles
        segment_names: Dictionary of segment names
        recommendations: Dictionary of marketing recommendations

    Returns:
        Dictionary of exported file paths
    """
    # Logger removed - redundant with "STEP 11: EXPORT RESULTS" section header

    try:
        config = load_config()
        project_root = get_project_root()
        output_dir = project_root / config['paths']['processed_data']
        output_dir.mkdir(parents=True, exist_ok=True)

        export_paths = {}

        # 1. Customer segment assignments (CSV)
        logger.info("Exporting customer segment assignments")
        customer_segments = rfm_df[['customer_id', 'cluster']].copy()
        customer_segments['segment_name'] = customer_segments['cluster'].map(segment_names)
        customer_segments['loyalty_score'] = rfm_df['loyalty_score']
    
        # Add churn_risk if available
        config = load_config()
        churn_cfg = config.get('notebook3', {}).get('churn_risk', {})
        if churn_cfg.get('enabled', True):
            # Import the calculation function
            from n3g_segment_profiler import calculate_churn_risk_category
        
            customer_segments['churn_risk'] = rfm_df['recency_days'].apply(
                lambda x: calculate_churn_risk_category(x, config)
            )
            logger.info("Added churn_risk column based on recency thresholds")
    
        # Add legacy churn column only if it exists in the data (for backward compatibility)
        if 'churn' in rfm_df.columns:
            customer_segments['churn'] = rfm_df['churn']
            logger.info("Added legacy churn column from RFM data")

        csv_path = output_dir / 'customer_segments.csv'
        customer_segments.to_csv(csv_path, index=False)
        export_paths['customer_segments_csv'] = csv_path
        logger.info(f"Saved: {csv_path.name} ({len(customer_segments):,} records)")

        # 2. Segment profiles (JSON)
        logger.info("Exporting segment profiles")

        export_profiles = {}
        for cluster_id, profile in segment_profiles.items():
            name = segment_names[cluster_id]

            profile_export = {
                'cluster_id': int(cluster_id),
                'name': name,
                'size': int(profile['count']),
                'percentage': float(round(profile['percentage'], 2)),
                'metrics': {
                    'avg_recency_days': float(round(profile['avg_recency'], 1)),
                    'avg_frequency': float(round(profile['avg_frequency'], 2)),
                    'avg_monetary': float(round(profile['avg_monetary'], 2)),
                    'total_revenue': float(round(profile['total_revenue'], 2)),
                    'avg_loyalty_score': float(round(profile['avg_loyalty'], 2)) if profile['avg_loyalty'] is not None else None
                },
                'rfm_scores': {
                    'recency_score': float(round(profile['avg_recency_score'], 2)),
                    'frequency_score': float(round(profile['avg_frequency_score'], 2)),
                    'monetary_score': float(round(profile['avg_monetary_score'], 2))
                }
            }
        
            # Add churn_risk metrics if available
            churn_risk_data = profile.get('churn_risk')
            if churn_risk_data:
                profile_export['churn_risk'] = {
                    'dominant_category': churn_risk_data['dominant_category'],
                    'distribution': {k: int(v) for k, v in churn_risk_data['distribution'].items()},
                    'percentages': {k: float(round(v, 1)) for k, v in churn_risk_data['percentages'].items()},
                    'active_count': int(churn_risk_data['active_count']),
                    'at_risk_count': int(churn_risk_data['at_risk_count']),
                    'churned_count': int(churn_risk_data['churned_count'])
                }
        
            export_profiles[name] = profile_export

        export_profiles = _convert_to_serializable(export_profiles)

        json_path = output_dir / 'segment_profiles.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(export_profiles, f, indent=2, ensure_ascii=False)
        export_paths['segment_profiles_json'] = json_path
        logger.info(f"Saved: {json_path.name} ({len(export_profiles)} segments)")

        # 3. Marketing recommendations (TXT)
        logger.info("Exporting marketing recommendations")

        txt_path = output_dir / 'marketing_recommendations.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("CUSTOMER SEGMENT MARKETING RECOMMENDATIONS\n")
            f.write("=" * 80 + "\n\n")

            for cluster_id in sorted(segment_profiles.keys()):
                name = segment_names[cluster_id]
                profile = segment_profiles[cluster_id]
                recs = recommendations[cluster_id]

                f.write(f"\n{name} (Cluster {cluster_id})\n")
                f.write("-" * 80 + "\n\n")

                f.write("PROFILE:\n")
                f.write(f"Size: {profile['count']:,} customers ({profile['percentage']:.1f}%)\n")
                f.write(f"Revenue: ${profile['total_revenue']:,.2f}\n")
                f.write(f"Avg Recency: {profile['avg_recency']:.0f} days\n")
                f.write(f"Avg Frequency: {profile['avg_frequency']:.1f} orders\n")
                f.write(f"Avg Monetary: ${profile['avg_monetary']:,.2f}\n")

                f.write("\nRECOMMENDATIONS:\n")
                for i, rec in enumerate(recs, 1):
                    f.write(f"{i}. {rec}\n")
                f.write("\n")

            f.write("=" * 80 + "\n")

        export_paths['marketing_recommendations_txt'] = txt_path
        logger.info(f"Saved: {txt_path.name}")

        # 4. Segment summary (CSV)
        logger.info("Exporting segment summary")

        summary_data = []
        for cluster_id in sorted(segment_profiles.keys()):
            name = segment_names[cluster_id]
            profile = segment_profiles[cluster_id]
        
            summary_row = {
                'Segment Name': name,
                'Cluster ID': cluster_id,
                'Customer Count': profile['count'],
                'Percentage': f"{profile['percentage']:.1f}%",
                'Total Revenue': f"${profile['total_revenue']:,.2f}",
                'Avg Recency (days)': f"{profile['avg_recency']:.0f}",
                'Avg Frequency': f"{profile['avg_frequency']:.1f}",
                'Avg Monetary': f"${profile['avg_monetary']:,.2f}",
                'Loyalty Score': f"{profile['avg_loyalty']:.2f}" if profile['avg_loyalty'] else "N/A",
            }
        
            # Add churn_risk metrics if available
            churn_risk_data = profile.get('churn_risk')
            if churn_risk_data:
                summary_row['Dominant Risk'] = churn_risk_data['dominant_category']
                summary_row['Active Count'] = churn_risk_data['active_count']
                summary_row['At Risk Count'] = churn_risk_data['at_risk_count']
                summary_row['Inactive Count'] = churn_risk_data['distribution'].get('Inactive', 0)
                summary_row['Churned Count'] = churn_risk_data['churned_count']
                summary_row['Churn Rate'] = f"{churn_risk_data['percentages'].get('Churned', 0):.1f}%"
            else:
                # Legacy fallback
                summary_row['Churn Rate'] = f"{profile.get('churn_rate', 0):.1f}%" if profile.get('churn_rate') is not None else "N/A"
                summary_row['Churned Count'] = profile.get('churned_count', 0) if profile.get('churned_count') is not None else "N/A"
                summary_row['Active Count'] = profile.get('active_count', 0) if profile.get('active_count') is not None else "N/A"
        
            summary_data.append(summary_row)

        summary_df = pd.DataFrame(summary_data)
        summary_path = output_dir / 'segment_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        export_paths['segment_summary_csv'] = summary_path
        logger.info(f"Saved: {summary_path.name}")

        logger.info(f"Exported {len(export_paths)} files to: {output_dir}")

        return export_paths
    except Exception as e:
        logger.error(f"export_segment_data failed: {e}")
        raise


__all__ = [
    'export_segment_data',
]

if __name__ == "__main__":
    print("Testing n3k_export module...")
    print("[OK] Module loaded successfully")
