"""
n3i_insights.py - Business Insights and Recommendations

Generates actionable insights and marketing recommendations for segments
based on actual characteristics. Shows only the most critical insights.

UPDATED: Now uses churn_risk categories instead of recency proxy for better
business interpretation and actionable recommendations.

Key Features:
- Priority-filtered insights (top 3 per segment)
- Dynamic recommendations based on churn risk and RFM metrics
- Revenue impact assessment
- Churn risk-based targeting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any

from n3a_utils import setup_logger, load_config

logger = setup_logger(__name__)
_config = load_config()


def generate_segment_insights(
    profiles: Dict[int, Dict[str, Any]],
    segment_names: Dict[int, str],
    max_insights: int = 3
) -> Dict[int, Dict[str, List[str]]]:
    """
    Generate segment insights based on actual metrics including churn risk.
    Shows only critical insights for each segment (top N by priority).
    
    NOTE: Now uses churn_risk categories for more actionable insights.

    Args:
        profiles: Dictionary of segment profiles
        segment_names: Dictionary of segment names
        max_insights: Maximum insights to show per segment (default: 3)

    Returns:
        Dictionary with characteristics and insights per segment
    """
    # Logger removed - function purpose clear from notebook
    
    try:
        insights = {}

        total_revenue = sum(p['total_revenue'] for p in profiles.values())
        avg_recency = np.mean([p['avg_recency'] for p in profiles.values()])
        avg_frequency = np.mean([p['avg_frequency'] for p in profiles.values()])
        avg_monetary = np.mean([p['avg_monetary'] for p in profiles.values()])

        for cluster_id, profile in profiles.items():
            name = segment_names[cluster_id]

            characteristics = []
            strategic_insights = []
            insight_priorities = []

            # Extract metrics
            recency_days = profile['avg_recency']
            frequency = profile['avg_frequency']
            monetary = profile['avg_monetary']
            r_score = profile['avg_recency_score']
            f_score = profile['avg_frequency_score']
            m_score = profile['avg_monetary_score']
            size_pct = profile['percentage']
            revenue_pct = (profile['total_revenue'] / total_revenue) * 100
            count = profile['count']
            overall_score = (r_score + f_score + m_score) / 3
        
            # Get churn risk metrics
            churn_risk_data = profile.get('churn_risk', {})
            dominant_risk = churn_risk_data.get('dominant_category', 'Unknown')
            churned_pct = churn_risk_data.get('percentages', {}).get('Churned', 0)
            at_risk_pct = churn_risk_data.get('percentages', {}).get('At Risk', 0)
            inactive_pct = churn_risk_data.get('percentages', {}).get('Inactive', 0)
            active_pct = churn_risk_data.get('percentages', {}).get('Active', 0)
        
            # Fallback to recency-based logic if churn_risk disabled
            if not churn_risk_data:
                inactive = recency_days > 120
                highly_inactive = recency_days > 180
                at_risk = recency_days > 90 and recency_days <= 180
            else:
                # Use churn_risk data
                inactive = dominant_risk in ['Inactive', 'Churned']
                highly_inactive = dominant_risk == 'Churned' or churned_pct > 50
                at_risk = dominant_risk == 'At Risk' or at_risk_pct > 40

            # Key characteristics (always show top 3)
            characteristics.append(f"Size: {size_pct:.1f}% ({count:,} customers)")
            characteristics.append(f"Revenue: {revenue_pct:.1f}% (${profile['total_revenue']:,.0f})")
        
            if churn_risk_data:
                characteristics.append(f"Churn Risk: {dominant_risk} ({churned_pct + at_risk_pct:.0f}% at-risk/churned)")
            else:
                recency_label = "Very recent" if recency_days < 60 else "Recent" if recency_days < 120 else "Inactive"
                characteristics.append(f"{recency_label}: {recency_days:.0f} days since last purchase")

            # Strategic insights with priority scoring
            # CRITICAL: High-value customers at churn risk (Priority 10)
            if m_score >= 3.5 and (highly_inactive or churned_pct > 30):
                strategic_insights.append("CRITICAL: High-value customers at risk of loss")
                insight_priorities.append(10)
                strategic_insights.append(f"Immediate action needed: ${profile['total_revenue']:,.0f} in at-risk revenue")
                insight_priorities.append(9)

            # CRITICAL: Revenue exposure (Priority 9)
            elif at_risk and revenue_pct > 15:
                strategic_insights.append(f"RISK: {revenue_pct:.1f}% of revenue showing signs of disengagement")
                insight_priorities.append(9)

            # HIGH PRIORITY: Top customers to retain (Priority 8)
            if overall_score >= 4.0 and active_pct > 60:
                strategic_insights.append("Top priority: Your best customers - maximize retention")
                insight_priorities.append(8)

            # HIGH PRIORITY: New customer onboarding (Priority 7)
            elif r_score >= 4.5 and f_score <= 2.0:
                strategic_insights.append("New customer onboarding critical period")
                insight_priorities.append(7)

            # MEDIUM: Reliable foundation (Priority 6)
            elif r_score >= 4.0 and f_score >= 3.0 and active_pct > 50:
                strategic_insights.append("Reliable foundation: Strong engagement and retention")
                insight_priorities.append(6)

            # MEDIUM: Recovery opportunity (Priority 5)
            elif highly_inactive and m_score >= 3.0:
                strategic_insights.append("Difficult but worthwhile recovery opportunity")
                insight_priorities.append(5)

            # LOW: Frequency opportunity (Priority 4)
            if f_score < 3.0 and m_score >= 3.0 and len(strategic_insights) < max_insights + 1:
                strategic_insights.append("Opportunity: Increase purchase frequency for high spenders")
                insight_priorities.append(4)

            # LOW: Low recovery ROI (Priority 3)
            elif highly_inactive and m_score < 3.0 and len(strategic_insights) < max_insights + 1:
                strategic_insights.append("Low recovery ROI - minimize expensive spend")
                insight_priorities.append(3)

            # Sort by priority and take top N
            if len(strategic_insights) > max_insights:
                sorted_insights = sorted(
                    zip(strategic_insights, insight_priorities),
                    key=lambda x: x[1],
                    reverse=True
                )
                strategic_insights = [insight for insight, _ in sorted_insights[:max_insights]]

            insights[cluster_id] = {
                'characteristics': characteristics[:3],  # Top 3 characteristics
                'insights': strategic_insights[:max_insights]  # Top N insights
            }

        logger.info(f"Generated insights for {len(insights)} segments (max {max_insights} per segment)")
        return insights
    except Exception as e:
        logger.error(f"generate_segment_insights failed: {e}")
        raise


def create_marketing_recommendations(
    profiles: Dict[int, Dict[str, Any]],
    segment_names: Dict[int, str],
    max_recommendations: int = 5
) -> Dict[int, List[str]]:
    """
    Create segment-specific marketing recommendations based on churn risk.
    Generates tactics based on segment characteristics and risk categories.
    
    NOTE: Now uses churn_risk metrics for more targeted recommendations.

    Args:
        profiles: Dictionary of segment profiles
        segment_names: Dictionary of segment names
        max_recommendations: Maximum recommendations per segment (default: 5)

    Returns:
        Dictionary with recommendations per segment
    """
    # Logger removed - redundant with section header
    
    try:
        recommendations = {}

        for cluster_id, profile in profiles.items():
            name = segment_names[cluster_id]
            recs = []
            priorities = []

            r_score = profile['avg_recency_score']
            f_score = profile['avg_frequency_score']
            m_score = profile['avg_monetary_score']
            recency_days = profile['avg_recency']
            overall_score = (r_score + f_score + m_score) / 3
        
            # Get churn risk metrics
            churn_risk_data = profile.get('churn_risk', {})
            dominant_risk = churn_risk_data.get('dominant_category', 'Unknown')
            churned_pct = churn_risk_data.get('percentages', {}).get('Churned', 0)
            at_risk_pct = churn_risk_data.get('percentages', {}).get('At Risk', 0)
            inactive_pct = churn_risk_data.get('percentages', {}).get('Inactive', 0)
            active_pct = churn_risk_data.get('percentages', {}).get('Active', 0)
        
            # Fallback to recency-based logic if churn_risk disabled
            if not churn_risk_data:
                inactive = recency_days > 120
                highly_inactive = recency_days > 180
            else:
                inactive = dominant_risk in ['Inactive', 'Churned'] or inactive_pct + churned_pct > 50
                highly_inactive = dominant_risk == 'Churned' or churned_pct > 50

            # Champions / Top customers (Priority 10)
            if overall_score >= 4.0 and active_pct > 60:
                recs.append("Implement VIP program with exclusive benefits")
                priorities.append(10)
                recs.append("Personal account management or priority support")
                priorities.append(9)
                recs.append("Referral incentive program - leverage as brand advocates")
                priorities.append(8)

            # At-risk high-value customers (Priority 10)
            elif m_score >= 3.5 and (highly_inactive or churned_pct > 30):
                recs.append("Urgent: Executive-level outreach and relationship repair")
                priorities.append(10)
                recs.append("Exclusive 20-25% discount with premium gift")
                priorities.append(9)
                recs.append("Time-sensitive offer window before customer is lost")
                priorities.append(8)

            # Loyal customers (Priority 8)
            elif f_score >= 3.5 and m_score >= 2.5 and r_score >= 3.0:
                recs.append("Tiered loyalty program encouraging increased engagement")
                priorities.append(8)
                recs.append("Product bundle offers to increase order value")
                priorities.append(7)
                recs.append("Milestone and anniversary rewards")
                priorities.append(6)

            # New customers (Priority 7)
            elif r_score >= 4.5 and f_score <= 2.0:
                recs.append("Welcome series with brand story and value proposition")
                priorities.append(7)
                recs.append("First repeat purchase incentive (time-limited)")
                priorities.append(6)
                recs.append("Educational content about product usage")
                priorities.append(5)

            # High-value inactive/churned customers (Priority 9)
            elif (highly_inactive or churned_pct > 40) and m_score >= 3.0:
                recs.append("Targeted win-back campaign with 30-40% discount")
                priorities.append(9)
                recs.append("Email series highlighting new offerings")
                priorities.append(7)
                recs.append("Survey to understand reason for inactivity")
                priorities.append(6)

            # At-risk customers (medium priority) (Priority 7)
            elif (at_risk_pct > 40 or dominant_risk == 'At Risk') and m_score >= 2.5:
                recs.append("Re-engagement campaign with 15-20% discount")
                priorities.append(7)
                recs.append("Highlight new products and improvements")
                priorities.append(6)
                recs.append("Loyalty points or rewards program")
                priorities.append(5)

            # Low-value churned (Priority 3)
            elif highly_inactive and m_score < 3.0:
                recs.append("Low-cost automated win-back emails only")
                priorities.append(3)
                recs.append("Focus on preventing other segments from becoming inactive")
                priorities.append(2)

            # Moderate segments (Priority 5-6)
            elif overall_score >= 2.5 and overall_score < 4.0:
                if dominant_risk in ['At Risk', 'Inactive'] or at_risk_pct > 30:
                    recs.append("Re-engagement campaign with 15-20% discount")
                    priorities.append(6)
                    recs.append("Highlight new products and improvements")
                    priorities.append(5)
                else:
                    recs.append("Upsell and cross-sell campaigns")
                    priorities.append(6)
                    recs.append("Loyalty points or rewards program")
                    priorities.append(5)

            # Default if no specific tactics (Priority 4)
            if len(recs) == 0:
                recs.append("Regular email campaigns with relevant content")
                priorities.append(4)
                recs.append("Segment-specific offers based on purchase history")
                priorities.append(3)

            # Add investment guidance based on churn risk
            if churn_risk_data:
                if churned_pct + inactive_pct > 60:
                    recs.append("Reduce spend on expensive channels, focus on high-ROI tactics")
                    priorities.append(2)
                elif active_pct > 60:
                    recs.append("Maximize investment - highest ROI segment for retention")
                    priorities.append(7)
            else:
                # Fallback to recency-based guidance
                if highly_inactive:
                    recs.append("Reduce spend on expensive channels, focus on high-ROI tactics")
                    priorities.append(2)
                elif not inactive:
                    recs.append("Maximize investment - highest ROI segment for retention")
                    priorities.append(7)

            # Sort by priority and take top N
            if len(recs) > max_recommendations:
                sorted_recs = sorted(
                    zip(recs, priorities),
                    key=lambda x: x[1],
                    reverse=True
                )
                recs = [rec for rec, _ in sorted_recs[:max_recommendations]]

            recommendations[cluster_id] = recs[:max_recommendations]

        logger.info(f"Created recommendations for {len(recommendations)} segments (max {max_recommendations} per segment)")
        return recommendations
    except Exception as e:
        logger.error(f"create_marketing_recommendations failed: {e}")
        raise


__all__ = [
    'generate_segment_insights',
    'create_marketing_recommendations',
]

if __name__ == "__main__":
    print("Testing n3i_insights module...")
    print("Module loaded successfully")
