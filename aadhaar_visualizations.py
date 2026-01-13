#!/usr/bin/env python3
"""
Aadhaar Data Visualization Module
Creates comprehensive visualizations for problem identification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

class AadhaarVisualizer:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        plt.style.use('seaborn-v0_8')
        
    def create_trend_visualizations(self):
        """Create trend analysis visualizations"""
        print("📊 Creating trend visualizations...")
        
        # Prepare monthly data
        bio_monthly = self.analyzer.biometric_data.groupby(['month', 'state']).agg({
            'bio_age_5_17': 'sum',
            'bio_age_17_': 'sum'
        }).reset_index()
        bio_monthly['total_bio'] = bio_monthly['bio_age_5_17'] + bio_monthly['bio_age_17_']
        
        # Top 10 states by biometric updates
        top_states = bio_monthly.groupby('state')['total_bio'].sum().nlargest(10).index
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # 1. Time series for top states
        for state in top_states[:5]:
            state_data = bio_monthly[bio_monthly['state'] == state]
            axes[0,0].plot(state_data['month'].astype(str), state_data['total_bio'], 
                          marker='o', label=state, linewidth=2)
        
        axes[0,0].set_title('Biometric Updates Trend - Top 5 States', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Month')
        axes[0,0].set_ylabel('Total Biometric Updates')
        axes[0,0].legend()
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Growth rate analysis
        bio_monthly['growth_rate'] = bio_monthly.groupby('state')['total_bio'].pct_change()
        growth_data = bio_monthly.dropna()
        
        axes[0,1].scatter(growth_data['total_bio'], growth_data['growth_rate'], 
                         alpha=0.6, s=50)
        axes[0,1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0,1].axhline(y=2, color='orange', linestyle='--', alpha=0.7, label='200% growth')
        axes[0,1].axhline(y=-0.5, color='orange', linestyle='--', alpha=0.7, label='-50% decline')
        axes[0,1].set_title('Growth Rate vs Volume Analysis', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Total Biometric Updates')
        axes[0,1].set_ylabel('Growth Rate')
        axes[0,1].legend()
        
        # 3. State-wise comparison
        state_totals = bio_monthly.groupby('state')['total_bio'].sum().sort_values(ascending=True)
        
        axes[1,0].barh(range(len(state_totals)), state_totals.values, color='skyblue')
        axes[1,0].set_yticks(range(len(state_totals)))
        axes[1,0].set_yticklabels(state_totals.index, fontsize=8)
        axes[1,0].set_title('State-wise Total Biometric Updates', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Total Updates')
        
        # 4. Age group distribution
        age_dist = self.analyzer.biometric_data.groupby('state').agg({
            'bio_age_5_17': 'sum',
            'bio_age_17_': 'sum'
        })
        age_dist['youth_pct'] = (age_dist['bio_age_5_17'] / (age_dist['bio_age_5_17'] + age_dist['bio_age_17_'])) * 100
        
        axes[1,1].hist(age_dist['youth_pct'], bins=20, color='lightcoral', alpha=0.7, edgecolor='black')
        axes[1,1].axvline(age_dist['youth_pct'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {age_dist["youth_pct"].mean():.1f}%')
        axes[1,1].set_title('Distribution of Youth Percentage in Biometric Updates', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Youth Percentage (5-17 years)')
        axes[1,1].set_ylabel('Number of States')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig('aadhaar_trend_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_anomaly_visualizations(self):
        """Create anomaly detection visualizations"""
        print("🚨 Creating anomaly detection visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # 1. Z-Score Analysis
        bio_state_totals = self.analyzer.biometric_data.groupby('state').agg({
            'bio_age_5_17': 'sum',
            'bio_age_17_': 'sum'
        })
        bio_state_totals['total_bio'] = bio_state_totals['bio_age_5_17'] + bio_state_totals['bio_age_17_']
        bio_state_totals['z_score'] = np.abs(stats.zscore(bio_state_totals['total_bio']))
        
        colors = ['red' if z > 2 else 'blue' for z in bio_state_totals['z_score']]
        axes[0,0].scatter(bio_state_totals['total_bio'], bio_state_totals['z_score'], 
                         c=colors, alpha=0.7, s=60)
        axes[0,0].axhline(y=2, color='red', linestyle='--', label='Outlier Threshold (Z=2)')
        axes[0,0].set_title('Z-Score Anomaly Detection - Biometric Updates', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Total Biometric Updates')
        axes[0,0].set_ylabel('Z-Score')
        axes[0,0].legend()
        
        # Add state labels for outliers
        outliers = bio_state_totals[bio_state_totals['z_score'] > 2]
        for state, row in outliers.iterrows():
            axes[0,0].annotate(state, (row['total_bio'], row['z_score']), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 2. Box plot for outlier detection
        demo_state_totals = self.analyzer.demographic_data.groupby('state').agg({
            'demo_age_5_17': 'sum',
            'demo_age_17_': 'sum'
        })
        demo_state_totals['total_demo'] = demo_state_totals['demo_age_5_17'] + demo_state_totals['demo_age_17_']
        
        axes[0,1].boxplot(demo_state_totals['total_demo'], vert=True)
        axes[0,1].set_title('Demographic Updates - Box Plot (IQR Method)', fontsize=14, fontweight='bold')
        axes[0,1].set_ylabel('Total Demographic Updates')
        
        # 3. Ratio Analysis Visualization
        bio_totals = self.analyzer.biometric_data.groupby('state')['bio_age_5_17', 'bio_age_17_'].sum()
        demo_totals = self.analyzer.demographic_data.groupby('state')['demo_age_5_17', 'demo_age_17_'].sum()
        enroll_totals = self.analyzer.enrollment_data.groupby('state')['age_0_5', 'age_5_17', 'age_18_greater'].sum()
        
        bio_totals['total_bio'] = bio_totals['bio_age_5_17'] + bio_totals['bio_age_17_']
        demo_totals['total_demo'] = demo_totals['demo_age_5_17'] + demo_totals['demo_age_17_']
        enroll_totals['total_enroll'] = enroll_totals['age_0_5'] + enroll_totals['age_5_17'] + enroll_totals['age_18_greater']
        
        ratio_data = pd.merge(bio_totals[['total_bio']], enroll_totals[['total_enroll']], 
                             left_index=True, right_index=True, how='outer').fillna(0)
        ratio_data['update_to_enroll_ratio'] = ratio_data['total_bio'] / (ratio_data['total_enroll'] + 1)
        
        colors = ['red' if r > 2 else 'green' for r in ratio_data['update_to_enroll_ratio']]
        axes[1,0].scatter(ratio_data['total_enroll'], ratio_data['update_to_enroll_ratio'], 
                         c=colors, alpha=0.7, s=60)
        axes[1,0].axhline(y=2, color='red', linestyle='--', label='Problem Threshold (Ratio > 2)')
        axes[1,0].set_title('Update-to-Enrollment Ratio Analysis', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Total Enrollments')
        axes[1,0].set_ylabel('Update-to-Enrollment Ratio')
        axes[1,0].legend()
        
        # 4. Heatmap of state performance
        performance_data = pd.merge(bio_totals[['total_bio']], demo_totals[['total_demo']], 
                                   left_index=True, right_index=True, how='outer').fillna(0)
        performance_data = pd.merge(performance_data, enroll_totals[['total_enroll']], 
                                   left_index=True, right_index=True, how='outer').fillna(0)
        
        # Normalize data for heatmap
        performance_normalized = performance_data.div(performance_data.max())
        
        # Select top 15 states for readability
        top_15_states = performance_data.sum(axis=1).nlargest(15).index
        heatmap_data = performance_normalized.loc[top_15_states]
        
        im = axes[1,1].imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto')
        axes[1,1].set_xticks(range(len(heatmap_data.columns)))
        axes[1,1].set_xticklabels(heatmap_data.columns, rotation=45)
        axes[1,1].set_yticks(range(len(heatmap_data.index)))
        axes[1,1].set_yticklabels(heatmap_data.index, fontsize=8)
        axes[1,1].set_title('State Performance Heatmap (Normalized)', fontsize=14, fontweight='bold')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1,1])
        
        plt.tight_layout()
        plt.savefig('aadhaar_anomaly_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_interactive_dashboard(self):
        """Create interactive Plotly dashboard"""
        print("🎯 Creating interactive dashboard...")
        
        # Prepare data
        bio_state_totals = self.analyzer.biometric_data.groupby('state').agg({
            'bio_age_5_17': 'sum',
            'bio_age_17_': 'sum'
        })
        bio_state_totals['total_bio'] = bio_state_totals['bio_age_5_17'] + bio_state_totals['bio_age_17_']
        bio_state_totals = bio_state_totals.reset_index()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('State-wise Biometric Updates', 'Age Group Distribution', 
                           'Geographic Distribution', 'Trend Analysis'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # 1. Bar chart
        fig.add_trace(
            go.Bar(x=bio_state_totals['state'], y=bio_state_totals['total_bio'],
                   name='Biometric Updates', marker_color='skyblue'),
            row=1, col=1
        )
        
        # 2. Pie chart for age distribution
        total_youth = self.analyzer.biometric_data['bio_age_5_17'].sum()
        total_adult = self.analyzer.biometric_data['bio_age_17_'].sum()
        
        fig.add_trace(
            go.Pie(labels=['Youth (5-17)', 'Adult (17+)'], 
                   values=[total_youth, total_adult],
                   name="Age Distribution"),
            row=1, col=2
        )
        
        # 3. Scatter plot for anomaly detection
        bio_state_totals['z_score'] = np.abs(stats.zscore(bio_state_totals['total_bio']))
        colors = ['red' if z > 2 else 'blue' for z in bio_state_totals['z_score']]
        
        fig.add_trace(
            go.Scatter(x=bio_state_totals['total_bio'], y=bio_state_totals['z_score'],
                      mode='markers', marker=dict(color=colors, size=8),
                      text=bio_state_totals['state'], name='States'),
            row=2, col=1
        )
        
        # 4. Monthly trend
        bio_monthly = self.analyzer.biometric_data.groupby(['month', 'state']).agg({
            'bio_age_5_17': 'sum',
            'bio_age_17_': 'sum'
        }).reset_index()
        bio_monthly['total_bio'] = bio_monthly['bio_age_5_17'] + bio_monthly['bio_age_17_']
        
        # Show trend for top 5 states
        top_5_states = bio_state_totals.nlargest(5, 'total_bio')['state'].tolist()
        
        for state in top_5_states:
            state_data = bio_monthly[bio_monthly['state'] == state]
            fig.add_trace(
                go.Scatter(x=state_data['month'].astype(str), y=state_data['total_bio'],
                          mode='lines+markers', name=state),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Aadhaar Data Analysis Dashboard",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        # Save as HTML
        fig.write_html("aadhaar_dashboard.html")
        print("✅ Interactive dashboard saved as 'aadhaar_dashboard.html'")
        
    def create_problem_summary_report(self):
        """Create visual problem summary report"""
        print("📋 Creating problem summary report...")
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        
        # Problem categories
        problem_categories = {
            'Data Quality Issues': 0,
            'Anomalous Patterns': 0,
            'Ratio Imbalances': 0,
            'Geographic Disparities': 0,
            'Age Distribution Issues': 0
        }
        
        # Count problems by category (simplified categorization)
        for problem in self.analyzer.problems_found:
            if 'negative' in problem.lower() or 'null' in problem.lower():
                problem_categories['Data Quality Issues'] += 1
            elif 'outlier' in problem.lower() or 'abnormal' in problem.lower():
                problem_categories['Anomalous Patterns'] += 1
            elif 'ratio' in problem.lower():
                problem_categories['Ratio Imbalances'] += 1
            elif 'state' in problem.lower() or 'underperform' in problem.lower():
                problem_categories['Geographic Disparities'] += 1
            elif 'age' in problem.lower():
                problem_categories['Age Distribution Issues'] += 1
        
        # Problem category chart
        categories = list(problem_categories.keys())
        counts = list(problem_categories.values())
        
        bars = axes[0].bar(categories, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        axes[0].set_title('Problems Identified by Category', fontsize=16, fontweight='bold')
        axes[0].set_ylabel('Number of Problems')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            if count > 0:
                axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           str(count), ha='center', va='bottom', fontweight='bold')
        
        # Severity assessment
        severity_data = {
            'Critical': len([p for p in self.analyzer.problems_found if 'outlier' in p.lower()]),
            'High': len([p for p in self.analyzer.problems_found if 'abnormal' in p.lower()]),
            'Medium': len([p for p in self.analyzer.problems_found if 'ratio' in p.lower()]),
            'Low': len([p for p in self.analyzer.problems_found if 'pattern' in p.lower()])
        }
        
        # Pie chart for severity
        severity_labels = list(severity_data.keys())
        severity_values = list(severity_data.values())
        colors = ['#FF4757', '#FF6348', '#FFA502', '#2ED573']
        
        wedges, texts, autotexts = axes[1].pie(severity_values, labels=severity_labels, 
                                              colors=colors, autopct='%1.1f%%',
                                              startangle=90)
        axes[1].set_title('Problem Severity Distribution', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('aadhaar_problem_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Found {len(self.analyzer.problems_found)} total problems")
        print(f"✅ Visualizations saved as PNG files")

if __name__ == "__main__":
    from aadhaar_data_analysis import AadhaarDataAnalyzer
    
    # Run analysis first
    analyzer = AadhaarDataAnalyzer()
    analyzer.run_complete_analysis()
    
    # Create visualizations
    visualizer = AadhaarVisualizer(analyzer)
    visualizer.create_trend_visualizations()
    visualizer.create_anomaly_visualizations()
    visualizer.create_interactive_dashboard()
    visualizer.create_problem_summary_report()