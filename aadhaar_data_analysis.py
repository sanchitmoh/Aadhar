#!/usr/bin/env python3
"""
Comprehensive Aadhaar Data Analysis for Inconsistencies and Abnormalities
Author: Data Analysis Framework
Purpose: Identify problems in Aadhaar enrollment, biometric, and demographic data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import glob
import os
from datetime import datetime
import requests
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()
API_KEY = os.getenv('AadhaarStatewise', '579b464db66ec23bdd000001a6537618e84d40aa5ae04945c503592f')

class AadhaarDataAnalyzer:
    def __init__(self):
        self.biometric_data = None
        self.demographic_data = None
        self.enrollment_data = None
        self.combined_data = None
        self.problems_found = []
        
    def load_data(self):
        """Load all Aadhaar datasets"""
        print("📊 Loading Aadhaar datasets...")
        
        # Load biometric data
        bio_files = glob.glob('api_data_aadhar_biometric/*.csv')
        bio_dfs = []
        for file in bio_files:
            df = pd.read_csv(file)
            bio_dfs.append(df)
        self.biometric_data = pd.concat(bio_dfs, ignore_index=True)
        self.biometric_data['data_type'] = 'biometric'
        
        # Load demographic data
        demo_files = glob.glob('api_data_aadhar_demographic/*.csv')
        demo_dfs = []
        for file in demo_files:
            df = pd.read_csv(file)
            demo_dfs.append(df)
        self.demographic_data = pd.concat(demo_dfs, ignore_index=True)
        self.demographic_data['data_type'] = 'demographic'
        
        # Load enrollment data
        enroll_files = glob.glob('api_data_aadhar_enrolment/api_data_aadhar_enrolment/*.csv')
        enroll_dfs = []
        for file in enroll_files:
            df = pd.read_csv(file)
            enroll_dfs.append(df)
        self.enrollment_data = pd.concat(enroll_dfs, ignore_index=True)
        self.enrollment_data['data_type'] = 'enrollment'
        
        print(f"✅ Loaded {len(self.biometric_data)} biometric records")
        print(f"✅ Loaded {len(self.demographic_data)} demographic records")
        print(f"✅ Loaded {len(self.enrollment_data)} enrollment records")
        
    def data_quality_check(self):
        """Step 1: Basic data quality assessment"""
        print("\n🔍 STEP 1: DATA QUALITY ASSESSMENT")
        print("="*50)
        
        datasets = {
            'Biometric': self.biometric_data,
            'Demographic': self.demographic_data,
            'Enrollment': self.enrollment_data
        }
        
        for name, df in datasets.items():
            print(f"\n{name} Data Quality:")
            print(f"  Total records: {len(df):,}")
            print(f"  Null values: {df.isnull().sum().sum()}")
            print(f"  Duplicate rows: {df.duplicated().sum()}")
            print(f"  Unique states: {df['state'].nunique()}")
            print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
            
            # Check for negative values in numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            negative_counts = (df[numeric_cols] < 0).sum()
            if negative_counts.any():
                print(f"  ⚠️  Negative values found: {negative_counts[negative_counts > 0].to_dict()}")
                self.problems_found.append(f"{name}: Negative values in {negative_counts[negative_counts > 0].index.tolist()}")
    
    def trend_analysis(self):
        """Step 2: Trend Analysis - Identify baseline patterns"""
        print("\n📈 STEP 2: TREND ANALYSIS")
        print("="*50)
        
        # Convert date columns with proper format
        for df in [self.biometric_data, self.demographic_data, self.enrollment_data]:
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
            df['month'] = df['date'].dt.to_period('M')
        
        # Biometric trends
        bio_monthly = self.biometric_data.groupby(['month', 'state']).agg({
            'bio_age_5_17': 'sum',
            'bio_age_17_': 'sum'
        }).reset_index()
        
        # Identify sudden drops or spikes
        bio_monthly['total_bio'] = bio_monthly['bio_age_5_17'] + bio_monthly['bio_age_17_']
        bio_monthly['growth_rate'] = bio_monthly.groupby('state')['total_bio'].pct_change()
        
        # Flag abnormal growth rates (>200% or <-50%)
        abnormal_growth = bio_monthly[
            (bio_monthly['growth_rate'] > 2.0) | (bio_monthly['growth_rate'] < -0.5)
        ]
        
        if not abnormal_growth.empty:
            print("⚠️  ABNORMAL BIOMETRIC UPDATE PATTERNS:")
            for _, row in abnormal_growth.head(10).iterrows():
                print(f"  {row['state']} ({row['month']}): {row['growth_rate']:.1%} growth")
            self.problems_found.append("Abnormal biometric update growth patterns detected")
    
    def anomaly_detection(self):
        """Step 3: Statistical Anomaly Detection using Z-Score and IQR"""
        print("\n🚨 STEP 3: ANOMALY DETECTION")
        print("="*50)
        
        # Z-Score Analysis for Biometric Updates
        bio_state_totals = self.biometric_data.groupby('state').agg({
            'bio_age_5_17': 'sum',
            'bio_age_17_': 'sum'
        })
        bio_state_totals['total_bio'] = bio_state_totals['bio_age_5_17'] + bio_state_totals['bio_age_17_']
        
        # Calculate Z-scores
        bio_state_totals['z_score'] = np.abs(stats.zscore(bio_state_totals['total_bio']))
        
        # Identify outliers (Z-score > 2)
        bio_outliers = bio_state_totals[bio_state_totals['z_score'] > 2].sort_values('z_score', ascending=False)
        
        if not bio_outliers.empty:
            print("🔥 BIOMETRIC UPDATE OUTLIERS (Z-score > 2):")
            for state, row in bio_outliers.head(5).iterrows():
                print(f"  {state}: {row['total_bio']:,} updates (Z-score: {row['z_score']:.2f})")
            self.problems_found.append(f"High biometric update outliers: {list(bio_outliers.head(3).index)}")
        
        # IQR Analysis for Demographic Updates
        demo_state_totals = self.demographic_data.groupby('state').agg({
            'demo_age_5_17': 'sum',
            'demo_age_17_': 'sum'
        })
        demo_state_totals['total_demo'] = demo_state_totals['demo_age_5_17'] + demo_state_totals['demo_age_17_']
        
        Q1 = demo_state_totals['total_demo'].quantile(0.25)
        Q3 = demo_state_totals['total_demo'].quantile(0.75)
        IQR = Q3 - Q1
        
        # Identify outliers using IQR method
        demo_outliers = demo_state_totals[
            (demo_state_totals['total_demo'] < Q1 - 1.5 * IQR) |
            (demo_state_totals['total_demo'] > Q3 + 1.5 * IQR)
        ]
        
        if not demo_outliers.empty:
            print("\n🔥 DEMOGRAPHIC UPDATE OUTLIERS (IQR Method):")
            for state, row in demo_outliers.iterrows():
                print(f"  {state}: {row['total_demo']:,} updates")
            self.problems_found.append(f"Demographic update outliers: {list(demo_outliers.index)}")
    
    def ratio_efficiency_analysis(self):
        """Step 4: Ratio & Efficiency Analysis"""
        print("\n⚖️  STEP 4: RATIO & EFFICIENCY ANALYSIS")
        print("="*50)
        
        # Calculate state-wise totals
        bio_totals = self.biometric_data.groupby('state').agg({
            'bio_age_5_17': 'sum',
            'bio_age_17_': 'sum'
        })
        bio_totals['total_bio'] = bio_totals['bio_age_5_17'] + bio_totals['bio_age_17_']
        
        demo_totals = self.demographic_data.groupby('state').agg({
            'demo_age_5_17': 'sum',
            'demo_age_17_': 'sum'
        })
        demo_totals['total_demo'] = demo_totals['demo_age_5_17'] + demo_totals['demo_age_17_']
        
        enroll_totals = self.enrollment_data.groupby('state').agg({
            'age_0_5': 'sum',
            'age_5_17': 'sum',
            'age_18_greater': 'sum'
        })
        enroll_totals['total_enroll'] = enroll_totals['age_0_5'] + enroll_totals['age_5_17'] + enroll_totals['age_18_greater']
        
        # Merge for ratio analysis
        ratio_analysis = pd.merge(bio_totals, demo_totals, left_index=True, right_index=True, how='outer')
        ratio_analysis = pd.merge(ratio_analysis, enroll_totals, left_index=True, right_index=True, how='outer')
        ratio_analysis = ratio_analysis.fillna(0)
        
        # Calculate key ratios
        ratio_analysis['bio_to_enroll_ratio'] = ratio_analysis['total_bio'] / (ratio_analysis['total_enroll'] + 1)
        ratio_analysis['demo_to_enroll_ratio'] = ratio_analysis['total_demo'] / (ratio_analysis['total_enroll'] + 1)
        ratio_analysis['update_to_enroll_ratio'] = (ratio_analysis['total_bio'] + ratio_analysis['total_demo']) / (ratio_analysis['total_enroll'] + 1)
        
        # Identify problematic ratios
        high_update_ratio = ratio_analysis[ratio_analysis['update_to_enroll_ratio'] > 2.0].sort_values('update_to_enroll_ratio', ascending=False)
        
        if not high_update_ratio.empty:
            print("⚠️  HIGH UPDATE-TO-ENROLLMENT RATIOS (>2.0):")
            for state, row in high_update_ratio.head(5).iterrows():
                print(f"  {state}: {row['update_to_enroll_ratio']:.2f} (Updates: {row['total_bio'] + row['total_demo']:,.0f}, Enrollments: {row['total_enroll']:,.0f})")
            self.problems_found.append("States with high update-to-enrollment ratios indicating data quality issues")
        
        # Biometric vs Demographic ratio analysis
        ratio_analysis['bio_to_demo_ratio'] = ratio_analysis['total_bio'] / (ratio_analysis['total_demo'] + 1)
        extreme_bio_demo = ratio_analysis[
            (ratio_analysis['bio_to_demo_ratio'] > 5) | (ratio_analysis['bio_to_demo_ratio'] < 0.2)
        ]
        
        if not extreme_bio_demo.empty:
            print("\n⚠️  EXTREME BIOMETRIC-TO-DEMOGRAPHIC RATIOS:")
            for state, row in extreme_bio_demo.iterrows():
                print(f"  {state}: {row['bio_to_demo_ratio']:.2f}")
            self.problems_found.append("Extreme biometric-to-demographic ratios detected")
    
    def spatial_comparison(self):
        """Step 5: State/District-wise Spatial Analysis"""
        print("\n🗺️  STEP 5: SPATIAL COMPARISON ANALYSIS")
        print("="*50)
        
        # State-wise performance ranking
        state_performance = self.biometric_data.groupby('state').agg({
            'bio_age_5_17': 'sum',
            'bio_age_17_': 'sum'
        })
        state_performance['total_bio'] = state_performance['bio_age_5_17'] + state_performance['bio_age_17_']
        state_performance['rank'] = state_performance['total_bio'].rank(ascending=False)
        
        # Identify consistently underperforming states (bottom 10%)
        bottom_10_percent = int(len(state_performance) * 0.1)
        underperforming = state_performance.nsmallest(max(3, bottom_10_percent), 'total_bio')
        
        print("📉 UNDERPERFORMING STATES (Bottom 10%):")
        for state, row in underperforming.iterrows():
            print(f"  {state}: {row['total_bio']:,} biometric updates")
        
        if not underperforming.empty:
            self.problems_found.append(f"Underperforming states: {list(underperforming.index)}")
        
        # District-level analysis within states
        district_analysis = self.biometric_data.groupby(['state', 'district']).agg({
            'bio_age_5_17': 'sum',
            'bio_age_17_': 'sum'
        })
        district_analysis['total_bio'] = district_analysis['bio_age_5_17'] + district_analysis['bio_age_17_']
        
        # Find districts with extreme values within each state
        district_analysis['state_rank'] = district_analysis.groupby('state')['total_bio'].rank(ascending=False)
        district_analysis['state_percentile'] = district_analysis.groupby('state')['total_bio'].rank(pct=True)
        
        extreme_districts = district_analysis[
            (district_analysis['state_percentile'] > 0.95) | (district_analysis['state_percentile'] < 0.05)
        ]
        
        print(f"\n🎯 EXTREME DISTRICTS (Top/Bottom 5% within states): {len(extreme_districts)} found")
    
    def demographic_segmentation(self):
        """Step 6: Age-based Demographic Analysis"""
        print("\n👥 STEP 6: DEMOGRAPHIC SEGMENTATION ANALYSIS")
        print("="*50)
        
        # Age group analysis for biometric updates
        bio_age_analysis = self.biometric_data.groupby('state').agg({
            'bio_age_5_17': 'sum',
            'bio_age_17_': 'sum'
        })
        
        bio_age_analysis['total'] = bio_age_analysis['bio_age_5_17'] + bio_age_analysis['bio_age_17_']
        bio_age_analysis['youth_percentage'] = (bio_age_analysis['bio_age_5_17'] / bio_age_analysis['total']) * 100
        bio_age_analysis['adult_percentage'] = (bio_age_analysis['bio_age_17_'] / bio_age_analysis['total']) * 100
        
        # Identify states with unusual age distribution
        unusual_youth = bio_age_analysis[
            (bio_age_analysis['youth_percentage'] > 80) | (bio_age_analysis['youth_percentage'] < 20)
        ]
        
        if not unusual_youth.empty:
            print("⚠️  UNUSUAL AGE DISTRIBUTION IN BIOMETRIC UPDATES:")
            for state, row in unusual_youth.iterrows():
                print(f"  {state}: {row['youth_percentage']:.1f}% youth, {row['adult_percentage']:.1f}% adult")
            self.problems_found.append("Unusual age distribution patterns in biometric updates")
        
        # Similar analysis for demographic updates
        demo_age_analysis = self.demographic_data.groupby('state').agg({
            'demo_age_5_17': 'sum',
            'demo_age_17_': 'sum'
        })
        
        demo_age_analysis['total'] = demo_age_analysis['demo_age_5_17'] + demo_age_analysis['demo_age_17_']
        demo_age_analysis['youth_percentage'] = (demo_age_analysis['demo_age_5_17'] / demo_age_analysis['total']) * 100
        
        # Compare biometric vs demographic age patterns
        age_comparison = pd.merge(
            bio_age_analysis[['youth_percentage']], 
            demo_age_analysis[['youth_percentage']], 
            left_index=True, right_index=True, 
            suffixes=('_bio', '_demo')
        )
        
        age_comparison['age_pattern_diff'] = abs(age_comparison['youth_percentage_bio'] - age_comparison['youth_percentage_demo'])
        large_differences = age_comparison[age_comparison['age_pattern_diff'] > 20]
        
        if not large_differences.empty:
            print("\n⚠️  LARGE AGE PATTERN DIFFERENCES (Bio vs Demo):")
            for state, row in large_differences.iterrows():
                print(f"  {state}: {row['age_pattern_diff']:.1f}% difference")
            self.problems_found.append("Large age pattern differences between biometric and demographic updates")
    
    def correlation_analysis(self):
        """Step 7: Correlation Analysis"""
        print("\n🔗 STEP 7: CORRELATION ANALYSIS")
        print("="*50)
        
        # Create state-level correlation matrix - only numeric columns
        bio_state = self.biometric_data.groupby('state')[['bio_age_5_17', 'bio_age_17_']].sum()
        demo_state = self.demographic_data.groupby('state')[['demo_age_5_17', 'demo_age_17_']].sum()
        enroll_state = self.enrollment_data.groupby('state')[['age_0_5', 'age_5_17', 'age_18_greater']].sum()
        
        # Merge all datasets
        correlation_data = pd.merge(bio_state, demo_state, left_index=True, right_index=True, how='outer')
        correlation_data = pd.merge(correlation_data, enroll_state, left_index=True, right_index=True, how='outer')
        correlation_data = correlation_data.fillna(0)
        
        # Calculate correlations
        correlation_matrix = correlation_data.corr()
        
        # Identify strong correlations (>0.8 or <-0.8)
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.8:
                    strong_correlations.append({
                        'var1': correlation_matrix.columns[i],
                        'var2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        if strong_correlations:
            print("🔗 STRONG CORRELATIONS FOUND:")
            for corr in strong_correlations[:5]:
                print(f"  {corr['var1']} ↔ {corr['var2']}: {corr['correlation']:.3f}")
        else:
            print("🔗 No strong correlations (>0.8) found between variables")
    
    def generate_insights_report(self):
        """Generate comprehensive insights report"""
        print("\n📋 COMPREHENSIVE INSIGHTS REPORT")
        print("="*60)
        
        print(f"\n🔍 TOTAL PROBLEMS IDENTIFIED: {len(self.problems_found)}")
        print("-" * 40)
        
        for i, problem in enumerate(self.problems_found, 1):
            print(f"{i}. {problem}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS FOR ADVANCEMENT:")
        print("-" * 40)
        
        recommendations = [
            "Implement real-time anomaly detection system using Z-score monitoring",
            "Set up automated alerts for states with high update-to-enrollment ratios",
            "Deploy predictive models to forecast biometric update loads",
            "Establish regional performance benchmarks and monitoring dashboards",
            "Implement age-group specific quality control measures",
            "Create correlation-based fraud detection algorithms",
            "Set up seasonal trend analysis for resource planning",
            "Develop district-level performance optimization strategies"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        print("\n🎯 ADVANCED TECHNIQUES TO IMPLEMENT:")
        print("-" * 40)
        print("• Machine Learning: Isolation Forest for anomaly detection")
        print("• Time Series: ARIMA models for trend forecasting")
        print("• Clustering: K-means for state performance grouping")
        print("• Statistical: Control charts for process monitoring")
        print("• Geospatial: Hotspot analysis for resource allocation")
        print("• Network Analysis: Inter-state migration pattern detection")
    
    def run_complete_analysis(self):
        """Execute complete analysis pipeline"""
        print("🚀 STARTING COMPREHENSIVE AADHAAR DATA ANALYSIS")
        print("="*60)
        
        self.load_data()
        self.data_quality_check()
        self.trend_analysis()
        self.anomaly_detection()
        self.ratio_efficiency_analysis()
        self.spatial_comparison()
        self.demographic_segmentation()
        self.correlation_analysis()
        self.generate_insights_report()
        
        print("\n✅ ANALYSIS COMPLETE!")
        return self.problems_found

if __name__ == "__main__":
    analyzer = AadhaarDataAnalyzer()
    problems = analyzer.run_complete_analysis()