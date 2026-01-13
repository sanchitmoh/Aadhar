#!/usr/bin/env python3
"""
Advanced Aadhaar Analytics Implementation
Implements: Isolation Forest, ARIMA, K-means, Control Charts, Geospatial Analysis, Network Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import glob
from datetime import datetime
import networkx as nx
from scipy import stats

# Machine Learning imports
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Time Series imports
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("⚠️  Install statsmodels for ARIMA: pip install statsmodels")

warnings.filterwarnings('ignore')

class AdvancedAadhaarAnalytics:
    def __init__(self):
        self.biometric_data = None
        self.demographic_data = None
        self.enrollment_data = None
        self.state_performance_data = None
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare data for advanced analytics"""
        print("🔄 Loading data for advanced analytics...")
        
        # Load biometric data
        bio_files = glob.glob('api_data_aadhar_biometric/*.csv')
        bio_dfs = [pd.read_csv(file) for file in bio_files]
        self.biometric_data = pd.concat(bio_dfs, ignore_index=True)
        
        # Load demographic data
        demo_files = glob.glob('api_data_aadhar_demographic/*.csv')
        demo_dfs = [pd.read_csv(file) for file in demo_files]
        self.demographic_data = pd.concat(demo_dfs, ignore_index=True)
        
        # Load enrollment data
        enroll_files = glob.glob('api_data_aadhar_enrolment/api_data_aadhar_enrolment/*.csv')
        enroll_dfs = [pd.read_csv(file) for file in enroll_files]
        self.enrollment_data = pd.concat(enroll_dfs, ignore_index=True)
        
        # Convert dates
        for df in [self.biometric_data, self.demographic_data, self.enrollment_data]:
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
        
        # Create state performance dataset
        self._create_state_performance_data()
        
        print(f"✅ Data loaded: {len(self.biometric_data):,} bio, {len(self.demographic_data):,} demo, {len(self.enrollment_data):,} enroll")
        
    def _create_state_performance_data(self):
        """Create comprehensive state performance dataset"""
        # Aggregate by state
        bio_state = self.biometric_data.groupby('state').agg({
            'bio_age_5_17': ['sum', 'mean', 'std'],
            'bio_age_17_': ['sum', 'mean', 'std']
        }).round(2)
        bio_state.columns = ['bio_youth_sum', 'bio_youth_mean', 'bio_youth_std', 
                            'bio_adult_sum', 'bio_adult_mean', 'bio_adult_std']
        
        demo_state = self.demographic_data.groupby('state').agg({
            'demo_age_5_17': ['sum', 'mean', 'std'],
            'demo_age_17_': ['sum', 'mean', 'std']
        }).round(2)
        demo_state.columns = ['demo_youth_sum', 'demo_youth_mean', 'demo_youth_std',
                             'demo_adult_sum', 'demo_adult_mean', 'demo_adult_std']
        
        enroll_state = self.enrollment_data.groupby('state').agg({
            'age_0_5': ['sum', 'mean', 'std'],
            'age_5_17': ['sum', 'mean', 'std'],
            'age_18_greater': ['sum', 'mean', 'std']
        }).round(2)
        enroll_state.columns = ['enroll_child_sum', 'enroll_child_mean', 'enroll_child_std',
                               'enroll_youth_sum', 'enroll_youth_mean', 'enroll_youth_std',
                               'enroll_adult_sum', 'enroll_adult_mean', 'enroll_adult_std']
        
        # Merge all data
        self.state_performance_data = pd.merge(bio_state, demo_state, left_index=True, right_index=True, how='outer')
        self.state_performance_data = pd.merge(self.state_performance_data, enroll_state, left_index=True, right_index=True, how='outer')
        self.state_performance_data = self.state_performance_data.fillna(0)
        
        # Calculate derived metrics
        self.state_performance_data['total_bio'] = (self.state_performance_data['bio_youth_sum'] + 
                                                   self.state_performance_data['bio_adult_sum'])
        self.state_performance_data['total_demo'] = (self.state_performance_data['demo_youth_sum'] + 
                                                    self.state_performance_data['demo_adult_sum'])
        self.state_performance_data['total_enroll'] = (self.state_performance_data['enroll_child_sum'] + 
                                                      self.state_performance_data['enroll_youth_sum'] + 
                                                      self.state_performance_data['enroll_adult_sum'])
        
        # Performance ratios
        self.state_performance_data['bio_demo_ratio'] = (self.state_performance_data['total_bio'] / 
                                                        (self.state_performance_data['total_demo'] + 1))
        self.state_performance_data['update_enroll_ratio'] = ((self.state_performance_data['total_bio'] + 
                                                              self.state_performance_data['total_demo']) / 
                                                             (self.state_performance_data['total_enroll'] + 1))
        
    def isolation_forest_anomaly_detection(self):
        """1. Machine Learning: Isolation Forest for anomaly detection"""
        print("\n🤖 MACHINE LEARNING: ISOLATION FOREST ANOMALY DETECTION")
        print("="*60)
        
        # Prepare features
        features = ['total_bio', 'total_demo', 'total_enroll', 'bio_demo_ratio', 'update_enroll_ratio']
        X = self.state_performance_data[features].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
        anomaly_labels = iso_forest.fit_predict(X_scaled)
        anomaly_scores = iso_forest.decision_function(X_scaled)
        
        # Create results
        results_df = self.state_performance_data.copy()
        results_df['anomaly_label'] = anomaly_labels
        results_df['anomaly_score'] = anomaly_scores
        results_df['is_anomaly'] = anomaly_labels == -1
        
        anomalies = results_df[results_df['is_anomaly']].sort_values('anomaly_score')
        
        print(f"🔍 ISOLATION FOREST RESULTS:")
        print(f"   Anomalies detected: {len(anomalies)} out of {len(results_df)} states")
        
        if not anomalies.empty:
            print(f"\n🚨 ANOMALOUS STATES:")
            for state, row in anomalies.head(5).iterrows():
                print(f"   {state}: Score={row['anomaly_score']:.3f}")
        
        self.results['isolation_forest'] = {
            'results': results_df,
            'anomalies': anomalies,
            'features': features
        }
        
        # Create visualization
        self._plot_isolation_forest(results_df, features)
        
    def _plot_isolation_forest(self, results_df, features):
        """Plot Isolation Forest results"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # 1. Anomaly scores
        normal_scores = results_df[results_df['anomaly_label'] == 1]['anomaly_score']
        anomaly_scores = results_df[results_df['anomaly_label'] == -1]['anomaly_score']
        
        axes[0,0].hist(normal_scores, bins=20, alpha=0.7, label='Normal', color='green')
        axes[0,0].hist(anomaly_scores, bins=20, alpha=0.7, label='Anomaly', color='red')
        axes[0,0].set_title('Isolation Forest: Anomaly Score Distribution', fontsize=14, fontweight='bold')
        axes[0,0].legend()
        
        # 2. PCA visualization
        pca = PCA(n_components=2)
        X_clean = results_df[features].replace([np.inf, -np.inf], 0).fillna(0)
        X_pca = pca.fit_transform(StandardScaler().fit_transform(X_clean))
        
        colors = ['red' if x == -1 else 'blue' for x in results_df['anomaly_label']]
        axes[0,1].scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.7)
        axes[0,1].set_title('PCA: Anomaly Visualization', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[0,1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        
        # 3. Feature comparison
        comparison_data = []
        for feature in features:
            normal_mean = results_df[results_df['anomaly_label'] == 1][feature].mean()
            anomaly_mean = results_df[results_df['anomaly_label'] == -1][feature].mean()
            comparison_data.append([feature, normal_mean, anomaly_mean])
        
        comp_df = pd.DataFrame(comparison_data, columns=['Feature', 'Normal', 'Anomaly'])
        x = np.arange(len(features))
        width = 0.35
        
        axes[1,0].bar(x - width/2, comp_df['Normal'], width, label='Normal', alpha=0.7)
        axes[1,0].bar(x + width/2, comp_df['Anomaly'], width, label='Anomaly', alpha=0.7)
        axes[1,0].set_title('Normal vs Anomaly Feature Comparison', fontsize=14, fontweight='bold')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(features, rotation=45)
        axes[1,0].legend()
        
        # 4. Top anomalies
        top_anomalies = results_df[results_df['is_anomaly']].nsmallest(10, 'anomaly_score')
        if not top_anomalies.empty:
            axes[1,1].barh(range(len(top_anomalies)), top_anomalies['anomaly_score'], color='red', alpha=0.7)
            axes[1,1].set_yticks(range(len(top_anomalies)))
            axes[1,1].set_yticklabels(top_anomalies.index, fontsize=10)
            axes[1,1].set_title('Top Anomalous States', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('isolation_forest_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def arima_forecasting(self):
        """2. Time Series: ARIMA forecasting"""
        print("\n📈 TIME SERIES: ARIMA FORECASTING")
        print("="*60)
        
        if not STATSMODELS_AVAILABLE:
            print("⚠️  ARIMA requires statsmodels. Install with: pip install statsmodels")
            return
        
        # Prepare monthly data
        bio_monthly = self.biometric_data.groupby('date').agg({
            'bio_age_5_17': 'sum',
            'bio_age_17_': 'sum'
        })
        bio_monthly['total_bio'] = bio_monthly['bio_age_5_17'] + bio_monthly['bio_age_17_']
        bio_monthly = bio_monthly.resample('M').sum()
        
        ts_data = bio_monthly['total_bio'].dropna()
        
        if len(ts_data) < 10:
            print("⚠️  Insufficient data for ARIMA")
            return
        
        print(f"📊 Time series: {len(ts_data)} observations")
        
        # Test stationarity
        adf_result = adfuller(ts_data)
        print(f"   ADF p-value: {adf_result[1]:.4f}")
        print(f"   Series is {'stationary' if adf_result[1] < 0.05 else 'non-stationary'}")
        
        # Fit ARIMA
        try:
            best_aic = np.inf
            best_model = None
            best_order = None
            
            for p in range(3):
                for d in range(2):
                    for q in range(3):
                        try:
                            model = ARIMA(ts_data, order=(p, d, q))
                            fitted = model.fit()
                            if fitted.aic < best_aic:
                                best_aic = fitted.aic
                                best_model = fitted
                                best_order = (p, d, q)
                        except:
                            continue
            
            if best_model:
                print(f"   Best ARIMA{best_order}, AIC: {best_aic:.2f}")
                
                # Forecast
                forecast_steps = 6
                forecast = best_model.forecast(steps=forecast_steps)
                forecast_ci = best_model.get_forecast(steps=forecast_steps).conf_int()
                
                last_date = ts_data.index[-1]
                forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                             periods=forecast_steps, freq='M')
                
                self.results['arima'] = {
                    'model': best_model,
                    'forecast': forecast,
                    'forecast_ci': forecast_ci,
                    'forecast_dates': forecast_dates,
                    'historical': ts_data,
                    'order': best_order
                }
                
                print(f"✅ Forecast generated for {forecast_steps} months")
                self._plot_arima_forecast(ts_data, forecast, forecast_ci, forecast_dates)
                
        except Exception as e:
            print(f"❌ ARIMA error: {e}")