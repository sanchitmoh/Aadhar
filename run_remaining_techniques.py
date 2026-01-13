#!/usr/bin/env python3
"""
Run Remaining 3 Advanced Techniques: Control Charts, Geospatial, Network Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import glob
from datetime import datetime
import networkx as nx
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

class RemainingTechniques:
    def __init__(self):
        self.data = {}
        self.results = {}
        
    def load_data(self):
        """Load all datasets"""
        print("🔄 Loading data...")
        
        # Load biometric data
        bio_files = glob.glob('api_data_aadhar_biometric/*.csv')
        bio_dfs = [pd.read_csv(file) for file in bio_files]
        self.data['biometric'] = pd.concat(bio_dfs, ignore_index=True)
        
        # Load demographic data
        demo_files = glob.glob('api_data_aadhar_demographic/*.csv')
        demo_dfs = [pd.read_csv(file) for file in demo_files]
        self.data['demographic'] = pd.concat(demo_dfs, ignore_index=True)
        
        # Load enrollment data
        enroll_files = glob.glob('api_data_aadhar_enrolment/api_data_aadhar_enrolment/*.csv')
        enroll_dfs = [pd.read_csv(file) for file in enroll_files]
        self.data['enrollment'] = pd.concat(enroll_dfs, ignore_index=True)
        
        # Convert dates
        for key, df in self.data.items():
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
        
        # Create state performance data
        self._create_state_performance_data()
        
        print(f"✅ Data loaded successfully")
        
    def _create_state_performance_data(self):
        """Create state performance dataset"""
        # Aggregate data by state
        bio_agg = self.data['biometric'].groupby('state').agg({
            'bio_age_5_17': 'sum',
            'bio_age_17_': 'sum'
        }).fillna(0)
        
        demo_agg = self.data['demographic'].groupby('state').agg({
            'demo_age_5_17': 'sum',
            'demo_age_17_': 'sum'
        }).fillna(0)
        
        enroll_agg = self.data['enrollment'].groupby('state').agg({
            'age_0_5': 'sum',
            'age_5_17': 'sum',
            'age_18_greater': 'sum'
        }).fillna(0)
        
        # Merge data
        self.data['state_performance'] = pd.merge(bio_agg, demo_agg, left_index=True, right_index=True, how='outer')
        self.data['state_performance'] = pd.merge(self.data['state_performance'], enroll_agg, left_index=True, right_index=True, how='outer')
        self.data['state_performance'] = self.data['state_performance'].fillna(0)
        
        # Calculate totals
        self.data['state_performance']['total_bio'] = (self.data['state_performance']['bio_age_5_17'] + 
                                                      self.data['state_performance']['bio_age_17_'])
        self.data['state_performance']['total_demo'] = (self.data['state_performance']['demo_age_5_17'] + 
                                                       self.data['state_performance']['demo_age_17_'])
        self.data['state_performance']['total_enroll'] = (self.data['state_performance']['age_0_5'] + 
                                                         self.data['state_performance']['age_5_17'] + 
                                                         self.data['state_performance']['age_18_greater'])
        
    def technique_4_control_charts(self):
        """4. Statistical Process Control Charts"""
        print("\n📊 TECHNIQUE 4: STATISTICAL PROCESS CONTROL CHARTS")
        print("="*60)
        
        # Prepare daily data
        bio_daily = self.data['biometric'].groupby(['date', 'state']).agg({
            'bio_age_5_17': 'sum',
            'bio_age_17_': 'sum'
        }).reset_index()
        bio_daily['total_bio'] = bio_daily['bio_age_5_17'] + bio_daily['bio_age_17_']
        
        # Top 5 states
        top_states = self.data['state_performance'].nlargest(5, 'total_bio').index.tolist()
        
        control_results = {}
        
        for state in top_states:
            state_data = bio_daily[bio_daily['state'] == state]['total_bio']
            
            if len(state_data) < 5:
                continue
                
            mean_val = state_data.mean()
            std_val = state_data.std()
            ucl = mean_val + 3 * std_val
            lcl = max(0, mean_val - 3 * std_val)
            
            out_of_control = state_data[(state_data > ucl) | (state_data < lcl)]
            
            control_results[state] = {
                'data': state_data,
                'mean': mean_val,
                'ucl': ucl,
                'lcl': lcl,
                'out_of_control_pct': len(out_of_control) / len(state_data) * 100
            }
            
            print(f"   {state}: {len(out_of_control)/len(state_data)*100:.1f}% out-of-control")
        
        self.results['control_charts'] = control_results
        self._plot_control_charts(control_results)
        
    def _plot_control_charts(self, control_results):
        """Plot control charts"""
        n_states = len(control_results)
        fig, axes = plt.subplots(n_states, 1, figsize=(15, 4*n_states))
        fig.suptitle('STATISTICAL PROCESS CONTROL CHARTS', fontsize=16, fontweight='bold')
        
        if n_states == 1:
            axes = [axes]
        
        for i, (state, results) in enumerate(control_results.items()):
            data = results['data']
            mean_val = results['mean']
            ucl = results['ucl']
            lcl = results['lcl']
            
            axes[i].plot(range(len(data)), data.values, 'bo-', markersize=4, alpha=0.7)
            axes[i].axhline(y=mean_val, color='green', linestyle='-', linewidth=2, label='Mean')
            axes[i].axhline(y=ucl, color='red', linestyle='--', linewidth=2, label='UCL')
            axes[i].axhline(y=lcl, color='red', linestyle='--', linewidth=2, label='LCL')
            
            # Highlight out-of-control points
            out_of_control_mask = (data > ucl) | (data < lcl)
            if out_of_control_mask.any():
                axes[i].scatter(range(len(data)), data.where(out_of_control_mask), 
                               color='red', s=50, zorder=5, label='Out of Control')
            
            axes[i].set_title(f'Control Chart - {state}', fontsize=12, fontweight='bold')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            axes[i].fill_between(range(len(data)), lcl, ucl, alpha=0.1, color='green')
        
        plt.tight_layout()
        plt.savefig('4_control_charts.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def technique_5_geospatial_hotspots(self):
        """5. Geospatial Hotspot Analysis"""
        print("\n🗺️  TECHNIQUE 5: GEOSPATIAL HOTSPOT ANALYSIS")
        print("="*60)
        
        hotspot_data = self.data['state_performance'].copy()
        
        # Calculate hotspot metrics
        hotspot_data['update_intensity'] = (hotspot_data['total_bio'] + hotspot_data['total_demo']) / 1000
        hotspot_data['update_enroll_ratio'] = ((hotspot_data['total_bio'] + hotspot_data['total_demo']) / 
                                              (hotspot_data['total_enroll'] + 1))
        
        # Normalize for scoring
        hotspot_data['intensity_norm'] = (hotspot_data['update_intensity'] - hotspot_data['update_intensity'].min()) / (hotspot_data['update_intensity'].max() - hotspot_data['update_intensity'].min())
        hotspot_data['ratio_norm'] = (hotspot_data['update_enroll_ratio'] - hotspot_data['update_enroll_ratio'].min()) / (hotspot_data['update_enroll_ratio'].max() - hotspot_data['update_enroll_ratio'].min())
        
        # Resource need score
        hotspot_data['resource_need_score'] = (hotspot_data['intensity_norm'] * 0.6 + 
                                              hotspot_data['ratio_norm'] * 0.4)
        
        # Categorize
        hotspot_data['resource_category'] = pd.cut(
            hotspot_data['resource_need_score'],
            bins=4,
            labels=['Low Need', 'Medium Need', 'High Need', 'Critical Need']
        )
        
        print(f"📍 HOTSPOT RESULTS:")
        for category in ['Critical Need', 'High Need', 'Medium Need', 'Low Need']:
            states = hotspot_data[hotspot_data['resource_category'] == category]
            if not states.empty:
                print(f"   {category}: {len(states)} states")
        
        self.results['hotspots'] = hotspot_data
        self._plot_hotspots(hotspot_data)
        
    def _plot_hotspots(self, hotspot_data):
        """Plot hotspot analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('GEOSPATIAL HOTSPOT ANALYSIS', fontsize=16, fontweight='bold')
        
        # 1. Resource need distribution
        axes[0,0].hist(hotspot_data['resource_need_score'], bins=20, color='skyblue', alpha=0.7)
        axes[0,0].set_title('Resource Need Score Distribution')
        axes[0,0].set_xlabel('Resource Need Score')
        axes[0,0].set_ylabel('Number of States')
        
        # 2. Top need states
        top_need = hotspot_data.nlargest(10, 'resource_need_score')
        axes[0,1].barh(range(len(top_need)), top_need['resource_need_score'], color='red', alpha=0.7)
        axes[0,1].set_yticks(range(len(top_need)))
        axes[0,1].set_yticklabels(top_need.index, fontsize=10)
        axes[0,1].set_title('Top 10 Resource Need States')
        
        # 3. Category pie chart
        category_counts = hotspot_data['resource_category'].value_counts()
        axes[1,0].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
        axes[1,0].set_title('Resource Need Categories')
        
        # 4. Intensity vs Need scatter
        axes[1,1].scatter(hotspot_data['update_intensity'], hotspot_data['resource_need_score'], alpha=0.7)
        axes[1,1].set_title('Update Intensity vs Resource Need')
        axes[1,1].set_xlabel('Update Intensity')
        axes[1,1].set_ylabel('Resource Need Score')
        
        plt.tight_layout()
        plt.savefig('5_geospatial_hotspots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def technique_6_network_analysis(self):
        """6. Network Analysis"""
        print("\n🕸️  TECHNIQUE 6: NETWORK ANALYSIS")
        print("="*60)
        
        # Create similarity network
        features = ['total_bio', 'total_demo', 'total_enroll']
        state_features = self.data['state_performance'][features].fillna(0)
        state_features_norm = (state_features - state_features.mean()) / (state_features.std() + 1e-8)
        
        # Correlation matrix
        similarity_matrix = state_features_norm.T.corr()
        
        # Create network
        G = nx.Graph()
        
        # Add nodes
        for state in similarity_matrix.index:
            total_updates = self.data['state_performance'].loc[state, 'total_bio'] + self.data['state_performance'].loc[state, 'total_demo']
            G.add_node(state, total_updates=total_updates)
        
        # Add edges for similar states
        threshold = 0.7
        for i, state1 in enumerate(similarity_matrix.index):
            for j, state2 in enumerate(similarity_matrix.columns):
                if i < j and similarity_matrix.iloc[i, j] > threshold:
                    G.add_edge(state1, state2, weight=similarity_matrix.iloc[i, j])
        
        print(f"🔗 NETWORK RESULTS:")
        print(f"   Nodes: {G.number_of_nodes()}")
        print(f"   Edges: {G.number_of_edges()}")
        
        # Calculate centrality
        centrality_metrics = {}
        if G.number_of_edges() > 0:
            centrality_metrics['degree'] = nx.degree_centrality(G)
            centrality_metrics['betweenness'] = nx.betweenness_centrality(G)
            centrality_metrics['closeness'] = nx.closeness_centrality(G)
            
            for metric, values in centrality_metrics.items():
                if values:
                    top_state = max(values.items(), key=lambda x: x[1])
                    print(f"   Top {metric}: {top_state[0]} ({top_state[1]:.3f})")
        
        self.results['network'] = {
            'graph': G,
            'centrality': centrality_metrics,
            'similarity_matrix': similarity_matrix
        }
        
        self._plot_network(G, centrality_metrics, similarity_matrix)
        
    def _plot_network(self, G, centrality_metrics, similarity_matrix):
        """Plot network analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('NETWORK ANALYSIS', fontsize=16, fontweight='bold')
        
        # 1. Network graph
        if G.number_of_nodes() > 0:
            pos = nx.spring_layout(G, k=1, iterations=50)
            node_sizes = [max(50, min(1000, G.nodes[node]['total_updates'] / 10000)) for node in G.nodes()]
            
            nx.draw(G, pos, ax=axes[0,0], node_size=node_sizes, with_labels=True, 
                   font_size=8, node_color='skyblue', edge_color='gray', alpha=0.7)
            axes[0,0].set_title('Inter-State Similarity Network')
        
        # 2. Similarity heatmap (top 15 states)
        top_states = self.data['state_performance'].nlargest(15, 'total_bio').index
        sim_subset = similarity_matrix.loc[top_states, top_states]
        
        im = axes[0,1].imshow(sim_subset.values, cmap='coolwarm', vmin=-1, vmax=1)
        axes[0,1].set_xticks(range(len(sim_subset.columns)))
        axes[0,1].set_xticklabels(sim_subset.columns, rotation=45, fontsize=8)
        axes[0,1].set_yticks(range(len(sim_subset.index)))
        axes[0,1].set_yticklabels(sim_subset.index, fontsize=8)
        axes[0,1].set_title('State Similarity Matrix')
        plt.colorbar(im, ax=axes[0,1])
        
        # 3. Centrality comparison
        if centrality_metrics and centrality_metrics['degree']:
            centrality_df = pd.DataFrame(centrality_metrics).fillna(0)
            top_central = centrality_df.nlargest(10, 'degree')
            
            x = np.arange(len(top_central))
            width = 0.25
            
            axes[1,0].bar(x - width, top_central['degree'], width, label='Degree', alpha=0.7)
            axes[1,0].bar(x, top_central['betweenness'], width, label='Betweenness', alpha=0.7)
            axes[1,0].bar(x + width, top_central['closeness'], width, label='Closeness', alpha=0.7)
            
            axes[1,0].set_title('Centrality Measures')
            axes[1,0].set_xticks(x)
            axes[1,0].set_xticklabels(top_central.index, rotation=45, fontsize=8)
            axes[1,0].legend()
        
        # 4. Degree distribution
        if G.number_of_nodes() > 0:
            degrees = [G.degree(n) for n in G.nodes()]
            axes[1,1].hist(degrees, bins=max(1, len(set(degrees))), color='lightgreen', alpha=0.7)
            axes[1,1].set_title('Degree Distribution')
            axes[1,1].set_xlabel('Degree')
            axes[1,1].set_ylabel('Number of States')
        
        plt.tight_layout()
        plt.savefig('6_network_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def run_remaining_techniques(self):
        """Run techniques 4-6"""
        print("🚀 RUNNING REMAINING 3 ADVANCED TECHNIQUES")
        print("="*60)
        
        self.load_data()
        
        self.technique_4_control_charts()
        self.technique_5_geospatial_hotspots()
        self.technique_6_network_analysis()
        
        print("\n✅ ALL REMAINING TECHNIQUES COMPLETED!")
        print("Generated files:")
        print("• 4_control_charts.png")
        print("• 5_geospatial_hotspots.png")
        print("• 6_network_analysis.png")

if __name__ == "__main__":
    analyzer = RemainingTechniques()
    analyzer.run_remaining_techniques()