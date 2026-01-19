"""
Football Player Style Clustering - Training Pipeline
Trains KMeans model on FIFA player attribute data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import json
from pathlib import Path
import os

class PlayerStyleTrainer:
    """Trains and saves player style clustering model"""
    
    # Style dimension definitions - UPDATE THESE based on explore_data.py output
    STYLE_DIMENSIONS = {
        'pace': ['movement_acceleration', 'movement_sprint_speed', 
                'acceleration', 'sprint_speed'],
        'dribbling': ['skill_dribbling', 'skill_ball_control', 
                     'movement_agility', 'movement_balance',
                     'dribbling', 'ball_control', 'agility', 'balance'],
        'creativity': ['attacking_short_passing', 'skill_long_passing', 
                      'mentality_vision', 'skill_curve',
                      'short_passing', 'long_passing', 'vision', 'curve'],
        'finishing': ['attacking_finishing', 'power_shot_power', 
                     'mentality_positioning',
                     'finishing', 'shot_power', 'positioning'],
        'defense': ['mentality_interceptions', 'defending_standing_tackle', 
                   'defending_sliding_tackle', 'mentality_aggression',
                   'interceptions', 'standing_tackle', 'sliding_tackle', 
                   'aggression'],
        'physicality': ['power_strength', 'power_stamina', 'power_jumping',
                       'strength', 'stamina', 'jumping']
    }
    
    # Cluster interpretations
    CLUSTER_LABELS = {
        0: "Creative Playmaker",
        1: "Ball Winning Midfielder",
        2: "Explosive Winger",
        3: "Target Man",
        4: "Defensive Center Back",
        5: "Box-to-Box Midfielder"
    }
    
    def __init__(self, data_path: str = 'FC26.csv', n_clusters: int = 6):
        self.data_path = data_path
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.feature_columns = []
        
    def load_data(self) -> pd.DataFrame:
        """Load and validate FC26 dataset"""
        print(f"📊 Loading dataset from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        
        print(f"✓ Loaded {len(df):,} players with {len(df.columns)} columns")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create style dimension features"""
        print("⚙️  Engineering style dimensions...")
        
        feature_df = pd.DataFrame()
        self.feature_columns = []
        
        for dimension, possible_attributes in self.STYLE_DIMENSIONS.items():
            # Find which attributes actually exist in the dataframe
            available_attrs = [col for col in possible_attributes if col in df.columns]
            
            if not available_attrs:
                print(f"  ⚠️  Warning: No columns found for {dimension}")
                print(f"      Looked for: {possible_attributes[:3]}...")
                feature_df[dimension] = 50.0
            else:
                # Calculate mean of available attributes
                feature_df[dimension] = df[available_attrs].mean(axis=1)
                self.feature_columns.append(dimension)
                print(f"  ✓ {dimension:12s}: using {len(available_attrs)} attributes")
        
        # Handle missing values
        feature_df = feature_df.fillna(feature_df.mean())
        
        print(f"✓ Created {len(feature_df.columns)} style dimensions")
        return feature_df
    
    def train(self, df: pd.DataFrame) -> dict:
        """Train clustering model"""
        print("🤖 Training KMeans clustering model...")
        
        # Engineer features
        features = self.engineer_features(df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(features)
        
        # Train KMeans
        self.kmeans.fit(X_scaled)
        
        # Get cluster assignments
        df['cluster'] = self.kmeans.labels_
        
        # Calculate cluster statistics
        stats = self._calculate_cluster_stats(df, features)
        
        print(f"✓ Model trained with {self.n_clusters} clusters")
        print(f"  Inertia: {self.kmeans.inertia_:.2f}")
        
        return stats
    
    def _calculate_cluster_stats(self, df: pd.DataFrame, 
                                 features: pd.DataFrame) -> dict:
        """Calculate statistics for each cluster"""
        stats = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = df['cluster'] == cluster_id
            cluster_features = features[cluster_mask]
            
            # Get sample players
            sample_players = []
            if 'short_name' in df.columns:
                sample_players = df[cluster_mask]['short_name'].head(5).tolist()
            elif 'name' in df.columns:
                sample_players = df[cluster_mask]['name'].head(5).tolist()
            
            stats[cluster_id] = {
                'count': int(cluster_mask.sum()),
                'avg_overall': float(df[cluster_mask]['overall'].mean()) 
                              if 'overall' in df.columns else None,
                'style_profile': cluster_features.mean().to_dict(),
                'label': self.CLUSTER_LABELS.get(cluster_id, f"Cluster {cluster_id}"),
                'sample_players': sample_players
            }
        
        return stats
    
    def save_artifacts(self, output_dir: str = 'models'):
        """Save trained model artifacts"""
        print(f"💾 Saving model artifacts to {output_dir}/...")
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save scaler
        joblib.dump(self.scaler, f'{output_dir}/scaler.pkl')
        print(f"  ✓ Saved scaler.pkl")
        
        # Save KMeans model
        joblib.dump(self.kmeans, f'{output_dir}/kmeans.pkl')
        print(f"  ✓ Saved kmeans.pkl")
        
        # Save cluster labels
        with open(f'{output_dir}/cluster_labels.json', 'w') as f:
            json.dump(self.CLUSTER_LABELS, f, indent=2)
        print(f"  ✓ Saved cluster_labels.json")
        
        # Save feature column mapping
        feature_info = {
            'feature_columns': self.feature_columns,
            'style_dimensions': self.STYLE_DIMENSIONS
        }
        with open(f'{output_dir}/feature_info.json', 'w') as f:
            json.dump(feature_info, f, indent=2)
        print(f"  ✓ Saved feature_info.json")
        
        print("✓ All artifacts saved successfully")
    
    def visualize_clusters(self, df: pd.DataFrame, features: pd.DataFrame):
        """Create cluster visualization"""
        try:
            from sklearn.decomposition import PCA
            import matplotlib.pyplot as plt
            
            print("📊 Creating cluster visualization...")
            
            # Reduce to 2D for visualization
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(self.scaler.transform(features))
            
            # Plot
            plt.figure(figsize=(14, 10))
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                                c=df['cluster'], cmap='tab10', alpha=0.6, s=30)
            plt.colorbar(scatter, label='Cluster')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            plt.title('Player Style Clusters (PCA Projection)', fontsize=14, fontweight='bold')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig('models/cluster_visualization.png', dpi=300, bbox_inches='tight')
            print("✓ Visualization saved to models/cluster_visualization.png")
            
        except ImportError:
            print("⚠️  Matplotlib not available, skipping visualization")
        except Exception as e:
            print(f"⚠️  Could not create visualization: {e}")


def main():
    """Main training pipeline"""
    print("=" * 70)
    print("⚽ Football Player Style Clustering - Training Pipeline")
    print("=" * 70)
    
    # Check if FC26.csv exists
    if not os.path.exists('FC26.csv'):
        print("\n❌ Error: FC26.csv not found in current directory!")
        print("\nPlease ensure FC26.csv is in the same directory as this script.")
        print("\nRun 'python explore_data.py' to check your data first.")
        return
    
    # Initialize trainer
    trainer = PlayerStyleTrainer(
        data_path='FC26.csv',
        n_clusters=6
    )
    
    # Load data
    df = trainer.load_data()
    
    # Train model
    stats = trainer.train(df)
    
    # Print cluster summary
    print("\n📋 Cluster Summary:")
    print("=" * 70)
    for cluster_id, info in stats.items():
        print(f"\nCluster {cluster_id}: {info['label']}")
        print(f"  Players: {info['count']:,}")
        if info['avg_overall']:
            print(f"  Avg Rating: {info['avg_overall']:.1f}")
        if info['sample_players']:
            print(f"  Sample: {', '.join(info['sample_players'][:3])}")
    
    # Save artifacts
    trainer.save_artifacts()
    
    # Optional visualization
    features = trainer.engineer_features(df)
    trainer.visualize_clusters(df, features)
    
    print("\n" + "=" * 70)
    print("✅ Training complete! Model ready for deployment.")
    print("=" * 70)
    print("\nNext step: Run the API")
    print("  uvicorn api:app --reload")


if __name__ == '__main__':
    main()