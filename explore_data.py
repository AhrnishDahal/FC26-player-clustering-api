"""
Data Exploration Script for FC26.csv
Run this FIRST to understand your dataset structure
"""

import pandas as pd
import numpy as np

def explore_dataset(filepath='FC26.csv'):
    """Explore and validate the FC26 dataset"""
    
    print("=" * 70)
    print("⚽ FC26 Dataset Exploration")
    print("=" * 70)
    
    # Load dataset
    print(f"\n📂 Loading {filepath}...")
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Successfully loaded {len(df):,} players")
    except FileNotFoundError:
        print(f"❌ Error: {filepath} not found in current directory")
        return
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    # Basic info
    print(f"\n📊 Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Show first few rows
    print("\n📋 First 3 rows:")
    print(df.head(3))
    
    # Column names
    print(f"\n📝 All Columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:3d}. {col}")
    
    # Check for required style attributes
    print("\n🔍 Checking for Required Attributes...")
    
    required_attributes = {
        'Pace': ['movement_acceleration', 'movement_sprint_speed', 
                'acceleration', 'sprint_speed', 'pace'],
        'Dribbling': ['skill_dribbling', 'skill_ball_control', 
                     'movement_agility', 'movement_balance',
                     'dribbling', 'ball_control', 'agility', 'balance'],
        'Creativity': ['attacking_short_passing', 'skill_long_passing', 
                      'mentality_vision', 'skill_curve',
                      'short_passing', 'long_passing', 'vision', 'curve'],
        'Finishing': ['attacking_finishing', 'power_shot_power', 
                     'mentality_positioning',
                     'finishing', 'shot_power', 'positioning'],
        'Defense': ['mentality_interceptions', 'defending_standing_tackle', 
                   'defending_sliding_tackle', 'mentality_aggression',
                   'interceptions', 'standing_tackle', 'sliding_tackle', 
                   'aggression', 'defensive_awareness'],
        'Physicality': ['power_strength', 'power_stamina', 'power_jumping',
                       'strength', 'stamina', 'jumping']
    }
    
    found_mapping = {}
    missing_dimensions = []
    
    for dimension, possible_cols in required_attributes.items():
        found = [col for col in possible_cols if col in df.columns]
        if found:
            found_mapping[dimension] = found
            print(f"  ✓ {dimension:12s}: {', '.join(found)}")
        else:
            missing_dimensions.append(dimension)
            print(f"  ✗ {dimension:12s}: NOT FOUND")
    
    if missing_dimensions:
        print(f"\n⚠️  Missing dimensions: {', '.join(missing_dimensions)}")
        print("\nSearching for similar column names...")
        
        for dim in missing_dimensions:
            print(f"\n  {dim}:")
            keywords = dimension.lower()
            matches = [col for col in df.columns if keywords[:4] in col.lower()]
            if matches:
                print(f"    Possible matches: {', '.join(matches[:5])}")
    
    # Check for player identification columns
    print("\n👤 Player Identification Columns:")
    id_cols = ['name', 'short_name', 'long_name', 'player_name', 'full_name']
    found_id = [col for col in id_cols if col in df.columns]
    if found_id:
        print(f"  ✓ Found: {', '.join(found_id)}")
    else:
        print("  ⚠️  No standard name column found")
        print(f"  First 10 columns: {', '.join(df.columns[:10])}...")
    
    # Check for metadata
    print("\n📈 Metadata Columns:")
    meta_cols = ['age', 'overall', 'potential', 'player_positions', 
                'positions', 'position', 'club', 'nationality']
    found_meta = [col for col in meta_cols if col in df.columns]
    if found_meta:
        print(f"  ✓ Found: {', '.join(found_meta)}")
    
    # Data quality
    print("\n🔬 Data Quality Check:")
    print(f"  Total missing values: {df.isnull().sum().sum():,}")
    print(f"  Columns with missing data: {(df.isnull().sum() > 0).sum()}")
    
    # Sample player
    if 'short_name' in df.columns or 'name' in df.columns:
        name_col = 'short_name' if 'short_name' in df.columns else 'name'
        print(f"\n⭐ Sample Player: {df[name_col].iloc[0]}")
        sample_cols = [col for col in df.columns if col in found_id + found_meta][:8]
        if sample_cols:
            print(df[sample_cols].iloc[0])
    
    # Save column mapping
    print("\n💾 Saving column mapping to 'column_mapping.txt'...")
    with open('column_mapping.txt', 'w') as f:
        f.write("FC26 Dataset Column Mapping\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total Players: {len(df):,}\n")
        f.write(f"Total Columns: {len(df.columns)}\n\n")
        f.write("Found Attribute Mapping:\n")
        for dim, cols in found_mapping.items():
            f.write(f"  {dim}: {', '.join(cols)}\n")
        f.write("\nAll Columns:\n")
        for col in df.columns:
            f.write(f"  - {col}\n")
    
    print("✓ Column mapping saved!")
    
    print("\n" + "=" * 70)
    print("✅ Exploration complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review column_mapping.txt")
    print("  2. If all dimensions found, run: python train_model.py")
    print("  3. If dimensions missing, update STYLE_DIMENSIONS in train_model.py")
    
    return df, found_mapping


if __name__ == '__main__':
    df, mapping = explore_dataset()