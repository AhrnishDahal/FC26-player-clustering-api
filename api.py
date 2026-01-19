# api.py - FastAPI with Built-in Web UI
"""
Football Scouting API - FastAPI Service with Web Interface
Provides endpoints for player style prediction and similarity search
Includes a beautiful web UI accessible at root URL
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class PlayerAttributes(BaseModel):
    """Player attributes for style prediction"""
    movement_acceleration: float = Field(default=50, ge=0, le=100)
    movement_sprint_speed: float = Field(default=50, ge=0, le=100)
    skill_dribbling: float = Field(default=50, ge=0, le=100)
    skill_ball_control: float = Field(default=50, ge=0, le=100)
    movement_agility: float = Field(default=50, ge=0, le=100)
    movement_balance: float = Field(default=50, ge=0, le=100)
    attacking_short_passing: float = Field(default=50, ge=0, le=100)
    skill_long_passing: float = Field(default=50, ge=0, le=100)
    mentality_vision: float = Field(default=50, ge=0, le=100)
    skill_curve: float = Field(default=50, ge=0, le=100)
    attacking_finishing: float = Field(default=50, ge=0, le=100)
    power_shot_power: float = Field(default=50, ge=0, le=100)
    mentality_positioning: float = Field(default=50, ge=0, le=100)
    mentality_interceptions: float = Field(default=50, ge=0, le=100)
    defending_standing_tackle: float = Field(default=50, ge=0, le=100)
    defending_sliding_tackle: float = Field(default=50, ge=0, le=100)
    mentality_aggression: float = Field(default=50, ge=0, le=100)
    power_strength: float = Field(default=50, ge=0, le=100)
    power_stamina: float = Field(default=50, ge=0, le=100)
    power_jumping: float = Field(default=50, ge=0, le=100)


class PlayerPrediction(BaseModel):
    """Cluster prediction response"""
    cluster_id: int
    style: str


class SimilarPlayerRequest(BaseModel):
    """Request for similar players"""
    player_name: str = Field(..., min_length=1)
    top_n: int = Field(default=5, ge=1, le=20)


class SimilarPlayerResponse(BaseModel):
    """Response with similar players"""
    similar_players: List[str]


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Football Player Style Clustering API",
    description="Professional scouting intelligence system for player style analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model artifacts
scaler = None
kmeans = None
cluster_labels = None
player_data = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_style_dimensions(attributes: dict) -> np.ndarray:
    """Convert raw FIFA attributes to 6 style dimensions"""
    style_dims = {
        'pace': ['movement_acceleration', 'movement_sprint_speed'],
        'dribbling': ['skill_dribbling', 'skill_ball_control', 
                     'movement_agility', 'movement_balance'],
        'creativity': ['attacking_short_passing', 'skill_long_passing', 
                      'mentality_vision', 'skill_curve'],
        'finishing': ['attacking_finishing', 'power_shot_power', 
                     'mentality_positioning'],
        'defense': ['mentality_interceptions', 'defending_standing_tackle', 
                   'defending_sliding_tackle', 'mentality_aggression'],
        'physicality': ['power_strength', 'power_stamina', 'power_jumping']
    }
    
    dimensions = []
    for dimension, attrs in style_dims.items():
        values = [attributes.get(attr, 50) for attr in attrs]
        dimensions.append(np.mean(values))
    
    return np.array(dimensions)


def create_style_dimensions_batch(df: pd.DataFrame) -> np.ndarray:
    """Create style dimensions for entire dataframe"""
    style_dims = {
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
    
    feature_matrix = []
    for dimension, attrs in style_dims.items():
        available_attrs = [col for col in attrs if col in df.columns]
        if available_attrs:
            feature_matrix.append(df[available_attrs].mean(axis=1).values)
        else:
            feature_matrix.append(np.full(len(df), 50))
    
    return np.array(feature_matrix).T


# ============================================================================
# WEB UI HTML
# ============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>⚽ Football Player Style Clustering</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }
        
        .header h1 {
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .card {
            background: white;
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
            border-bottom: 2px solid #e0e0e0;
        }
        
        .tab {
            padding: 15px 30px;
            background: none;
            border: none;
            font-size: 1.1em;
            cursor: pointer;
            transition: all 0.3s;
            border-bottom: 3px solid transparent;
            color: #666;
        }
        
        .tab.active {
            color: #667eea;
            border-bottom-color: #667eea;
            font-weight: 600;
        }
        
        .tab:hover {
            color: #667eea;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
            animation: fadeIn 0.3s;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        
        input[type="text"],
        input[type="number"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 1em;
            transition: border-color 0.3s;
        }
        
        input:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .attribute-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .attribute-item {
            display: flex;
            flex-direction: column;
        }
        
        .attribute-item label {
            font-size: 0.9em;
            margin-bottom: 5px;
        }
        
        .attribute-item input {
            padding: 10px;
        }
        
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 10px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        .result {
            margin-top: 30px;
            padding: 25px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 15px;
            display: none;
        }
        
        .result.show {
            display: block;
            animation: slideUp 0.4s;
        }
        
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .result h3 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.5em;
        }
        
        .result-content {
            font-size: 1.1em;
            line-height: 1.6;
        }
        
        .cluster-badge {
            display: inline-block;
            padding: 10px 20px;
            background: #667eea;
            color: white;
            border-radius: 25px;
            font-weight: 600;
            margin-top: 10px;
        }
        
        .player-list {
            list-style: none;
            padding: 0;
        }
        
        .player-list li {
            padding: 12px;
            margin: 8px 0;
            background: white;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .loading.show {
            display: block;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error {
            background: #fee;
            color: #c33;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            display: none;
        }
        
        .error.show {
            display: block;
        }
        
        .quick-select {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        
        .quick-select button {
            padding: 10px 20px;
            font-size: 0.9em;
            background: #f0f0f0;
            color: #333;
            box-shadow: none;
        }
        
        .quick-select button:hover {
            background: #667eea;
            color: white;
        }
        
        .info-box {
            background: #e3f2fd;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 4px solid #2196f3;
        }
        
        .info-box h4 {
            color: #1976d2;
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚽ Football Player Style Clustering</h1>
            <p>ML-Powered Player Scouting Intelligence System</p>
        </div>
        
        <div class="card">
            <div class="tabs">
                <button class="tab active" onclick="switchTab('predict')">🎯 Predict Style</button>
                <button class="tab" onclick="switchTab('similar')">🔍 Find Similar Players</button>
                <button class="tab" onclick="switchTab('clusters')">📊 View All Styles</button>
            </div>
            
            <!-- Predict Style Tab -->
            <div id="predict-tab" class="tab-content active">
                <div class="info-box">
                    <h4>How it works</h4>
                    <p>Enter player attributes (0-100) to predict their playing style. The ML model will classify them into one of 6 distinct archetypes.</p>
                </div>
                
                <div class="quick-select">
                    <button onclick="loadPreset('winger')">⚡ Explosive Winger</button>
                    <button onclick="loadPreset('playmaker')">🎨 Creative Playmaker</button>
                    <button onclick="loadPreset('defender')">🛡️ Defender</button>
                    <button onclick="loadPreset('striker')">⚽ Striker</button>
                </div>
                
                <form id="predict-form">
                    <h3 style="margin-bottom: 20px;">Player Attributes</h3>
                    <div class="attribute-grid">
                        <div class="attribute-item">
                            <label>Acceleration</label>
                            <input type="number" name="movement_acceleration" min="0" max="100" value="70" required>
                        </div>
                        <div class="attribute-item">
                            <label>Sprint Speed</label>
                            <input type="number" name="movement_sprint_speed" min="0" max="100" value="70" required>
                        </div>
                        <div class="attribute-item">
                            <label>Dribbling</label>
                            <input type="number" name="skill_dribbling" min="0" max="100" value="70" required>
                        </div>
                        <div class="attribute-item">
                            <label>Ball Control</label>
                            <input type="number" name="skill_ball_control" min="0" max="100" value="70" required>
                        </div>
                        <div class="attribute-item">
                            <label>Agility</label>
                            <input type="number" name="movement_agility" min="0" max="100" value="70" required>
                        </div>
                        <div class="attribute-item">
                            <label>Balance</label>
                            <input type="number" name="movement_balance" min="0" max="100" value="70" required>
                        </div>
                        <div class="attribute-item">
                            <label>Short Passing</label>
                            <input type="number" name="attacking_short_passing" min="0" max="100" value="70" required>
                        </div>
                        <div class="attribute-item">
                            <label>Long Passing</label>
                            <input type="number" name="skill_long_passing" min="0" max="100" value="60" required>
                        </div>
                        <div class="attribute-item">
                            <label>Vision</label>
                            <input type="number" name="mentality_vision" min="0" max="100" value="65" required>
                        </div>
                        <div class="attribute-item">
                            <label>Curve</label>
                            <input type="number" name="skill_curve" min="0" max="100" value="65" required>
                        </div>
                        <div class="attribute-item">
                            <label>Finishing</label>
                            <input type="number" name="attacking_finishing" min="0" max="100" value="70" required>
                        </div>
                        <div class="attribute-item">
                            <label>Shot Power</label>
                            <input type="number" name="power_shot_power" min="0" max="100" value="75" required>
                        </div>
                        <div class="attribute-item">
                            <label>Positioning</label>
                            <input type="number" name="mentality_positioning" min="0" max="100" value="70" required>
                        </div>
                        <div class="attribute-item">
                            <label>Interceptions</label>
                            <input type="number" name="mentality_interceptions" min="0" max="100" value="50" required>
                        </div>
                        <div class="attribute-item">
                            <label>Standing Tackle</label>
                            <input type="number" name="defending_standing_tackle" min="0" max="100" value="50" required>
                        </div>
                        <div class="attribute-item">
                            <label>Sliding Tackle</label>
                            <input type="number" name="defending_sliding_tackle" min="0" max="100" value="50" required>
                        </div>
                        <div class="attribute-item">
                            <label>Aggression</label>
                            <input type="number" name="mentality_aggression" min="0" max="100" value="60" required>
                        </div>
                        <div class="attribute-item">
                            <label>Strength</label>
                            <input type="number" name="power_strength" min="0" max="100" value="70" required>
                        </div>
                        <div class="attribute-item">
                            <label>Stamina</label>
                            <input type="number" name="power_stamina" min="0" max="100" value="75" required>
                        </div>
                        <div class="attribute-item">
                            <label>Jumping</label>
                            <input type="number" name="power_jumping" min="0" max="100" value="70" required>
                        </div>
                    </div>
                    
                    <button type="submit">🔮 Predict Player Style</button>
                </form>
                
                <div class="loading" id="predict-loading">
                    <div class="spinner"></div>
                    <p style="margin-top: 10px;">Analyzing player attributes...</p>
                </div>
                
                <div class="error" id="predict-error"></div>
                
                <div class="result" id="predict-result">
                    <h3>🎯 Prediction Result</h3>
                    <div class="result-content" id="predict-content"></div>
                </div>
            </div>
            
            <!-- Find Similar Tab -->
            <div id="similar-tab" class="tab-content">
                <div class="info-box">
                    <h4>Find Similar Players</h4>
                    <p>Enter a player's name to find players with similar playing styles based on their attributes.</p>
                </div>
                
                <form id="similar-form">
                    <div class="form-group">
                        <label>Player Name</label>
                        <input type="text" name="player_name" placeholder="e.g., Mbappé, Messi, Ronaldo" required>
                    </div>
                    
                    <div class="form-group">
                        <label>Number of Similar Players</label>
                        <input type="number" name="top_n" min="1" max="20" value="5" required>
                    </div>
                    
                    <button type="submit">🔍 Find Similar Players</button>
                </form>
                
                <div class="loading" id="similar-loading">
                    <div class="spinner"></div>
                    <p style="margin-top: 10px;">Searching for similar players...</p>
                </div>
                
                <div class="error" id="similar-error"></div>
                
                <div class="result" id="similar-result">
                    <h3>🔍 Similar Players Found</h3>
                    <div class="result-content" id="similar-content"></div>
                </div>
            </div>
            
            <!-- View Clusters Tab -->
            <div id="clusters-tab" class="tab-content">
                <div class="info-box">
                    <h4>Player Style Archetypes</h4>
                    <p>These are the 6 distinct playing styles discovered by the ML clustering algorithm.</p>
                </div>
                
                <div class="loading" id="clusters-loading">
                    <div class="spinner"></div>
                    <p style="margin-top: 10px;">Loading cluster information...</p>
                </div>
                
                <div class="result show" id="clusters-result" style="display: none;">
                    <div id="clusters-content"></div>
                </div>
            </div>
        </div>
        
        <div class="card" style="text-align: center; padding: 20px;">
            <p style="color: #666;">
                Built with ❤️ using FastAPI, scikit-learn, and KMeans Clustering<br>
                <a href="/docs" style="color: #667eea; text-decoration: none;">📚 API Documentation</a> | 
                <a href="/health" style="color: #667eea; text-decoration: none;">💚 Health Check</a>
            </p>
        </div>
    </div>
    
    <script>
        // Tab switching
        function switchTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab
            document.getElementById(tabName + '-tab').classList.add('active');
            event.target.classList.add('active');
            
            // Load clusters if clusters tab is selected
            if (tabName === 'clusters') {
                loadClusters();
            }
        }
        
        // Preset player profiles
        const presets = {
            winger: {
                movement_acceleration: 90,
                movement_sprint_speed: 92,
                skill_dribbling: 88,
                skill_ball_control: 85,
                movement_agility: 88,
                movement_balance: 82,
                attacking_short_passing: 75,
                skill_long_passing: 65,
                mentality_vision: 70,
                skill_curve: 72,
                attacking_finishing: 80,
                power_shot_power: 82,
                mentality_positioning: 85,
                mentality_interceptions: 35,
                defending_standing_tackle: 30,
                defending_sliding_tackle: 28,
                mentality_aggression: 50,
                power_strength: 65,
                power_stamina: 88,
                power_jumping: 70
            },
            playmaker: {
                movement_acceleration: 70,
                movement_sprint_speed: 68,
                skill_dribbling: 85,
                skill_ball_control: 90,
                movement_agility: 80,
                movement_balance: 75,
                attacking_short_passing: 92,
                skill_long_passing: 88,
                mentality_vision: 95,
                skill_curve: 85,
                attacking_finishing: 65,
                power_shot_power: 70,
                mentality_positioning: 75,
                mentality_interceptions: 65,
                defending_standing_tackle: 55,
                defending_sliding_tackle: 50,
                mentality_aggression: 60,
                power_strength: 60,
                power_stamina: 75,
                power_jumping: 60
            },
            defender: {
                movement_acceleration: 65,
                movement_sprint_speed: 70,
                skill_dribbling: 50,
                skill_ball_control: 60,
                movement_agility: 62,
                movement_balance: 70,
                attacking_short_passing: 65,
                skill_long_passing: 60,
                mentality_vision: 60,
                skill_curve: 50,
                attacking_finishing: 35,
                power_shot_power: 55,
                mentality_positioning: 70,
                mentality_interceptions: 88,
                defending_standing_tackle: 90,
                defending_sliding_tackle: 85,
                mentality_aggression: 82,
                power_strength: 85,
                power_stamina: 78,
                power_jumping: 88
            },
            striker: {
                movement_acceleration: 80,
                movement_sprint_speed: 85,
                skill_dribbling: 75,
                skill_ball_control: 78,
                movement_agility: 72,
                movement_balance: 70,
                attacking_short_passing: 70,
                skill_long_passing: 55,
                mentality_vision: 65,
                skill_curve: 68,
                attacking_finishing: 92,
                power_shot_power: 88,
                mentality_positioning: 90,
                mentality_interceptions: 30,
                defending_standing_tackle: 32,
                defending_sliding_tackle: 28,
                mentality_aggression: 65,
                power_strength: 80,
                power_stamina: 80,
                power_jumping: 85
            }
        };
        
        function loadPreset(type) {
            const preset = presets[type];
            const form = document.getElementById('predict-form');
            
            Object.keys(preset).forEach(key => {
                const input = form.querySelector(`input[name="${key}"]`);
                if (input) {
                    input.value = preset[key];
                }
            });
        }
        
        // Predict form submission
        document.getElementById('predict-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const data = {};
            formData.forEach((value, key) => {
                data[key] = parseFloat(value);
            });
            
            // Show loading
            document.getElementById('predict-loading').classList.add('show');
            document.getElementById('predict-result').classList.remove('show');
            document.getElementById('predict-error').classList.remove('show');
            
            try {
                const response = await fetch('/cluster', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });
                
                if (!response.ok) {
                    throw new Error('Prediction failed');
                }
                
                const result = await response.json();
                
                // Display result
                document.getElementById('predict-content').innerHTML = `
                    <p style="font-size: 1.2em;">This player is classified as:</p>
                    <div class="cluster-badge">
                        ${result.style}
                    </div>
                    <p style="margin-top: 20px; color: #666;">
                        <strong>Cluster ID:</strong> ${result.cluster_id}
                    </p>
                `;
                
                document.getElementById('predict-result').classList.add('show');
            } catch (error) {
                document.getElementById('predict-error').textContent = 
                    'Error: Unable to predict player style. Please try again.';
                document.getElementById('predict-error').classList.add('show');
            } finally {
                document.getElementById('predict-loading').classList.remove('show');
            }
        });
        
        // Similar players form submission
        document.getElementById('similar-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const data = {
                player_name: formData.get('player_name'),
                top_n: parseInt(formData.get('top_n'))
            };
            
            // Show loading
            document.getElementById('similar-loading').classList.add('show');
            document.getElementById('similar-result').classList.remove('show');
            document.getElementById('similar-error').classList.remove('show');
            
            try {
                const response = await fetch('/similar_players', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });
                
                if (!response.ok) {
                    throw new Error('Search failed');
                }
                
                const result = await response.json();
                
                // Display result
                let html = `<p style="font-size: 1.1em; margin-bottom: 15px;">
                    Players similar to <strong>${data.player_name}</strong>:
                </p>
                <ul class="player-list">`;
                
                result.similar_players.forEach((player, index) => {
                    html += `<li>🎯 ${index + 1}. ${player}</li>`;
                });
                
                html += '</ul>';
                
                document.getElementById('similar-content').innerHTML = html;
                document.getElementById('similar-result').classList.add('show');
            } catch (error) {
                document.getElementById('similar-error').textContent = 
                    'Error: Player not found or search failed. Please check the player name and try again.';
                document.getElementById('similar-error').classList.add('show');
            } finally {
                document.getElementById('similar-loading').classList.remove('show');
            }
        });
        
        // Load clusters
        async function loadClusters() {
            const loading = document.getElementById('clusters-loading');
            const result = document.getElementById('clusters-result');
            const content = document.getElementById('clusters-content');
            
            loading.classList.add('show');
            
            try {
                const response = await fetch('/clusters');
                if (!response.ok) {
                    throw new Error('Failed to load clusters');
                }
                
                const clusters = await response.json();
                
                let html = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">';
                
                const descriptions = {
                    "Creative Playmaker": "High creativity, vision, and passing. Masters of orchestrating attacks.",
                    "Ball Winning Midfielder": "Defensive specialists with high interceptions and tackling.",
                    "Explosive Winger": "Speed demons with exceptional pace and dribbling ability.",
                    "Target Man": "Physical strikers who dominate in the air and hold up play.",
                    "Defensive Center Back": "Defensive rocks with strength and positioning.",
                    "Box-to-Box Midfielder": "Balanced all-rounders who excel in all areas."
                };
                
                const icons = {
                    "Creative Playmaker": "🎨",
                    "Ball Winning Midfielder": "🛡️",
                    "Explosive Winger": "⚡",
                    "Target Man": "🎯",
                    "Defensive Center Back": "🏰",
                    "Box-to-Box Midfielder": "⚙️"
                };
                
                Object.entries(clusters).forEach(([id, name]) => {
                    html += `
                        <div style="background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);">
                            <div style="font-size: 3em; margin-bottom: 10px;">${icons[name] || '⚽'}</div>
                            <h4 style="color: #667eea; margin-bottom: 10px; font-size: 1.3em;">${name}</h4>
                            <p style="color: #666; line-height: 1.6;">${descriptions[name] || 'Unique playing style'}</p>
                            <div style="margin-top: 15px; padding-top: 15px; border-top: 2px solid #f0f0f0;">
                                <span style="color: #999; font-size: 0.9em;">Cluster ID: ${id}</span>
                            </div>
                        </div>
                    `;
                });
                
                html += '</div>';
                
                content.innerHTML = html;
                result.style.display = 'block';
            } catch (error) {
                content.innerHTML = '<p style="color: #c33;">Error loading cluster information.</p>';
                result.style.display = 'block';
            } finally {
                loading.classList.remove('show');
            }
        }
    </script>
</body>
</html>
"""


# ============================================================================
# STARTUP & ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def load_models():
    """Load model artifacts on startup"""
    global scaler, kmeans, cluster_labels, player_data
    
    try:
        print("🔄 Loading model artifacts...")
        
        scaler = joblib.load('models/scaler.pkl')
        print("  ✓ Loaded scaler")
        
        kmeans = joblib.load('models/kmeans.pkl')
        print("  ✓ Loaded KMeans model")
        
        with open('models/cluster_labels.json', 'r') as f:
            cluster_labels = json.load(f)
        print("  ✓ Loaded cluster labels")
        
        # Load player data for similarity search
        player_data = pd.read_csv('FC26.csv')
        print(f"  ✓ Loaded {len(player_data):,} players from FC26.csv")
        
        print("✅ All models loaded successfully")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Model files not found - {e}")
        print("   Please run 'python train_model.py' first!")
        raise
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web UI"""
    return HTML_TEMPLATE


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "models_loaded": scaler is not None and kmeans is not None,
        "total_players": len(player_data) if player_data is not None else 0
    }


@app.get("/clusters")
async def get_clusters() -> Dict[str, str]:
    """List all player style clusters"""
    return cluster_labels


@app.post("/cluster", response_model=PlayerPrediction)
async def predict_cluster(attributes: PlayerAttributes):
    """
    Predict player style cluster from attributes
    
    Input: FIFA player attributes
    Output: Cluster ID and style label
    """
    try:
        # Convert to style dimensions
        features = create_style_dimensions(attributes.dict())
        
        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Predict cluster
        cluster_id = int(kmeans.predict(features_scaled)[0])
        style = cluster_labels[str(cluster_id)]
        
        return PlayerPrediction(
            cluster_id=cluster_id,
            style=style
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/similar_players", response_model=SimilarPlayerResponse)
async def find_similar_players(request: SimilarPlayerRequest):
    """
    Find players with similar playing style
    
    Input: Player name and number of similar players
    Output: List of similar player names
    """
    try:
        # Try multiple name columns
        name_columns = ['short_name', 'name', 'long_name', 'player_name']
        player_mask = None
        name_col_used = None
        
        for name_col in name_columns:
            if name_col in player_data.columns:
                player_mask = player_data[name_col].str.contains(
                    request.player_name, case=False, na=False
                )
                if player_mask.any():
                    name_col_used = name_col
                    break
        
        if player_mask is None or not player_mask.any():
            raise HTTPException(
                status_code=404, 
                detail=f"Player '{request.player_name}' not found in dataset"
            )
        
        # Get player features
        player_row = player_data[player_mask].iloc[0]
        player_features = create_style_dimensions(player_row.to_dict())
        player_scaled = scaler.transform(player_features.reshape(1, -1))
        
        # Calculate all player features
        all_features = create_style_dimensions_batch(player_data)
        all_scaled = scaler.transform(all_features)
        
        # Calculate distances
        distances = np.linalg.norm(all_scaled - player_scaled, axis=1)
        
        # Get top N similar (excluding the player itself)
        similar_indices = np.argsort(distances)[1:request.top_n + 1]
        similar_names = player_data.iloc[similar_indices][name_col_used].tolist()
        
        return SimilarPlayerResponse(similar_players=similar_names)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding similar players: {str(e)}")


@app.get("/player/{player_name}")
async def get_player_profile(player_name: str):
    """Get detailed player profile including cluster assignment"""
    try:
        # Try multiple name columns
        name_columns = ['short_name', 'name', 'long_name', 'player_name']
        player_mask = None
        name_col_used = None
        
        for name_col in name_columns:
            if name_col in player_data.columns:
                player_mask = player_data[name_col].str.contains(
                    player_name, case=False, na=False
                )
                if player_mask.any():
                    name_col_used = name_col
                    break
        
        if player_mask is None or not player_mask.any():
            raise HTTPException(
                status_code=404,
                detail=f"Player '{player_name}' not found"
            )
        
        player = player_data[player_mask].iloc[0]
        
        # Get cluster prediction
        features = create_style_dimensions(player.to_dict())
        features_scaled = scaler.transform(features.reshape(1, -1))
        cluster_id = int(kmeans.predict(features_scaled)[0])
        
        return {
            "name": player[name_col_used],
            "age": int(player['age']) if 'age' in player else None,
            "overall": int(player['overall']) if 'overall' in player else None,
            "positions": player['player_positions'] if 'player_positions' in player else 
                        (player['positions'] if 'positions' in player else None),
            "cluster_id": cluster_id,
            "style": cluster_labels[str(cluster_id)]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching player: {str(e)}")