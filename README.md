### Project Overview
This project solves the subjectivity in traditional football scouting by providing an objective, data-driven player classification systemn by analyzing over 18,000 players from the FC 26 dataset.

🤖 ML Engine: KMeans Clustering (k=6) trained on engineered style dimensions.

📊 Data Scale: 18,000+ unique player profiles.

⚡ Performance: <100ms API response time.

🎨 User Experience: Built-in interactive Web UI (no need for /docs).


### Features
## 🎯 Style Prediction & Search
Instant Classification: Input 20 player attributes to get an immediate archetype.

Similarity Engine: Find the "Top N" similar players using Euclidean distance in the style space.

Archetype Dictionary: Detailed breakdown of all 6 styles (e.g., Creative Playmaker, Target Man, etc.).

## 📊 The 6 Player Archetypes
Creative Playmaker: High vision, passing, and technical curve.

Ball Winning Midfielder: Defensive specialists focused on interceptions.

Explosive Winger: Pure pace, agility, and dribbling focus.

Target Man: Physical strikers with high strength and positioning.

Defensive Center Back: Dominant in tackling and physical presence.

Box-to-Box Midfielder: Highly balanced all-rounders.

## METHODOLOGY

<img width="278" height="176" alt="image" src="https://github.com/user-attachments/assets/04ba5798-b1df-4caf-81bf-c71349a50dee" />

### 🚀 Quick Start1. Installation 
# Clone the repository
git clone https://github.com/AhrnishDahal/FC26-player-clustering-api.git
cd FC26-player-clustering-api

1. Run explore_data.py for basic overview of complete csv data
2. Run the Pipeline
3. Train the model (creates /models folder)
python train_model.py

# Start the API
uvicorn api:app --reload
Visit: http://localhost:8000


# SCREEN FOR SIMILAR PLAYERS
<img width="458" height="424" alt="screen" src="https://github.com/user-attachments/assets/44192121-aeab-4a63-b372-48542697f687" />

# SCREEN FOR STYLE OF PLAYER BY ATTRIBUTES
<img width="472" height="485" alt="screenstyle" src="https://github.com/user-attachments/assets/4c4c668d-fbad-48c4-9719-35ca87867ab1" />

# SCREEN FOR SEEING STYLES CLASSIFICATION
<img width="472" height="423" alt="styles" src="https://github.com/user-attachments/assets/7cf9a00f-6b77-49dc-b80f-7353732dcd1a" />





--Developed by AhrnishDahal


If you like this project, please give it a ⭐!
