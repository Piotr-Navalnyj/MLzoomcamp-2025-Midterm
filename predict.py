import pandas as pd
import numpy as np
from xgboost import XGBRegressor

#
# 1. Load trained XGBoost model
# 
model = XGBRegressor()
model.load_model("xgb_model.json")
print("Model loaded successfully!")


# 2. Load dataset to reconstruct preprocessing
# (We need the same dummy columns!)

df = pd.read_csv(
    r"Data.csv",
    sep=';',
    encoding='latin1'
)

# Keep consistent columns
keep_cols = [
    'age','height_cm','weight_kg','overall','potential','value_eur','wage_eur',
    'international_reputation','weak_foot','skill_moves','pace','shooting',
    'passing','dribbling','defending','physic','player_positions'
]
df = df[keep_cols].copy()

# Clean missing values
df = df.fillna(0)

# Extract main position
df["main_position"] = df["player_positions"].str.split(",").str[0].str.strip()
df = df.drop("player_positions", axis=1)

# One-hot all positions in training dataset
df_dummies = pd.get_dummies(df.drop("value_eur", axis=1), columns=["main_position"], drop_first=True)

# Save the column structure (important!)
feature_columns = df_dummies.columns.tolist()



# 3. Example player to predict
# You can replace ALL values below with your own input


new_player = {
    "age": 25,
    "height_cm": 180,
    "weight_kg": 75,
    "overall": 82,
    "potential": 88,
    "wage_eur": 50000,
    "international_reputation": 3,
    "weak_foot": 4,
    "skill_moves": 4,
    "pace": 85,
    "shooting": 78,
    "passing": 80,
    "dribbling": 83,
    "defending": 55,
    "physic": 70,
    "main_position": "ST"
}

new_df = pd.DataFrame([new_player])


# 4. Apply SAME preprocessing to new player


# One-hot encode the position
new_df = pd.get_dummies(new_df, columns=["main_position"], drop_first=True)

# Ensure all missing dummy columns exist
for col in feature_columns:
    if col not in new_df.columns:
        new_df[col] = 0

# Reorder columns exactly like training X
new_df = new_df[feature_columns]

# Convert to numpy
X_new = new_df.values



# 5. Predict value


predicted_value = model.predict(X_new)[0]

print(f"\nPredicted player market value: €{predicted_value:,.0f}\n")
