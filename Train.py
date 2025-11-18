import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# 1. LOAD DATA

df = pd.read_csv(
    r"Data.csv",
    sep=';',
    encoding='latin1'
)


keep_cols = [
    'age', 'height_cm', 'weight_kg', 'overall', 'potential', 'value_eur', 'wage_eur',
    'international_reputation', 'weak_foot', 'skill_moves', 'pace', 'shooting',
    'passing', 'dribbling', 'defending', 'physic', 'player_positions'
]
df = df[keep_cols].copy()


df = df.fillna(0)


df["main_position"] = df["player_positions"].str.split(",").str[0].str.strip()
df = df.drop("player_positions", axis=1)


df = pd.get_dummies(df, columns=["main_position"], drop_first=True)


df = df.astype({col: int for col in df.select_dtypes(include=["bool"]).columns})



# 2. TRAIN / VAL / TEST SPLIT (60% / 20% / 20%)
=
n = len(df)
np.random.seed(42)
idx = np.random.permutation(n)

n_train = int(0.6 * n)
n_val   = int(0.2 * n)
n_test  = n - n_train - n_val

train_idx = idx[:n_train]
val_idx   = idx[n_train:n_train + n_val]
test_idx  = idx[n_train + n_val:]

df_train = df.iloc[train_idx].reset_index(drop=True)
df_val   = df.iloc[val_idx].reset_index(drop=True)
df_test  = df.iloc[test_idx].reset_index(drop=True)



# 3. FEATURES + TARGET

y_train = df_train["value_eur"].values
y_val   = df_val["value_eur"].values
y_test  = df_test["value_eur"].values

X_train = df_train.drop("value_eur", axis=1).values
X_val   = df_val.drop("value_eur", axis=1).values
X_test  = df_test.drop("value_eur", axis=1).values



# 4. TRAIN XGBOOST

xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

print("\nTraining XGBoost...")
xgb.fit(X_train, y_train)

y_pred_val = xgb.predict(X_val)
y_pred_test = xgb.predict(X_test)



# 5. METRICS

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


mae_val, rmse_val, r2_val = evaluate(y_val, y_pred_val)
mae_test, rmse_test, r2_test = evaluate(y_test, y_pred_test)



# 6. PRINT RESULTS

print("\n===== XGBOOST VALIDATION =====")
print("RMSE:", rmse_val)
print("MAE:", mae_val)
print("R² :", r2_val)

print("\n===== XGBOOST TEST =====")
print("RMSE:", rmse_test)
print("MAE:", mae_test)
print("R² :", r2_test)

xgb.save_model("xgb_model.json")
print("\nModel saved as xgb_model.json")
