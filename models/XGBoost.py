# ... [PREVIOUS IMPORTS AND CONFIGS] ...
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# STEP 1: Load and Clean Data
battery_files = [
    "battery00.csv", "battery01.csv", "battery10.csv", "battery11.csv",
    "battery20.csv", "battery21.csv", "battery22.csv", "battery23.csv",
    "battery30.csv", "battery31.csv"
]

battery_data = []

for battery_id, file in enumerate(battery_files):
    if os.path.exists(file):
        print(f"[INFO] Found and loading: {file}")
        df = pd.read_csv(file, on_bad_lines='skip', low_memory=False)

        df['battery_id'] = battery_id
        df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
        df['time'] = pd.to_numeric(df['time'], errors='coerce')
        df['mode'] = pd.to_numeric(df['mode'], errors='coerce')
        df['voltage_charger'] = pd.to_numeric(df['voltage_charger'], errors='coerce')
        df['temperature_battery'] = pd.to_numeric(df['temperature_battery'], errors='coerce')
        df['current_load'] = pd.to_numeric(df.get('current_load'), errors='coerce')
        df['voltage_load'] = pd.to_numeric(df.get('voltage_load'), errors='coerce')

        df = df.dropna(subset=['start_time', 'time', 'mode', 'voltage_charger', 'temperature_battery'])
        df['actual_time'] = df['start_time'] + pd.to_timedelta(df['time'], unit='s')
        battery_data.append(df)
    else:
        print(f"[WARNING] File not found: {file}")

all_data = pd.concat(battery_data, ignore_index=True)

# STEP 2: Correlation Heatmap
all_data['start_time_numeric'] = (
    all_data['start_time'] - all_data['start_time'].min()
).dt.total_seconds()

heatmap_cols = ['start_time_numeric', 'time', 'mode', 'voltage_charger', 'temperature_battery']
plt.figure(figsize=(8, 6))
sns.heatmap(all_data[heatmap_cols].dropna().corr(), annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Correlation Heatmap of Battery Features")
plt.tight_layout()
plt.savefig("Correlation Heatmap of Battery Features")

# STEP 3: Discharge Segmentation + Feature Engineering
discharge_df = all_data[all_data['mode'] == -1].copy()
discharge_df = discharge_df.sort_values(['battery_id', 'actual_time'])
discharge_df['time_gap'] = discharge_df.groupby('battery_id')['actual_time'].diff().dt.total_seconds()
discharge_df['cycle_change'] = (discharge_df['time_gap'] > 15) | (discharge_df['time_gap'].isna())
discharge_df['cycle_id'] = discharge_df.groupby('battery_id')['cycle_change'].cumsum()

def estimate_capacity(group):
    dt = group['actual_time'].diff().dt.total_seconds().fillna(0)
    return (group['current_load'] * dt).sum() / 3600

def calculate_tiedvd(group):
    if group['voltage_load'].isna().all():
        return np.nan
    group = group.sort_values('actual_time')
    return (group['actual_time'].iloc[-1] - group['actual_time'].iloc[0]).total_seconds()

def calculate_thermal_influence(group):
    return group['temperature_battery'].max() - group['temperature_battery'].min()

cycle_data = []
for (battery_id, cycle_id), group in discharge_df.groupby(['battery_id', 'cycle_id']):
    group = group.sort_values('actual_time')
    if group.shape[0] < 5:
        continue

    capacity = estimate_capacity(group)
    tiedvd = calculate_tiedvd(group)
    delta_temp = calculate_thermal_influence(group)
    temp_mean = group['temperature_battery'].mean()
    duration = (group['actual_time'].max() - group['actual_time'].min()).total_seconds()
    voltage_start = group['voltage_load'].iloc[0]
    voltage_end = group['voltage_load'].iloc[-1]
    voltage_range = voltage_start - voltage_end

    cycle_data.append({
        'battery_id': battery_id,
        'cycle_id': cycle_id,
        'capacity': capacity,
        'tiedvd': tiedvd,
        'thermal_influence': delta_temp,
        'temperature_mean': temp_mean,
        'duration_sec': duration,
        'voltage_range': voltage_range
    })

cycle_summary_df = pd.DataFrame(cycle_data)

# SoH calculation
cycle_summary_df['soh'] = cycle_summary_df.groupby('battery_id')['capacity'].transform(
    lambda x: (x / x.max()) * 100
)

# Filter
filtered_df = cycle_summary_df[
    (cycle_summary_df['capacity'] > 0.01) &
    (cycle_summary_df['thermal_influence'].between(-10, 120)) &
    (cycle_summary_df['tiedvd'].between(0, 1000)) &
    (cycle_summary_df['duration_sec'] >= 10)
].copy()

filtered_df['cycle_number'] = filtered_df.groupby('battery_id').cumcount()

# degradation samples

synthetic_degraded = pd.DataFrame({
    'capacity': np.linspace(0.6, 0.4, 50),
    'thermal_influence': np.linspace(80, 110, 50),
    'tiedvd': np.linspace(300, 50, 50),
    'voltage_range': np.linspace(0.6, 0.3, 50),
    'soh': np.linspace(85, 45, 50)
})

features = ['capacity', 'thermal_influence', 'tiedvd', 'voltage_range']
training_df = pd.concat([filtered_df[features + ['soh']], synthetic_degraded], ignore_index=True)

# Train XGBoost Model
X = training_df[features].values
y = training_df['soh'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("\nXGBoost Regression Results")
print(f"MAE:   {mae:.4f}")
print(f"MSE:   {mse:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"MAPE:  {mape:.4f}")
print("\nTraining SoH Range:")
print(f"Min: {training_df['soh'].min():.2f}%")
print(f"Max: {training_df['soh'].max():.2f}%")

# Plot: Actual vs Predicted SoH
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.4)
plt.xlabel("Actual SOH (%)")
plt.ylabel("Predicted SOH (%)")
plt.title("XGBoost Regression: Actual vs Predicted SOH")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Actual_vs_Predicted_SoH")

# Residual Visualization
sorted_idx = np.argsort(y_pred)
plt.figure(figsize=(10, 6))
plt.plot(y_pred[sorted_idx], y_pred[sorted_idx], 'r-', label='Predicted Value')
plt.scatter(y_pred[sorted_idx], y_test[sorted_idx], color='blue', label='Actual Value', zorder=3)
for i in range(len(sorted_idx)):
    plt.plot([y_pred[sorted_idx][i], y_pred[sorted_idx][i]], [y_pred[sorted_idx][i], y_test[sorted_idx][i]], 'k-', alpha=0.4)
plt.xlabel("Predicted SOH")
plt.ylabel("Actual SOH")
plt.title("Prediction Error Visualisation")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Error_Visualisation")

# Simulated Tesla Scenario
print("\nSimulated Tesla Usage Scenario")
n_cycles = 1500
sim_df = pd.DataFrame({
    'cycle_number': np.arange(n_cycles),
    'capacity': np.linspace(1.0, 0.2, n_cycles),
    'thermal_influence': np.linspace(40, 120, n_cycles),
    'tiedvd': np.linspace(900, 20, n_cycles),
    'voltage_range': np.linspace(1.7, 0.2, n_cycles)
})

sim_df_scaled = scaler.transform(sim_df[features])
sim_df['predicted_soh'] = model.predict(sim_df_scaled)

second_life_point = sim_df[sim_df['predicted_soh'] < 70].head(1)
if not second_life_point.empty:
    cycle = int(second_life_point['cycle_number'].values[0])
    days = cycle * 10.3
    years = days / 365
    print(f"\nSecond Life Reached At:\nCycle - {cycle}\nDay {days:.0f} ({years:.1f} years)")
else:
    print("\nSecond life threshold not reached in simulation.")

print("\nSample simulated SoH predictions:")
print(sim_df[['cycle_number', 'capacity', 'voltage_range', 'thermal_influence', 'tiedvd', 'predicted_soh']].head(10))
print(sim_df[['cycle_number', 'predicted_soh']].tail(10))

# Simulated SoH over Time
plt.figure(figsize=(10, 6))
plt.plot(sim_df['cycle_number'], sim_df['predicted_soh'], label="Predicted SoH", color='purple')
plt.axhline(70, color='red', linestyle='--', label="Second Life Threshold")
plt.xlabel("Cycle Number")
plt.ylabel("Predicted SoH (%)")
plt.title("Simulated Tesla Usage: SoH over Time")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Simulated_Tesla_SoH_Over_Time")

# Final model output 
def run_model():
    return {
        "model_name": "XGBoost",
        "y_true": y_test.tolist(),
        "y_pred": y_pred.tolist()
    }
