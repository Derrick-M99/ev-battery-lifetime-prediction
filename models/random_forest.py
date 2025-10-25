import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# Load and Clean Data
battery_files = [
    "battery00.csv", "battery01.csv", "battery10.csv", "battery11.csv",
    "battery20.csv", "battery21.csv", "battery22.csv", "battery23.csv",
    "battery30.csv", "battery31.csv"
]

battery_data = []

for battery_id, file in enumerate(battery_files):
    if os.path.exists(file):
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

all_data = pd.concat(battery_data, ignore_index=True)

# Correlation Heatmap
all_data['start_time_numeric'] = (
    all_data['start_time'] - all_data['start_time'].min()
).dt.total_seconds()

heatmap_cols = ['start_time_numeric', 'time', 'mode', 'voltage_charger', 'temperature_battery']
heatmap_df = all_data[heatmap_cols].dropna()

plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Correlation Heatmap of Battery Features")
plt.tight_layout()
plt.show()

# SDischarge Cycle Segmentation
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
    first_voltage = group['voltage_load'].iloc[0]
    last_voltage = group['voltage_load'].iloc[-1]
    if abs(first_voltage - last_voltage) < 0.2:
        return np.nan
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

    cycle_data.append({
        'battery_id': battery_id,
        'cycle_id': cycle_id,
        'capacity': capacity,
        'tiedvd': tiedvd,
        'thermal_influence': delta_temp,
        'temperature_mean': temp_mean,
        'duration_sec': duration
    })

cycle_summary_df = pd.DataFrame(cycle_data)

# SoH calculation
cycle_summary_df['soh'] = cycle_summary_df.groupby('battery_id')['capacity'].transform(
    lambda x: (x / x.max()) * 100
)

print("\nCycles per battery BEFORE filtering:")
print(cycle_summary_df.groupby('battery_id').size())

print("\nDescriptive statistics of cycle features (before filtering):")
print(cycle_summary_df[['capacity', 'thermal_influence', 'tiedvd', 'duration_sec']].describe())


# STEP 4: Filtering 
filtered_df = cycle_summary_df[
    (cycle_summary_df['capacity'] > 0.01) &
    (cycle_summary_df['thermal_influence'].between(-10, 120)) &
    (cycle_summary_df['tiedvd'].between(0, 1000)) &
    (cycle_summary_df['duration_sec'] >= 10)
].copy()


# Add cycle number 
filtered_df['cycle_number'] = filtered_df.groupby('battery_id').cumcount()

# First figure: capacity and thermal_influence
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 8))
fig1.suptitle("Feature Distributions", fontsize=16)

features_part1 = ['capacity', 'thermal_influence']

for i, feature in enumerate(features_part1):
    axes1[i, 0].hist(cycle_summary_df[feature].dropna(), bins=50, edgecolor='black', color='skyblue')
    axes1[i, 0].set_title(f'{feature} (Before Filtering)')
    axes1[i, 0].set_xlabel(feature)
    axes1[i, 0].set_ylabel('Frequency')
    axes1[i, 0].grid(True, alpha=0.3)

    axes1[i, 1].hist(filtered_df[feature].dropna(), bins=50, edgecolor='black', color='salmon')
    axes1[i, 1].set_title(f'{feature} (After Filtering)')
    axes1[i, 1].set_xlabel(feature)
    axes1[i, 1].set_ylabel('Frequency')
    axes1[i, 1].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Second figure: tiedvd and duration_sec
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8))
fig2.suptitle("Feature Distributions", fontsize=16)

features_part2 = ['tiedvd', 'duration_sec']

for i, feature in enumerate(features_part2):
    axes2[i, 0].hist(cycle_summary_df[feature].dropna(), bins=50, edgecolor='black', color='skyblue')
    axes2[i, 0].set_title(f'{feature} (Before Filtering)')
    axes2[i, 0].set_xlabel(feature)
    axes2[i, 0].set_ylabel('Frequency')
    axes2[i, 0].grid(True, alpha=0.3)

    axes2[i, 1].hist(filtered_df[feature].dropna(), bins=50, edgecolor='black', color='salmon')
    axes2[i, 1].set_title(f'{feature} (After Filtering)')
    axes2[i, 1].set_xlabel(feature)
    axes2[i, 1].set_ylabel('Frequency')
    axes2[i, 1].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Plot Trends with Clean 20-Cycle Averages 

colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
marker_style = '*'  # marker to be used in all plots

# Create 20-cycle bins
filtered_df['cycle_bin'] = (filtered_df['cycle_number'] // 20) * 20

# Count entries per bin and filter only full bins 
bin_counts = filtered_df.groupby(['battery_id', 'cycle_bin']).size().reset_index(name='count')
valid_bins = bin_counts[bin_counts['count'] == 20][['battery_id', 'cycle_bin']]

# Join with original to filter out incomplete bins
filtered_df = filtered_df.merge(valid_bins, on=['battery_id', 'cycle_bin'])

# Now compute binned averages
binned_df = filtered_df.groupby(['battery_id', 'cycle_bin'])[
    ['capacity', 'soh', 'thermal_influence', 'tiedvd']
].mean().reset_index()


fig1, axes1 = plt.subplots(1, 2, figsize=(18, 6))

for i, (battery_id, group) in enumerate(binned_df.groupby('battery_id')):
    color = colors[i % len(colors)]
    x = group['cycle_bin']

    axes1[0].plot(x, group['capacity'], label=f'Battery {battery_id}', color=color, marker=marker_style)
    axes1[1].plot(x, group['soh'], label=f'Battery {battery_id}', color=color, marker=marker_style)

axes1[0].set_title('Capacity vs Cycle Number')
axes1[0].set_xlabel('Cycle Number')
axes1[0].set_ylabel('Estimated Capacity (Ah)')
axes1[0].grid(True, alpha=0.3)

axes1[1].set_title('State of Health (SoH) vs Cycle Number')
axes1[1].set_xlabel('Cycle Number')
axes1[1].set_ylabel('SoH (%)')
axes1[1].grid(True, alpha=0.3)

axes1[1].legend(title='Battery ID', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 0.95, 1])
plt.show()


fig2, axes2 = plt.subplots(1, 2, figsize=(18, 6))

for i, (battery_id, group) in enumerate(binned_df.groupby('battery_id')):
    color = colors[i % len(colors)]
    x = group['cycle_bin']

    axes2[0].plot(x, group['thermal_influence'], label=f'Battery {battery_id}', color=color, marker=marker_style)
    axes2[1].plot(x, group['tiedvd'], label=f'Battery {battery_id}', color=color, marker=marker_style)

axes2[0].set_title('Thermal Influence (ΔT) vs Cycle Number')
axes2[0].set_xlabel('Cycle Number')
axes2[0].set_ylabel('ΔT (°C)')
axes2[0].grid(True, alpha=0.3)

axes2[1].set_title('TIEDVD vs Cycle Number')
axes2[1].set_xlabel('Cycle Number')
axes2[1].set_ylabel('TIEDVD (seconds)')
axes2[1].grid(True, alpha=0.3)

axes2[1].legend(title='Battery ID', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 0.95, 1])
plt.show()

import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree = None

    def fit(self, X, y):
        self.features_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.tree = self._build_tree(X, y, depth=0)

    def _gini(self, y):
        m = len(y)
        if m == 0:
            return 0
        counts = Counter(y)
        return 1 - sum((n / m) ** 2 for n in counts.values())

    def _best_split(self, X, y):
        best_gini = 1.0
        best_idx, best_val = None, None
        m, n = X.shape
        features = np.random.choice(n, self.max_features or n, replace=False)

        for idx in features:
            thresholds = np.unique(X[:, idx])
            for val in thresholds:
                left = y[X[:, idx] <= val]
                right = y[X[:, idx] > val]
                if len(left) == 0 or len(right) == 0:
                    continue
                gini = (len(left) / m) * self._gini(left) + (len(right) / m) * self._gini(right)
                if gini < best_gini:
                    best_gini, best_idx, best_val = gini, idx, val

        return best_idx, best_val

    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(set(y)) == 1:
            return Counter(y).most_common(1)[0][0]

        idx, val = self._best_split(X, y)
        if idx is None:
            return Counter(y).most_common(1)[0][0]

        left_mask = X[:, idx] <= val
        right_mask = X[:, idx] > val

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return (idx, val, left_subtree, right_subtree)

    def _predict_one(self, x, node):
        if not isinstance(node, tuple):
            return node
        idx, val, left, right = node
        return self._predict_one(x, left if x[idx] <= val else right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

class RandomForest:
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_features=self.max_features)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.array([Counter(tree_preds[:, i]).most_common(1)[0][0] for i in range(X.shape[0])])

    def predict_proba(self, X, positive_class='Healthy'):
        # probability is the proportion of trees that predict the positive class
        preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(preds == positive_class, axis=0)

# Create Health Labels from filtered_df
filtered_df['health_label'] = np.where(filtered_df['soh'] >= 85, 'Healthy', 'Degraded')

# Select Features and Target
features = ['capacity', 'thermal_influence', 'tiedvd']
X = filtered_df[features].values
y = filtered_df['health_label'].values

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train 
rf_model = RandomForest(n_estimators=10, max_depth=8, max_features=2)
rf_model.fit(X_train, y_train)

# Predictions
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)
y_scores = rf_model.predict_proba(X_test)  

from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import LabelBinarizer

# Accuracy
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

# Classification Metrics
recall = recall_score(y_test, y_test_pred, pos_label='Healthy')
precision = precision_score(y_test, y_test_pred, pos_label='Healthy')
f1 = f1_score(y_test, y_test_pred, pos_label='Healthy')
cm = confusion_matrix(y_test, y_test_pred)

print("\nClassification Metrics (Random Forest):")
print(f"Training Accuracy: {train_acc * 100:.2f}%")
print(f"Testing Accuracy:  {test_acc * 100:.2f}%")
print(f"Recall:            {recall * 100:.2f}%")
print(f"Precision:         {precision * 100:.2f}%")
print(f"F1 Score:          {f1 * 100:.2f}%")

# Confusion Matrix Plot
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.show()

# Binarize labels for ROC/AUC, PR curves
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test_bin, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve - Random Forest')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test_bin, y_scores)
avg_precision = average_precision_score(y_test_bin, y_scores)

plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, color='green', lw=2,
         label=f'Avg Precision = {avg_precision:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Random Forest')
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# F1 Score vs Threshold
thresholds_f1 = np.linspace(0.0, 1.0, 100)
f1_scores = []

for thresh in thresholds_f1:
    preds_thresh = (y_scores >= thresh).astype(int)
    f1_thresh = f1_score(y_test_bin, preds_thresh)
    f1_scores.append(f1_thresh)

plt.figure(figsize=(8, 6))
plt.plot(thresholds_f1, f1_scores, color='purple', lw=2)
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.title("F1 Score vs Classification Threshold - Random Forest")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
