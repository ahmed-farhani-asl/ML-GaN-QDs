import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import numpy as np

# Load data
full_data = pd.read_excel('input/dataset.xlsx')  # Load full dataset

# Remove rows where Bandgap is 0
filtered_data = full_data[full_data['Bandgap (ev)'] != 0]

# Drop unnecessary columns in the program (not modifying original dataset)
unwanted_columns = [
    'QDs', 'Ga Source', 'N Source', 'Wavelength (nm)', 'Solvent_mmol', 'GaN Volume fraction', 'height / diameter ratio', 'Width (nm)', 'Aspect ratio (Height / Width)', 'water / surfactant molar ratio', 'Solvent', 'Buffer', 'Surface', 'Characterization', 'Citation', 'DOI'
]
filtered_data = filtered_data.drop(columns=unwanted_columns)

# Extract features (X) and target (y)
X = filtered_data.drop(columns=['Bandgap (ev)'])
y = filtered_data['Bandgap (ev)']

# Split data (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
print(len(y_test))

# Feature selection
selector = SelectKBest(score_func=f_regression, k='all')
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(n_estimators=100),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100)
}

# Train and evaluate models
results = {}
predictions_store = {}
feature_importance_data = {}

def evaluate_model(name, mse, r2, adjusted_r2, dataset_type):
    accuracy_desc = "Acceptable" if r2 > 0.7 and adjusted_r2 > 0.7 and mse < 1.0 else "Not Acceptable"
    print(f"{name} ({dataset_type}): MSE={mse:.4f}, R²={r2:.4f}, Adjusted R²={adjusted_r2:.4f} ({accuracy_desc})")
    return accuracy_desc

optimal_params_all_models = {}

for name, model in models.items():
    model.fit(X_train_selected, y_train)
    
    # Training metrics
    train_predictions = model.predict(X_train_selected)
    train_mse = mean_squared_error(y_train, train_predictions)
    train_r2 = r2_score(y_train, train_predictions)
    train_adjusted_r2 = 1 - (1 - train_r2) * (len(y_train) - 1) / (len(y_train) - X_train_selected.shape[1] - 1)
    accuracy_desc_train = evaluate_model(name, train_mse, train_r2, train_adjusted_r2, "Training")
    
    # Testing metrics
    test_predictions = model.predict(X_test_selected)
    predictions_store[name] = test_predictions
    test_mse = mean_squared_error(y_test, test_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    test_adjusted_r2 = 1 - (1 - test_r2) * (len(y_test) - 1) / (len(y_test) - X_test_selected.shape[1] - 1)
    accuracy_desc_test = evaluate_model(name, test_mse, test_r2, test_adjusted_r2, "Testing")
    
    results[name] = {
        'Train MSE': train_mse, 'Train R²': train_r2, 'Train Adjusted R²': train_adjusted_r2,
        'Train Accuracy': accuracy_desc_train,
        'Test MSE': test_mse, 'Test R²': test_r2, 'Test Adjusted R²': test_adjusted_r2,
        'Test Accuracy': accuracy_desc_test
    }

    # Predicting optimal conditions for desired bandgap ranges
    predicted_bandgaps = model.predict(X_train_selected)
    ranges = [(2.5, 3.5), (4.0, 5.5)]
    for r_min, r_max in ranges:
        within_range_indices = (predicted_bandgaps >= r_min) & (predicted_bandgaps <= r_max)
        optimal_params = X_train.loc[within_range_indices]
        range_str = f"{r_min}-{r_max}".replace('.', '_')
        optimal_params.to_csv(f'output/optimal_parameters_{name.replace(" ", "_")}_range_{range_str}.csv', index=False)
        optimal_params_all_models[f"{name}_range_{range_str}"] = optimal_params

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_ * 1000
        importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        feature_importance_data[name] = importance_df

# Create feature importance plots
num_models = len(feature_importance_data)
fig, axes = plt.subplots(num_models, 1, figsize=(12, 6 * num_models))
fig.tight_layout()
if num_models == 1:
    axes = [axes]

for ax, (model_name, importance_df) in zip(axes, feature_importance_data.items()):
    ax.bar(importance_df['Feature'], importance_df['Importance'], alpha=0.9)
    ax.set_title(f'Feature Importance ({model_name})')
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance (scaled)')
    ax.tick_params(axis='x', rotation=90)
    
plt.tight_layout()
plt.savefig('output/feature_importance_comparison_subplots.png')
plt.show()


# Plot predicted vs actual for each model and save
plt.figure(figsize=(12, 8))
for i, (name, predictions) in enumerate(predictions_store.items()):
    plt.subplot(2, 2, i + 1)
    plt.scatter(y_test, predictions, alpha=0.7, edgecolor='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f'{name}: Predicted vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
plt.tight_layout()
plt.savefig('output/predicted_vs_actual.png')
plt.show()


# Save results to CSV
results_df = pd.DataFrame(results).T
results_df.to_csv('output/model_performance.csv')
