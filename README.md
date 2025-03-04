
GaN Quantum Dots Bandgap Prediction using Machine Learning

Project Description:
This project applies various machine learning (ML) regression models to predict the bandgap energy of GaN quantum dots (QDs) based on synthesis parameter data. The primary objectives of this project are:
Regression Modeling: Establish a quantitative relationship between synthesis parameters (represented by matrix X) and bandgap energy values (represented by vector Y).
Model Comparison: Train and evaluate multiple ML models to predict the bandgap energy, using 75-80% of the dataset for training and the remaining portion for testing.
Feature Importance Analysis: Identify the most influential synthesis parameters contributing to higher bandgap values.
Optimal Parameter Estimation: Predict optimal synthesis parameter values for obtaining a desired bandgap, particularly targeting higher bandgap energies for UV detector applications.

Dataset (./input/dataset.xlsx):
X: Matrix containing synthesis parameter data.
Y: Vector containing corresponding bandgap energy values.

Data preprocessing and splitting into training and testing sets are included in the pipeline.

Models Implemented:
Linear Regression
Decision Tree Regressor
Random Forest Regressor
Gradient Boosting Regressor

Performance Evaluation
The models are evaluated using:
Mean Absolute Error (MAE)
Mean Squared Error (MSE)
R-squared Score (R²)

Results
The model performances are compared, and the best-performing models are selected for further analysis.

Feature Importance
Feature importance is analyzed to determine the synthesis parameters with the most significant impact on the bandgap values.

Optimal Parameter Prediction
For a given target bandgap value, the best-performing models are used to predict the optimal synthesis parameters.

Installation:
pip install numpy matplotlib sklearn pandas

Usage
Clone the repository:
git clone https://github.com/ahmed-farhani-asl/ML-GaN-QDs.git

Navigate to the project directory
cd GaN-QDs-Bandgap-Prediction

Run the main script:
python main.py

Author:
Ahmad Farhani Asl

Acknowledgments
Special thanks to Mr. Bahram Dalvand for guidance and support during this project.

