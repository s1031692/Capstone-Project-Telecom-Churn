# Telecom Churn Prediction

This repository contains the code and analysis for predicting customer churn in the telecom industry. The goal is to build predictive models to identify high-value customers at high risk of churn and understand the main indicators of churn.

## Problem Statement

In the telecom industry, customer churn is a significant challenge, and retaining highly profitable customers is a top business goal. This project focuses on the prepaid customer model, where customers can churn without notice. The objective is to predict churn for high-value customers based on their usage patterns and other features in the first three months (good and action phases).

## Data

The dataset (`telecom_churn_data.csv`) contains customer-level information for four consecutive months (June, July, August, and September). The data includes various features such as call patterns, mobile internet usage, recharge details, and customer demographics.

## Project Overview

The main objectives of this project are:

1. **Data Preprocessing**: Load and preprocess the telecom churn data, including handling missing values, encoding categorical variables, and filtering high-value customers.

2. **Exploratory Data Analysis (EDA)**: Perform EDA to understand the data distribution, visualize feature correlations, and gain insights into the relationship between features and customer churn.

3. **Model Development**: Develop a machine learning model to predict customer churn using techniques like Random Forest and Logistic Regression. Handle class imbalance using techniques like SMOTE.

4. **Feature Importance Analysis**: Identify the most important features contributing to customer churn and analyze their relationship with churn.

5. **Churn Management Strategies**: Recommend strategies to manage customer churn based on the analysis of important features and their impact on churn.

6. **Business Impact Quantification**: Quantify the potential impact of implementing the recommended strategies, including estimating the expected reduction in churn rate and potential revenue savings.

7. **Visualization and Presentation**: Visualize the findings and present the key insights, recommended strategies, and expected business impact.


## Approach

1. **Data Understanding and Preparation**
   - Filter high-value customers based on the 70th percentile of average recharge amount.
   - Tag churners based on the usage in the churn phase (month 9).
   - Remove attributes related to the churn phase.

2. **Exploratory Data Analysis (EDA)**
   - Handle missing values.
   - Encode categorical variables.
   - Visualize churn distribution.
   - Visualize feature distributions and correlations.

3. **Model Development**
   - Split the data into train and test sets.
   - Handle class imbalance using SMOTE.
   - Train a Random Forest model for churn prediction.
   - Evaluate model performance using various metrics.

4. **Model Selection and Feature Importance**
   - Train a Logistic Regression model.
   - Identify important features using the Logistic Regression model coefficients.
   - Visualize the top important features.

5. **Business Impact and Recommendations**
   - Analyze important features and provide insights.
   - Recommend strategies to manage customer churn based on the analysis.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn

## Usage

1. Clone the repository or download the code files.
2. Place the `telecom_churn_data.csv` file in the same directory as the code.
3. Run the `telecom_churn_prediction.py` script.
