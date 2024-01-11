# Climate Dynamics Analysis Project

## Overview
This repository houses a comprehensive analysis project focused on climate dynamics, exploring paleo-present climate data to uncover patterns, relationships, and future predictions. The project employs various analytical techniques, including statistical visualization, temporal analyses, and predictive modeling.

## Project Structure

- data: Contains the climate change dataset in CSV format 
(climate_change_data.csv).

- notebooks: Includes Jupyter notebooks used for data analysis and modeling.
01_data_preparation.ipynb: Data loading, fusion, and initial exploration.
02_visualizations.ipynb: Visualization of data distribution, temporal trends, and relationships.
03_linear_regression_model.ipynb: Implementation and evaluation of a linear regression model.
04_tensorflow_linear_regression.ipynb: Application of TensorFlow for linear regression and evaluation.
05_knn_regression_model.ipynb: Implementation and evaluation of a K-Nearest Neighbors (KNN) regression model.

- output: Stores exported data files, such as CSV and JSON, and model evaluation outputs.
Getting Started 

## Getting Started

1 - Installing Dependencies

Use the following command to install the required Python libraries:

pip install numpy pandas matplotlib seaborn tensorflow scikit-learn


2 - Clone the project repository from GitHub:

git clone https://github.com/AnaFMello/Project-2-PFDA.git

3- Run the Jupyter Notebook:

jupyter notebook


## Data Exploration

The initial steps involve loading climate change data from 'climate_change_data.csv' into a Pandas DataFrame. 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

Load data

df = pd.read_csv('climate_change_data.csv')

Basic EDA

df.head()
df.shape
df.describe()
df.info()
df.dtypes
df.isnull().sum()
df.duplicated().sum()
df['Country'].value_counts()
df['Location'].value_counts()


## Data Visualization

Various visualizations are employed to understand the distribution and trends of key variables, including temperature, CO2 emissions, sea level rise, precipitation, humidity, and wind speed.

Distribution plots

sns.distplot(df['Temperature'])
plt.show()

Histograms

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.hist(df['Temperature'])
plt.xlabel('Temperature')
plt.subplot(1, 2, 2)
plt.hist(df['CO2 Emissions'])
plt.xlabel('CO2 Emissions')

Line plots

plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
sns.lineplot(x='Year', y='Temperature', data=df)
plt.subplot(1, 2, 2)
sns.lineplot(x='Year', y='CO2 Emissions', data=df)

Correlation matrix and pair plots

numeric_columns = df.select_dtypes(include=['number']).columns
correlations = df[numeric_columns].corr()
sns.pairplot(df)


## Predictive Modeling

The project explores predictive modeling using linear regression and machine learning techniques (TensorFlow-based linear regression and K-Nearest Neighbors). Evaluation metrics such as Mean Squared Error (MSE) are utilized for model assessment.

Linear Regression

...

TensorFlow-based Linear Regression

...

K-Nearest Neighbors Regression

...

# Conclusion

The insights gained from this analysis contribute to our understanding of climate dynamics. Findings from this project, documented in a Jupyter notebook, provide a foundation for continued research and exploration into the complexities of climate change.

# References

References have been included within the academic style in the Jupyter notebook. Please refer to the notebook for detailed academic citations.