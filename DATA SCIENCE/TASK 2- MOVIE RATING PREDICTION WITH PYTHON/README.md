Project Overview
This project aims to predict the IMDb ratings of Indian movies using various features such as genre, director, actors, duration, year of release, and votes. The dataset contains information about Indian films, including their ratings and other attributes. The project involves data cleaning, feature engineering, exploratory data analysis (EDA), model training, and evaluation.


Steps Involved:-
Data Loading: Load the dataset using pandas.
Data Cleaning:
Handle missing values.
Convert data types for relevant columns.
Remove duplicates.
Exploratory Data Analysis (EDA):
Analyze the distribution of various features using plots.
Identify the top genres, directors, rated films, and popular films.
Visualize the impact of various features on the rating.
Feature Engineering:
Target encoding for categorical features (Genre, Director, Actors, Name).
Feature Selection:
Compute mutual information to identify important features.
Visualize the information gain for each feature.
Correlation Analysis:
Visualize the correlation matrix of the features.
Data Splitting:
Split the data into training and testing sets.
Model Training:
Train a Linear Regression model on the training set.
Performance Evaluation:
Evaluate the model using Mean Squared Error (MSE) and R-squared value.
Visualize the actual vs predicted ratings.
Plot the residual errors.


Visualization Examples
Top 10 Genres:
Bar plot showing the top 10 genres by count.
Top 10 Directors:
Bar plot showing the top 10 directors by rating.
Top 10 Rated Films:
Bar plot showing the top 10 rated films.
Top 10 Popular Films:
Bar plot showing the top 10 popular films by votes.
Top 10 Most Performing Actors:
Bar plot showing the top 10 most performing actors by the number of films.
Distributions:
Histograms and distribution plots for rating, year, duration, and votes.
Scatter Plots:
Scatter plots showing the impact of duration and votes on the rating.
Regression plot showing the impact of year on rating.
Information Gain:
Bar plot showing the information gain for each feature.
Correlation Matrix:
Heatmap showing the correlation between features.
