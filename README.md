# Student Grade Analysis & Prediction

This project involves analyzing and predicting student grades based on various attributes. It utilizes data visualization, exploratory data analysis (EDA), and machine learning algorithms to gain insights into factors influencing student performance.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Data Visualization](#data-visualization)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis)
- [Machine Learning Algorithms](#machine-learning-algorithms)

## Overview

This project analyzes and predicts student grades based on various attributes using data visualization and machine learning. It includes exploratory data analysis (EDA), visualization of factors influencing grades, and the implementation of machine learning algorithms for predictive modeling. The project is developed in Python and leverages popular libraries such as Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.

## Dataset
dataset file saved as student-mat.csv
DATASET DESIGN
1. School GP (assuming it refers to the name of the school)
2. Sex F (indicating the student is female)
3. Age 18
4. Address U (assuming it stands for urban)
5. Famsize GT3 (indicating the family size is greater than 3)
6. Pstatus A (assuming it stands for living with both parents)
7. Medu 4 (mother’s education level is 4, which typically corresponds to higher
education)
8. Fedu 4 (father’s education level is 4, which typically corresponds to higher education)
9. Mjob at_home (mother’s occupation is “at home”)
10. Fjob teacher (father’s occupation is a “teacher”)
11. Reason course (the reason for choosing the school is related to a specific course)
12. Guardian mother (the student’s guardian is her mother)
13. Traveltime 2 (assumed to be the travel time to school on a scale of 1 to 4)
14. Studytime 2 (assumed to be the weekly study time on a scale of 1 to 4)
15. Failures 0 (the number of past class failures)
16. Schoolsup yes (indicating the student receives extra educational support from the
school)
17. Famsup no (indicating the student does not receive family educational support)
18. Dalc 1 (assumed to represent weekday alcohol consumption on a scale of 1 to 5, with
1 indicating very low consumption)
19. Walc 1 (assumed to represent weekend alcohol consumption on a scale of 1 to 5, with
1 indicating very low consumption)
20. Health 3 (assumed to represent the student’s selfperceived health on a scale of 1 to 5)
21. Absences 6 (the number of school absences)
22. G1 5 (the student’s grade in the first period)
23. G2 6 (the student’s grade in the second period)
24. G3 6 (the student’s final grade)

## Data Visualization

Explore visualizations created from the dataset to better understand its characteristics. Include charts, graphs, and heatmaps.

## Exploratory Data Analysis (EDA)

Summarize key insights gained from the EDA process. Discuss any patterns or relationships observed in the data.

## Machine Learning Algorithms

In the provided code, the following machine learning algorithms are used for predicting the final grades (`G3`):

1. Linear Regression
2. ElasticNet Regression
3. Random Forest Regressor
4. Extra Trees Regressor
5. Support Vector Regressor (SVM)
6. Gradient Boosting Regressor

These algorithms are trained on the training set (`X_train`, `y_train`) and evaluated on the testing set (`X_test`, `y_test`). Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) are used as metrics to evaluate the performance of each algorithm. The results are then compared, and the model with the lowest MAE or RMSE is considered the best-performing one for the given dataset.



