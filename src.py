#!/usr/bin/env python
# coding utf8
# # Student Grade Analysis & Prediction
# # Import Libraries
# In[1]
Import pandas as pd
Import numpy as np
Import matplotlib.pyplot as plt
Import seaborn as sns
Get_ipython().run_line_magic(‘matplotlib’, ‘inline’)
# # Dataset
# In[2]
Stud= pd.read_csv(‘studentmat.csv’) # Read the dataset
# In[3]
Print(‘Total number of students’,len(stud))
# In[4]
Stud[‘G3’].describe()
# In[5]
Stud.info() # Information on dataset
# In[6]
Stud.columns # Dataset Columns
# In[7]
Stud.describe() # Dataset description
# In[8]
Stud.head() # First 5 values of dataset
# In[9]
Stud.tail() # Last 5 values of dataset
# In[10]
Stud.isnull().any() # To check any null values present in dataset
# In[13]
Import cufflinks as cf
Cf.go_offline()
# In[14]
Stud.iplot() # Plot for the all attributes
# In[15]
Stud.iplot(kind=’scatter’,x=’age’,y=’G3’,mode=’markers’,size=8) # Plot for age vs G3
#In[16]
Stud.iplot(kind=’box’)
# In[17]
Stud[‘G3’].iplot(kind=’hist’,bins=100,color=’blue’)
# # Data Visualization
# In[18]
Sns.heatmap(stud.isnull(),cmap=”rainbow”,yticklabels=False) # To check anynull
values present in dataset pictorially
# In[19]
Sns.heatmap(stud.isnull(),cmap=”viridis”,yticklabels=False) # Map color – viridis
# There are no null values in the given dataset
# # Student’s Sex
# In[20]
F_stud = len(stud[stud[‘sex’] == ‘F’]) # Number of female students
Print(‘Number of female students’,f_stud)
M_stud = len(stud[stud[‘sex’] == ‘M’]) # Number of male students
Print(‘Number of male students’,m_stud)
# In[21]
Sns.set_style(‘whitegrid’) # male & female student representaion on countplot
Sns.countplot(x=’sex’,data=stud,palette=’plasma’)
# The gender distribution is pretty even. # # Age of Students
# In[22]
B = sns.kdeplot(stud[‘age’]) # Kernel Density Estimations
b.axes.set_title(‘Ages of students’)
b.set_xlabel(‘Age’)
b.set_ylabel(‘Count’)
plt.show()
# In[23]
B = sns.countplot(x=’age’,hue=’sex’, data=stud, palette=’inferno’)
b.axes.set_title(‘Number of Male & Female students in different age groups’)
b.set_xlabel(“Age”)
b.set_ylabel(“Count”)
plt.show()
# The student age seems to be ranging from 1519, where gender distribution is prettyevenin each age group. # The age group above 19 may be outliers, year back students or droupouts. # # Students from Urban & Rural Areas
# In[24]
U_stud = len(stud[stud[‘address’] == ‘U’]) # Number of urban areas students
Print(‘Number of Urban students’,u_stud)
R_stud = len(stud[stud[‘address’] == ‘R’]) # Number of rural areas students
Print(‘Number of Rural students’,r_stud)
# In[25]
Sns.set_style(‘whitegrid’)
Sns.countplot(x=’address’,data=stud,palette=’magma’) # urban & rural representaiononcountplot
# Approximately 77.72% students come from urban region and 22.28%fromrural region. 
# In[26]
Sns.countplot(x=’address’,hue=’G3’,data=stud,palette=’Oranges’)
# # EDA – Exploratory Data Analysis
# ### 1. Does age affect final grade?
# In[27]
B= sns.boxplot(x=’age’, y=’G3’,data=stud,palette=’gist_heat’)
b.axes.set_title(‘Age vs Final Grade’)
# Plotting the distribution rather than statistics would help us better understand the data. # The above plot shows that the median grades of the three age groups(15,16,17) are similar. Note the skewness of age group 19. (may be due to sample size). Age group 20 seemstoscore highest grades among all. 
# In[28]
B = sns.swarmplot(x=’age’, y=’G3’,hue=’sex’, data=stud,palette=’PiYG’)
b.axes.set_title(‘Does age affect final grade?’)
# ## 2. Do urban students perform better than rural students?
# In[29]
# Grade distribution by address
Sns.kdeplot(stud.loc[stud[‘address’] == ‘U’, ‘G3’], label=’Urban’, shade = True)
Sns.kdeplot(stud.loc[stud[‘address’] == ‘R’, ‘G3’], label=’Rural’, shade = True)
Plt.title(‘Do urban students score higher than rural students?’)
Plt.xlabel(‘Grade’);
Plt.ylabel(‘Density’)
Plt.show()
# The above graph clearly shows there is not much difference between the grades basedonlocation. # In[30]
Stud.corr()[‘G3’].sort_values()
# ## Encoding categorical variables using LabelEncoder()
# In[31]
From sklearn.preprocessing import LabelEncoder
Le=LabelEncoder()
Stud.iloc[,0]=le.fit_transform(stud.iloc[,0])
Stud.iloc[,1]=le.fit_transform(stud.iloc[,1])
Stud.iloc[,3]=le.fit_transform(stud.iloc[,3])
Stud.iloc[,4]=le.fit_transform(stud.iloc[,4])
Stud.iloc[,5]=le.fit_transform(stud.iloc[,5])
Stud.iloc[,8]=le.fit_transform(stud.iloc[,8])
Stud.iloc[,9]=le.fit_transform(stud.iloc[,9])
Stud.iloc[,10]=le.fit_transform(stud.iloc[,10])
Stud.iloc[,11]=le.fit_transform(stud.iloc[,11])
Stud.iloc[,15]=le.fit_transform(stud.iloc[,15])
Stud.iloc[,16]=le.fit_transform(stud.iloc[,16])
Stud.iloc[,17]=le.fit_transform(stud.iloc[,17])
Stud.iloc[,18]=le.fit_transform(stud.iloc[,18])
Stud.iloc[,19]=le.fit_transform(stud.iloc[,19])
Stud.iloc[,20]=le.fit_transform(stud.iloc[,20])
Stud.iloc[,21]=le.fit_transform(stud.iloc[,21])
Stud.iloc[,22]=le.fit_transform(stud.iloc[,22])
# In[32]
Stud.head()
# In[33]
Stud.tail()
# In[34]
Stud.corr()[‘G3’].sort_values() # Correlation wrt G3
# In[35]
# drop the school and grade columns
Stud = stud.drop([‘school’, ‘G1’, ‘G2’], axis=’columns’)
# Although G1 and G2 which are period grades of a student and are highly correlatedtothefinal grade G3, we drop them. It is more difficult to predict G3 without G2 and G1, but suchprediction is much more useful because we want to find other factors affect the grade. 
# In[36]
# Find correlations with the Grade
Most_correlated = stud.corr().abs()[‘G3’].sort_values(ascending=False)
# Maintain the top 8 most correlation features with Grade
Most_correlated = most_correlated[9]
Most_correlated
# In[37]
Stud = stud.loc[,most_correlated.index]
Stud.head()
# ### Failure Attribute
# In[38]
B = sns.swarmplot(x=stud[‘failures’],y=stud[‘G3’],palette=’autumn’)
b.axes.set_title(‘Previous Failures vs Final Grade(G3)’)
# Observation Student with less previous failures usually score higher
# ### Family Education Attribute ( Fedu + Medu )
# In[39]
Fa_edu = stud[‘Fedu’] + stud[‘Medu’]
B = sns.swarmplot(x=fa_edu,y=stud[‘G3’],palette=’summer’)
b.axes.set_title(‘Family Education vs Final Grade(G3)’)
# Observation Educated families result in higher grades
# ### Wish to go for Higher Education Attribute
# In[40]
B = sns.boxplot(x=stud[‘higher’],y=stud[‘G3’],palette=’binary’)
b.axes.set_title(‘Higher Education vs Final Grade(G3)’)
# Observation Students who wish to go for higher studies score more
# ## Going Out with Friends Attribute
# In[41]
B = sns.countplot(x=stud[‘goout’],palette=’OrRd’)
b.axes.set_title(‘Go Out vs Final Grade(G3)’)
# Observation The students have an average score when it comes to going out with friends.
# In[42]
B = sns.swarmplot(x=stud[‘goout’],y=stud[‘G3’],palette=’autumn’)
b.axes.set_title(‘Go Out vs Final Grade(G3)’)
# Observation Students who go out a lot score less
# ### Romantic relationship Attribute
# In[43]
B = sns.swarmplot(x=stud[‘romantic’],y=stud[‘G3’],palette=’YlOrBr’)
b.axes.set_title(‘Romantic Relationship vs Final Grade(G3)’)
# Here romantic attribute with value 0 means no relationship and value with 1 meansinrelationship
#
# Observation Students with no romantic relationship score higher
# ### Reason Attribute
# In[44]
B = sns.countplot(x=’reason’,data=stud,palette=’gist_rainbow’) # Reason to choosethisschool
b.axes.set_title(‘Reason vs Students Count’)
# In[45]
B = sns.swarmplot(x=’reason’, y=’G3’, data=stud,palette=’gist_rainbow’)
b.axes.set_title(‘Reason vs Final grade’)
# Observation The students have an equally distributed average score when it comestoreason attribute. # # Machine Learning Algorithms
# In[46]
# Standard ML Models for comparison
From sklearn.linear_model import LinearRegression
From sklearn.linear_model import ElasticNet
From sklearn.ensemble import RandomForestRegressor
From sklearn.ensemble import ExtraTreesRegressor
From sklearn.ensemble import GradientBoostingRegressor
From sklearn.svm import SVR
# Splitting data into training/testing
From sklearn.model_selection import train_test_split
From sklearn.preprocessing import MinMaxScaler
# Metrics
From sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
# Distributions
Import scipy
# In[47]
# splitting the data into training and testing data (75% and 25%)
# we mention the random state to achieve the same split everytime we run the code
X_train, X_test, y_train, y_test = train_test_split(stud, stud[‘G3’], test_size =0.25, random_state=42)
# In[48]
X_train.head()
# ## MAE – Mean Absolute Error & RMSE – Root Mean Square Error
# In[49]
# Calculate mae and rmse
Def evaluate_predictions(predictions, true)
Mae = np.mean(abs(predictions – true))
Rmse = np.sqrt(np.mean((predictions – true) 2))
Return mae, rmse
# In[50]
# find the median
Median_pred = X_train[‘G3’].median()
# create a list with all values as median
Median_preds = [median_pred for _ in range(len(X_test))]
# store the true G3 values for passing into the function
True = X_test[‘G3’]
# In[51]
# Display the naive baseline metrics
Mb_mae, mb_rmse = evaluate_predictions(median_preds, true)
Print(‘Median Baseline MAE {.4f}’.format(mb_mae))
Print(‘Median Baseline RMSE {.4f}’.format(mb_rmse))
# In[52]
# Evaluate several ml models by training on training set and testing on testing set
Def evaluate(X_train, X_test, y_train, y_test)
# Names of models
Model_name_list = [‘Linear Regression’, ‘ElasticNet Regression’, ‘Random Forest’, ‘Extra Trees’, ‘SVM’, ‘Gradient Boosted’, ‘Baseline’]
X_train = X_train.drop(‘G3’, axis=’columns’)
X_test = X_test.drop(‘G3’, axis=’columns’)
# Instantiate the models
Model1 = LinearRegression()
Model2 = ElasticNet(alpha=1.0, l1_ratio=0.5)
Model3 = RandomForestRegressor(n_estimators=100)
Model4 = ExtraTreesRegressor(n_estimators=100)
Model5 = SVR(kernel=’rbf’, degree=3, C=1.0, gamma=’auto’)
Model6 = GradientBoostingRegressor(n_estimators=50)
# Dataframe for results
Results = pd.DataFrame(columns=[‘mae’, ‘rmse’], index = model_name_list)
# Train and predict with each model
For i, model in enumerate([model1, model2, model3, model4, model5, model6])
Model.fit(X_train, y_train)
Predictions = model.predict(X_test)
# Metrics
Mae = np.mean(abs(predictions – y_test))
Rmse = np.sqrt(np.mean((predictions – y_test) 2))
# Insert results into the dataframe
Model_name = model_name_list[i]
Results.loc[model_name, ] = [mae, rmse]
# Median Value Baseline Metrics
Baseline = np.median(y_train)
Baseline_mae = np.mean(abs(baseline – y_test))
Baseline_rmse = np.sqrt(np.mean((baseline – y_test) 2))
Results.loc[‘Baseline’, ] = [baseline_mae, baseline_rmse]
Return results
# In[53]
Results = evaluate(X_train, X_test, y_train, y_test)
Results
# In[54]
Plt.figure(figsize=(12, 7))
# Root mean squared error
Ax = plt.subplot(1, 2, 1)
Results.sort_values(‘mae’, ascending = True).plot.bar(y = ‘mae’, color = ‘violet’, ax =ax)
Plt.title(‘Model Mean Absolute Error’)
Plt.ylabel(‘MAE’)
# Median absolute percentage error
Ax = plt.subplot(1, 2, 2)
Results.sort_values(‘rmse’, ascending = True).plot.bar(y = ‘rmse’, color = ‘pink’, ax =ax)
Plt.title(‘Model Root Mean Squared Error’)
Plt.ylabel(‘RMSE’)
Plt.show()
# Conclusion As we see both Model Mean Absolute Error & Model Root Mean SquaredError that the linear regression is performing the best in both cases.
