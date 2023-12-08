#!/usr/bin/env python
# coding: utf-8

# # Employee churn model 

# In[144]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[145]:


df=pd.read_csv("HR-Employee-Attrition.csv")


# In[146]:


df.head()


# #### Data dictionary

# In[147]:


df.size


# In[148]:


df.shape


# In[149]:


df.info()


# In[150]:


df.describe()


# ###### checking for null and duplicated values

# In[151]:


df.isnull().sum()


# In[152]:


df.duplicated().sum()


# In[153]:


for i in df.columns:
    print(i)


# In[154]:


len(df.query("Over18 == 'N'"))


# In[155]:


df.drop('Over18',axis=1, inplace=True) # Dropped because the column is not needed,as an indivisual must be over 18 in order to work 


# In[156]:


len(df.query("EmployeeCount != 1"))


# In[157]:


df.drop('EmployeeCount',axis=1, inplace=True) 


# In[158]:


df.drop('StandardHours',axis=1, inplace=True) 


# In[159]:


df.drop('EmployeeNumber',axis=1, inplace=True) 


# In[160]:


print(df['BusinessTravel'].unique())


# In[161]:


print(df['Department'].unique())


# In[162]:


l=[df.value_counts(['Department'])]
l


# In[163]:


df.value_counts(['JobRole'])


# # Data visualization

# #### Distribution of Attrition

# In[164]:


sns.countplot(x="Attrition" , data=df,palette="Set2")


# #### Checking the overall rating values given by the employees

# In[165]:


emp_ratings=["WorkLifeBalance",'RelationshipSatisfaction','JobSatisfaction','EnvironmentSatisfaction']Checking the overall rating values given by the employees
df[emp_ratings].hist(figsize=(7,7) ,color='#8da0cb')
plt.show()


# #### Gender vs JobSatisfaction

# In[166]:


df['Gender'].value_counts()


# In[167]:


g_rating = df.groupby("Gender")["JobSatisfaction"].value_counts().to_frame().unstack()


# In[168]:


g_r = g_rating.reset_index()

# Rename the columns for clarity
g_r.columns = ['Gender', '1', '2','3','4']

print(g_r)


# In[169]:


x=g_r['Gender'].tolist()
for i in range(0,2):
    r = g_r.iloc[i, 1:]# Create a pie chart
    plt.figure(figsize=(4, 4))
    plt.pie(r, labels=r.index, autopct='%1.1f%%', startangle=140,colors=color_palette)
    print('Rating Distribution for Gender ',x[i])
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    plt.show()
    


# ##### people who have left the company within a year of workin

# In[170]:


len(df.query("YearsAtCompany == 0"))


# In[171]:


len(df.query("(YearsAtCompany == 0) and (Attrition =='Yes')"))


# #### Attrition by each department

# In[172]:


plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.countplot(data=df, x='Department', hue='Attrition',palette='Set2')
plt.title('Employee Attrition by Department')
plt.xlabel('Department')
plt.ylabel('Count')
plt.legend(title='Attrition')


# In[173]:


plt.figure(figsize=(10, 6))

# Create a subset of the dataframe where OverTime is "Yes"
overtime_yes = df[df['OverTime'] == 'Yes']

# Create a countplot to visualize attrition with respect to overtime
sns.countplot(data=overtime_yes, x='OverTime', hue='Attrition', palette='Set2')

plt.title('Employee Attrition with Respect to Overtime')
plt.xlabel('Overtime')
plt.ylabel('Count')
plt.legend(title='Attrition')
plt.show()


# #### Relation between travel and attrition 

# In[174]:


business_travel_ct = pd.crosstab(df['BusinessTravel'], df['Attrition'])
business_travel_ct.plot(kind='bar', stacked=True,color=['#8dd3c7','#fc8d62'])
plt.title('Attrition by Business Travel')
plt.ylabel('Count')
plt.xlabel('Business Travel')
plt.xticks(rotation=0)
plt.legend(title='Attrition')
plt.show()


# In[175]:


businessdata = df.groupby("BusinessTravel")["Attrition"].value_counts()

businessdata


# #### Imbalanced class distribution

# In[176]:


sns.countplot(data=df, x='Attrition',color='#a6d854')
plt.title('Class Distribution')
plt.show()


# #### Summary of the categorical variables

# In[177]:


df.describe(include="object").T


# * Most frequent in this company is a person who travels rarely, from the Research & Development department , educated in Life Sciences , Male, Married, Sales Executive and no OverTime
# 
# 

# In[178]:


df['JobRole'].value_counts().plot.bar(cmap='Set3')
plt.title('Number of Employee by JobRole')
plt.ylabel('Count')
plt.xlabel('JobRole')



# In[179]:


plt.figure(figsize=(10, 6))
ax=sns.countplot(x='JobRole', hue='Attrition',data=df, palette='Set2')
plt.title('Attrition by JobRole')
plt.xlabel('JobRole')
plt.ylabel('Count')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

plt.show()


# In[180]:


attrition = df[df['Attrition'] == 'Yes']
no_attrition = df[df['Attrition']=='No']


# In[181]:


sns.kdeplot(attrition['DistanceFromHome'], label='Employee who left', fill=True, color='palegreen')
sns.kdeplot(no_attrition['DistanceFromHome'], label='Employee who stayes',fill=True, color='salmon')


# In[182]:


sns.kdeplot(attrition['MonthlyIncome'], label='Employee who left', fill=True, color='palegreen')
sns.kdeplot(no_attrition['MonthlyIncome'], label='Employee who stayes',fill=True, color='salmon')


# In[183]:


sns.kdeplot(attrition['Age'], label='Employee who left', fill=True, color='palegreen')
sns.kdeplot(no_attrition['Age'], label='Employee who stayes',fill=True, color='salmon')


# In[184]:


sns.kdeplot(attrition['YearsAtCompany'], label='Employee who left', fill=True, color='palegreen')
sns.kdeplot(no_attrition['YearsAtCompany'], label='Employee who stayes',fill=True, color='salmon')


# #### Job Satisfaction per department

# In[185]:


ratings = df.groupby("Department")["JobSatisfaction"].value_counts().to_frame().unstack()


# In[186]:


ratings


# In[187]:


type(ratings)


# In[188]:


ratings_reset = ratings.reset_index()

# Rename the columns for clarity
ratings_reset.columns = ['Department', '1', '2','3','4']

print(ratings_reset)


# In[189]:


ratings_reset.head(1)


# In[190]:


import plotly.graph_objects as go


# In[191]:


num_colors = len(ratings_reset.columns) - 1
color_palette = plt.cm.Set3(np.linspace(0, 1, num_colors))


# In[192]:


x=ratings_reset['Department'].tolist()
for i in range(0,3):
    ratings = ratings_reset.iloc[i, 1:]# Create a pie chart
    plt.figure(figsize=(4, 4))
    plt.pie(ratings, labels=ratings.index, autopct='%1.1f%%', startangle=140 ,colors=color_palette)
    print('Rating Distribution for Department ',x[i])
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    plt.show()


# In[193]:


dfr=df.query("Department == 'Research & Development'")


# In[194]:


len(dfr.query("JobSatisfaction ==1 and Attrition== 'Yes'"))


# ## Encoding

# In[195]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Create a label encoder object
le = LabelEncoder()


# In[196]:


df.head()


# In[197]:


le_count = 0
for col in df.columns[1:]:
    if df[col].dtype == 'object':
            le.fit(df[col])
            df[col] = le.transform(df[col])
            le_count += 1
print('{} columns were label encoded.'.format(le_count))


# In[198]:


from sklearn.preprocessing import StandardScaler


# In[199]:


scaler = StandardScaler()


# In[200]:


numerical_columns = ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'PercentSalaryHike', 'PerformanceRating', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])


# In[201]:


df.head()


# In[202]:


df.info()


# In[203]:


correlation = df.corr()
print(correlation["Attrition"].sort_values(ascending=False))


# In[204]:


from sklearn.preprocessing import MinMaxScaler


# In[205]:


scaler = MinMaxScaler()
scol = list(df.columns)
scol.remove('Attrition')
for col in scol:
    df[col] = df[col].astype(float)
    df[[col]] = scaler.fit_transform(df[[col]])
df['Attrition'] = pd.to_numeric(df['Attrition'], downcast='float')
df.head()


# In[206]:


from sklearn.model_selection import train_test_split


# ### Re Sampling the imbalanced data

# In[207]:


from sklearn.utils import resample


# In[208]:


df_majority = df[df['Attrition'] == 0.0]
df_minority = df[df['Attrition'] == 1.0]


# In[209]:


df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)


# In[210]:


df_emp = pd.concat([df_majority, df_minority_upsampled])


# In[293]:


df_emp["Attrition"].value_counts()


# In[296]:


df_emp.shape


# ### Model building 

# * Step1 : specify dependent and target variable 

# In[211]:


X=df_emp.drop('Attrition',axis=1)


# In[212]:


Y=df_emp['Attrition']


# In[213]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.40,random_state=45)


# In[214]:


y_test.head()


# ### Logistic Regression

# In[215]:


from sklearn.linear_model import LogisticRegression
log= LogisticRegression()


# In[216]:


log.fit(x_train,y_train)


# In[217]:


y_pred_lr=log.predict(x_test) 


# In[218]:


y_pred_lr


# In[219]:


from sklearn.metrics import accuracy_score,recall_score
from sklearn.metrics import f1_score


# In[220]:


log_a=accuracy_score(y_pred_lr, y_test)
log_a


# In[221]:


f1_lr = f1_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)


# In[222]:


print("Accuracy",log_a)
print("Recall",recall_lr)
print("f1score",f1_lr)


# ### Decision tree

# In[237]:


from sklearn.tree import DecisionTreeClassifier


# In[238]:


dc=DecisionTreeClassifier(criterion='gini',max_depth=3,min_samples_split=7,random_state=10)
dc.fit(x_train,y_train)


# In[239]:


y_pred_dc=dc.predict(x_test)
y_pred_dc


# In[240]:


a_dt=accuracy_score(y_test,y_pred_dc)


# In[241]:


recall_dt = recall_score(y_test, y_pred_dc)
f1_dt=f1_score(y_test,y_pred_dc)


# In[242]:


print("Accuracy ", a_dt)
print("recall ", recall_dt)
print("f1score",f1_dt)


# ### Random forest

# In[223]:


# Number of trees in random forest
n_estimators = [100, 125, 150, 175]
# Number of features to consider at every split
max_features = ['auto', 'sqrt', 'log2']
# Maximum number of levels in tree
max_depth = [2,4]
# Minimum number of samples required to split a node
min_samples_split = [2,  5, 8]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# criterion
criterion = ['gini', 'entropy']


# In[224]:


param_grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap,
    'criterion': criterion
}
print(param_grid)


# In[225]:


from sklearn.ensemble import RandomForestClassifier


# In[226]:


rf_Model = RandomForestClassifier()


# In[227]:


from sklearn.model_selection import GridSearchCV
rf_Grid = GridSearchCV(estimator = rf_Model, param_grid = param_grid, cv = 3, verbose=2, n_jobs = 4)


# In[228]:


grid_fit = rf_Grid.fit(x_train, y_train)


# In[229]:


#rf_opt = grid_fit.best_estimator_
#lf = RandomForestClassifier(n_estimators = 500, max_depth = 4, max_features = 3, bootstrap = True, random_state = 18).fit(x_train, y_train)
rf_Grid.best_params_


# In[230]:


rf_opt=RandomForestClassifier(n_estimators = 100, max_depth = 4, max_features = 'sqrt',bootstrap = True, random_state = 45)


# In[231]:


rf_opt.fit(x_train, y_train)


# In[232]:


y_pred_rf=rf_opt.predict(x_test)


# In[233]:


rf_a=accuracy_score(y_pred_rf, y_test)


# In[234]:


f1_rf = f1_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)


# In[235]:


y_train_pred = rf_opt.predict(x_train)
y_test_pred = rf_opt.predict(x_test)
print("Accuracy ", rf_a)
print("recall ", recall_rf)
print("f1score", f1_rf)


# In[236]:


train= accuracy_score(y_train, y_train_pred)
print("Training Accuracy:", train)

test= accuracy_score(y_test, y_test_pred)
print("Testing Accuracy:", test)


# ### Neural Network

# In[243]:


from tensorflow import keras


# In[244]:


from tensorflow.keras import layers


# In[245]:


# Create a simple neural network model
model = keras.Sequential([
    layers.Input(shape=(x_train.shape[1],)),  # Input layer
    layers.Dense(64, activation='relu'),      # Hidden layer with 64 neurons and ReLU activation
    layers.Dense(1, activation='sigmoid')      # Output layer with sigmoid activation (binary classification)
])



# In[291]:


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[292]:


# Fit the model on the training data
model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)

# Predict on the test data
y_pred_n = model.predict(x_test)
y_pred = np.round(y_pred_n).flatten()  # Convert probabilities to binary predictions


# In[248]:


# Calculate the accuracy of the predictions
a_nn = accuracy_score(y_test, y_pred)


# In[249]:


recall_nn= recall_score(y_test, y_pred)
f1_nn=f1_score(y_test,y_pred)


# In[250]:


print("Accuracy ", a_nn)
print("recall ", recall_nn)
print("f1score",f1_nn)


# ## Support vector machines: linear Non linear

# In[251]:


from sklearn.svm import SVC

# Create a linear SVM model
svm_model = SVC(kernel='linear', C=1.0, random_state=42)


# In[252]:


svm_model.fit(x_train, y_train)


# In[253]:


y_pred_svm=svm_model.predict(x_test)


# In[254]:


accuracy = accuracy_score(y_test, y_pred_svm)
print("Accuracy:", accuracy)


# In[255]:


recall_svm= recall_score(y_test, y_pred_svm)

f1_svm=f1_score(y_test,y_pred_svm)


# In[256]:


print(recall_svm)
print(f1_svm)


# In[257]:


svm_nl = SVC(kernel='rbf', C=1.0 gamma='scale', random_state=42)


# In[258]:


svm_nl.fit(x_train, y_train)


# In[259]:


y_pred_snl=svm_nl.predict(x_test)


# In[260]:


a_snl = accuracy_score(y_test, y_pred_snl)


# In[261]:


recall_snl= recall_score(y_test, y_pred_snl)

f1_snl=f1_score(y_test,y_pred_snl)

acc.append(a_snl)
fs.append(f1_snl)
rec.append(recall_snl)
# In[262]:


print("Accuracy ", a_snl)
print("recall ", recall_snl)
print("f1score",f1_snl)


# ## Gradient Boosting Classifier

# In[263]:


from sklearn.ensemble import GradientBoostingClassifier

# Create a Gradient Boosting Classifier model
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.4, random_state=45)


# In[264]:


gb_model.fit(x_train, y_train)


# In[265]:


y_pred_gb=gb_model.predict(x_test)


# In[266]:


predicted_prob = gb_model.predict_proba(x_test)
type(predicted_prob)


# In[267]:


df_prob = pd.DataFrame(data=predicted_prob, columns=['Class 0','Class 1'])

df_prob.drop('Class 0',axis=1, inplace=True) 
df_prob.head()


# In[268]:


a_gb = accuracy_score(y_test, y_pred_gb)


# In[269]:


recall_gb= recall_score(y_test, y_pred_gb)

f1_gb=f1_score(y_test,y_pred_gb)


# In[270]:


print("Accuracy:", accuracy)
print("Recall :",recall_gb)
print("F1score :",f1_gb)


# In[271]:


acc=[log_a,rf_a,a_dt,a_nn,a_snl,a_gb]


# In[272]:


fs=[f1_lr,f1_rf,f1_dt,f1_nn,f1_snl,f1_gb]


# In[273]:


rec=[recall_lr,recall_rf,recall_dt,recall_nn,recall_snl,recall_gb]


# In[274]:


models = ['Logistic Regression', 'Random Forest', 'Decision Tree','Neural Network','SVM','Gradient Boosting Classifier']


# Create a dictionary to hold the metrics
metrics_dict = {
    'Model': models,
    'Accuracy': acc,
    'F1-Score': fs,
    'Recall': rec
}



# In[275]:


# Convert the dictionary into a DataFrame
metrics = pd.DataFrame(metrics_dict)

# Sort the DataFrame based on your preferred metric
metrics= metrics.sort_values(by='Recall', ascending=False)

# Print the sorted comparison table
print(metrics)


# In[288]:


# Plotting the metrics
plt.figure(figsize=(10, 6))
plt.bar(metrics.Model, metrics['Recall'], color='palegreen', label='Recall')

# Adding labels and title
plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Comparison of Metrics for Different Models')
plt.xticks(rotation=45)

plt.show()


# ### Create a risk column 

# In[276]:


def assign_risk_category(label):
    if label < 0.6:
        return 'Low-risk'
    elif 0.6 <= label <= 0.8:
        return 'Medium-risk'
    else:
        return 'High-risk'


# In[277]:


# Apply the function to create the 'Risk Category' column
df_prob['Risk Category'] = df_prob['Class 1'].apply(assign_risk_category)

# Print the DataFrame with the newly added 'Risk Category' column
print(df_prob)


# In[278]:


df_prob.value_counts(['Risk Category'])


# In[279]:


import pickle


# In[280]:


# Save the trained model as a pickle string
saved_model = pickle.dumps(gb_model)

# Load the pickled model
loaded_gb_model = pickle.loads(saved_model)

# Use the loaded pickled model to make predictions
p = loaded_gb_model.predict(x_test)


# In[ ]:




