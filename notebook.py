# -*- coding: utf-8 -*-
"""Submission Akhir: Menyelesaikan Permasalahan Institusi Pendidikan.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1J38FsMcV9BWCEh5tXhGX3jMKbtFw4DJV

# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

- Nama: Arini Arumsari
- Email: ariniarum98@gmail.com
- Id Dicoding: -

## Persiapan

### Menyiapkan library yang dibutuhkan
"""

# Commented out IPython magic to ensure Python compatibility.
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import rcParams
# %matplotlib inline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from scipy.stats import boxcox
from imblearn import under_sampling, over_sampling
import gdown
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, fbeta_score, make_scorer
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import cross_validate, RandomizedSearchCV, GridSearchCV, HalvingGridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, reset_parameter, LGBMClassifier

"""### Menyiapkan data yang akan digunakan"""

df = pd.read_csv("data.csv", sep=";")
df.sample(5)

"""## Data Understanding"""

df.shape

# Mengecek apakah ada data yang NaN/Null
print(df.isnull().values.any())
print(df.isna().sum())

"""tidak ada data yang Null"""

# Mengecek Datatype tiap kolom berserta non-null kolom
df.info()

df.describe(include='object')

df.describe()

df.duplicated().sum()

"""tidak terdapat data duplikated

## Data Preparation / Preprocessing
"""

df['Status'].value_counts()

df = df[df.Status!='Enrolled']

status_counts = df['Status'].value_counts()
labels = status_counts.index
sizes = status_counts.values

# Plotting
plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, startangle=90, autopct='%1.1f%%', textprops={'fontsize': 14})
plt.title('Distribution of Status')
plt.axis('equal')
plt.show()

# Mengubah kolom 'Status' menjadi numerik
df['Status']=df['Status'].map({'Dropout':0,'Graduate':1})

plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), annot=True, cmap='jet', linecolor='black', linewidth=1)
plt.show()

df.corr()['Status']

"""Mengeliminasi variabel yang mempunyai korelasi kecil terhadap "Status" dan tidak akan dipakai untuk melatih model


"""

df.drop(columns=['Marital_status',
                          'Age_at_enrollment',
                          'Application_mode',
                          'Application_order',
                          'Course',
                          'Previous_qualification',
                          'Nacionality',
                          'Mothers_qualification',
                          'Fathers_qualification',
                          'Mothers_occupation',
                          'Fathers_occupation',
                          'Educational_special_needs',
                          'International',
                          'Curricular_units_1st_sem_evaluations',
                          'Curricular_units_1st_sem_without_evaluations',
                          'Curricular_units_2nd_sem_evaluations',
                          'Curricular_units_2nd_sem_without_evaluations',
                          'Unemployment_rate',
                          'Inflation_rate',
                         'GDP'], axis=1, inplace=True)
df.info()

correl = df.corr()['Status']
sorted_corr = correl.abs().sort_values(ascending=False)

plt.figure(figsize=(15, 10))
plt.bar(sorted_corr.index, sorted_corr.values)
plt.xlabel('Features')
plt.ylabel('Correlation with Status')
plt.title('Features Sorted by Correlation with Status')
plt.xticks(rotation=90)
plt.show()

correl.head()

"""Mengubah nilai beberapa variabel dari numerik menjadi string


"""

df['Gender'] = df['Gender'].astype(str).replace({'0': 'Male', '1': 'Female'})
df['Displaced'] = df['Displaced'].astype(str).replace({'0': 'No', '1': 'Yes'})
df['Debtor'] = df['Debtor'].astype(str).replace({'0': 'No', '1': 'Yes'})
df['Scholarship_holder'] = df['Scholarship_holder'].astype(str).replace({'0': 'No', '1': 'Yes'})
df['Tuition_fees_up_to_date'] = df['Tuition_fees_up_to_date'].astype(str).replace({'0': 'No', '1': 'Yes'})
df['Daytime_evening_attendance'] = df['Daytime_evening_attendance'].astype(str).replace({'0': 'Evening', '1': 'Daytime'})
df['Status'] = df['Status'].astype(str).replace({'0': 'Dropout', '1': 'Graduate'})

def categorical_plot(features, df, segment_feature=None):
    fig, ax = plt.subplots(len(features), 1,figsize=(10,20))
    for i, feature in enumerate(features):
        if segment_feature:
            sns.countplot(data=df, y=segment_feature, hue=feature, ax=ax[i])
        else:
            sns.countplot(data=df, x=feature, ax=ax[i])
    plt.tight_layout()
    plt.show()

categorical_plot(

features=[
        'Displaced',
        'Debtor',
        'Gender',
        'Scholarship_holder',
        'Tuition_fees_up_to_date',
        'Daytime_evening_attendance'
    ],
    df=df,
    segment_feature="Status"
)

"""## Modeling"""

category_cols = df.select_dtypes(exclude=['int32','int64','float32','float64'])
category_cols.head()

df.describe()

import os
os.makedirs("model")

import joblib

def save_encoders(features, encoder):
    for feature in features:
        joblib.dump(encoder, "model/encoder_{}.joblib".format(feature))

features_to_encode = ['Daytime_evening_attendance',
                     'Displaced',
                     'Debtor',
                     'Tuition_fees_up_to_date',
                     'Gender',
                     'Scholarship_holder']

label_encoder = LabelEncoder()

# Label encode columns and save encoders
for column in features_to_encode:
    df[column] = label_encoder.fit_transform(df[column])

save_encoders(features_to_encode, label_encoder)

def save_scalers(features, scaler):
    for feature in features:
        joblib.dump(scaler, "model/scaler_{}.joblib".format(feature))

features_to_scale = ['Admission_grade',
                     'Previous_qualification_grade',
                     'Curricular_units_1st_sem_approved',
                     'Curricular_units_1st_sem_grade',
                     'Curricular_units_1st_sem_enrolled',
                     'Curricular_units_1st_sem_credited',
                     'Curricular_units_2nd_sem_approved',
                     'Curricular_units_2nd_sem_grade',
                     'Curricular_units_2nd_sem_enrolled',
                     'Curricular_units_2nd_sem_credited']

scaler = StandardScaler()

# Fit scaler to columns and save scalers
for column in features_to_scale:
    scaled_feature = scaler.fit_transform(df[[column]])
    scaled_feature = scaled_feature.reshape(-1, 1)  # Reshape the scaled feature
    df[column] = scaled_feature
    save_scalers([column], scaler)

X=np.array(df.drop(['Status'],axis=1))
y=np.array(df['Status'])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
print(X_train.shape)
print(X_test.shape)

encoder = LabelEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train)
joblib.dump(encoder, "model/encoder_target.joblib")

y_test = encoder.transform(y_test)

"""## Model Decision Tree

"""

# Initialize the Decision Tree classifier
clf_dt = DecisionTreeClassifier()

# Define the grid of hyperparameters
param_grid1 = {
    'min_samples_leaf': [1, 10, 100],
    'max_depth': [1, 10, 20, 30],
    'criterion': ['gini', 'entropy']
}

# Initialize GridSearchCV
gs1 = GridSearchCV(
    estimator=clf_dt,
    param_grid=param_grid1,
    cv=5,
    n_jobs=-1,
    scoring='accuracy'
)

# Train the classifier using GridSearchCV
clf_dt_grid = gs1.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf_dt_grid.predict(X_test)

# Print the best parameters found by GridSearchCV
print("Best parameters:", clf_dt_grid.best_params_)

# Calculate and print the test accuracy
test_accuracy = accuracy_score(y_test, y_pred)
print("The test accuracy score of Decision Tree Classifier is", test_accuracy)

"""## Model Logistic Regression

"""

# Initialize the Logistic Regression classifier
clf_lr = LogisticRegression()

# Define the grid of hyperparameters
param_grid1 = {
    'C' :[0.1, 1, 10, 100],
    'max_iter': [100, 150, 250, 400],
    'multi_class': ['auto'],
    'solver': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'],
    'tol': [1e-4, 1e-5, 1e-6]
}

# Initialize GridSearchCV
gs1 = GridSearchCV(
    estimator=clf_lr,
    param_grid=param_grid1,
    cv=5,
    n_jobs=-1,
    scoring='accuracy'
)

# Train the classifier using GridSearchCV
clf_lr_grid = gs1.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf_lr_grid.predict(X_test)

# Print the best parameters found by GridSearchCV
print("Best parameters:", clf_lr_grid.best_params_)

# Calculate and print the test accuracy
test_accuracy = accuracy_score(y_test, y_pred)
print("The test accuracy score of Logistic Regression is", test_accuracy)

"""## Model Random Forest

"""

# Initialize the Random Forest classifier
clf_rf = RandomForestClassifier()

# Define the grid of hyperparameters
param_grid1 = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 5, 7, 9],
    'criterion': ['gini', 'entropy']
}

# Initialize GridSearchCV
gs1 = GridSearchCV(
    estimator=clf_rf,
    param_grid=param_grid1,
    cv=5,
    n_jobs=-1,
    scoring='accuracy'
)

# Train the classifier using GridSearchCV
clf_rf_grid = gs1.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf_rf_grid.predict(X_test)

# Print the best parameters found by GridSearchCV
print("Best parameters:", clf_rf_grid.best_params_)

# Calculate and print the test accuracy
test_accuracy = accuracy_score(y_test, y_pred)
print("The test accuracy score of Random Forest Classifier is", test_accuracy)

"""## Model Gradient Boosting

"""

# Initialize the Gradient Boosting classifier
clf_gb = GradientBoostingClassifier()

# Define the grid of hyperparameters
param_grid1 = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
}

# Initialize GridSearchCV
gs1 = GridSearchCV(
    estimator=clf_gb,
    param_grid=param_grid1,
    cv=5,
    scoring='accuracy'
)

# Train the classifier using GridSearchCV
clf_gb_grid = gs1.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf_gb_grid.predict(X_test)

# Print the best parameters found by GridSearchCV
print("Best parameters:", clf_gb_grid.best_params_)

# Calculate and print the test accuracy
test_accuracy = accuracy_score(y_test, y_pred)
print("The test accuracy score of Gradient Boosting Classifier is", test_accuracy)

"""# Evaluation

## Logistic Regression
"""

lr_cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix of Logistric Regression:")
print(lr_cm)

plt.figure(figsize=(6, 4))
sns.heatmap(lr_cm, annot=True, cmap='Spectral', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('Actual labels')
plt.title('Confusion Matrix of Logistic Regression')
plt.show()

print("The Classification Report of Logistic Regression")
print(classification_report(y_test, y_pred))

"""## Decision Tree

"""

dt_cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix of Decision Tree Classifier:")
print(dt_cm)

plt.figure(figsize=(6, 4))
sns.heatmap(dt_cm, annot=True, cmap='Spectral', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('Actual labels')
plt.title('Confusion Matrix of Decision Tree Classifier')
plt.show()

print("The Classification Report of Decision Tree Classifier")
print(classification_report(y_test, y_pred, zero_division=1))

"""## Random Forest

"""

rf_cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix of Random Forest Classifier:")
print(rf_cm)

plt.figure(figsize=(6, 4))
sns.heatmap(rf_cm, annot=True, cmap='Spectral', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('Actual labels')
plt.title('Confusion Matrix of Random Forest Classifier')
plt.show()

print("The Classification Report of Random Forest Classifier")
print(classification_report(y_test, y_pred))

"""## Gradient Boosting

"""

gb_cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix of Gradient Boosting Classifier:")
print(gb_cm)

plt.figure(figsize=(6, 4))
sns.heatmap(gb_cm, annot=True, cmap='Spectral', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('Actual labels')
plt.title('Confusion Matrix of Gradient Boosting Classifier')
plt.show()

print("The Classification Report of Gradient Boosting Classifier")
print(classification_report(y_test, y_pred))

"""# Deployment

"""

joblib.dump(clf_rf_grid, 'Random_Forest_Model.joblib')

df.to_csv('df_clean.csv', index=False)  # Optional: exclude the index column