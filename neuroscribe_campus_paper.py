"""
Original file is located at
    https://colab.research.google.com/drive/1-J9xrlnztcuA780Q8jeFkGC5Hl6EaYfO
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
from google.colab import files
uploaded = files.upload()

df = pd.read_csv('Placement_Data_Full_Class.csv')
print(df.head(7))

# Display basic information
print(df.info())
print(df.columns)
#drop slno column
if 'sl_no' in df.columns:
    df=df.drop('sl_no', axis=1)
else:
  print("sl_no not found")
# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values:\n", missing_values[missing_values > 0])

# Remove duplicate rows
df = df.drop_duplicates()

# Display data types to verify if categorical columns exist
print("\nData Types in the Dataset:\n", df.dtypes)

# Check for object (categorical) columns explicitly
categorical_cols = df.select_dtypes(include=['object','category']).columns.tolist()
print("\nCategorical Columns Found:", categorical_cols)

# If categorical columns exist, display descriptive statistics
if len(categorical_cols) > 0:
    print("\nDescriptive Statistics (Categorical Features):\n", df[categorical_cols].describe())
else:
    print("\nNo categorical columns found in the dataset.")

print("\nDescriptive Statistics (Numerical Features):\n", df.describe())

# ---------------------------
# Detect Data Types
# ---------------------------
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
# ---------------------------
# Visualize Categorical Features
# ---------------------------
if len(categorical_cols) > 0:
    for col in categorical_cols:
        plt.figure(figsize=(5, 3))
        sns.countplot(x=col, data=df, palette='Set2')
        plt.title(f'Distribution of {col}', fontsize=14)
        plt.xlabel(f'{col}', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
else:
    print("No categorical columns to visualize.")

# ---------------------------
# Visualize Numerical Features
# ---------------------------

# Distribution Plots
for col in numerical_cols:
    plt.figure(figsize=(6, 3))
    sns.histplot(df[col], kde=True, bins=20, color='skyblue')
    plt.title(f'Distribution of {col}', fontsize=14)
    plt.xlabel(f'{col}', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()





df['gender'].value_counts()
df['gender'].value_counts()

placed_counts = df.groupby(['gender', 'status'])['status'].count().unstack()

# Create the bar chart using pandas plot
placed_counts.plot(kind='bar', rot=0)
plt.title('Placement Status by Gender')
plt.xlabel('Gender')
plt.ylabel('Number of Students')
plt.legend(title='Placement Status')
plt.figure(figsize=(4,2))
plt.show()

hsc_s_status_counts = df.groupby(['hsc_s', 'status'])['status'].count().unstack()

hsc_s_status_counts.plot(kind='bar', rot=0)


plt.xlabel('Higher Secondary Specialization (hsc_s)')
plt.ylabel('Number of Students')
plt.title('Placement Status by Higher Secondary Specialization')
plt.legend(title='Placement Status')
plt.figure(figsize=(4,2))
plt.show()

hsc_s_status_counts = df.groupby(['degree_t', 'status'])['status'].count().unstack()

hsc_s_status_counts.plot(kind='bar', rot=0)

plt.xlabel('Field of Degree (degree_t)')
plt.ylabel('Number of Students')
plt.title('Placement Status by Field of Degree')

plt.legend(title='Placement Status')
plt.figure(figsize=(4,2))
plt.show()

hsc_s_status_counts = df.groupby(['workex', 'status'])['status'].count().unstack()

hsc_s_status_counts.plot(kind='bar', rot=0)

plt.xlabel('Work Experience (workex)')
plt.ylabel('Number of Students')
plt.title('Placement Status by Work Experience')

plt.legend(title='Placement Status')
plt.figure(figsize=(4,2))
plt.show()

hsc_s_status_counts = df.groupby(['specialisation', 'status'])['status'].count().unstack()

hsc_s_status_counts.plot(kind='bar', rot=0)

plt.xlabel('Specialisation (specialisation)')
plt.ylabel('Number of Students')
plt.title('Placement Status by Specialisation')

plt.legend(title='Placement Status')
plt.figure(figsize=(4,2))
plt.show()




# Boxplots for Outlier Detection
features = ['salary', 'degree_p', 'etest_p', 'mba_p','hsc_p','ssc_p']

# Plot boxplots for each feature separated by 'status'
for feature in features:
    plt.figure(figsize=(4, 5))
    sns.boxplot(x='status', y=feature, data=df)
    plt.title(f'{feature} by Placement Status')
    plt.tight_layout()
    plt.show()


# Correlation Heatmap
# ---------------------------
if len(numerical_cols) > 1:
    plt.figure(figsize=(8,8))
    heatmap = sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, square=True)
    heatmap.set_title('Correlation Heatmap of Numerical Features', fontsize=16)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12, rotation=0)
    plt.show()
else:
    print("Not enough numerical columns for a correlation heatmap.")

"""shape of data"""

print("DataFrame shape:", df.shape)

df['salary']=df['salary'].fillna(df['salary'].median())

"""**Dropping Outliers**"""

print(df.shape)
# Handle outliers in 'hsc_p'
Q1 = df['hsc_p'].quantile(0.25)
Q3 = df['hsc_p'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_hsc = df[(df['hsc_p'] < lower_bound) | (df['hsc_p'] > upper_bound)].index
df.drop(outliers_hsc, inplace=True)

# Reset index
df.reset_index(drop=True, inplace=True)

# Handle outliers in 'degree_p'
Q1 = df['degree_p'].quantile(0.25)
Q3 = df['degree_p'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_degree = df[(df['degree_p'] < lower_bound) | (df['degree_p'] > upper_bound)].index
df.drop(outliers_degree, inplace=True)

# Reset index again
df.reset_index(drop=True, inplace=True)

features_to_plot = ['degree_p', 'hsc_p']
for feature in features_to_plot:
    plt.figure(figsize=(5, 5))
    sns.boxplot(x='status', y=feature, data=df)
    plt.title(f'Boxplot of {feature} by Placement Status (Outliers Removed)')
    plt.tight_layout()
    plt.show()
print(df.shape)
#ENCODING
# Label Encoding
df['gender'] = df['gender'].map({'M': 0, 'F': 1})
df['ssc_b'] = df['ssc_b'].map({'Central': 0, 'Others': 1})
df['hsc_b'] = df['hsc_b'].map({'Central': 0, 'Others': 1})
df['hsc_s'] = df['hsc_s'].map({'Commerce': 0, 'Science': 1, 'Arts': 2})
df['degree_t'] = df['degree_t'].map({'Sci&Tech': 0, 'Comm&Mgmt': 1, 'Others': 2})
df['workex'] = df['workex'].map({'No': 0, 'Yes': 1})
df['status'] = df['status'].map({'Placed': 0, 'Not Placed': 1})
df['specialisation'] = df['specialisation'].map({'Mkt&Fin': 0, 'Mkt&HR': 1})

# handling missing values in the salary column
from sklearn.impute import SimpleImputer
import numpy as np
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer.fit(df[['salary']])
df['salary'] = imputer.transform(df[['salary']])
df.info()
plt.figure(figsize=(5, 5))
sns.boxplot(x='status', y='salary', data=df)
plt.title(f'Boxplot of {feature} by Placement Status (Outliers Removed)')
plt.tight_layout()
plt.show()

"""SPLITTING"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
X = df.drop('status', axis=1)
y = df['status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y,random_state = 0)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

"""Models"""

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Define your models and parameter grids
models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000), {
        'C': [0.1, 1, 10]
    }),
    "K-Nearest Neighbors": (KNeighborsClassifier(), {
        'n_neighbors': [3, 5, 7]
    }),
    "Support Vector Machine": (SVC(probability=True), {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }),
    "Random Forest": (RandomForestClassifier(), {
        'n_estimators': [100, 200],
        'max_depth': [None, 10]
    }),
    "Gradient Boosting": (GradientBoostingClassifier(), {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }),
    "XGBoost": (XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    })
}

best_models = {}

for name, (model, param_grid) in models.items():
    print(f"\n\n{name} Results")

    # Scale data if model requires it
    if name in ["Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine", "Gradient Boosting", "XGBoost"]:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled, X_test_scaled = X_train, X_test

    # Grid Search CV
    grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid.fit(X_train_scaled, y_train)
    best_model = grid.best_estimator_
    best_models[name] = (best_model, X_train_scaled, X_test_scaled)
    print("Best Parameters:", grid.best_params_)

    # Predictions & Metrics
    predictions = best_model.predict(X_test_scaled)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions, average='macro'))
    print("Recall:", recall_score(y_test, predictions, average='macro'))
    print("F1 Score:", f1_score(y_test, predictions, average='macro'))
    print("Classification Report:\n", classification_report(y_test, predictions))

    # Confusion Matrix Plot
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

!pip install lime shap

import lime.lime_tabular
import shap
import matplotlib.pyplot as plt # Ensure matplotlib is imported for SHAP plots
import numpy as np # Import numpy
from IPython.display import display # Ensure display is available for Jupyter

# Pick one test instance to explain
instance_index = 0

# Class names for binary classification
class_names = ['Not Placed', 'Placed']

# LIME Explainer (only needs to be created once)
# We use the scaled training data from one of the trained models to initialize LIME
# Assuming all models use the same scaling approach or that the first model's scaled data is representative
# Pass the training data as a NumPy array directly to the explainer
# Ensure the training data passed to LIME explainer is a NumPy array

# Access the scaled training data (NumPy array) from the first model in best_models
# best_models.values() gives a list of tuples (model, X_train_scaled, X_test_scaled)
# We take the first tuple [0], and the second element which is X_train_scaled [1]
# Ensure the training data used for the LIME explainer is a NumPy array
first_model_X_train_scaled_array = list(best_models.values())[0][1]

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    # Use the scaled training data (NumPy array) from the first model
    # Ensure this is a NumPy array for the explainer
    first_model_X_train_scaled_array,
    feature_names=X.columns.tolist(), # Use original DataFrame columns for feature names
    class_names=class_names,
    mode='classification'
)

# Loop through the best trained models
for name, (model, X_train_scaled_model, X_test_scaled_model) in best_models.items():
    print(f"\n--- {name} ---")

    # Ensure X_test_scaled_model is treated correctly.
    # Access the specific instance from the scaled test data.
    # Use .iloc for integer-based indexing on DataFrames, then .values to get the NumPy array for LIME/SHAP.
    if isinstance(X_test_scaled_model, pd.DataFrame):
         instance = X_test_scaled_model.iloc[instance_index].values
    else: # Assume it's already a NumPy array
         instance = X_test_scaled_model[instance_index, :]


    # ---- LIME ----
    # Use the actual best model object for prediction
    # lime_explainer was initialized with a representative scaled training set
    print(f"LIME explanation for {name}:")
    # Ensure the instance passed to explain_instance is a NumPy array
    lime_exp = lime_explainer.explain_instance(
        instance, # Pass the instance as a NumPy array
        model.predict_proba,
        num_features=10
    )

    # Assuming display is available in the environment (like Jupyter)
    try:
        # LIME's show_in_notebook returns HTML, display it
        # Ensure you have matplotlib and seaborn imported earlier for plots if show_table=False
        # show_in_notebook is preferred in Jupyter environments
        lime_exp.show_in_notebook(show_table=True)
    except ImportError:
        # Fallback if display is not available (e.g., plain script)
        print(lime_exp.as_html())


    # ---- SHAP ----
    print(f"SHAP explanation for {name}:")
    try:
        # Use the actual best model object and the scaled training/test data (NumPy arrays) for SHAP
        # SHAP typically works well with NumPy arrays for tree-based models and linear models after scaling
        # Ensure X_train_scaled_model and X_test_scaled_model are NumPy arrays
        # We now ensure the data passed to SHAP is a NumPy array
        if isinstance(X_train_scaled_model, pd.DataFrame):
             X_train_scaled_array_for_shap = X_train_scaled_model.values
        else:
             X_train_scaled_array_for_shap = X_train_scaled_model

        if isinstance(X_test_scaled_model, pd.DataFrame):
             X_test_scaled_array_for_shap = X_test_scaled_model.values
        else:
             X_test_scaled_array_for_shap = X_test_scaled_model

        shap_explainer = shap.Explainer(model, X_train_scaled_array_for_shap)
        shap_values = shap_explainer(X_test_scaled_array_for_shap)

        # Ensure SHAP plots are displayed
        shap.plots.beeswarm(shap_values, max_display=10)
        plt.title(f"SHAP Beeswarm Plot for {name}") # Add title for clarity
        # Assuming display is available for SHAP plots
        try:
            # Display the current matplotlib figure
            display(plt.gcf())
        except ImportError:
             # Fallback if display is not available
             plt.show()
        plt.close() # Close the plot figure after displaying
    except Exception as e:
        print(f"SHAP not supported for {name} or encountered an error: {e}")