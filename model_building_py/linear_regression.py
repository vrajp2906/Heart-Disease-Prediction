# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score  # Import recall_score

def logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)  
    return accuracy, precision, recall  

def main(in_directory):
    df = pd.read_csv(in_directory)
    
    # Select relevant variables
    #selected_columns = ['BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'DiffWalking', 'Sex', 'AgeCategory', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'HeartDisease']
    #selected_columns = ['BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'DiffWalking', 'Sex', 'AgeCategory', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'HeartDisease']
    selected_columns = ['HeartDisease', 'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'Sex', 'Diabetic', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer']
    df = df[selected_columns]

    
    #y = df['HeartDisease'].map({'Yes': 1, 'No': 0})
    y = df['HeartDisease']
    X = df.drop('HeartDisease', axis=1)
    
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ])
    X = preprocessor.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    accuracy_lr, precision_lr, recall_lr = logistic_regression(X_train, y_train, X_test, y_test)  

    
    print(f'Logistic Regression Accuracy: {accuracy_lr:.3f}, Precision: {precision_lr:.3f}, Recall: {recall_lr:.3f}')  

if __name__ == '__main__':
    in_directory = '../db/heart_data_cleaned.csv'  
    main(in_directory)
