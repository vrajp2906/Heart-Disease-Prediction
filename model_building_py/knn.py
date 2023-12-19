# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier

def knn_classifier(X_train, y_train, X_test, y_test):
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    return accuracy, precision, recall

def main(in_directory):
    df = pd.read_csv(in_directory)
    
    #df_sampled = df;
    df_sampled = df.sample(frac=1, random_state=42)

    # Select relevant variables
    #selected_columns = ['Smoking', 'Stroke', 'DiffWalking', 'Sex', 'AgeCategory', 'Diabetic', 'PhysicalActivity', 'KidneyDisease', 'HeartDisease']
    '''
    selected_columns = [
    'HeartDisease', 'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 
    'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory', 
    'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime', 
    'Asthma', 'KidneyDisease', 'SkinCancer']
    '''
    
    selected_columns = ['HeartDisease', 'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'Sex', 'Diabetic', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer']

    df_sampled = df_sampled[selected_columns]

    # Data preprocessing
    #y = df_sampled['HeartDisease'].map({'Yes': 1, 'No': 0})
    y = df_sampled['HeartDisease'].map({'Yes': 1, 'No': 0})
    X = df_sampled.drop('HeartDisease', axis=1)
    
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ])
    X = preprocessor.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # KNN Model training and evaluation
    accuracy_knn, precision_knn, recall_knn = knn_classifier(X_train, y_train, X_test, y_test)

    print(f'KNN Classifier Accuracy: {accuracy_knn:.3f}, Precision: {precision_knn:.3f}, Recall: {recall_knn:.3f}')

if __name__ == '__main__':
    in_directory = '../db/heart_data.csv'  # Replace with the path to your dataset
    main(in_directory)


