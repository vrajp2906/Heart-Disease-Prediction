import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

def xg_boost(X_train, y_train, X_test, y_test):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    param_grid = {
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [100, 200],
        'gamma': [0, 0.1, 0.5]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)
    return evaluate_metrics(y_test, predictions, best_model, X_test)

def evaluate_metrics(y_test, predictions, model, X_test):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return accuracy, precision, recall, auc

def main(in_directory):
    df = pd.read_csv(in_directory)
    
    
    df_sampled = df.sample(frac=1.0, random_state=42)

    # Select relevant variables
    #selected_columns = ['Smoking', 'Stroke', 'DiffWalking', 'Sex', 'AgeCategory', 'Diabetic', 'PhysicalActivity', 'KidneyDisease', 'HeartDisease']
    selected_columns = ['HeartDisease', 'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'Sex', 'Diabetic', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer']
    df_sampled = df_sampled[selected_columns]

    # Data preprocessing
    #y = df_sampled['HeartDisease'].map({'Yes': 1, 'No': 0})
    y = df_sampled['HeartDisease']
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
    
    # XGBoost model training and evaluation
    accuracy_xgb, precision_xgb, recall_xgb, auc_xgb = xg_boost(X_train, y_train, X_test, y_test)

   
    print(f'XGBoost Classifier Accuracy: {accuracy_xgb:.3f}, Precision: {precision_xgb:.3f}, Recall: {recall_xgb:.3f}, AUC: {auc_xgb:.3f}')

if __name__ == '__main__':
    in_directory = '../db/heart_data_cleaned.csv' 
    main(in_directory)
