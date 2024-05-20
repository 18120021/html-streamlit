def score():
    import pandas as pd
    import numpy as np
    import warnings
    warnings.filterwarnings("ignore", category=Warning)
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.model_selection import GridSearchCV, cross_val_score

    # Load dataset
    traffic_data_df = pd.read_csv('network_traffic_5G_final_data.csv')

    # Drop rows with missing values
    traffic_data_df.dropna(inplace=True)

    # Shuffle dataset to introduce randomness
    traffic_data_df = traffic_data_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Select features and target variable
    features = ['packet_loss', 'throughput', 'latency', 'jitter', 'ping', 'packet_length']
    target = 'traffic_type'

    # Encode categorical target variable 'traffic_type' using LabelEncoder
    label_encoder = LabelEncoder()
    traffic_data_df[target] = label_encoder.fit_transform(traffic_data_df[target])

    # Prepare the feature matrix and target vector
    X = traffic_data_df[features]
    y = traffic_data_df[target]

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [0.001, 0.01, 0.1, 1],
        'kernel': ['rbf']
    }

    # Initialize SVM classifier
    svm_model = SVC()

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_scaled, y)

    # Get the best hyperparameters
    best_params = grid_search.best_params_

    # Initialize SVM classifier with the best hyperparameters
    best_svm_model = SVC(**best_params)

    # Fit the model
    best_svm_model.fit(X_scaled, y)

    # Perform cross-validation on the SVM model
    cv_scores = cross_val_score(best_svm_model, X_scaled, y, cv=5, scoring='accuracy')

    # Calculate mean cross-validation accuracy
    mean_cv_accuracy = np.mean(cv_scores)

    # Predictions using the voting classifier
    y_pred_voting = best_svm_model.predict(X_scaled)

    # Calculate metrics
    accuracy_voting = accuracy_score(y, y_pred_voting)
    precision_voting = precision_score(y, y_pred_voting, average='weighted')
    recall_voting = recall_score(y, y_pred_voting, average='weighted')
    f1_voting = f1_score(y, y_pred_voting, average='weighted')

    # Create a DataFrame with the metrics data
    metrics_data = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "Cross-Val"],
    "Value": [accuracy_voting, precision_voting, recall_voting, f1_voting, mean_cv_accuracy]
    }

    metrics_df = pd.DataFrame(metrics_data)
    
    return metrics_df