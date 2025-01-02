import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import data_preprocessing as data_pre
import predict_price as pp
def visualize_relationships(data):
    """
    Create scatter plots to visualize relationships between features and price.
    """
    plt.figure(figsize=(12, 6))
    
    # Price vs. Distance
    plt.subplot(2, 3, 1)
    sns.scatterplot(data=data, x='center_dist', y='price')
    plt.title('Price vs. Center Distance')
    plt.xlabel('Distance (dist)')
    plt.ylabel('Price')
    
    # Price vs. Metro Distance
    plt.subplot(2, 3, 2)
    sns.scatterplot(data=data, x='metro_dist', y='price')
    plt.title('Price vs. Metro Distance')
    plt.xlabel('Metro Distance (metro_dist)')
    plt.ylabel('Price')
    
    # Price vs. Guest Satisfaction
    plt.subplot(2, 3, 3)
    sns.scatterplot(data=data, x='guest_satisfaction', y='price')
    plt.title('Price vs. Guest Satisfaction')
    plt.xlabel('Guest Satisfaction (guest_satisfaction)')
    plt.ylabel('Price')
    
    # Price vs. Cleanliness Rating
    plt.subplot(2, 3, 4)
    sns.scatterplot(data=data, x='cleanliness_rating', y='price')
    plt.title('Price vs. Cleanliness Rating')
    plt.xlabel('Cleanliness Rating (cleanliness_rating)')
    plt.ylabel('Price')
    
    # Price vs. Attraction Index Normalized
    plt.subplot(2, 3, 5)
    sns.scatterplot(data=data, x='attr_index_norm', y='price')
    plt.title('Price vs. Attraction Index Normalized')
    plt.xlabel('Attraction Index Normalized (attr_index_norm)')
    plt.ylabel('Price')
    
    # Price vs. Restaurant Index Normalized
    plt.subplot(2, 3, 6)
    sns.scatterplot(data=data, x='rest_index_norm', y='price')
    plt.title('Price vs. Restaurant Index Normalized')
    plt.xlabel('Restaurant Index Normalized (rest_index_norm)')
    plt.ylabel('Price')
    
    plt.tight_layout()
    plt.show()

def calculate_custom_accuracy(y_true, y_pred, percentage):
    """
    Calculate the custom accuracy within a given percentage.
    """
    differences = np.abs((y_true - y_pred) / y_true) * 100
    within_percentage = differences <= percentage
    accuracy = np.mean(within_percentage) * 100
    return accuracy

def load_and_preprocess_data(data_dir):
    """
    Load and preprocess the data.
    """
    data = data_pre.combine_csv_files(data_dir)
    clean_data = data_pre.preprocess_data(data)
    return clean_data


def prepare_model_data(data, room):
    """
    Prepare the data for modeling by filtering and defining features and target variable.
    """
    # Filter for specified room type
    data = data[data['room_type'] == room]

    # Define the independent variables (X) and dependent variable (y)
    X = data[['attr_index_norm', 'rest_index_norm', 'metro_dist','center_dist', 'cleanliness_rating', 'guest_satisfaction', 'bedrooms', 'person_capacity'] + 
             [col for col in data.columns if col.startswith('city_')]]
    y = data['price']

    return X, y

def train_and_evaluate_lr(X_train, X_test, y_train, y_test):
    """
    Train and evaluate Linear Regression model.
    """
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    custom_accuracy_lr = calculate_custom_accuracy(y_test, y_pred_lr, percentage=15)

    print(f'Linear Regression - Mean Absolute Error: {mae_lr}')
    print(f'Linear Regression - R^2 Score: {r2_lr}')
    print(f'Linear Regression - Custom Accuracy (within 15%): {custom_accuracy_lr}%')

    coefficients = pd.DataFrame(lr_model.coef_, X_train.columns, columns=['Coefficient'])
    print(coefficients)

    # Plot Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_lr, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')
    plt.title('Linear Regression: Actual vs Predicted Prices')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.show()
    
    return custom_accuracy_lr
    
def train_and_evaluate_rf(X_train, X_test, y_train, y_test):
    """
    Train and evaluate Random Forest model without hyperparameter tuning.
    """
    rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    custom_accuracy_rf = calculate_custom_accuracy(y_test, y_pred_rf, percentage=15)

    print(f'Random Forest - Mean Absolute Error: {mae_rf}')
    print(f'Random Forest - R^2 Score: {r2_rf}')
    print(f'Random Forest - Custom Accuracy (within 15%): {custom_accuracy_rf}%')

    feature_importances_rf = pd.DataFrame(rf_model.feature_importances_, X_train.columns, columns=['Importance']).sort_values('Importance', ascending=False)
    print(feature_importances_rf)

    # Plot Feature Importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances_rf['Importance'], y=feature_importances_rf.index)
    plt.title('Random Forest: Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

    # Plot Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_rf, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')
    plt.title('Random Forest: Actual vs Predicted Prices')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.show()

    return rf_model, custom_accuracy_rf

def train_and_evaluate_gb(X_train, X_test, y_train, y_test):
    """
    Train and evaluate Gradient Boosting model without hyperparameter tuning.
    """
    gb_model = GradientBoostingRegressor(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=3)
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)

    mae_gb = mean_absolute_error(y_test, y_pred_gb)
    r2_gb = r2_score(y_test, y_pred_gb)
    custom_accuracy_gb = calculate_custom_accuracy(y_test, y_pred_gb, percentage=15)

    print(f'Gradient Boosting - Mean Absolute Error: {mae_gb}')
    print(f'Gradient Boosting - R^2 Score: {r2_gb}')
    print(f'Gradient Boosting - Custom Accuracy (within 15%): {custom_accuracy_gb}%')

    feature_importances_gb = pd.DataFrame(gb_model.feature_importances_, X_train.columns, columns=['Importance']).sort_values('Importance', ascending=False)
    print(feature_importances_gb)

    # Plot Feature Importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances_gb['Importance'], y=feature_importances_gb.index)
    plt.title('Gradient Boosting: Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

    # Plot Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_gb, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')
    plt.title('Gradient Boosting: Actual vs Predicted Prices')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.show()

    return custom_accuracy_gb

def predict_price_with_rf(lat, long, bedrooms, city, person_capacity, models_rf, rf_model):
    """
    Predict the price of an Airbnb listing using the trained Random Forest model.
    
    Parameters:
    lat (float): Latitude of the location.
    long (float): Longitude of the location.
    bedrooms (int): Number of bedrooms.
    city (str): City name.
    person_capacity (int): Person capacity of the listing.
    models_rf (dict): A dictionary containing the trained Random Forest models for feature prediction.
    rf_model (RandomForestRegressor): Trained Random Forest model for price prediction.
    
    Returns:
    float: Predicted price of the Airbnb listing.
    """
    # Predict other features based on lat and long
    predicted_features = pp.predict_features(models_rf, lat, long)

    # Create a DataFrame for the prediction
    input_data = {
        'lat': [lat],
        'long': [long],
        'bedrooms': [bedrooms],
        'person_capacity': [person_capacity],
        'city_' + city: [1]  # One-hot encode the city
    }

    # Add predicted features to the input data
    input_data.update(predicted_features)

    # Ensure all required columns are present
    required_columns = rf_model.feature_names_in_

    # Initialize missing city columns with 0
    for col in required_columns:
        if col not in input_data:
            input_data[col] = [0]

    # Convert to DataFrame and ensure the correct order of columns
    input_df = pd.DataFrame(input_data)
    input_df = input_df[required_columns]

    # Predict the price
    predicted_price = rf_model.predict(input_df)[0]

    return predicted_price