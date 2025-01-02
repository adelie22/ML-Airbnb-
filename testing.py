import machine_learning as ml
import data_preprocessing as dp
import predict_price as pp
from sklearn.model_selection import train_test_split

def test_methods(data_dir='data', room_type='Private room', lr_threshold=51, rf_threshold=70, gb_threshold=53):
    """
    Function to test the methods for loading data, preprocessing, training, and evaluating models.
    It also predicts the price using the trained Random Forest model.
    
    Parameters:
    data_dir (str): Directory where the data files are stored.
    room_type (str): Type of room to filter the data.
    lr_threshold (int): Threshold for Linear Regression model accuracy.
    rf_threshold (int): Threshold for Random Forest model accuracy.
    gb_threshold (int): Threshold for Gradient Boosting model accuracy.
    """
    # Load and preprocess data
    clean_data = ml.load_and_preprocess_data(data_dir)
    
    # Prepare model data
    X, y = ml.prepare_model_data(clean_data, room_type)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate Linear Regression model
    lr_accuracy = ml.train_and_evaluate_lr(X_train, X_test, y_train, y_test)
    assert lr_accuracy <= lr_threshold, f"Linear Regression accuracy {lr_accuracy} is above the threshold {lr_threshold}"
    
    # Train and evaluate Random Forest model
    rf_model, rf_accuracy = ml.train_and_evaluate_rf(X_train, X_test, y_train, y_test)
    assert rf_accuracy <= rf_threshold, f"Random Forest accuracy {rf_accuracy} is above the threshold {rf_threshold}"
    
    # Train and evaluate Gradient Boosting model
    gb_accuracy = ml.train_and_evaluate_gb(X_train, X_test, y_train, y_test)
    assert gb_accuracy <= gb_threshold, f"Gradient Boosting accuracy {gb_accuracy} is above the threshold {gb_threshold}"
    
    print("All model accuracy tests passed.")
    
    # Train Random Forest models for feature prediction
    models_rf = pp.train_models_for_features_rf(clean_data)
    
    # Predict the price using the trained Random Forest model
    predicted_price = ml.predict_price_with_rf(52.41772, 4.90569, 1, 'amsterdam', 2, models_rf, rf_model)
    print(f"Predicted price: ${predicted_price:.2f}")
    
    assert predicted_price <= 250
    
    print("Prediction successful.")

def test_preprocess_data(data_dir='data'):
    """
    Function to test data loading and preprocessing.
    It verifies if the expected columns are present in the preprocessed data.
    
    Parameters:
    data_dir (str): Directory where the data files are stored.
    """
    # Load and preprocess data
    clean_data = ml.load_and_preprocess_data(data_dir)
    
    # Expected columns after preprocessing
    expected_columns = {'price', 'room_type', 'room_shared', 'room_private', 'person_capacity',
       'superhost', 'multi', 'business', 'cleanliness_rating',
       'guest_satisfaction', 'bedrooms', 'center_dist', 'metro_dist',
       'attr_index', 'attr_index_norm', 'rest_index', 'rest_index_norm',
       'long', 'lat', 'day', 'city_amsterdam', 'city_athens', 'city_barcelona',
       'city_berlin', 'city_budapest', 'city_lisbon', 'city_london',
       'city_paris', 'city_rome', 'city_vienna'}
    
    # Assert that all expected columns are present after preprocessing
    assert expected_columns.issubset(set(clean_data.columns)), "Not all expected columns are present after preprocessing data."
    
    print("Preprocessing test passed.")

def main():
    """
    Main function to run the test methods and preprocessing test.
    """
    test_preprocess_data()
    test_methods()
    

if __name__ == "__main__":
    main()