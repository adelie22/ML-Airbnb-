import pandas as pd
from sklearn.model_selection import train_test_split
from data_preprocessing import combine_csv_files, preprocess_data
import machine_learning as ml
from predict_price import predict_features, train_models_for_features_rf, visualize_airbnb_locations, get_city_from_one_hot

def menu():
    data_dir = 'data'
    datas = None
    clean_data = None
    X = None
    y = None
    models_rf = None
    rf_model = None

    while True:
        print("\nMenu:")
        print("1. Load and preprocess data")
        print("2. Visualize relationships")
        print("3. Train and evaluate Linear Regression model")
        print("4. Train and evaluate Random Forest model")
        print("5. Train and evaluate Gradient Boosting model")
        print("6. Predict price")
        print("7. Visualize Airbnb locations")
        print("8. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            clean_data = preprocess_data(combine_csv_files(data_dir))
            datas = combine_csv_files(data_dir)
            print("Data loaded and preprocessed.")
        
        elif choice == '2':
            if datas is not None:
                ml.visualize_relationships(datas)
            else:
                print("Please load and preprocess the data first.")
        
        elif choice == '3':
            if clean_data is not None:
                room = input("Enter room type (e.g., 'Private room'): ")
                X, y = ml.prepare_model_data(clean_data, room)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                ml.train_and_evaluate_lr(X_train, X_test, y_train, y_test)
            else:
                print("Please load and preprocess the data first.")
        
        elif choice == '4':
            if clean_data is not None:
                room = input("Enter room type (e.g., 'Private room'): ")
                X, y = ml.prepare_model_data(clean_data, room)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                rf_model, rf_accuracy = ml.train_and_evaluate_rf(X_train, X_test, y_train, y_test)
                models_rf = train_models_for_features_rf(datas)
            else:
                print("Please load and preprocess the data first.")
        
        elif choice == '5':
            if clean_data is not None:
                room = input("Enter room type (e.g., 'Private room'): ")
                X, y = ml.prepare_model_data(clean_data, room)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                ml.train_and_evaluate_gb(X_train, X_test, y_train, y_test)
            else:
                print("Please load and preprocess the data first.")
        
        elif choice == '6':
            if rf_model is not None and models_rf is not None:
                lat = float(input("Enter latitude: "))
                long = float(input("Enter longitude: "))
                bedrooms = int(input("Enter number of bedrooms: "))
                city = input("Enter city name (e.g., 'amsterdam'): ")
                person_capacity = int(input("Enter person capacity: "))
                predicted_price = ml.predict_price_with_rf(lat, long, bedrooms, city, person_capacity, models_rf, rf_model)
                print(f'Predicted price: ${predicted_price:.2f}')
            else:
                print("Please train the Random Forest model first.")
        
        elif choice == '7':
            if clean_data is not None:
                visualize_airbnb_locations(clean_data)
            else:
                print("Please load and preprocess the data first.")
        elif choice == '8':
            break   
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    menu()