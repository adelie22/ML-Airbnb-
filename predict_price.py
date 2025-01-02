import pandas as pd
import folium
from folium.plugins import MarkerCluster, MousePosition
import webbrowser
import os
import data_preprocessing as data_pre
import machine_learning as ml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def get_city_from_one_hot(row, city_columns):
    """
    Given a row and the list of city columns, return the city name.
    """
    for city in city_columns:
        if row[city] == 1:
            return city.split('_')[1]  # Extract city name from column name
    return "Unknown"

def visualize_airbnb_locations(data):
    """
    Create an interactive map to visualize Airbnb locations.
    """
    # Create a map centered around the average location
    center_lat = data['lat'].mean()
    center_long = data['long'].mean()
    airbnb_map = folium.Map(location=[center_lat, center_long], zoom_start=12)

    # Extract city columns
    city_columns = [col for col in data.columns if col.startswith('city_')]

    # Add markers for each Airbnb listing
    marker_cluster = MarkerCluster().add_to(airbnb_map)
    for idx, row in data.iterrows():
        city = get_city_from_one_hot(row, city_columns)
        folium.Marker(
            location=[row['lat'], row['long']],
            popup=f"Price: ${row['price']}<br>Room Type: {row['room_type']}<br>City: {city}"
        ).add_to(marker_cluster)
    
    # Add Mouse Position plugin to display coordinates
    formatter = "function(num) {return L.Util.formatNum(num, 5);};"
    mouse_position = MousePosition(
        position='topright',
        separator=' Long: ',
        empty_string='NaN',
        lng_first=False,
        num_digits=5,
        prefix='Lat:',
        lat_formatter=formatter,
        lng_formatter=formatter,
    )
    airbnb_map.add_child(mouse_position)

    # Add LatLngPopup plugin to allow clicking on the map to place a marker with coordinates
    airbnb_map.add_child(folium.LatLngPopup())

    # Function to add a marker on map click
    def on_map_click(e):
        folium.Marker(
            location=[e.latlng[0], e.latlng[1]],
            popup=f"Lat: {e.latlng[0]}, Long: {e.latlng[1]}"
        ).add_to(airbnb_map)

    airbnb_map.add_child(folium.ClickForMarker(popup="Lat: {lat}, Long: {lng}"))

    # Save the map to an HTML file
    map_file = "airbnb_locations_map.html"
    airbnb_map.save(map_file)
    # Open the HTML file in the default web browser
    webbrowser.open(f'file://{os.path.abspath(map_file)}')


def train_models_for_features_rf(data):
    """
    Train Random Forest regression models to predict attr_index_norm, rest_index_norm, metro_dist, and center_dist based on lat and long.
    
    Parameters:
    data (pd.DataFrame): DataFrame containing the training data.
    
    Returns:
    dict: A dictionary containing the trained models.
    """
    
    features = ['attr_index_norm', 'rest_index_norm', 'metro_dist', 'center_dist']
    models = {}
    
    for feature in features:
        X = data[['lat', 'long']]
        y = data[feature]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(random_state=42, n_estimators=100)
        model.fit(X_train, y_train)
        
        print(f'Model for {feature} trained successfully.')
        
        models[feature] = model
    
    return models


def predict_features(models, lat, long):
    """
    Predict attr_index_norm, rest_index_norm, metro_dist, and center_dist based on given latitude and longitude.
    
    Parameters:
    models (dict): A dictionary containing the trained models.
    lat (float): Latitude of the location.
    long (float): Longitude of the location.
    
    Returns:
    dict: A dictionary containing the predicted values.
    """
    features = ['attr_index_norm', 'rest_index_norm', 'metro_dist', 'center_dist']
    predictions = {}
    
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({'lat': [lat], 'long': [long]})

    for feature in features:
        model = models[feature]
        prediction = model.predict(input_data)[0]
        predictions[feature] = prediction

    return predictions