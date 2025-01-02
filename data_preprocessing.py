import pandas as pd
import os


def combine_csv_files(data_dir):
    """
    Combines multiple CSV files from a specified directory into a single DataFrame.
    
    Also rename columns, drop unnecessary columns, and handle missing values.
    
    Each CSV file is assumed to have a filename in the format 'city_day.csv'.
    The resulting DataFrame will have additional columns for 'city' and 'day'
    extracted from the filenames.
    
    
    
    Parameters:
    data_dir (str): Path to the directory containing the CSV files.
    
    Returns:
    pd.DataFrame: Combined DataFrame with data from all CSV files.
    """
    
    # Check if directory exists
    if not os.path.isdir(data_dir):
        raise ValueError(f"The directory {data_dir} does not exist.")
    
    # List all files in the directory
    file_list = os.listdir(data_dir)
    
    # Initialize an empty list to store DataFrames
    df_list = []

    # Iterate over each file in the directory
    for file_name in file_list:
        if file_name.endswith('.csv'):  # Ensure that the file is a CSV
            file_path = os.path.join(data_dir, file_name)
            
            try:
                # Extract city and day from the file name
                city = file_name.split('_')[0]
                day = file_name.split('_')[1].split('.')[0]  # Removing the '.csv' extension
                
                # Read the CSV file into a DataFrame
                df = pd.read_csv(file_path)
                
                # Add 'city' and 'day' columns
                df['city'] = city
                df['day'] = day
            
                # Append the DataFrame to the list
                df_list.append(df)
            
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
    
    # Check if CSV files exist in directory
    if not df_list:
        raise ValueError(f"No CSV files found in the directory {data_dir}.")
    
    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # Define a dictionary with the old column names as keys and new column names as values
    column_name_mapping = {
        'realSum': 'price',
        'host_is_superhost': 'superhost',
        'biz': 'business',
        'guest_satisfaction_overall': 'guest_satisfaction',
        'dist': 'center_dist',
        'lng': 'long'
        # Add more mappings as needed
    }

    # Rename the columns
    combined_df.rename(columns=column_name_mapping, inplace=True)
    
    # Drop the 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in combined_df.columns:
        combined_df.drop(columns=['Unnamed: 0'], inplace=True)
    
    # Drop rows with any NaN values
    combined_df = combined_df.dropna()
    
    return combined_df


def preprocess_data(cleaned_data):
    """
    Cleans the given DataFrame by removing the highest 25% of the price data,
    and one-hot encoding the 'city' column.
    
    Parameters:
    data (pd.DataFrame): The DataFrame to clean.
    
    Returns:
    pd.DataFrame: Processed DataFrame ready for modeling.
    """
    
    # Remove the highest 25% of the price data
    price_75th_percentile = cleaned_data['price'].quantile(0.75)
    filtered_data = cleaned_data[cleaned_data['price'] <= price_75th_percentile]
    
    # One-hot encode the 'city' column
    processed_data = pd.get_dummies(filtered_data, columns=['city'])

    return processed_data