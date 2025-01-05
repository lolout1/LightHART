import os
import pandas as pd

def clean_and_rewrite(file_path, output_folder):
    """
    Cleans the file by splitting its single column into Timestamp, X, Y, Z columns,
    and saves it to the specified output folder with the same name.

    Args:
        file_path (str): Path to the input file.
        output_folder (str): Directory where the cleaned file will be saved.
    """
    try:
        # Load the file
        data = pd.read_csv(file_path, header=None)

        # Check if it's a single-column file (needs splitting)
        if data.shape[1] == 1:
            # Split by the semicolon delimiter
            data_split = data[0].str.split(';', expand=True)
            # Assign proper column names
            data_split.columns = ['Timestamp', 'X', 'Y', 'Z']
        elif data.shape[1] == 4 and list(data.iloc[0]) == ['Timestamp', 'X', 'Y', 'Z']:
            print(f"File {file_path} is already structured properly.")
            return
        else:
            print(f"File {file_path} has an unexpected structure and was skipped.")
            return

        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Save the cleaned file to the output folder with the same name
        output_path = os.path.join(output_folder, os.path.basename(file_path))
        data_split.to_csv(output_path, index=False)

        print(f"File {file_path} cleaned and saved to {output_path}.")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# Input folder containing files to clean
input_folder = "data/smartfallmm/needs_cleaning/young/accelerometer/watch"
output_folder = input_folder  # Rewriting files in the same folder

# List all files in the input folder
file_list = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.csv')]

# Process each file
for file_path in file_list:
    clean_and_rewrite(file_path, output_folder)

