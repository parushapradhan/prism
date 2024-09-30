import pandas as pd

# Function to apply rules. Replace this with your actual rules logic.
def apply_rules(row):
    # Rule 7: EMA_20 = Bin_2 AND Volume = Bin_2
    if row['EMA_20'] == 'Bin_2' and row['Volume'] == 'Bin_2':
        return 1
    
    # Rule 8: EMA_20 = Bin_2 AND RSI_05 = Bin_2 AND Volume = Bin_1
    if row['EMA_20'] == 'Bin_2' and row['RSI_05'] == 'Bin_2' and row['Volume'] == 'Bin_1':
        return 1
    
    # Rule 9: EMA_20 = Bin_2 AND RSI_09 = Bin_3 AND High = 'Bin_2'
    if row['EMA_20'] == 'Bin_2' and row['RSI_09'] == 'Bin_3' and row['High'] == 'Bin_2':
        return 1
    
    # Default return if none of the rules match
    return 0

# Function to apply the rules and add a Predicted_Value column
def calculate_predictions(df):
    df['Predicted_Value'] = df.apply(apply_rules, axis=1)
    return df

# Function to calculate accuracy
def calculate_accuracy(df):
    correct_predictions = (df['Predicted_Value'] == df['Target']).sum()
    accuracy = correct_predictions / len(df)
    return accuracy

# Load the validation data
validation_file_path = './new_val.csv'  # Update this path as needed

# Load the data into a pandas DataFrame
validation_df = pd.read_csv(validation_file_path)

# Calculate predictions using the defined rules
validation_df = calculate_predictions(validation_df)

# Calculate accuracy based on the 'Target' column
accuracy = calculate_accuracy(validation_df)

# Save the updated DataFrame with predictions to a new CSV file
output_file_path = './Binned_Training200.csv'
validation_df.to_csv(output_file_path, index=False)

print(f"Accuracy of the rules on the validation set: {accuracy * 100:.2f}%")
print(f"Validation results with predictions saved to: {output_file_path}")
