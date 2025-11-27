import pandas as pd
import numpy as np
from urllib.parse import urlparse

def combine_csv_files_with_links(file1, file2, output_file):
    """
    Combine two CSV files, remove duplicates, and create an Excel file with clickable LinkedIn URLs
    and properly formatted columns.
    
    Parameters:
    file1 (str): Path to the first CSV file
    file2 (str): Path to the second CSV file
    output_file (str): Path for the output Excel file
    """
    
    # Read both CSV files
    print(f"Reading {file1}...")
    df1 = pd.read_csv(file1)
    
    print(f"Reading {file2}...")
    df2 = pd.read_csv(file2)
    
    
    # Combine the dataframes
    print("Combining dataframes...")
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    # Check for duplicates based on linkedinUrl
    print("Checking for duplicates...")
    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['linkedinUrl'], keep='first')
    final_count = len(combined_df)
    duplicates_removed = initial_count - final_count
    
    print(f"Removed {duplicates_removed} duplicate entries.")
    
    # Create a function to make URLs clickable
    def make_clickable(url):
        # Check if the URL is valid and not empty
        if pd.isna(url) or str(url).strip() == '':
            return ''
        
        url = str(url).strip()
        
        # Add https:// if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        return f'=HYPERLINK("{url}", "View Profile")'
    
    # Apply the function to the linkedinUrl column
    print("Making URLs clickable...")
    combined_df['linkedinUrl'] = combined_df['linkedinUrl'].apply(make_clickable)
    
    # Save to Excel with formatted columns
    print(f"Saving to {output_file}...")
    
    # Create a Pandas Excel writer using XlsxWriter as the engine
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        # Write the dataframe to Excel
        combined_df.to_excel(writer, sheet_name='Profiles', index=False)
        
        # Get the workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['Profiles']
        
        # Add a header format
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        # Write the column headers with the defined format
        for col_num, value in enumerate(combined_df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Adjust column widths based on content
        for idx, col in enumerate(combined_df.columns):
            max_len = max(
                combined_df[col].astype(str).apply(len).max(),
                len(col)
            ) + 2  # Add a little extra space
            
            # Set a maximum width to avoid extremely wide columns
            max_width = 50
            worksheet.set_column(idx, idx, min(max_len, max_width))
        
        # Add autofilter to the header row
        worksheet.autofilter(0, 0, 0, len(combined_df.columns) - 1)
        
        # Freeze the first row
        worksheet.freeze_panes(1, 0)
    
    print(f"Successfully created {output_file} with {final_count} unique profiles.")
    print("The LinkedIn URLs are clickable hyperlinks in the Excel file.")

# Example usage
if __name__ == "__main__":
    # Replace these with your actual file paths
    file2 = "final_cleaned_dataset_rt.csv"  # Replace with your first CSV file path
    file1 = "cleaned_dataset_url.csv"
  
    # Replace with your second CSV file path
    output_file = "combined_profiles_fr.xlsx"  # Output Excel file name
    
    combine_csv_files_with_links(file1, file2, output_file)