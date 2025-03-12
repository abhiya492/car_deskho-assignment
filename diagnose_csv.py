import pandas as pd
import os
import sys

def analyze_csv(filepath):
    """Analyze a CSV file and print diagnostic information"""
    print(f"\n{'='*60}")
    print(f"ANALYZING CSV FILE: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    try:
        # Try different encoding options
        encodings = ['utf-8', 'latin1', 'utf-8-sig', 'cp1252']
        df = None
        successful_encoding = None
        
        for encoding in encodings:
            try:
                print(f"Trying to read with {encoding} encoding...")
                df = pd.read_csv(filepath, encoding=encoding)
                successful_encoding = encoding
                print(f"SUCCESS! Read with {encoding} encoding")
                break
            except Exception as e:
                print(f"Failed with {encoding} encoding: {str(e)}")
        
        if df is None:
            print("Could not read the CSV file with any encoding.")
            return
            
        # Basic info
        print(f"\nRows: {len(df)}, Columns: {len(df.columns)}")
        print(f"Successful encoding: {successful_encoding}")
        
        # Column analysis
        print("\nCOLUMN NAMES AND TYPES:")
        for col in df.columns:
            print(f"  {repr(col)} ({type(col).__name__}) - {df[col].dtype}")
            
        # Check for whitespace in column names
        whitespace_cols = [col for col in df.columns if str(col).strip() != str(col)]
        if whitespace_cols:
            print("\nWARNING: These columns have leading/trailing whitespace:")
            for col in whitespace_cols:
                print(f"  '{col}'")
                
        # Check for columns needed for visualization
        key_cols = ['region', 'price', 'bedrooms', 'bathrooms']
        print("\nKEY COLUMNS FOR VISUALIZATION:")
        for col_name in key_cols:
            # Case insensitive search
            found = False
            for actual_col in df.columns:
                if str(actual_col).lower() == col_name.lower():
                    found = True
                    print(f"  ✅ Found '{col_name}' as '{actual_col}'")
                    break
            if not found:
                print(f"  ❌ Missing '{col_name}'")
        
        # Sample data
        print("\nSAMPLE DATA (first 3 rows):")
        print(df.head(3))
        
        # Data types summary
        print("\nDATA TYPES SUMMARY:")
        print(df.dtypes.value_counts())
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print("\nMISSING VALUES:")
            for col, count in missing[missing > 0].items():
                print(f"  {col}: {count} missing values")
                
        print(f"\n{'='*60}")
        print("DIAGNOSIS COMPLETE")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error analyzing CSV: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python diagnose_csv.py <path_to_csv_file>")
    else:
        filepath = sys.argv[1]
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
        else:
            analyze_csv(filepath) 