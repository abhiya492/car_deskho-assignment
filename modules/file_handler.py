import pandas as pd
import io
import os
from typing import Dict, Any, Tuple, Optional

class FileHandler:
    def __init__(self):
        self._dataframe = None
        self._file_info = None
    
    def validate_csv(self, file) -> Tuple[bool, str]:
        """Validate if the uploaded file is a valid CSV"""
        try:
            # Read the first few lines to check format
            if hasattr(file, 'name'):
                file_extension = os.path.splitext(file.name)[1].lower()
                if file_extension != '.csv':
                    return False, "Uploaded file is not a CSV file."
            
            # Check file size (max 25MB)
            if hasattr(file, 'size') and file.size > 25 * 1024 * 1024:  # 25MB in bytes
                return False, "File size exceeds the maximum limit of 25MB."
                
            return True, "Valid CSV file"
        except Exception as e:
            return False, f"Error validating file: {str(e)}"
    
    def load_csv(self, file) -> Tuple[bool, str]:
        """Load and validate a CSV file."""
        try:
            # Check file size (25MB limit)
            if os.path.getsize(file.name) > 25 * 1024 * 1024:
                return False, "File size exceeds 25MB limit"
            
            # Try to read the CSV file
            try:
                self._dataframe = pd.read_csv(file.name)
                print(f"CSV read successfully with {len(self._dataframe)} rows and {len(self._dataframe.columns)} columns")
            except pd.errors.ParserError:
                # Try with different encoding options
                try:
                    self._dataframe = pd.read_csv(file.name, encoding='latin1')
                    print("CSV read with latin1 encoding")
                except:
                    try:
                        self._dataframe = pd.read_csv(file.name, encoding='utf-8-sig')
                        print("CSV read with utf-8-sig encoding")
                    except:
                        return False, "Unable to parse CSV file with multiple encoding attempts"
            
            # Clean column names - trim whitespace and convert to string
            self._dataframe.columns = [str(col).strip() for col in self._dataframe.columns]
            
            # Store file information
            self._file_info = {
                "filename": os.path.basename(file.name),
                "rows": len(self._dataframe),
                "columns": len(self._dataframe.columns),
                "column_names": list(self._dataframe.columns),
                "column_types": self._dataframe.dtypes.to_dict(),
                "numeric_columns": self._dataframe.select_dtypes(include=['int64', 'float64']).columns.tolist(),
                "categorical_columns": self._dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
            }
            
            print(f"Processed columns: {self._file_info['column_names']}")
            
            return True, "File loaded successfully"
            
        except pd.errors.EmptyDataError:
            return False, "The file is empty"
        except Exception as e:
            return False, f"Error loading file: {str(e)}"
    
    def generate_file_info(self) -> None:
        """Generate basic information about the loaded DataFrame"""
        if self._dataframe is not None:
            self._file_info = {
                "filename": self._file_info["filename"],
                "rows": len(self._dataframe),
                "columns": len(self._dataframe.columns),
                "column_names": list(self._dataframe.columns),
                "data_types": {col: str(dtype) for col, dtype in self._dataframe.dtypes.items()},
                "memory_usage": f"{self._dataframe.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB",
                "sample": self._dataframe.head(5).to_dict(orient='records'),
                "numeric_columns": list(self._dataframe.select_dtypes(include=['number']).columns),
                "text_columns": list(self._dataframe.select_dtypes(include=['object']).columns),
                "has_null": self._dataframe.isnull().any().any(),
                "stats": self._generate_basic_stats()
            }
    
    def _generate_basic_stats(self) -> Dict[str, Any]:
        """Generate basic statistics for numeric columns"""
        if self._dataframe is None:
            return {}
        
        stats = {}
        numeric_df = self._dataframe.select_dtypes(include=['number'])
        
        if not numeric_df.empty:
            stats["description"] = numeric_df.describe().to_dict()
        
        return stats
    
    def get_dataframe(self) -> pd.DataFrame:
        """Return the loaded dataframe."""
        return self._dataframe
    
    def get_file_info(self) -> Dict[str, Any]:
        """Return information about the loaded file."""
        return self._file_info
    
    def get_column_data(self, column_name: str) -> Tuple[bool, Any]:
        """Get data for a specific column"""
        if self._dataframe is None:
            return False, "No CSV file loaded"
        
        if column_name not in self._dataframe.columns:
            return False, f"Column '{column_name}' not found in the CSV file"
        
        return True, self._dataframe[column_name]
    
    def get_column_summary(self) -> Dict[str, Any]:
        """Return summary statistics for the dataframe."""
        if self._dataframe is None:
            return {}
            
        summary = {
            "numeric_summary": self._dataframe.describe().to_dict(),
            "missing_values": self._dataframe.isnull().sum().to_dict(),
            "unique_values": {col: self._dataframe[col].nunique() 
                            for col in self._dataframe.columns}
        }
        return summary