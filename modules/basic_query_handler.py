import pandas as pd
import re
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import matplotlib.pyplot as plt

class BasicQueryHandler:
    """A basic query handler that uses simple pattern matching instead of an LLM"""
    
    def __init__(self):
        self.supported_queries = {
            "average": r"(?:average|avg|mean).*?(price|sqft|bedrooms|bathrooms|year_built|lot_size)",
            "maximum": r"(?:maximum|max|highest).*?(price|sqft|bedrooms|bathrooms|year_built|lot_size)",
            "minimum": r"(?:minimum|min|lowest).*?(price|sqft|bedrooms|bathrooms|year_built|lot_size)",
            "count": r"(?:count|how many).*?(bedrooms|bathrooms|garage|fireplace|pool)",
            "region": r"(?:region|area|location)",
            "chart": r"(?:chart|plot|graph|visualize|visualization)"
        }
        # Default column mapping (will be updated on first query)
        self.column_map = {}
        
        # Column aliases for common variations in CSV files
        self.column_aliases = {
            'region': ['region', 'location', 'area', 'neighborhood'],
            'sqft': ['sqft', 'square_feet', 'sq_ft', 'square_footage', 'size'],
            'garage': ['garage', 'garage_spaces', 'garages'],
            'fireplace': ['fireplace', 'fireplaces'],
            'pool': ['pool', 'has_pool'],
            'price': ['price', 'listing_price', 'sale_price', 'value']
        }

    def _get_actual_column_name(self, df: pd.DataFrame, col_name: str) -> str:
        """Get the actual column name from the dataframe, handling case insensitivity and aliases"""
        # If we've already mapped this column name, return the actual column name
        if col_name in self.column_map:
            return self.column_map[col_name]
        
        # Try to find a direct match (case-insensitive)
        col_name_lower = col_name.lower()
        for actual_col in df.columns:
            if actual_col.lower() == col_name_lower:
                self.column_map[col_name] = actual_col
                print(f"Found direct match for '{col_name}': '{actual_col}'")
                return actual_col
        
        # If no direct match, try aliases
        if col_name_lower in self.column_aliases:
            aliases = self.column_aliases[col_name_lower]
            for alias in aliases:
                for actual_col in df.columns:
                    if actual_col.lower() == alias.lower():
                        self.column_map[col_name] = actual_col
                        print(f"Found match for '{col_name}' using alias '{alias}': '{actual_col}'")
                        return actual_col
        
        # If no match found, return the original name (which will cause a controlled error later)
        print(f"No match found for column '{col_name}'")
        return col_name
        
    def _check_column_exists(self, df: pd.DataFrame, col_name: str) -> bool:
        """Check if a column exists in the dataframe, handling case insensitivity"""
        try:
            actual_col = self._get_actual_column_name(df, col_name)
            return actual_col in df.columns
        except:
            return False
            
    def _safe_groupby(self, df: pd.DataFrame, groupby_col: str, value_col: str) -> pd.DataFrame:
        """Safely perform a groupby operation, handling missing columns"""
        try:
            # Get actual column names
            actual_groupby_col = self._get_actual_column_name(df, groupby_col)
            actual_value_col = self._get_actual_column_name(df, value_col)
            
            # Check if columns exist
            if actual_groupby_col not in df.columns:
                print(f"Column not found: {groupby_col} (looking for {actual_groupby_col})")
                print(f"Available columns: {df.columns.tolist()}")
                return None
                
            if actual_value_col not in df.columns:
                print(f"Column not found: {value_col} (looking for {actual_value_col})")
                print(f"Available columns: {df.columns.tolist()}")
                return None
                
            # Perform the groupby
            return df.groupby(actual_groupby_col)[actual_value_col].mean().reset_index()
        except Exception as e:
            print(f"Error in groupby: {str(e)}")
            return None

    def process_query(self, query: str, df: pd.DataFrame, df_info: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Process a query and return an answer with optional visualization info"""
        query = query.lower()
        
        # Initialize column map with the dataframe's actual column names
        if not self.column_map:
            for col in df.columns:
                self.column_map[col.lower()] = col
        
        # Print available columns for debugging
        print(f"Available columns: {df.columns.tolist()}")
        
        # Try to match the query to our supported patterns
        for query_type, pattern in self.supported_queries.items():
            if re.search(pattern, query):
                if query_type == "average":
                    return self._handle_average_query(query, df)
                elif query_type == "maximum":
                    return self._handle_max_query(query, df)
                elif query_type == "minimum":
                    return self._handle_min_query(query, df)
                elif query_type == "count":
                    return self._handle_count_query(query, df)
                elif query_type == "region":
                    return self._handle_region_query(query, df)
                elif query_type == "chart":
                    return self._handle_chart_query(query, df)
        
        # If no pattern matched, return a default response
        return "I'm sorry, I don't understand that query. Please try a simpler question about averages, counts, or maximums/minimums of properties in the dataset.", None

    def _handle_average_query(self, query: str, df: pd.DataFrame) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Handle queries about averages"""
        for col in ["price", "sqft", "bedrooms", "bathrooms", "year_built", "lot_size"]:
            if col in query:
                actual_col = self._get_actual_column_name(df, col)
                
                if actual_col not in df.columns:
                    return f"I couldn't find the '{col}' column in your data. Available columns are: {', '.join(df.columns.tolist())}", None
                
                avg_value = df[actual_col].mean()
                formatted_value = f"${avg_value:,.2f}" if col == "price" else f"{avg_value:.2f}"
                
                # Create visualization info if it's a good candidate for visualization
                viz_info = None
                if "by region" in query or "by area" in query:
                    # Check if region column exists
                    region_col = self._get_actual_column_name(df, "region")
                    
                    if region_col not in df.columns:
                        no_region_msg = f"I couldn't find a 'region' column in your data to group by. Available columns are: {', '.join(df.columns.tolist())}"
                        return f"The average {col} overall is {formatted_value}. {no_region_msg}", None
                    
                    # Now safely do the groupby
                    region_avgs = self._safe_groupby(df, "region", col)
                    
                    if region_avgs is None:
                        return f"The average {col} overall is {formatted_value}. I couldn't group by region due to an error.", None
                    
                    viz_info = {
                        "type": "bar",
                        "columns": [region_col, actual_col],
                        "title": f"Average {col} by region",
                        "params": {}
                    }
                    
                    # Create a more detailed answer with regional breakdown
                    answer = f"The average {col} overall is {formatted_value}.\n\nBreakdown by region:\n"
                    for _, row in region_avgs.iterrows():
                        region_val = f"${row[actual_col]:,.2f}" if col == "price" else f"{row[actual_col]:.2f}"
                        answer += f"- {row[region_col]}: {region_val}\n"
                    return answer, viz_info
                
                return f"The average {col} is {formatted_value}.", viz_info
                
        return "Please specify what property you want the average of (price, sqft, bedrooms, etc.).", None

    def _handle_max_query(self, query: str, df: pd.DataFrame) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Handle queries about maximums"""
        for col in ["price", "sqft", "bedrooms", "bathrooms", "year_built", "lot_size"]:
            if col in query:
                actual_col = self._get_actual_column_name(df, col)
                
                if actual_col not in df.columns:
                    return f"I couldn't find the '{col}' column in your data. Available columns are: {', '.join(df.columns.tolist())}", None
                
                max_value = df[actual_col].max()
                max_row = df[df[actual_col] == max_value].iloc[0]
                formatted_value = f"${max_value:,.2f}" if col == "price" else f"{max_value:.2f}"
                
                # Get the actual column names for the result
                id_col = self._get_actual_column_name(df, "id")
                bed_col = self._get_actual_column_name(df, "bedrooms")
                bath_col = self._get_actual_column_name(df, "bathrooms")
                region_col = self._get_actual_column_name(df, "region")
                
                # Check which columns exist for the result message
                result = f"The maximum {col} is {formatted_value}"
                details = []
                
                if id_col in df.columns:
                    details.append(f"Property ID: {max_row[id_col]}")
                if bed_col in df.columns:
                    details.append(f"{max_row[bed_col]} bed")
                if bath_col in df.columns:
                    details.append(f"{max_row[bath_col]} bath")
                if region_col in df.columns:
                    details.append(f"{max_row[region_col]} region")
                
                if details:
                    result += f" ({', '.join(details)})"
                    
                return result + ".", None
                
        return "Please specify what property you want the maximum of (price, sqft, bedrooms, etc.).", None

    def _handle_min_query(self, query: str, df: pd.DataFrame) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Handle queries about minimums"""
        for col in ["price", "sqft", "bedrooms", "bathrooms", "year_built", "lot_size"]:
            if col in query:
                actual_col = self._get_actual_column_name(df, col)
                
                if actual_col not in df.columns:
                    return f"I couldn't find the '{col}' column in your data. Available columns are: {', '.join(df.columns.tolist())}", None
                
                min_value = df[actual_col].min()
                min_row = df[df[actual_col] == min_value].iloc[0]
                formatted_value = f"${min_value:,.2f}" if col == "price" else f"{min_value:.2f}"
                
                # Get the actual column names for the result
                id_col = self._get_actual_column_name(df, "id")
                bed_col = self._get_actual_column_name(df, "bedrooms")
                bath_col = self._get_actual_column_name(df, "bathrooms")
                region_col = self._get_actual_column_name(df, "region")
                
                # Check which columns exist for the result message
                result = f"The minimum {col} is {formatted_value}"
                details = []
                
                if id_col in df.columns:
                    details.append(f"Property ID: {min_row[id_col]}")
                if bed_col in df.columns:
                    details.append(f"{min_row[bed_col]} bed")
                if bath_col in df.columns:
                    details.append(f"{min_row[bath_col]} bath")
                if region_col in df.columns:
                    details.append(f"{min_row[region_col]} region")
                
                if details:
                    result += f" ({', '.join(details)})"
                    
                return result + ".", None
                
        return "Please specify what property you want the minimum of (price, sqft, bedrooms, etc.).", None

    def _handle_count_query(self, query: str, df: pd.DataFrame) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Handle count queries"""
        if "bedrooms" in query:
            bed_col = self._get_actual_column_name(df, "bedrooms")
            
            if bed_col not in df.columns:
                return f"I couldn't find a 'bedrooms' column in your data. Available columns are: {', '.join(df.columns.tolist())}", None
                
            counts = df[bed_col].value_counts().sort_index()
            viz_info = {
                "type": "bar",
                "columns": [bed_col, "count"],
                "title": "Count of properties by bedroom count",
                "params": {}
            }
            answer = "Count of properties by bedroom count:\n"
            for bed, count in counts.items():
                answer += f"- {bed} bedrooms: {count} properties\n"
            return answer, viz_info
            
        elif "bathrooms" in query:
            bath_col = self._get_actual_column_name(df, "bathrooms")
            
            if bath_col not in df.columns:
                return f"I couldn't find a 'bathrooms' column in your data. Available columns are: {', '.join(df.columns.tolist())}", None
                
            counts = df[bath_col].value_counts().sort_index()
            viz_info = {
                "type": "bar",
                "columns": [bath_col, "count"],
                "title": "Count of properties by bathroom count",
                "params": {}
            }
            answer = "Count of properties by bathroom count:\n"
            for bath, count in counts.items():
                answer += f"- {bath} bathrooms: {count} properties\n"
            return answer, viz_info
            
        elif "garage" in query:
            garage_col = self._get_actual_column_name(df, "garage")
            
            if garage_col not in df.columns:
                return f"I couldn't find a 'garage' column in your data. Available columns are: {', '.join(df.columns.tolist())}", None
                
            counts = df[garage_col].value_counts().sort_index()
            viz_info = {
                "type": "bar",
                "columns": [garage_col, "count"],
                "title": "Count of properties by garage spaces",
                "params": {}
            }
            answer = "Count of properties by garage spaces:\n"
            for garage, count in counts.items():
                answer += f"- {garage} garage spaces: {count} properties\n"
            return answer, viz_info
            
        elif "fireplace" in query or "fireplaces" in query:
            fireplace_col = self._get_actual_column_name(df, "fireplace")
            
            if fireplace_col not in df.columns:
                return f"I couldn't find a 'fireplace' column in your data. Available columns are: {', '.join(df.columns.tolist())}", None
                
            counts = df[fireplace_col].value_counts().sort_index()
            viz_info = {
                "type": "bar",
                "columns": [fireplace_col, "count"],
                "title": "Count of properties by number of fireplaces",
                "params": {}
            }
            answer = "Count of properties by number of fireplaces:\n"
            for fireplace, count in counts.items():
                answer += f"- {fireplace} fireplaces: {count} properties\n"
            return answer, viz_info
            
        elif "pool" in query:
            pool_col = self._get_actual_column_name(df, "pool")
            
            if pool_col not in df.columns:
                return f"I couldn't find a 'pool' column in your data. Available columns are: {', '.join(df.columns.tolist())}", None
                
            has_pool = (df[pool_col] == 1).sum()
            no_pool = (df[pool_col] == 0).sum()
            viz_info = {
                "type": "pie",
                "columns": [pool_col, "count"],
                "title": "Properties with/without pools",
                "params": {}
            }
            return f"There are {has_pool} properties with a pool and {no_pool} properties without a pool.", viz_info
            
        return "Please specify what feature you want to count (bedrooms, bathrooms, garage, fireplace, pool).", None

    def _handle_region_query(self, query: str, df: pd.DataFrame) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Handle region-related queries"""
        region_col = self._get_actual_column_name(df, "region")
        
        if region_col not in df.columns:
            return f"I couldn't find a 'region' column in your data. Available columns are: {', '.join(df.columns.tolist())}", None
            
        if "count" in query or "how many" in query:
            counts = df[region_col].value_counts()
            viz_info = {
                "type": "bar",
                "columns": [region_col, "count"],
                "title": "Count of properties by region",
                "params": {}
            }
            answer = "Count of properties by region:\n"
            for region, count in counts.items():
                answer += f"- {region}: {count} properties\n"
            return answer, viz_info
            
        elif "average price" in query or "avg price" in query:
            price_col = self._get_actual_column_name(df, "price")
            
            if price_col not in df.columns:
                return f"I couldn't find a 'price' column in your data. Available columns are: {', '.join(df.columns.tolist())}", None
                
            region_avgs = self._safe_groupby(df, "region", "price")
            
            if region_avgs is None:
                return "I couldn't calculate average prices by region due to an error.", None
                
            viz_info = {
                "type": "bar",
                "columns": [region_col, price_col],
                "title": "Average price by region",
                "params": {}
            }
            answer = "Average price by region:\n"
            for _, row in region_avgs.iterrows():
                answer += f"- {row[region_col]}: ${row[price_col]:,.2f}\n"
            return answer, viz_info
            
        return f"The data contains properties in different regions. You can ask about counts or average prices by region.", None

    def _handle_chart_query(self, query: str, df: pd.DataFrame) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Handle visualization queries"""
        # Try to extract information from the query
        viz_type = "bar"  # Default visualization type
        
        # Determine chart type from query
        if "bar" in query:
            viz_type = "bar"
        elif "pie" in query:
            viz_type = "pie"
        elif "line" in query:
            viz_type = "line"
        elif "scatter" in query:
            viz_type = "scatter"
        
        # Look for specific columns mentioned in the query
        columns = []
        for col in df.columns:
            if col.lower() in query.lower():
                columns.append(col)
        
        # If no specific columns were found, use columns based on viz type
        if not columns:
            if viz_type in ["bar", "pie"]:
                # Look for categorical columns first for these viz types
                for col in df.columns:
                    if df[col].dtype == 'object' or df[col].nunique() < 10:
                        columns.append(col)
                        break
                
                # If we found a category column, add a numeric column if needed
                if columns and viz_type == "bar":
                    for col in df.columns:
                        if pd.api.types.is_numeric_dtype(df[col].dtype) and col != columns[0]:
                            columns.append(col)
                            break
            
            elif viz_type in ["line", "scatter"]:
                # Find a good x and y axis
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col].dtype):
                        columns.append(col)
                        if len(columns) >= 2:
                            break
            
            # If still couldn't find appropriate columns
            if not columns:
                columns = df.columns[:min(2, len(df.columns))].tolist()
        
        # Create visualization info using the new format
        x_column = columns[0] if columns else None
        y_column = columns[1] if len(columns) > 1 else None
        
        viz_info = {
            "type": viz_type,
            "x_column": x_column,
            "y_column": y_column,
            "title": f"{viz_type.capitalize()} Chart of {' vs '.join(columns)}"
        }
        
        answer = f"Here's a {viz_type} chart visualizing "
        if len(columns) == 1:
            answer += f"the distribution of {columns[0]}."
        else:
            answer += f"{columns[0]} vs {columns[1]}."
        
        return answer, viz_info 