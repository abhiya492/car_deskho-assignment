import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional, Tuple, List, Union
import io
import base64

class Visualizer:
    def __init__(self):
        self.supported_plot_types = [
            'bar', 'line', 'scatter', 'pie', 'histogram', 
            'box', 'violin', 'heatmap', 'area'
        ]
        self.default_width = 800
        self.default_height = 500
        
        # Define column name mappings to handle different CSV formats
        self.column_mappings = {
            # Common alternate names for geographic data
            'region': ['region', 'location', 'area', 'city', 'state', 'country', 'zone'],
            'country': ['country', 'nation', 'land'],
            'city': ['city', 'town', 'municipality'],
            
            # Common alternate names for categorical data
            'category': ['category', 'type', 'class', 'group', 'classification', 'property_type'],
            'product': ['product', 'item', 'good', 'merchandise', 'service'],
            'status': ['status', 'condition', 'state', 'health'],
            
            # Common alternate names for numeric data
            'value': ['value', 'price', 'cost', 'amount', 'sum', 'total'],
            'quantity': ['quantity', 'count', 'number', 'amount', 'total'],
            'age': ['age', 'years', 'duration', 'period'],
            
            # Common alternate names for time data
            'date': ['date', 'day', 'time', 'timestamp', 'year_built'],
            'year': ['year', 'yr', 'annual', 'year_built'],
            'month': ['month', 'mo'],
            
            # Common alternate names for property data
            'size': ['size', 'square_feet', 'area', 'square_footage', 'sq_ft', 'sqft'],
            'rooms': ['rooms', 'bedrooms', 'bathrooms', 'beds', 'baths'],
        }
        
        # Visualization debug flag
        self.debug = True
    
    def find_column_by_type(self, df: pd.DataFrame, preferred_types: List[str]) -> Optional[str]:
        """Find a column in the dataframe that matches any of the preferred types"""
        for col_type in preferred_types:
            for col_name in df.columns:
                for mapped_col in self.column_mappings.get(col_type, [col_type]):
                    if mapped_col.lower() in col_name.lower():
                        if self.debug:
                            print(f"Found column '{col_name}' matching type '{col_type}'")
                        return col_name
        return None

    def create_visualization(
        self, 
        data: pd.DataFrame, 
        viz_type: str, 
        x_column: Optional[str] = None, 
        y_column: Optional[str] = None,
        color_by: Optional[str] = None,
        title: str = "Visualization"
    ) -> Dict[str, Any]:
        """Create a visualization based on the data and type."""
        try:
            print(f"Creating visualization of type: {viz_type}")
            print(f"Data columns: {list(data.columns)}")
            print(f"Requested columns - x: {x_column}, y: {y_column}, color: {color_by}")
            
            # If columns weren't specified, try to find appropriate columns
            if not x_column:
                # Try to find category-like columns for x axis
                x_column = self.find_column_by_type(data, ['category', 'region', 'product', 'status'])
            
            if not y_column:
                # Try to find numeric columns for y axis
                y_column = self.find_column_by_type(data, ['value', 'quantity', 'price', 'size'])
            
            if not color_by and viz_type not in ['pie', 'count']:
                # Try to find a categorical column for color
                color_by = self.find_column_by_type(data, ['category', 'status'])
            
            print(f"Using columns - x: {x_column}, y: {y_column}, color: {color_by}")
                
            if viz_type == "bar":
                return self._create_bar_chart(data, x_column, y_column, color_by, title)
            elif viz_type == "line":
                return self._create_line_chart(data, x_column, y_column, color_by, title)
            elif viz_type == "scatter":
                return self._create_scatter_plot(data, x_column, y_column, color_by, title)
            elif viz_type == "pie":
                return self._create_pie_chart(data, x_column, y_column, title)
            elif viz_type == "count":
                # Special case for counting occurrences of values in a column
                if x_column:
                    return self._create_count_based_viz(data, x_column, color_by, title, viz_type="bar")
                else:
                    return {"error": "No column specified for count visualization"}
            else:
                return {"error": f"Unsupported visualization type: {viz_type}"}
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error creating visualization: {str(e)}")
            print(error_details)
            return {
                "error": f"Failed to create visualization: {str(e)}",
                "details": error_details,
                "figure": self._create_error_figure(f"Visualization Error: {str(e)}")
            }
    
    def execute_visualization_code(self, dataframe: pd.DataFrame, code: str) -> Optional[Any]:
        """Execute the Python code generated by the LLM to create a visualization"""
        if dataframe is None or dataframe.empty or not code:
            return None
        
        try:
            # Create a local variables dict with the dataframe and libraries
            local_vars = {
                'df': dataframe,
                'pd': pd,
                'plt': plt,
                'px': px,
                'go': go
            }
            
            # Execute the code
            exec(code, {}, local_vars)
            
            # Check if a plotly figure was created
            if 'fig' in local_vars and isinstance(local_vars['fig'], (go.Figure, px.Figure)):
                return local_vars['fig']
            
            # If no plotly figure was created but matplotlib was used
            if 'plt' in local_vars and plt.get_fignums():
                # Convert matplotlib figure to plotly
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                plt.close()  # Close the matplotlib figure
                
                # Create a simple plotly figure with the image
                fig = go.Figure()
                img_str = base64.b64encode(buf.read()).decode()
                
                fig.add_layout_image(
                    dict(
                        source=f'data:image/png;base64,{img_str}',
                        x=0,
                        y=0,
                        xref="paper",
                        yref="paper",
                        sizex=1,
                        sizey=1,
                        sizing="stretch"
                    )
                )
                
                fig.update_layout(
                    width=800,
                    height=600,
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                
                return fig
            
            return None
        
        except Exception as e:
            print(f"Error executing visualization code: {str(e)}")
            return None
    
    def create_visualization_from_info(self, dataframe: pd.DataFrame, viz_info: Dict[str, Any]) -> Optional[Any]:
        """Bridge method to maintain compatibility with old code while using new system"""
        print(f"Creating visualization from info: {viz_info}")
        
        # Extract parameters from viz_info
        viz_type = viz_info.get("type", "bar").lower()
        x_column = viz_info.get("x_column")
        y_column = viz_info.get("y_column")
        color_by = viz_info.get("color_by")
        title = viz_info.get("title", "Data Visualization")
        
        # Call the new visualization system
        result = self.create_visualization(
            data=dataframe,
            viz_type=viz_type,
            x_column=x_column,
            y_column=y_column,
            color_by=color_by,
            title=title
        )
        
        # Convert new system result to old format (just return the figure)
        if result and "figure" in result:
            return result["figure"]
        
        # If there was an error or no figure, return None as old method did
        return None
    
    def _create_bar_chart(self, df: pd.DataFrame, columns: list, title: str, params: Dict[str, Any]) -> go.Figure:
        """Create a bar chart."""
        try:
            print(f"Creating bar chart with columns: {columns}, df shape: {df.shape}")
            
            if len(columns) == 1:
                # Single column - show value counts
                value_counts = df[columns[0]].value_counts().sort_index()
                # Create a plotly figure manually to ensure it works
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[str(x) for x in value_counts.index],  # Convert to strings to avoid issues
                    y=value_counts.values,
                    marker_color='steelblue'
                ))
                fig.update_layout(
                    title=title or f"Count of {columns[0]}",
                    xaxis_title=columns[0],
                    yaxis_title="Count"
                )
                print(f"Created bar chart with {len(value_counts)} categories")
                return fig
            
            elif len(columns) == 2 and columns[1] == "count":
                # Special case for count visualizations
                # This handles the case where the second column is named 'count' but doesn't exist in df
                value_counts = df[columns[0]].value_counts().sort_index()
                # Create a plotly figure manually
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[str(x) for x in value_counts.index],  # Convert to strings to avoid issues
                    y=value_counts.values,
                    marker_color='steelblue'
                ))
                fig.update_layout(
                    title=title or f"Count of {columns[0]}",
                    xaxis_title=columns[0],
                    yaxis_title="Count"
                )
                print(f"Created count-based bar chart with {len(value_counts)} categories")
                return fig
            
            else:
                # Two columns - one for x, one for y
                try:
                    # Try using px.bar first (this is what was failing before)
                    fig = px.bar(df, x=columns[0], y=columns[1])
                    print("Created bar chart with px.bar")
                    return fig
                except Exception as e:
                    print(f"Error with px.bar: {str(e)}, falling back to go.Figure")
                    # Fall back to manual creation
                    x_values = df[columns[0]].tolist()
                    y_values = df[columns[1]].tolist()
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=[str(x) for x in x_values],  # Convert to strings to avoid issues
                        y=y_values,
                        marker_color='steelblue'
                    ))
                    fig.update_layout(
                        title=title or f"{columns[1]} by {columns[0]}",
                        xaxis_title=columns[0],
                        yaxis_title=columns[1]
                    )
                    print("Created bar chart with go.Figure")
                    return fig
        except Exception as e:
            print(f"Error in _create_bar_chart: {str(e)}")
            # Return a simple empty figure rather than None
            fig = go.Figure()
            fig.update_layout(
                title="Error creating chart",
                annotations=[
                    dict(
                        text=f"Error: {str(e)}",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5
                    )
                ]
            )
            return fig
        
    def _create_line_chart(self, df: pd.DataFrame, columns: list, title: str, params: Dict[str, Any]) -> go.Figure:
        """Create a line chart."""
        try:
            if len(columns) == 1:
                # Single column - show value counts over unique values
                value_counts = df[columns[0]].value_counts().sort_index()
                fig = px.line(x=value_counts.index, y=value_counts.values)
                fig.update_layout(xaxis_title=columns[0], yaxis_title="Count")
                return fig
            elif len(columns) >= 2:
                # Check if the second column is a special "count" placeholder
                if columns[1] == "count" and "count" not in df.columns:
                    value_counts = df[columns[0]].value_counts().sort_index()
                    fig = px.line(x=value_counts.index, y=value_counts.values)
                    fig.update_layout(xaxis_title=columns[0], yaxis_title="Count")
                    return fig
                else:
                    # Use the columns as provided
                    return px.line(df, x=columns[0], y=columns[1:])
            else:
                # Fallback to a default column if none provided
                return px.line(df, x=df.columns[0])
        except Exception as e:
            print(f"Error in line chart creation: {str(e)}")
            # Create a very simple line chart as fallback
            try:
                x_col = df.select_dtypes(include=['number']).columns[0]
                fig = px.line(df, x=x_col)
                return fig
            except:
                # If all else fails, return None
                return None
        
    def _create_scatter_plot(self, df: pd.DataFrame, columns: list, title: str, params: Dict[str, Any]) -> go.Figure:
        """Create a scatter plot."""
        color_col = params.get('color')
        if color_col and color_col in df.columns:
            return px.scatter(df, x=columns[0], y=columns[1], color=color_col)
        return px.scatter(df, x=columns[0], y=columns[1])
        
    def _create_histogram(self, df: pd.DataFrame, columns: list, title: str, params: Dict[str, Any]) -> go.Figure:
        """Create a histogram."""
        return px.histogram(df, x=columns[0], nbins=params.get('nbins', 30))
        
    def _create_box_plot(self, df: pd.DataFrame, columns: list, title: str, params: Dict[str, Any]) -> go.Figure:
        """Create a box plot."""
        if len(columns) == 2:
            return px.box(df, x=columns[0], y=columns[1])
        return px.box(df, y=columns[0])
        
    def _create_pie_chart(self, df: pd.DataFrame, columns: list, title: str, params: Dict[str, Any]) -> go.Figure:
        """Create a pie chart."""
        try:
            print(f"Creating pie chart with columns: {columns}, df shape: {df.shape}")
            
            if len(columns) == 2 and columns[1] == "count":
                # Special case for count-based pie chart
                value_counts = df[columns[0]].value_counts()
                # Create a plotly figure manually
                fig = go.Figure()
                fig.add_trace(go.Pie(
                    labels=[str(x) for x in value_counts.index],  # Convert to strings to avoid issues
                    values=value_counts.values,
                    textinfo='percent+label'
                ))
                fig.update_layout(
                    title=title or f"Distribution of {columns[0]}"
                )
                print(f"Created count-based pie chart with {len(value_counts)} categories")
                return fig
            
            elif len(columns) == 2:
                # First try the regular method
                try:
                    # Group by first column and sum second column
                    values = df.groupby(columns[0])[columns[1]].sum()
                    fig = go.Figure()
                    fig.add_trace(go.Pie(
                        labels=[str(x) for x in values.index],  # Convert to strings
                        values=values.values,
                        textinfo='percent+label'
                    ))
                    fig.update_layout(
                        title=title or f"{columns[1]} by {columns[0]}"
                    )
                    print(f"Created pie chart with {len(values)} categories")
                    return fig
                except Exception as e:
                    print(f"Error with regular pie chart: {str(e)}, falling back to count-based")
                    # Fall back to count-based
                    value_counts = df[columns[0]].value_counts()
                    fig = go.Figure()
                    fig.add_trace(go.Pie(
                        labels=[str(x) for x in value_counts.index],
                        values=value_counts.values,
                        textinfo='percent+label'
                    ))
                    fig.update_layout(
                        title=title or f"Distribution of {columns[0]}"
                    )
                    return fig
            
            # Single column - show value counts
            value_counts = df[columns[0]].value_counts()
            fig = go.Figure()
            fig.add_trace(go.Pie(
                labels=[str(x) for x in value_counts.index],
                values=value_counts.values,
                textinfo='percent+label'
            ))
            fig.update_layout(
                title=title or f"Distribution of {columns[0]}"
            )
            print(f"Created simple pie chart with {len(value_counts)} categories")
            return fig
            
        except Exception as e:
            print(f"Error in _create_pie_chart: {str(e)}")
            # Return a simple empty figure rather than None
            fig = go.Figure()
            fig.update_layout(
                title="Error creating chart",
                annotations=[
                    dict(
                        text=f"Error: {str(e)}",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5
                    )
                ]
            )
            return fig

    def _create_error_figure(self, message: str) -> go.Figure:
        """Create a simple figure displaying an error message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=14, color="red")
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return fig

    def _create_bar_chart(self, data: pd.DataFrame, x_column: Optional[str], 
                         y_column: Optional[str], color_by: Optional[str], 
                         title: str) -> Dict[str, Any]:
        """Create a bar chart."""
        try:
            if x_column is None:
                print("No x_column found for bar chart, creating empty figure")
                return {
                    "figure": self._create_error_figure("No suitable column found for X axis"),
                    "warning": "No suitable column found for X axis"
                }
            
            # Case 1: We have both x and y columns
            if x_column and y_column:
                print(f"Creating bar chart with x={x_column}, y={y_column}, color={color_by}")
                fig = px.bar(
                    data, 
                    x=x_column, 
                    y=y_column,
                    color=color_by,
                    title=title
                )
                return {"figure": fig}
            
            # Case 2: We only have x_column, so we do a count-based visualization
            else:
                print(f"Creating count-based bar chart with x={x_column}, color={color_by}")
                value_counts = data[x_column].value_counts().reset_index()
                value_counts.columns = [x_column, 'count']
                
                fig = px.bar(
                    value_counts, 
                    x=x_column, 
                    y='count',
                    title=f"{title} (Count of {x_column})"
                )
                return {"figure": fig}
                
        except Exception as e:
            print(f"Error creating bar chart: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return {
                "error": f"Failed to create bar chart: {str(e)}",
                "figure": self._create_error_figure(f"Bar Chart Error: {str(e)}")
            }

    def _create_line_chart(self, data: pd.DataFrame, x_column: Optional[str], 
                          y_column: Optional[str], color_by: Optional[str], 
                          title: str) -> Dict[str, Any]:
        """Create a line chart."""
        try:
            if x_column is None or y_column is None:
                print("Missing x_column or y_column for line chart, creating empty figure")
                return {
                    "figure": self._create_error_figure("Missing X or Y columns for line chart"),
                    "warning": "Line charts require both X and Y columns"
                }
                
            fig = px.line(
                data, 
                x=x_column, 
                y=y_column,
                color=color_by,
                title=title
            )
            return {"figure": fig}
        except Exception as e:
            print(f"Error creating line chart: {str(e)}")
            return {
                "error": f"Failed to create line chart: {str(e)}",
                "figure": self._create_error_figure(f"Line Chart Error: {str(e)}")
            }
    
    def _create_scatter_plot(self, data: pd.DataFrame, x_column: Optional[str], 
                            y_column: Optional[str], color_by: Optional[str], 
                            title: str) -> Dict[str, Any]:
        """Create a scatter plot."""
        try:
            if x_column is None or y_column is None:
                print("Missing x_column or y_column for scatter plot, creating empty figure")
                return {
                    "figure": self._create_error_figure("Missing X or Y columns for scatter plot"),
                    "warning": "Scatter plots require both X and Y columns"
                }
                
            fig = px.scatter(
                data, 
                x=x_column, 
                y=y_column,
                color=color_by,
                title=title
            )
            return {"figure": fig}
        except Exception as e:
            print(f"Error creating scatter plot: {str(e)}")
            return {
                "error": f"Failed to create scatter plot: {str(e)}",
                "figure": self._create_error_figure(f"Scatter Plot Error: {str(e)}")
            }
    
    def _create_pie_chart(self, data: pd.DataFrame, names_column: Optional[str], 
                         values_column: Optional[str], title: str) -> Dict[str, Any]:
        """Create a pie chart."""
        try:
            if names_column is None:
                print("No names_column found for pie chart, creating empty figure")
                return {
                    "figure": self._create_error_figure("No suitable column found for pie chart categories"),
                    "warning": "No suitable column found for pie chart categories"
                }
            
            # If we have both names and values columns
            if values_column:
                print(f"Creating pie chart with names={names_column}, values={values_column}")
                fig = px.pie(
                    data, 
                    names=names_column, 
                    values=values_column,
                    title=title
                )
                return {"figure": fig}
            
            # If we only have names column, do a count-based pie chart
            else:
                print(f"Creating count-based pie chart with names={names_column}")
                value_counts = data[names_column].value_counts().reset_index()
                value_counts.columns = [names_column, 'count']
                
                fig = px.pie(
                    value_counts, 
                    names=names_column, 
                    values='count',
                    title=f"{title} (Count of {names_column})"
                )
                return {"figure": fig}
        except Exception as e:
            print(f"Error creating pie chart: {str(e)}")
            return {
                "error": f"Failed to create pie chart: {str(e)}",
                "figure": self._create_error_figure(f"Pie Chart Error: {str(e)}")
            }

    def _create_count_based_viz(self, data: pd.DataFrame, column: str, 
                               color_by: Optional[str], title: str,
                               viz_type: str = "bar") -> Dict[str, Any]:
        """Create a count-based visualization (specialized bar or pie chart)."""
        try:
            print(f"Creating count-based visualization for column: {column}")
            value_counts = data[column].value_counts().reset_index()
            value_counts.columns = [column, 'count']
            
            if viz_type == "bar":
                fig = px.bar(
                    value_counts, 
                    x=column, 
                    y='count',
                    title=f"{title} (Count of {column})"
                )
            elif viz_type == "pie":
                fig = px.pie(
                    value_counts, 
                    names=column, 
                    values='count',
                    title=f"{title} (Count of {column})"
                )
            else:
                return {"error": f"Unsupported count visualization type: {viz_type}"}
                
            return {"figure": fig}
        except Exception as e:
            print(f"Error creating count-based visualization: {str(e)}")
            return {
                "error": f"Failed to create count visualization: {str(e)}",
                "figure": self._create_error_figure(f"Count Visualization Error: {str(e)}")
            }