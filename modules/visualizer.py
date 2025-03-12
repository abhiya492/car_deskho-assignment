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
    
    def create_visualization(
        self, 
        dataframe: pd.DataFrame, 
        viz_type: str, 
        x_column: str, 
        y_column: Optional[str] = None, 
        title: str = "Data Visualization",
        color_column: Optional[str] = None
    ) -> Optional[Any]:
        """Create a visualization based on the specified parameters"""
        if dataframe is None or dataframe.empty:
            return None
        
        # Normalize visualization type
        viz_type = viz_type.lower().strip()
        
        # Validate inputs
        if x_column not in dataframe.columns:
            return None
        
        if y_column is not None and y_column not in dataframe.columns:
            return None
        
        if color_column is not None and color_column not in dataframe.columns:
            color_column = None
        
        # Create plotly figure based on visualization type
        fig = None
        
        try:
            if viz_type == 'bar':
                if y_column:
                    fig = px.bar(dataframe, x=x_column, y=y_column, title=title, color=color_column)
                else:
                    # Create count plot if no y_column specified
                    count_data = dataframe[x_column].value_counts().reset_index()
                    count_data.columns = [x_column, 'count']
                    fig = px.bar(count_data, x=x_column, y='count', title=title)
            
            elif viz_type == 'line':
                if y_column:
                    fig = px.line(dataframe, x=x_column, y=y_column, title=title, color=color_column)
                else:
                    return None  # Line chart requires y-axis
            
            elif viz_type == 'scatter':
                if y_column:
                    fig = px.scatter(dataframe, x=x_column, y=y_column, title=title, color=color_column)
                else:
                    return None  # Scatter plot requires y-axis
            
            elif viz_type == 'pie':
                # For pie charts, x_column represents the categories and y_column the values
                if y_column:
                    # Group by x_column and sum y_column values
                    pie_data = dataframe.groupby(x_column)[y_column].sum().reset_index()
                    fig = px.pie(pie_data, names=x_column, values=y_column, title=title)
                else:
                    # Create count-based pie chart if no y_column specified
                    count_data = dataframe[x_column].value_counts().reset_index()
                    count_data.columns = [x_column, 'count']
                    fig = px.pie(count_data, names=x_column, values='count', title=title)
            
            elif viz_type == 'histogram':
                fig = px.histogram(dataframe, x=x_column, title=title, color=color_column)
                if y_column:
                    # If y_column is provided, we can create a weighted histogram
                    fig = px.histogram(dataframe, x=x_column, y=y_column, title=title, color=color_column)
            
            elif viz_type == 'box':
                if y_column:
                    fig = px.box(dataframe, x=x_column, y=y_column, title=title, color=color_column)
                else:
                    fig = px.box(dataframe, x=x_column, title=title, color=color_column)
            
            elif viz_type == 'violin':
                if y_column:
                    fig = px.violin(dataframe, x=x_column, y=y_column, title=title, color=color_column)
                else:
                    fig = px.violin(dataframe, x=x_column, title=title, color=color_column)
            
            elif viz_type == 'heatmap':
                if y_column:
                    # For heatmap, we need to pivot the data
                    pivot_data = dataframe.pivot_table(
                        index=y_column, 
                        columns=x_column, 
                        values=color_column if color_column else 'count',
                        aggfunc='count' if not color_column else 'mean'
                    )
                    fig = px.imshow(pivot_data, title=title)
                else:
                    return None  # Heatmap requires y-axis
            
            elif viz_type == 'area':
                if y_column:
                    fig = px.area(dataframe, x=x_column, y=y_column, title=title, color=color_column)
                else:
                    return None  # Area chart requires y-axis
            
            else:
                # Default to bar chart if type not recognized
                if y_column:
                    fig = px.bar(dataframe, x=x_column, y=y_column, title=title, color=color_column)
                else:
                    count_data = dataframe[x_column].value_counts().reset_index()
                    count_data.columns = [x_column, 'count']
                    fig = px.bar(count_data, x=x_column, y='count', title=title)
            
            # Update layout for better appearance
            fig.update_layout(
                template="plotly_white",
                xaxis_title=x_column,
                yaxis_title=y_column if y_column else "Count",
                legend_title=color_column if color_column else None
            )
            
            return fig
        
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
            return None
    
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
    
    def create_visualization_from_info(self, df: pd.DataFrame, viz_info: Dict[str, Any]) -> Optional[go.Figure]:
        """Create a visualization based on the provided information."""
        try:
            viz_type = viz_info.get('type', '').lower()
            columns = viz_info.get('columns', [])
            title = viz_info.get('title', '')
            params = viz_info.get('params', {})
            
            if not viz_type or not columns:
                return None
                
            # Make sure all columns exist in the dataframe
            if not all(col in df.columns for col in columns):
                return None
            
            fig = None
            
            if viz_type == 'bar':
                fig = self._create_bar_chart(df, columns, title, params)
            elif viz_type == 'line':
                fig = self._create_line_chart(df, columns, title, params)
            elif viz_type == 'scatter':
                fig = self._create_scatter_plot(df, columns, title, params)
            elif viz_type == 'histogram':
                fig = self._create_histogram(df, columns, title, params)
            elif viz_type == 'box':
                fig = self._create_box_plot(df, columns, title, params)
            elif viz_type == 'pie':
                fig = self._create_pie_chart(df, columns, title, params)
                
            if fig:
                # Apply common styling
                fig.update_layout(
                    title=title,
                    width=self.default_width,
                    height=self.default_height,
                    template='plotly_white'
                )
                
            return fig
            
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
            return None
            
    def _create_bar_chart(self, df: pd.DataFrame, columns: list, title: str, params: Dict[str, Any]) -> go.Figure:
        """Create a bar chart."""
        if len(columns) == 1:
            # Single column - show value counts
            value_counts = df[columns[0]].value_counts()
            fig = px.bar(x=value_counts.index, y=value_counts.values)
        else:
            # Two columns - one for x, one for y
            fig = px.bar(df, x=columns[0], y=columns[1])
        return fig
        
    def _create_line_chart(self, df: pd.DataFrame, columns: list, title: str, params: Dict[str, Any]) -> go.Figure:
        """Create a line chart."""
        return px.line(df, x=columns[0], y=columns[1:])
        
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
        if len(columns) == 2:
            values = df.groupby(columns[0])[columns[1]].sum()
            return px.pie(values=values.values, names=values.index)
        value_counts = df[columns[0]].value_counts()
        return px.pie(values=value_counts.values, names=value_counts.index)