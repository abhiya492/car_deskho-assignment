import json
import httpx
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from pydantic import BaseModel, Field
import requests

class DataAnalysisQuery(BaseModel):
    """Model representing a data analysis query"""
    query: str = Field(..., description="The user's natural language query about the data")
    answer: str = Field("", description="The answer to the user's query")
    
    # Data operation fields
    requires_calculation: bool = Field(False, description="Whether the query requires numerical calculations")
    calculation_description: str = Field("", description="Description of the calculation needed")
    requires_filtering: bool = Field(False, description="Whether the query requires filtering the data")
    filter_conditions: List[str] = Field([], description="List of filter conditions if filtering is required")
    requires_grouping: bool = Field(False, description="Whether the query requires grouping the data")
    groupby_columns: List[str] = Field([], description="Columns to group by if grouping is required")
    requires_sorting: bool = Field(False, description="Whether the query requires sorting the data")
    sort_by_columns: List[str] = Field([], description="Columns to sort by if sorting is required")
    sort_ascending: bool = Field(True, description="Sort order, True for ascending, False for descending")
    
    # Visualization fields
    requires_visualization: bool = Field(False, description="Whether the query requires visualization")
    visualization_type: str = Field("", description="Type of visualization (e.g., bar, line, scatter)")
    x_axis: str = Field("", description="Column for x-axis in visualization")
    y_axis: str = Field("", description="Column for y-axis in visualization")
    
    # Metadata for the response
    confidence: float = Field(0.0, description="Confidence in the answer (0-1)")
    python_code: str = Field("", description="Python code to execute the query")

class QueryResponse(BaseModel):
    answer: str = Field(description="The answer to the user's question")
    visualization_needed: bool = Field(description="Whether a visualization would be helpful")
    viz_type: Optional[str] = Field(None, description="Type of visualization (bar, line, scatter, etc.)")
    viz_columns: Optional[List[str]] = Field(None, description="Columns to use in visualization")
    viz_title: Optional[str] = Field(None, description="Title for the visualization")
    viz_params: Optional[Dict[str, Any]] = Field(None, description="Additional visualization parameters")

class OllamaAgent:
    def __init__(self):
        self.model_name = "mistral-local-ultrafast"
        self.api_url = "http://localhost:11434/api/generate"
        
    def _generate_system_prompt(self, df_info: Dict[str, Any]) -> str:
        """Generate a system prompt with context about the dataframe."""
        return f"""You are a data analysis assistant. You have access to a CSV file with the following properties:
- Columns: {', '.join(df_info['column_types'].keys())}
- Numeric columns: {df_info['numeric_columns']}
- Categorical columns: {df_info['categorical_columns']}
- Number of rows: {df_info['rows']}

Your task is to:
1. Answer questions about the data briefly and precisely
2. Suggest visualizations when appropriate
3. Format responses as structured JSON

Keep answers concise and focused on the data."""

    def _format_query(self, user_query: str, df_info: Dict[str, Any]) -> str:
        """Format the complete query with system prompt and user question."""
        system_prompt = self._generate_system_prompt(df_info)
        return f"{system_prompt}\n\nQuestion: {user_query}\n\nRespond with JSON: {{\n'answer': 'brief answer',\n'visualization_needed': boolean,\n'viz_type': 'type if needed',\n'viz_columns': ['columns'],\n'viz_title': 'title',\n'viz_params': {{}}\n}}"

    async def answer_query(self, query: str, df: pd.DataFrame, df_info: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Process a user query and return an answer with optional visualization info."""
        try:
            formatted_query = self._format_query(query, df_info)
            
            # Send request to Ollama with optimized parameters
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": formatted_query,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_thread": 4,
                        "num_gpu": 1,
                        "num_ctx": 512,
                        "num_batch": 512,
                        "repeat_penalty": 1.1,
                        "top_k": 40,
                        "top_p": 0.95
                    }
                },
                timeout=30  # Add timeout to prevent hanging
            )
            
            if response.status_code != 200:
                return "Sorry, I encountered an error processing your question.", None
                
            # Parse the response
            llm_response = response.json()
            response_text = llm_response['response']
            
            # Extract JSON from response
            try:
                # Find JSON content between curly braces
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    parsed_response = QueryResponse(**json.loads(json_str))
                else:
                    return "I couldn't format the response properly.", None
                    
            except Exception as e:
                return f"Error parsing response: {str(e)}", None
            
            # Return the answer and visualization info if needed
            viz_info = None
            if parsed_response.visualization_needed and parsed_response.viz_type:
                viz_info = {
                    "type": parsed_response.viz_type,
                    "columns": parsed_response.viz_columns,
                    "title": parsed_response.viz_title,
                    "params": parsed_response.viz_params or {}
                }
                
            return parsed_response.answer, viz_info
            
        except requests.exceptions.Timeout:
            return "The request timed out. Please try again.", None
        except Exception as e:
            return f"Error processing query: {str(e)}", None