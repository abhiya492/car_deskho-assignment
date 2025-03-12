import json
import httpx
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from pydantic import BaseModel, Field
import asyncio

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
        self.max_retries = 3
        self.timeout = 120  # Increased timeout to 120 seconds
        self.client = httpx.AsyncClient(timeout=self.timeout)
        
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
        # Keep the prompt as short as possible to reduce memory usage
        return f"{system_prompt}\n\nQuestion: {user_query}\n\nRespond with JSON: {{\n'answer': 'brief answer',\n'visualization_needed': boolean,\n'viz_type': 'type if needed',\n'viz_columns': ['columns'],\n'viz_title': 'title',\n'viz_params': {{}}\n}}"

    async def _make_request_with_retry(self, formatted_query: str) -> Dict[str, Any]:
        """Make request to Ollama with retry logic"""
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    # Use minimal memory settings
                    response = await client.post(
                        self.api_url,
                        json={
                            "model": self.model_name,
                            "prompt": formatted_query,
                            "stream": False,
                            "options": {
                                "temperature": 0.5,
                                "num_thread": 1,        # Minimum threads
                                "num_gpu": 0,           # Disable GPU
                                "num_ctx": 512,         # Minimum context
                                "num_batch": 32,        # Very small batch size
                                "repeat_penalty": 1.1,
                                "top_k": 20,            # Reduced
                                "top_p": 0.9,
                                "numa": False,          # Fixed - was lowercase 'false' causing Python errors
                                "f16_kv": True,         # Fixed - was lowercase 'true' causing Python errors
                                "low_vram": True        # Fixed - was lowercase 'true' causing Python errors
                            }
                        }
                    )
                    
                    if response.status_code == 500:
                        print(f"Server error on attempt {attempt + 1}, response: {response.text}")
                        if attempt == self.max_retries - 1:
                            raise httpx.RequestError(f"Server error: {response.text}")
                        await asyncio.sleep(2)  # Longer delay for 500 errors
                        continue
                        
                    response.raise_for_status()
                    return response.json()
                    
            except httpx.TimeoutException:
                print(f"Timeout on attempt {attempt + 1}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2)  # Increased delay between retries
                continue
            except httpx.RequestError as e:
                print(f"Request error on attempt {attempt + 1}: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2)  # Increased delay between retries
                continue
        raise httpx.RequestError("Max retries exceeded")

    async def answer_query(self, query: str, df: pd.DataFrame, df_info: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Process a user query and return an answer with optional visualization info."""
        try:
            formatted_query = self._format_query(query, df_info)
            
            try:
                llm_response = await self._make_request_with_retry(formatted_query)
            except httpx.TimeoutException:
                return "The request timed out. The server might be busy. Please try again in a moment.", None
            except httpx.RequestError as e:
                return f"Error connecting to the language model server: {str(e)}. Please ensure the server is running.", None
            
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
                # Convert from old format (columns) to new format (x_column, y_column)
                viz_info = {
                    "type": parsed_response.viz_type,
                    "x_column": parsed_response.viz_columns[0] if parsed_response.viz_columns else None,
                    "y_column": parsed_response.viz_columns[1] if len(parsed_response.viz_columns) > 1 else None,
                    "color_by": parsed_response.viz_params.get("color_column") if parsed_response.viz_params else None,
                    "title": parsed_response.viz_title
                }
                
            return parsed_response.answer, viz_info
            
        except Exception as e:
            return f"Error processing query: {str(e)}", None