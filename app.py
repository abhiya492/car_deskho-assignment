import gradio as gr
import pandas as pd
import asyncio
import os
import sys

# Add modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from modules.file_handler import FileHandler
from modules.llm_agent import OllamaAgent
from modules.visualizer import Visualizer
from modules.basic_query_handler import BasicQueryHandler

# Initialize modules
file_handler = FileHandler()
llm_agent = OllamaAgent()
visualizer = Visualizer()
basic_handler = BasicQueryHandler()  # Fallback handler

async def process_query(query, file_state):
    """Process a user query about the data"""
    if not query.strip():
        return "Please enter a question about the data.", None
    
    if file_state is None or "dataframe" not in file_state:
        return "Please upload a CSV file first.", None
    
    dataframe = file_state["dataframe"]
    file_info = file_state["file_info"]
    
    # Try to process with LLM agent first
    try:
        answer, viz_info = await llm_agent.answer_query(query, dataframe, file_info)
        
        # If we got an error message from the LLM agent, fall back to basic handler
        if answer.startswith("Error") or answer.startswith("The request timed out"):
            print(f"LLM failed with message: {answer}. Falling back to basic handler.")
            answer, viz_info = basic_handler.process_query(query, dataframe, file_info)
    except Exception as e:
        # On exception, fall back to basic handler
        print(f"LLM agent exception: {str(e)}. Falling back to basic handler.")
        answer, viz_info = basic_handler.process_query(query, dataframe, file_info)
    
    # Create visualization if needed
    plot = None
    if viz_info:
        try:
            print(f"Creating visualization with info: {viz_info}")
            viz_result = visualizer.create_visualization(
                data=dataframe,
                viz_type=viz_info.get("type", "bar"),
                x_column=viz_info.get("x_column"),
                y_column=viz_info.get("y_column"),
                color_by=viz_info.get("color_by"),
                title=viz_info.get("title", "Data Visualization")
            )
            
            # Handle visualization result
            if viz_result:
                if "error" in viz_result:
                    print(f"Visualization error: {viz_result['error']}")
                    if "details" in viz_result:
                        print(f"Details: {viz_result['details']}")
                    
                    # If we have a fallback figure, use it
                    if "figure" in viz_result:
                        plot = viz_result["figure"]
                        answer += "\n\nNote: There was an issue with the visualization, but I'm showing a simplified version."
                    else:
                        answer += f"\n\nNote: I couldn't create a visualization: {viz_result['error']}"
                elif "figure" in viz_result:
                    plot = viz_result["figure"]
                    if "warning" in viz_result:
                        answer += f"\n\nNote: {viz_result['warning']}"
                else:
                    print("Visualization returned unexpected format")
                    answer += "\n\nNote: I couldn't create a visualization due to an unexpected result format."
            else:
                print("Visualization creation returned None")
                answer += "\n\nNote: I couldn't create a visualization for this query."
        except Exception as e:
            import traceback
            print(f"Visualization error: {str(e)}")
            traceback.print_exc()
            answer += f"\n\nNote: I tried to create a visualization but encountered an error: {str(e)}"
    
    return answer, plot

def upload_file(file):
    """Handle file upload and return dataframe info"""
    if file is None:
        return None, "Please upload a CSV file."
    
    # Load the CSV file
    success, message = file_handler.load_csv(file)
    if not success:
        return None, message
    
    dataframe = file_handler.get_dataframe()
    file_info = file_handler.get_file_info()
    
    # Debug information
    print("=" * 50)
    print(f"CSV File loaded: {file_info['filename']}")
    print(f"Columns: {list(dataframe.columns)}")
    print(f"Data types: {dataframe.dtypes}")
    print(f"Sample data:\n{dataframe.head(3)}")
    print("=" * 50)
    
    # Return the dataframe, info, and a success message
    return {
        "dataframe": dataframe,
        "file_info": file_info
    }, f"Successfully loaded {file_info['filename']} with {file_info['rows']} rows and {file_info['columns']} columns."

def show_data_preview(file_state):
    """Show a preview of the data"""
    if file_state is None or "dataframe" not in file_state:
        return "No data available. Please upload a CSV file."
    
    dataframe = file_state["dataframe"]
    if dataframe is None or dataframe.empty:
        return "The uploaded file contains no data."
    
    # Show the first 10 rows
    return dataframe.head(10).to_html()

# Define the Gradio interface
with gr.Blocks(title="CSV Question Answering & Visualization App") as app:
    gr.Markdown("# CSV Question Answering & Visualization App")
    gr.Markdown("Upload a CSV file, ask questions about the data, and visualize the results")
    
    # File state for storing uploaded file info
    file_state = gr.State(None)
    
    with gr.Row():
        with gr.Column(scale=1):
            # File upload section
            file_upload = gr.File(label="Upload CSV File (Max 25MB)")
            upload_button = gr.Button("Load CSV")
            upload_status = gr.Textbox(label="Upload Status", interactive=False)
        
        with gr.Column(scale=2):
            # Data preview section
            preview_button = gr.Button("Show Data Preview")
            data_preview = gr.HTML(label="Data Preview")
    
    # Query section
    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(
                label="Ask a question about your data",
                placeholder="e.g., What is the average price? Show me a bar chart of sales by region"
            )
            query_button = gr.Button("Submit Query")
    
    # Result section
    with gr.Row():
        with gr.Column():
            answer_output = gr.Textbox(label="Answer", interactive=False)
        with gr.Column():
            visualization_output = gr.Plot(label="Visualization")
    
    # Connect components
    upload_button.click(
        fn=upload_file,
        inputs=[file_upload],
        outputs=[file_state, upload_status]
    )
    
    preview_button.click(
        fn=show_data_preview,
        inputs=[file_state],
        outputs=[data_preview]
    )
    
    query_button.click(
        fn=process_query,
        inputs=[query_input, file_state],
        outputs=[answer_output, visualization_output]
    )
    
    # Also enable pressing Enter to submit query
    query_input.submit(
        fn=process_query,
        inputs=[query_input, file_state],
        outputs=[answer_output, visualization_output]
    )

# Launch the app
if __name__ == "__main__":
    app.launch()