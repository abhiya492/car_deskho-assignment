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

# Initialize modules
file_handler = FileHandler()
llm_agent = OllamaAgent()
visualizer = Visualizer()

async def process_query(query, file_state):
    """Process a user query about the data"""
    if not query.strip():
        return "Please enter a question about the data.", None
    
    if file_state is None or "dataframe" not in file_state:
        return "Please upload a CSV file first.", None
    
    dataframe = file_state["dataframe"]
    file_info = file_state["file_info"]
    
    # Process the query using the LLM agent
    answer, viz_info = await llm_agent.answer_query(query, dataframe, file_info)
    
    # Create visualization if needed
    plot = None
    if viz_info:
        plot = visualizer.create_visualization_from_info(dataframe, viz_info)
    
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