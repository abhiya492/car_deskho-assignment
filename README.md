# CSV Question Answering & Visualization App

A Gradio-based application that allows users to upload CSV files, ask questions about the data, and visualize the results using a local language model (Ollama).

## Features

- **CSV File Upload**: Upload and validate CSV files up to 25MB
- **Question Answering**: Ask questions about your data (numerical and textual) using a local LLM
- **Data Visualization**: Automatically generate visualizations based on your queries
- **Local Processing**: All computation and AI processing happens locally through Ollama

## Requirements

- Python 3.9+
- Ollama with a local model (we recommend using mistral-local-fast)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/csv-qa-app.git
cd csv-qa-app
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Make sure Ollama is installed and running with your model:
```bash
# In a separate terminal
ollama serve
# Create optimized model if needed
ollama create mistral-local-fast -f Modelfile
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your browser at http://127.0.0.1:7860

3. Upload a CSV file (you can use the sample_data.csv provided)

4. Ask questions about your data, for example:
   - "What's the average price in the dataset?"
   - "Show me a bar chart of sales by region"
   - "What's the correlation between price and square footage?"
   - "Which houses have more than 3 bedrooms and are under $300,000?"

## Project Structure

- `app.py`: Main Gradio application
- `modules/file_handler.py`: CSV file handling and validation
- `modules/llm_agent.py`: Integration with Ollama for question answering
- `modules/visualizer.py`: Data visualization utilities
- `Modelfile`: Configuration for the Ollama model
- `sample_data.csv`: Example housing dataset for testing

## Customization

- To use a different model, update the model name in `modules/llm_agent.py`
- Adjust visualization settings in `modules/visualizer.py`
- Modify the Gradio interface in `app.py`

## Troubleshooting

- If you encounter slow responses, try reducing the parameters in the Modelfile
- For visualization issues, check that the columns mentioned in your query exist in your dataset
- Make sure Ollama is running in a separate terminal with `ollama serve`

## License

This project is licensed under the MIT License - see the LICENSE file for details.
