# CSV Question Answering & Visualization App

A Gradio-based application that allows users to upload CSV files, ask questions about the data, and visualize the results using a local language model (Ollama).

![CSV-QA App Interface](screenshots/app_interface.png)

## Features

- **CSV File Upload**: Upload and validate CSV files up to 25MB
- **Question Answering**: Ask questions about your data (numerical and textual) using a local LLM
- **Data Visualization**: Automatically generate visualizations based on your queries
- **Local Processing**: All computation and AI processing happens locally through Ollama
- **Adaptive Analysis**: The application can adapt to different CSV structures with intelligent column name mapping
- **Error Handling**: Robust error handling and fallback mechanisms for visualization and queries

## Requirements

- Python 3.9+
- Ollama with a local model (we recommend using mistral)
- 8GB+ of RAM recommended (4GB minimum)

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
# Create the model using our Modelfile for optimized memory usage
ollama create mistral-local-ultrafast -f Modelfile
```

4. If you're on a system with limited memory (less than 8GB RAM), run the memory optimization script:
```bash
# Windows
restart-ollama-low-memory.bat

# macOS/Linux
chmod +x restart-ollama-low-memory.sh
./restart-ollama-low-memory.sh
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your browser at http://127.0.0.1:7862

3. Upload a CSV file (you can use the sample_data.csv provided)

4. Ask questions about your data, for example:
   - "What's the average price in the dataset?"
   - "Show me a bar chart of sales by region"
   - "What's the correlation between price and square footage?"
   - "Which houses have more than 3 bedrooms and are under $300,000?"
   - "Create a pie chart of property types"
   - "What location has the most expensive properties?"

## Supported Visualization Types

The app supports various visualization types:
- Bar charts
- Line charts
- Scatter plots
- Pie charts
- Count-based visualizations

Just request these in your query like: "show me a pie chart of property types" or "create a bar chart of prices by location"

## Column Name Mapping

One of the key features is the ability to recognize different column naming conventions. For example:
- "region", "location", "area", "city" are all recognized as geographical location columns
- "price", "cost", "amount", "value" are recognized as value columns
- "size", "square_feet", "area", "square_footage", "sq_ft" are recognized as size columns

This allows the app to work with a wide variety of CSV files without requiring exact column name matches.

## Project Structure

- `app.py`: Main Gradio application
- `modules/file_handler.py`: CSV file handling and validation
- `modules/llm_agent.py`: Integration with Ollama for question answering
- `modules/visualizer.py`: Data visualization utilities
- `modules/basic_query_handler.py`: Fallback query handler when LLM is unavailable
- `diagnose_csv.py`: Utility to diagnose issues in CSV files
- `Modelfile`: Configuration for the Ollama model
- `sample_data.csv`: Example housing dataset for testing

## Customization

- To use a different model, update the model name in `modules/llm_agent.py`
- Adjust visualization settings in `modules/visualizer.py`
- Modify the Gradio interface in `app.py`
- Add more column name mappings in `visualizer.py` and `basic_query_handler.py`

## Troubleshooting

### Memory Issues
- If you see "model requires more system memory than is available", use the provided memory optimization script
- For systems with very limited memory, edit the Modelfile to reduce parameters or use a smaller model

### Visualization Issues
- Make sure the column names in your query match the actual CSV or are in the supported mappings
- Check the application logs for specific error messages about visualization failures
- Use the `diagnose_csv.py` script to check your CSV file for potential issues

### Query Processing
- If the LLM is giving strange responses, try rephrasing your question
- For complex queries, break them down into simpler questions
- Check if the columns you're asking about actually exist in your data

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Gradio](https://www.gradio.app/) for the web interface framework
- [Ollama](https://ollama.ai/) for the local LLM capabilities
- [Plotly](https://plotly.com/) and [Pandas](https://pandas.pydata.org/) for data processing and visualization
