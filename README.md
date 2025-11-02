# Financial Report Analysis with RAG

This project implements a Retrieval Augmented Generation (RAG) system to analyze company quarterly/annual reports and predict stock movement potential. The system extracts information from financial reports, processes it using advanced NLP techniques, and provides a confidence score for potential stock movement.

## Features

- PDF document processing and text extraction
- Intelligent text chunking and embedding generation
- Vector database storage for efficient information retrieval
- Financial metrics extraction
- LLM-based analysis for stock movement prediction
- Confidence score generation with detailed reasoning

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

4. Place company reports (PDF format) in the `data` directory.

## Usage

Run the main script:
```bash
python main.py
```

## Project Structure

- `src/`
  - `pdf_processor.py`: Handles PDF loading and text extraction
  - `vector_store.py`: Manages document embeddings and vector storage
  - `stock_analyzer.py`: Implements LLM-based analysis
- `data/`: Store company reports here
- `main.py`: Main application script
- `requirements.txt`: Project dependencies

## Output

The system provides:
- Stock movement direction (up/down)
- Confidence score (0-100)
- Key factors influencing the prediction
- Potential risks to consider
- Detailed reasoning for the analysis