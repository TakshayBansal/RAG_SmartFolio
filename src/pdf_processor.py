from typing import List, Dict
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PDFProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )

    def load_and_split_pdf(self, file_path: str) -> List[str]:
        """
        Load PDF file and split it into chunks
        """
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        
        # Extract text from pages
        texts = [page.page_content for page in pages]
        
        # Split texts into smaller chunks
        chunks = []
        for text in texts:
            chunks.extend(self.text_splitter.split_text(text))
            
        return chunks

    def extract_financial_metrics(self, text: str) -> Dict:
        """
        Extract key financial metrics from the text
        """
        # TODO: Implement financial metrics extraction
        # This will be enhanced with regex patterns and financial metrics detection
        metrics = {
            'revenue': None,
            'net_income': None,
            'operating_margin': None,
            'debt_to_equity': None,
            'cash_flow': None,
            'projects': [],
            'investments': []
        }
        return metrics