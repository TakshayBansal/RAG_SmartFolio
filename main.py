import os
import json
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from src.document_processor import FinancialDocumentProcessor
from src.text_processor import ChunkProcessor
from src.vector_store import VectorStore
from src.finbert_analyzer import FinBERTAnalyzer

class FinancialDocumentAnalyzer:
    """Main class for analyzing financial documents and predicting stock movement"""
    
    def __init__(self, persist_directory: str = "./data/vectorstore"):
        # Load environment variables
        load_dotenv()
        
        # Initialize components
        self.document_processor = FinancialDocumentProcessor()
        self.chunk_processor = ChunkProcessor()
        self.vector_store = VectorStore(persist_directory=persist_directory)
        self.financial_analyzer = FinBERTAnalyzer()
        
        # Create necessary directories
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directories if they don't exist"""
        Path("./data/reports").mkdir(parents=True, exist_ok=True)
        Path("./data/vectorstore").mkdir(parents=True, exist_ok=True)
        Path("./data/analysis").mkdir(parents=True, exist_ok=True)
    
    def process_document(self,
                        file_path: str,
                        company_name: str,
                        document_type: str = "annual_report") -> Dict:
        """
        Process a financial document and store it in the vector database
        
        Args:
            file_path: Path to the PDF document
            company_name: Name of the company
            document_type: Type of document (e.g., "annual_report", "quarterly_report")
        """
        # Process the document
        doc_results = self.document_processor.load_and_process_pdf(file_path)
        
        # Process text chunks
        processed_results = self.chunk_processor.process_text(
            doc_results["full_text"],
            metadata={"company": company_name, "document_type": document_type}
        )
        
        # Store in vector database
        self.vector_store.add_documents(
            chunks=processed_results["processed_chunks"],
            source=company_name,
            document_type=document_type
        )
        
        return {
            "metrics": doc_results["metrics"],
            "projects": doc_results["projects"],
            "chunk_count": len(processed_results["processed_chunks"])
        }
    
    def analyze_company(self,
                       company_name: str) -> Dict:
        """
        Analyze a company's documents and predict stock movement
        
        Args:
            company_name: Name of the company to analyze
        """
        # Retrieve relevant document chunks
        financial_chunks = self.vector_store.search(
            query="financial performance metrics revenue profit margin",
            n_results=5,
            source=company_name
        )
        
        project_chunks = self.vector_store.search(
            query="new projects initiatives investments expansion plans",
            n_results=5,
            source=company_name
        )
        
        risk_chunks = self.vector_store.search(
            query="risks challenges concerns negative factors",
            n_results=5,
            source=company_name
        )
        
        # Combine all relevant chunks
        all_chunks = financial_chunks + project_chunks + risk_chunks
        
        # Analyze document using FinBERT
        analysis_result = self.financial_analyzer.analyze_document(
            document_chunks=all_chunks
        )
        
        # Save analysis results
        self._save_analysis(company_name, analysis_result)
        
        return analysis_result
    
    def _save_analysis(self, company_name: str, analysis: Dict):
        """Save analysis results to a file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./data/analysis/{company_name}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(analysis, f, indent=2)

def main():
    # Initialize analyzer
    analyzer = FinancialDocumentAnalyzer()
    
    # Example usage
    company_name = "Bharat Electronics"
    report_path = "./data/reports/bel.pdf"
    
    if os.path.exists(report_path):
        print(f"Processing report for {company_name}...")
        
        # Process document
        doc_result = analyzer.process_document(
            file_path=report_path,
            company_name=company_name
        )
        print(f"Processed document: {doc_result}")
        
        # Analyze company using FinBERT
        analysis = analyzer.analyze_company(
            company_name=company_name
        )
        
        # Print analysis results
        print("\nAnalysis Results:")
        print(f"Direction: {analysis['analysis']['direction']}")
        print(f"Confidence Score: {analysis['analysis']['confidence_score']}")
        print("\nKey Factors:")
        for factor in analysis['analysis']['key_factors']:
            print(f"- {factor}")
        print("\nRisks:")
        for risk in analysis['analysis']['risks']:
            print(f"- {risk}")
        print(f"\nReasoning: {analysis['analysis']['reasoning']}")
    else:
        print(f"Please place a financial report PDF in {report_path}")

if __name__ == "__main__":
    main()