import re
from typing import List, Dict, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class FinancialMetricsExtractor:
    """Extracts financial metrics from text using regex patterns"""
    
    def __init__(self):
        self.patterns = {
            'revenue': r'(?:revenue|sales)[^\n.]*?(?:USD|₹|€|\$)\s*(\d+(?:\.\d+)?(?:\s*[bmkMB]+)?)',
            'profit': r'(?:net profit|net income)[^\n.]*?(?:USD|₹|€|\$)\s*(\d+(?:\.\d+)?(?:\s*[bmkMB]+)?)',
            'margin': r'(?:profit margin|operating margin)[^\n.]*?(\d+(?:\.\d+)?)\s*%',
            'growth': r'(?:growth|increase)[^\n.]*?(\d+(?:\.\d+)?)\s*%',
            'debt_equity': r'debt[- ]to[- ]equity[^\n.]*?(\d+(?:\.\d+)?)',
            'eps': r'earnings per share[^\n.]*?(?:USD|₹|€|\$)\s*(\d+(?:\.\d+)?)',
        }
    
    def extract_metrics(self, text: str) -> Dict[str, Optional[str]]:
        """Extract financial metrics from text"""
        metrics = {}
        for key, pattern in self.patterns.items():
            match = re.search(pattern, text.lower())
            metrics[key] = match.group(1) if match else None
        return metrics

class ProjectsExtractor:
    """Extracts information about new projects and initiatives"""
    
    def __init__(self):
        self.project_patterns = [
            r'new (?:project|initiative|development)[^\n.]*?(?:\.|$)',
            r'announced (?:plans|development)[^\n.]*?(?:\.|$)',
            r'launching[^\n.]*?(?:\.|$)',
            r'investment in[^\n.]*?(?:\.|$)'
        ]
    
    def extract_projects(self, text: str) -> List[str]:
        """Extract project information from text"""
        projects = []
        for pattern in self.project_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                projects.append(match.group(0).strip())
        return list(set(projects))  # Remove duplicates

class FinancialDocumentProcessor:
    """Main class for processing financial documents"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        self.metrics_extractor = FinancialMetricsExtractor()
        self.projects_extractor = ProjectsExtractor()
    
    def load_and_process_pdf(self, file_path: str) -> Dict:
        """Load PDF and extract all relevant information"""
        # Load PDF
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        
        # Extract text from pages
        full_text = "\n".join([page.page_content for page in pages])
        
        # Split into chunks
        chunks = self.text_splitter.split_text(full_text)
        
        # Extract metrics and projects
        metrics = self.metrics_extractor.extract_metrics(full_text)
        projects = self.projects_extractor.extract_projects(full_text)
        
        # Prepare the results
        results = {
            "chunks": chunks,
            "metrics": metrics,
            "projects": projects,
            "full_text": full_text
        }
        
        return results

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze the sentiment of the text"""
        # This is a simple implementation that can be enhanced with more sophisticated NLP
        positive_words = ['growth', 'profit', 'increase', 'success', 'strong', 'positive',
                         'opportunity', 'improvement', 'innovative', 'excellent']
        negative_words = ['loss', 'decline', 'decrease', 'risk', 'challenge', 'difficult',
                         'uncertain', 'weakness', 'concern', 'negative']
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        total = positive_count + negative_count
        
        if total == 0:
            return {"positive": 0.5, "negative": 0.5}
        
        return {
            "positive": positive_count / total,
            "negative": negative_count / total
        }