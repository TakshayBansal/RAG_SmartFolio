from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from .embeddings import LocalEmbeddings

class TextProcessor:
    """Handles text chunking and embedding generation"""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embeddings: Optional[LocalEmbeddings] = None,
    ) -> None:
        self.embeddings = embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
    
    def create_chunks(self, text: str, metadata: Dict = None) -> List[Document]:
        """Split text into chunks and create Document objects"""
        texts = self.text_splitter.split_text(text)
        docs = [
            Document(page_content=t, metadata=metadata or {})
            for t in texts
        ]
        return docs
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if self.embeddings is None:
            self.embeddings = LocalEmbeddings()
        return self.embeddings.embed_documents(texts)
    
    def process_document(self, text: str, metadata: Dict = None) -> Dict:
        """Process a document: split into chunks and generate embeddings"""
        # Create chunks
        chunks = self.create_chunks(text, metadata)
        
        # Extract texts from chunks
        texts = [chunk.page_content for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.create_embeddings(texts)
        
        return {
            "chunks": chunks,
            "embeddings": embeddings
        }

class RelevanceScorer:
    """Scores text chunks for relevance to financial analysis"""
    
    def __init__(self):
        self.financial_keywords = {
            'high_importance': [
                'revenue', 'profit', 'margin', 'growth', 'earnings',
                'investment', 'expansion', 'acquisition', 'market share',
                'strategy', 'dividend', 'debt', 'cost reduction'
            ],
            'medium_importance': [
                'project', 'initiative', 'partnership', 'development',
                'research', 'innovation', 'technology', 'sustainability',
                'efficiency', 'optimization'
            ],
            'low_importance': [
                'company', 'business', 'industry', 'market', 'sector',
                'product', 'service', 'customer', 'employee'
            ]
        }
        
        self.weights = {
            'high_importance': 3.0,
            'medium_importance': 2.0,
            'low_importance': 1.0
        }
    
    def score_chunk(self, text: str) -> float:
        """Score a chunk of text based on keyword relevance"""
        text = text.lower()
        score = 0.0
        
        for importance, keywords in self.financial_keywords.items():
            weight = self.weights[importance]
            for keyword in keywords:
                if keyword in text:
                    score += weight
        
        # Normalize score
        max_possible_score = sum(
            self.weights[imp] * len(keywords)
            for imp, keywords in self.financial_keywords.items()
        )
        
        return score / max_possible_score if max_possible_score > 0 else 0.0

class ChunkProcessor:
    """Main class for processing and managing text chunks"""
    
    def __init__(self):
        self.text_processor = TextProcessor()
        self.relevance_scorer = RelevanceScorer()
    
    def process_text(self, text: str, metadata: Dict = None) -> Dict:
        """Process text and return chunks with relevance scores"""
        # Create chunks
        chunks = self.text_processor.create_chunks(text, metadata)
        
        # Score chunks for relevance
        scores = [
            self.relevance_scorer.score_chunk(chunk.page_content)
            for chunk in chunks
        ]
        
        # Combine everything into a structured format
        processed_chunks = []
        for chunk, score in zip(chunks, scores):
            processed_chunks.append({
                "content": chunk.page_content,
                "metadata": chunk.metadata,
                "relevance_score": score
            })
        
        # Sort chunks by relevance score
        processed_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return {
            "processed_chunks": processed_chunks,
            "relevance_scores": scores
        }