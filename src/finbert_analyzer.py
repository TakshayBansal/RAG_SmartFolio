from typing import Dict, List, Optional
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from datetime import datetime

class FinBERTAnalyzer:
    """Financial text analysis using FinBERT model"""
    
    def __init__(self):
        # Load FinBERT model and tokenizer
        self.model_name = "ProsusAI/finbert"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.labels = ["positive", "negative", "neutral"]
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of a piece of text using FinBERT"""
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Convert to dictionary
        sentiment_scores = {
            label: score.item()
            for label, score in zip(self.labels, predictions[0])
        }
        
        return sentiment_scores
    
    def analyze_chunks(self, chunks: List[Dict]) -> Dict:
        """Analyze a list of text chunks and aggregate the results"""
        # Initialize sentiment scores
        total_scores = {label: 0.0 for label in self.labels}
        chunk_scores = []
        
        # Analyze each chunk
        for chunk in chunks:
            sentiment = self._analyze_sentiment(chunk["content"])
            chunk_scores.append({
                "content": chunk["content"][:200] + "...",  # First 200 chars for reference
                "sentiment": sentiment
            })
            
            # Add to total scores
            for label in self.labels:
                total_scores[label] += sentiment[label]
        
        # Calculate average scores
        num_chunks = len(chunks)
        avg_scores = {
            label: score/num_chunks
            for label, score in total_scores.items()
        }
        
        return {
            "overall_sentiment": avg_scores,
            "chunk_analysis": chunk_scores
        }
    
    def predict_stock_movement(self, sentiment_analysis: Dict) -> Dict:
        """Predict stock movement based on sentiment analysis"""
        # Extract overall sentiment scores
        scores = sentiment_analysis["overall_sentiment"]
        
        # Calculate confidence score (0-100)
        confidence_score = (scores["positive"] - scores["negative"]) * 100
        
        # Determine direction and adjust confidence score
        if confidence_score > 0:
            direction = "up"
            confidence_score = min(confidence_score, 100)
        else:
            direction = "down"
            confidence_score = min(abs(confidence_score), 100)
        
        # Determine time horizon based on sentiment stability
        neutral_score = scores["neutral"]
        if neutral_score > 0.4:
            time_horizon = "short-term"  # High uncertainty
        elif neutral_score > 0.2:
            time_horizon = "mid-term"
        else:
            time_horizon = "long-term"  # Strong sentiment
        
        # Extract key factors from most positive/negative chunks
        chunk_analysis = sentiment_analysis["chunk_analysis"]
        pos_chunks = sorted(chunk_analysis, key=lambda x: x["sentiment"]["positive"], reverse=True)[:3]
        neg_chunks = sorted(chunk_analysis, key=lambda x: x["sentiment"]["negative"], reverse=True)[:3]
        
        return {
            "direction": direction,
            "confidence_score": confidence_score,
            "time_horizon": time_horizon,
            "key_factors": [chunk["content"] for chunk in pos_chunks],
            "risks": [chunk["content"] for chunk in neg_chunks],
            "reasoning": f"Analysis based on FinBERT sentiment scores - Positive: {scores['positive']:.2f}, "
                        f"Negative: {scores['negative']:.2f}, Neutral: {scores['neutral']:.2f}"
        }
    
    def analyze_document(self, document_chunks: List[Dict]) -> Dict:
        """
        Analyze a complete document and provide stock movement prediction
        
        Args:
            document_chunks: List of relevant document chunks from vector store
        """
        # Perform sentiment analysis
        sentiment_analysis = self.analyze_chunks(document_chunks)
        
        # Predict stock movement
        prediction = self.predict_stock_movement(sentiment_analysis)
        
        return {
            "sentiment_analysis": sentiment_analysis["overall_sentiment"],
            "analysis": prediction,
            "timestamp": datetime.now().isoformat()
        }