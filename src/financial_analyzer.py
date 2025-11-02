from typing import Dict, List, Optional
import json
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class FinancialMetrics(BaseModel):
    revenue: Optional[float] = Field(None, description="Revenue in millions/billions")
    revenue_growth: Optional[float] = Field(None, description="Revenue growth percentage")
    net_income: Optional[float] = Field(None, description="Net income in millions/billions")
    profit_margin: Optional[float] = Field(None, description="Profit margin percentage")
    operating_margin: Optional[float] = Field(None, description="Operating margin percentage")
    eps: Optional[float] = Field(None, description="Earnings per share")
    debt_to_equity: Optional[float] = Field(None, description="Debt to equity ratio")

class StockAnalysis(BaseModel):
    direction: str = Field(..., description="Expected stock movement direction (up/down)")
    confidence_score: float = Field(..., description="Confidence score (0-100)")
    key_factors: List[str] = Field(..., description="Key factors influencing the decision")
    risks: List[str] = Field(..., description="Potential risks to consider")
    reasoning: str = Field(..., description="Brief explanation for the analysis")
    time_horizon: str = Field(..., description="Expected time horizon for the prediction (short-term/mid-term/long-term)")

class FinancialAnalyzer:
    """Analyzes financial documents and provides stock movement predictions"""
    
    def __init__(self, model_name: str = "gpt-4-1106-preview", temperature: float = 0):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.metrics_parser = PydanticOutputParser(pydantic_object=FinancialMetrics)
        self.analysis_parser = PydanticOutputParser(pydantic_object=StockAnalysis)
        
        # Initialize prompts
        self._init_prompts()
    
    def _init_prompts(self):
        """Initialize the prompts for different analysis tasks"""
        self.metrics_extraction_prompt = ChatPromptTemplate.from_template("""
        You are a financial expert extracting key metrics from financial documents.
        Extract the following metrics from the text below. If a metric is not found, return null.
        Format your response as JSON matching this structure:
        {
            "revenue": float or null,
            "revenue_growth": float or null,
            "net_income": float or null,
            "profit_margin": float or null,
            "operating_margin": float or null,
            "eps": float or null,
            "debt_to_equity": float or null
        }
        
        Text:
        {text}
        
        Extract only the most recent values for each metric.
        Ensure all numeric values are converted to the same scale (preferably millions).
        """)
        
        self.stock_analysis_prompt = ChatPromptTemplate.from_template("""
        You are a professional financial analyst evaluating a company's potential stock performance.
        
        Financial Metrics:
        {metrics}
        
        Key Information from Reports:
        {context}
        
        Additional Market Context (if available):
        {market_context}
        
        Based on all available information, analyze the likelihood of stock price movement.
        Consider:
        1. Financial performance and trends
        2. Market position and competitive advantage
        3. Growth initiatives and expansion plans
        4. Industry trends and market conditions
        5. Risk factors and potential challenges
        
        Provide your analysis in JSON format with the following structure:
        {
            "direction": "up" or "down",
            "confidence_score": float between 0 and 100,
            "key_factors": list of main factors supporting the prediction,
            "risks": list of potential risks or concerns,
            "reasoning": brief explanation for the conclusion,
            "time_horizon": "short-term" (< 6 months), "mid-term" (6-18 months), or "long-term" (> 18 months)
        }
        
        Focus on data-driven insights and be conservative in your confidence scoring.
        """)
    
    def extract_metrics(self, text: str) -> FinancialMetrics:
        """Extract financial metrics from text"""
        chain = LLMChain(llm=self.llm, prompt=self.metrics_extraction_prompt)
        response = chain.run(text=text)
        
        try:
            metrics_dict = json.loads(response)
            return FinancialMetrics(**metrics_dict)
        except Exception as e:
            print(f"Error parsing metrics: {e}")
            return FinancialMetrics()
    
    def analyze_stock_potential(self,
                              metrics: FinancialMetrics,
                              context: str,
                              market_context: str = "") -> StockAnalysis:
        """Analyze stock movement potential"""
        chain = LLMChain(llm=self.llm, prompt=self.stock_analysis_prompt)
        
        # Convert metrics to a readable format
        metrics_str = json.dumps(metrics.dict(), indent=2)
        
        response = chain.run(
            metrics=metrics_str,
            context=context,
            market_context=market_context
        )
        
        try:
            analysis_dict = json.loads(response)
            return StockAnalysis(**analysis_dict)
        except Exception as e:
            print(f"Error parsing analysis: {e}")
            raise
    
    def analyze_document(self,
                        document_chunks: List[Dict],
                        market_context: str = "") -> Dict:
        """
        Analyze a complete document and provide stock movement prediction
        
        Args:
            document_chunks: List of relevant document chunks from vector store
            market_context: Additional market context (optional)
        """
        # Combine relevant chunks for metrics extraction
        full_text = "\n".join(chunk["content"] for chunk in document_chunks)
        
        # Extract metrics
        metrics = self.extract_metrics(full_text)
        
        # Analyze stock potential
        analysis = self.analyze_stock_potential(
            metrics=metrics,
            context=full_text,
            market_context=market_context
        )
        
        return {
            "metrics": metrics.dict(),
            "analysis": analysis.dict(),
            "timestamp": datetime.now().isoformat()
        }

class MarketContextAnalyzer:
    """Analyzes broader market context for more accurate predictions"""
    
    def __init__(self, model_name: str = "gpt-4-1106-preview"):
        self.llm = ChatOpenAI(model_name=model_name)
        self._init_prompt()
    
    def _init_prompt(self):
        self.market_context_prompt = ChatPromptTemplate.from_template("""
        Analyze the following market-related information and provide a concise summary
        focusing on factors that could impact the company's stock performance:
        
        Industry Information:
        {industry_info}
        
        Market Conditions:
        {market_conditions}
        
        Competitor Analysis:
        {competitor_info}
        
        Provide a concise summary (2-3 paragraphs) highlighting the most relevant factors
        that could impact stock performance. Focus on:
        1. Industry trends and challenges
        2. Market conditions and sentiment
        3. Competitive dynamics
        4. Regulatory environment (if relevant)
        """)
    
    def generate_market_context(self,
                              industry_info: str,
                              market_conditions: str,
                              competitor_info: str) -> str:
        """Generate relevant market context for stock analysis"""
        chain = LLMChain(llm=self.llm, prompt=self.market_context_prompt)
        
        context = chain.run(
            industry_info=industry_info,
            market_conditions=market_conditions,
            competitor_info=competitor_info
        )
        
        return context