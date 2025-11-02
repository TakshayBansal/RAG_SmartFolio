from typing import Dict
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

class StockAnalyzer:
    def __init__(self, model_name: str = "gpt-4-1106-preview"):
        self.llm = ChatOpenAI(model_name=model_name, temperature=0)
        self.template = """
        You are a financial expert analyzing company reports to predict stock movement potential.
        
        Based on the following information from a company's report, analyze the likelihood of the stock price going up or down.
        Consider these factors:
        1. Financial performance and metrics
        2. New projects and initiatives
        3. Market expansion and investments
        4. Industry trends and competition
        5. Risk factors and challenges
        
        Information from the report:
        {context}
        
        Additional financial metrics:
        {metrics}
        
        Please provide:
        1. A confidence score (0-100) for stock price movement (up/down)
        2. Key factors influencing your decision
        3. Potential risks to consider
        
        Format your response as JSON with the following structure:
        {
            "direction": "up/down",
            "confidence_score": number,
            "key_factors": [list of factors],
            "risks": [list of risks],
            "reasoning": "brief explanation"
        }
        """
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def analyze_stock_potential(self, context: str, metrics: Dict) -> Dict:
        """
        Analyze stock potential based on report context and metrics
        """
        response = self.chain.run(
            context=context,
            metrics=metrics
        )
        return response