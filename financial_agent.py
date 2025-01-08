from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai 
from openai import OpenAI
from dotenv import load_dotenv
import os 


load_dotenv()
api_key= os.getenv("OPEN_API_KEY")

client = OpenAI(api_key=api_key)


## Web search agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model= Groq(id="llama-3.3-70b-versatile"),
    tools = [DuckDuckGo()],
    instructions=["Always include sources"],
    markdown=True
)


## Financial Agent 

finance_agent = Agent(
    name = "Finance AI Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools =[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,company_news=True)],
    instructions = ["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True
)

multi_ai_agent = Agent(
    team=[web_search_agent,finance_agent],
    show_tool_calls=True,
    markdown=True,
)


multi_ai_agent.print_response("Summarize analyst recommendation and share the latest news for NVDA",stream=True)


