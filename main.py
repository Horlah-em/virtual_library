from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.agents import create_openai_functions_agent
from langchain import hub
import pinecone
import os
from langchain_pinecone import Pinecone
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder,ChatPromptTemplate
from dotenv import load_dotenv


load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")
TAVILY_API_KEY=os.environ.get("TAVILY_API_KEY")

# creates a retrieval QA chain
def create_chain():
    template = """
    You're a helpful, polite, fact_based agent for answering legal related questions.
    Your job is to provide legal assistance for lawyers by offering advice or guidance based on the information provided.
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Respond in the persona of a legal professional.
    if the user doesn't include country in the query, add Nigeria to the query.
    include a citation session at the end which lists all relevant documents you can find as bullet points.
   
    {context}
    Question: {question}
    Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY
    )
    docsearch = Pinecone(
        embedding=embeddings, index_name="supreme-courts-cluster")
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY), 
        retriever=docsearch.as_retriever(), chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    return qa

# create a retrieval tool
def create_retrieval_tool():
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY
    )
    docsearch = Pinecone.from_existing_index(
        embedding=embeddings, index_name="supreme-courts-cluster")
    retriever=docsearch.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "legal_document_search",
        "search old legal information in Nigeria",
    )
    return retriever_tool

# creates a search tool
def create_search_tool():
    search = TavilySearchResults(description="a search engine for recent legal search")
    return search

# creates an agent with a retrieval tool and a search tool
def create_agent():
    retriever_tool = create_retrieval_tool()
    search_tool = create_search_tool()
     # creates a list of tools
    tools = [search_tool, retriever_tool]
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    template_string = """
    You're a helpful, polite, fact_based agent for answering library related questions.
    Your task is to assist library users in automatic cataloguing and classification, indexing and immersive biobliographic teaching.
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Respond in the persona of a library professional.
    If possible, include a reference like the title of book and the autor's name.
    """
    messages = [
        SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], 
                                                          template=template_string)),
        MessagesPlaceholder(variable_name='chat_history', optional=True),
        HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')),
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ]
    prompt = ChatPromptTemplate.from_messages(messages=messages)
    #creates agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor

# invokes llm with agent
def run_llm_agent(query):
    agent_executor = create_agent()
    output = agent_executor.invoke({"input": query})
    return output

# invokes llm by using retrieval QA chain
def run_llm(query):
    qa = create_chain()
    result = qa({"query": query})
    return result

def create_chat_memory(chat_history):
    return ConversationBufferWindowMemory(memory_key="history", chat_memory= chat_history, k=3 )
