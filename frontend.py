import streamlit as st
from streamlit_chat import message
from streamlit_option_menu import option_menu
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SQLDatabase
from main import run_llm
from main import run_llm_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
st.set_page_config(page_title="Virtual Liberian", page_icon=":books:")


def init_database(user: str, password: str, host: str, port: str
                  , database: str) -> SQLDatabase:
  db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
  return SQLDatabase.from_uri(db_uri)
      
def get_sql_chain(db):
  template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: which 3 artists have the most tracks?
    SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
    Question: Name 10 artists
    SQL Query: SELECT Name FROM Artist LIMIT 10;
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
  prompt = ChatPromptTemplate.from_template(template)
  
  llm = ChatOpenAI(model="gpt-4-0125-preview")
#   llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
  
  def get_schema(_):
    return db.get_table_info()
  
  return (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm
    | StrOutputParser()
  )
    
def get_response(user_query: str, db: SQLDatabase, chat_history: list):
  sql_chain = get_sql_chain(db)
  
  template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""
  
  prompt = ChatPromptTemplate.from_template(template)
  
  llm = ChatOpenAI(model="gpt-4-0125-preview")
  #llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
  
  chain = (
    RunnablePassthrough.assign(query=sql_chain).assign(
      schema=lambda _: db.get_table_info(),
      response=lambda vars: db.run(vars["query"]),
    )
    | prompt
    | llm
    | StrOutputParser()
  )
  
  return chain.invoke({
    "question": user_query,
    "chat_history": chat_history,
  })
    
  
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
      AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
    ]

load_dotenv()









if "chat_history" not in st.session_state:
     st.session_state.chat_history =[
          AIMessage(content = "Hello! You can browse through the catalogue"),
     ]



selected = option_menu(
        menu_title= "Virtual Librarian",
        options= ["Home", "Catalogue", "Contact"],
        icons=["house", "book", "envelope"],
        menu_icon= "cast",
        default_index=0,
        orientation= "horizontal",)

if selected == "Home":
       prompt = st.chat_input("make your request...")

       if prompt:
          with st.spinner("Generating response..."):
               generated_response = run_llm_agent(query=prompt)
               # formatted_response = generated_response['result']
               formatted_response = generated_response['output']
               st.session_state["user_prompt_history"].append(prompt)
               st.session_state["chat_answers_history"].append(formatted_response)

          if st.session_state["chat_answers_history"]:

            for generated_answer, user_query in zip(st.session_state["chat_answers_history"],  st.session_state["user_prompt_history"]):
                message(user_query, is_user=True)
                message(generated_answer)

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

with st.sidebar:
        st.subheader("  CATALOGUE :memo:")
        st.write("Connect to DB to check Catalogues.")
        st.text_input("Host", value="localhost", key="Host")
        st.text_input("Port", value="3306", key ="Port")
        st.text_input("User", value="root", key="User")
        st.text_input("Password", type="password", value="", key= "Password")
        st.text_input("Database", value="MyFavour", key="Database")

        if st.button("Connect"):
             with st.spinner("Connecting to database"):
                  db = init_database(
                       st.session_state["User"],
                       st.session_state["Password"],
                       st.session_state["Host"],
                       st.session_state["Port"],
                       st.session_state["Database"]

                  )
                  st.session_state.db= db
                  st.success("connected to database!")







    




        


     

if selected == "Catalogue":
        
        st.title(f"{selected}")

        for message in st.session_state.chat_history:
          if isinstance(message, AIMessage):
               with st.chat_message("AI"):
                    st.markdown(message.content)
          elif isinstance(message, HumanMessage):
               with st.chat_message("Human"):
                    st.markdown(message.content)
        user_query = st.chat_input("Type a message...")
        if user_query is not None and user_query.strip() != "":
          st.session_state.chat_history.append(HumanMessage(content=user_query))

          with st.chat_message("Human"):
               st.markdown(user_query)

          with st.chat_message("AI"):
               response = get_response(user_query, st.session_state.chat_history)
               st.markdown(response)
if selected == "Contact":
        st.text(f"{selected} and About page.")
        with st.chat_message("Human"):
              st.markdown("Contact:")
              st.markdown("Name: Kingsley Moses")
              st.markdown("Telephone: 07946660845")
              st.markdown("Email: Kingsmoses44@gmail.com")
              st.markdown("Address: 110 middlesex street, London, E1 7ht")
        with st.chat_message("AI"):
              st.markdown("About: ")
              st.markdown("Vlibrary-bot is a conversational AI companion for an unparalleled digital reading experience. It’s a cutting-edge platform that harnesses advanced natural language processing to transform how readers explore and engage with literature. ")
              st.markdown("VLibrary-bot makes knowledge accessible through intuitive conversations. With a chat, the AI assistant can browse through vast catalogs spanning millions of titles across genres and subjects. The intelligent recommendation system understands reader’s preferences, reading habits, and interests, curating personalized suggestions tailored just for them.")

              
