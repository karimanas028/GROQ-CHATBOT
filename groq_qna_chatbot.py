import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

prompt=ChatPromptTemplate.from_messages([('system', 'You are a helpful AI assistant'),
('human','Question:{Question}')])

st.secrets['GROQ_API_KEY']

def generate_response(query, llm, temperature, max_tokens, api_key):
    model=ChatGroq(model_name=llm,temperature=temperature, groq_api_key= api_key, max_tokens=max_tokens)
    parser= StrOutputParser()
    chain= prompt | model | parser
    result=chain.invoke({'Question': query})
    return result

st.title('Q/A Chatbot using Groq')

query=st.text_input('What is your query?')

st.sidebar.title('Settings')

api=st.sidebar.text_input('Provide the API key', type= 'password')

llm=st.sidebar.selectbox('Select the LLM : ',[
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
    "gemma2-9b-it",
    "qwen/qwen3-32b",
])

temperature= st.sidebar.slider('Select Temperature: ',min_value=0.0, max_value=2.0,value=0.8)

max_tokens= st.sidebar.slider('Select the maximum tokens:', min_value=50, max_value=10000, value=1000)

if st.button('Answer'):
    if query and api:
        response= generate_response(query,llm,temperature,max_tokens,api)
        st.write(response)
    else:
        st.warning('Please enter enough details')
