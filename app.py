import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun # to search from the internet we will use duck duck go search run
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler

import os
from dotenv import load_dotenv
load_dotenv()

#Arxiv and wikipedia tools
Arxiv_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
arxiv=ArxivQueryRun(api_wrapper=Arxiv_wrapper)

Wiki_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki=WikipediaQueryRun(api_wrapper=Wiki_wrapper)

search=DuckDuckGoSearchRun(name="Search")


st.title(" 🦜 LangChain: Chat with search")
"""
In this example, we're using 'StreamlitCallbackHandler' to display the thoughts and actions in an iteractive Streamlit app.
Try more LangChain Streamlit agent examples at[github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""


# Side bar settings

st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter the Groq API key:",type="password")

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt:=st.chat_input(placeholder="What is Machine Learning"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(groq_api_key=api_key,model_name="Llama3-8b-8192",streaming=True)
    tools=[search,arxiv,wiki]
    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)
# It basically lies in how they handle our content and memory, how they generate prompts from the language
# zero_shot_react_description . It should not rely on the chat history. It makes the decison based on the current input only without coinsdering any previous action.
# if chat_zero_shot react description. It uses chat history to remember the context of the chat and history of the conversation, and it accepts the certain structure in the chat history and might raise an error if it does not find it.
    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)
