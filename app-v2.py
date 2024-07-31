from pandasai.llm.local_llm import LocalLLM
import streamlit as st 
import pandas as pd
from pandasai import Agent

model = LocalLLM(
    api_base="http://localhost:11434/v1",
    model="orca-mini:latest" 
) 

st.title("Análise de dado usando AI")

uploaded_file = st.sidebar.file_uploader(
    "Carregue um arquivo CSV",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head(5))

    agent = Agent(data, config={'llm': model})
    prompt = st.text_input("Converse com a IA")

    if st.button("Gerar"):
        if prompt:
            with st.spinner("Gerando Resposta..."):
                st.write(agent.chat(prompt))