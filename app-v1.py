from pandasai.llm.local_llm import LocalLLM
import streamlit as st 
import pandas as pd
from pandasai import SmartDataframe

model = LocalLLM(
    api_base="http://localhost:11434/v1",
    model="llama3" 
) 

st.title("Análise de dado usando AI")

updloaded_file = st.file_uploader("Carregue um arquivo Excel", type=['csv'])

if updloaded_file is not None:
    data = pd.read_csv(updloaded_file)
    st.write(data.head(3))

    df = SmartDataframe(data, config={"llm": model})
    prompt = st.text_area("Insira seu prompt:")

    if st.button("Gerar"):
        if prompt:
            with st.spinner("Gerando resposta..."):
                st.write(df.chat(prompt))
            
