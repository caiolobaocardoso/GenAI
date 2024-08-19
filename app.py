from dotenv import load_dotenv
import streamlit as st
import seaborn as sns
from langchain_anthropic import ChatAnthropic
from pandasai import SmartDataframe
import pandas as pd

load_dotenv()

st.title("Inteligência Artificial")

uploaded_file = st.sidebar.file_uploader("Carregue sua planilha", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head(3))
    model = ChatAnthropic(model="claude-3-haiku-20240307")
    df = SmartDataframe(data, config={"llm": model})

    prompt = st.text_area("O que está na sua mente?")

    if st.button("Gerar"):
        if prompt:
            with st.spinner("Gerando..."):
                st.write(df.chat(prompt))
