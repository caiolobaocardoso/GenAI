import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from langchain_anthropic import ChatAnthropic
from pandasai import SmartDataframe
import pandas as pd
from dotenv import load_dotenv
import base64

load_dotenv()

# Transformar a imagem de fundo em base 64
def base_64_backgroung(background):
    with open(background, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Faz o processamento pra IA burra entender formato de data
def process_dataframe(df):
    return df.astype(str)

# Configura o plano de fundo do streamlit
def set_background(png_file):
    bin_str = base_64_backgroung(png_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Colocar as configs da pagina do streamlit
st.set_page_config(page_title="Análise de Dados com AI", page_icon="🤖", layout="wide")

# Imagem de fundo - alterar apenas o path
set_background('/home/caio/workspace/genai-v2/background.jpg')

# CSS da pagina
st.markdown("""
<style>
    /* Style for the main content area */
    .css-1d391kg {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 10px;
    }
    /* Style for buttons */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    /* Style for text inputs */
    .stTextInput>div>div>input {
        background-color: #ffffff;
    }
    /* Style for main title */
    .main-title {
        color: #8E24AA;
        text-align: center;
        padding-top: 20px;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Título 
st.markdown("<h1 class='main-title'>Análise de dados usando IA</h1>", unsafe_allow_html=True)

# Colunas
col1, col2 = st.columns([1, 3])

# Sidebar - Seção de controle do dataset e tipo de gráfico
with st.sidebar:
    st.markdown("### Configurações")
    dataset_option = st.radio(
        "Escolha uma base de dados:",
        ("Faça um upload", "Penguins", "Iris", "Titanic")
    )

    if dataset_option == "Faça um upload":
        uploaded_file = st.file_uploader("Faça um upload de arquivo CSV ou Excel", type=["csv", "xlsx"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file, parse_dates=True)
                elif uploaded_file.name.endswith('.xlsx'):
                    data = pd.read_excel(uploaded_file, parse_dates=True)
                else:
                    st.warning("Formato Inválido. Por favor faça um upload de arquivo CSV ou Excel.")
                    data = None
                
                if data is not None:
                    data = process_dataframe(data)
            except Exception as e:
                st.error(f"Erro ao ler o arquivo: {str(e)}")
                data = None
        else:
            data = None
    else:
        data = sns.load_dataset(dataset_option.lower())
        data = process_dataframe(data)

    # Escolha do tipo de gráfico
    if data is not None:
        grafico_opcao = st.selectbox(
            "Escolha o tipo de gráfico",
            ("Selecione", "Gráfico de Barras", "Gráfico de Dispersão", "Gráfico de Linha", "Histograma", "Boxplot")
        )

        # Mostrar as opções de colunas de acordo com o gráfico escolhido
        if grafico_opcao == "Gráfico de Barras" or grafico_opcao == "Gráfico de Dispersão" or grafico_opcao == "Gráfico de Linha":
            coluna_x = st.selectbox("Selecione a coluna para o eixo X", data.columns.tolist())
            coluna_y = st.selectbox("Selecione a coluna para o eixo Y", data.columns.tolist())
        elif grafico_opcao == "Histograma" or grafico_opcao == "Boxplot":
            coluna = st.selectbox("Selecione a coluna", data.columns.tolist())
        else:
            coluna_x, coluna_y, coluna = None, None, None

        if st.button("Gerar Gráfico"):
            with col2:
                # Geração dos gráficos com base no tipo selecionado
                if grafico_opcao == "Gráfico de Barras":
                    st.markdown("### Gráfico de Barras")
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x=data[coluna_x], y=data[coluna_y])
                    st.pyplot(plt)

                elif grafico_opcao == "Gráfico de Dispersão":
                    st.markdown("### Gráfico de Dispersão")
                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(x=data[coluna_x], y=data[coluna_y])
                    st.pyplot(plt)

                elif grafico_opcao == "Gráfico de Linha":
                    st.markdown("### Gráfico de Linha")
                    plt.figure(figsize=(10, 6))
                    sns.lineplot(x=data[coluna_x], y=data[coluna_y])
                    st.pyplot(plt)

                elif grafico_opcao == "Histograma":
                    st.markdown("### Histograma")
                    plt.figure(figsize=(10, 6))
                    sns.histplot(data[coluna])
                    st.pyplot(plt)

                elif grafico_opcao == "Boxplot":
                    st.markdown("### Boxplot")
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(x=data[coluna])
                    st.pyplot(plt)

# Interação com IA via prompts
with col2:
    if data is not None:
        st.markdown("### Preview dos Dados")
        st.dataframe(data.head(5), use_container_width=True)

        # Configuração do modelo LLM
        model = ChatAnthropic(model="claude-3-haiku-20240307")
        df = SmartDataframe(data, config={"llm": model})

        st.markdown("### Pergunte para o Claude!")
        prompt = st.text_area("O que está pensando?", height=100)

        if st.button("Gerar"):
            if prompt:
                with st.spinner("Pensando..."):
                    try:
                        result = df.chat(prompt)
                        st.markdown("### Resultado")
                        st.write(result)
                    except Exception as e:
                        st.error(f"Desculpe, ocorreu um erro ao processar sua solicitação: {str(e)}")
                        st.info("Por favor, tente reformular sua pergunta ou verificar se os dados estão no formato correto.")
            else:
                st.warning("Preciso de algo para poder pensar!")
    elif dataset_option == "Faça um upload":
        st.info("Por favor, faça upload de um arquivo CSV ou Excel.")
