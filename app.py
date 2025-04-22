import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from PIL import Image

# Centraliza elementos e define layout da página
st.set_page_config(page_title="Diagnóstico Patológico", layout="centered")

# Carrega imagem
logo = Image.open("logo_engenharia.png")
st.image(logo, width=100)

# Título
st.markdown("<h1 style='text-align: center;'>🔍 Diagnóstico por Manifestação Patológica</h1>", unsafe_allow_html=True)
st.write("Digite abaixo a manifestação observada (ex: fissura em viga, infiltração na parede, manchas em fachada...)")

# Carregando base
df = pd.read_csv("base_normas_com_recomendacoes_consultas.csv")

# Pré-processamento
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

df["trecho_processado"] = df["trecho"].apply(preprocess)

# Vetorização
vetor = TfidfVectorizer()
matriz = vetor.fit_transform(df["trecho_processado"])

# Entrada
entrada = st.text_input("Descreva o problema:")

def gerar_resposta(idx):
    linha = df.iloc[idx]
    resposta = f"""🔍 **Manifestação:** {linha['manifestacao']}  
📘 **Segundo a {linha['norma']}, seção {linha['secao']}:**  
{linha['trecho']}  
✅ **Recomendações:** {linha['recomendacoes']}  
📄 **Consultas relacionadas:** {linha['consultas_relacionadas']}  
"""
    return resposta

if entrada:
    entrada_proc = preprocess(entrada)
    entrada_vec = vetor.transform([entrada_proc])
    similaridades = cosine_similarity(entrada_vec, matriz).flatten()
    indices = similaridades.argsort()[::-1]

    resultados_encontrados = False
    for idx in indices:
        if similaridades[idx] > 0.2:
            st.markdown(gerar_resposta(idx))
            resultados_encontrados = True

    if not resultados_encontrados:
        st.warning("Nenhum resultado encontrado para essa manifestação.")
