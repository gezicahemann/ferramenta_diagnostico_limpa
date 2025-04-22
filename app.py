import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64
import re

# Função para centralizar a logo e exibir com tamanho menor
def exibir_logo():
    with open("logo_engenharia.png", "rb") as img_file:
        logo_base64 = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{logo_base64}" alt="Logo" width="90"/>
        </div>
        """,
        unsafe_allow_html=True
    )

# Pré-processamento leve (sem spaCy)
def preprocessar(texto):
    texto = str(texto).lower()
    texto = re.sub(r"[^\w\s]", "", texto)
    palavras = texto.split()
    return " ".join([p for p in palavras if len(p) > 2])

# Carregar base de dados
df = pd.read_csv("base_normas_com_recomendacoes_consultas.csv")

# Criar coluna unificada para consulta
df["texto_unificado"] = (df["manifestacao"].astype(str) + " " + df["trecho"].astype(str)).apply(preprocessar)

# Verificação de dados válidos
if df["texto_unificado"].isnull().all():
    st.error("Erro: a base de dados não contém informações válidas para pesquisa.")
    st.stop()

# Vetorização
vetorizador = TfidfVectorizer()
matriz_tfidf = vetorizador.fit_transform(df["texto_unificado"])

# Interface
st.set_page_config(page_title="Diagnóstico Patológico", layout="centered")

st.markdown("""
    <style>
        .stTextInput > div > div > input {
            font-size: 18px;
        }
    </style>
    """, unsafe_allow_html=True)

exibir_logo()

st.markdown("<h1 style='text-align: center;'>🔎 Diagnóstico por Manifestação Patológica</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Digite abaixo a manifestação observada (ex: fissura em viga, infiltração na parede, manchas em fachada...)</p>", unsafe_allow_html=True)

entrada = st.text_input("Descreva o problema:")

def buscar_respostas(consulta):
    consulta_proc = preprocessar(consulta)
    if not consulta_proc.strip():
        return []

    consulta_vec = vetorizador.transform([consulta_proc])
    similaridades = cosine_similarity(consulta_vec, matriz_tfidf).flatten()
    indices_relevantes = similaridades.argsort()[::-1]

    respostas = []
    for i in indices_relevantes:
        if similaridades[i] > 0.25:  # Limite de relevância
            row = df.iloc[i]
            resposta = {
                "manifestacao": row["manifestacao"],
                "norma": row["norma"],
                "secao": row["secao"],
                "trecho": row["trecho"],
                "recomendacoes": row["recomendacoes"],
                "consultas": row["consultas_relacionadas"]
            }
            respostas.append(resposta)
    return respostas

# Exibição dos resultados
if entrada:
    resultados = buscar_respostas(entrada)
    if resultados:
        st.success("Resultados encontrados:")
        for r in resultados:
            texto_formatado = f"""
🔎 **Manifestação:** {r["manifestacao"]}  
📘 **Segundo a {r["norma"]}, seção {r["secao"]}:**  
{r["trecho"]}  

✅ **Recomendações:** {r["recomendacoes"]}  

📑 **Consultas relacionadas:** {r["consultas"]}
---
"""
            st.markdown(texto_formatado)
    else:
        st.warning("Nenhum resultado encontrado para essa manifestação.")

# Rodapé
st.markdown("<p style='text-align: center; font-size: 13px;'>Desenvolvido por Gézica Hemann | Engenharia Civil</p>", unsafe_allow_html=True)
