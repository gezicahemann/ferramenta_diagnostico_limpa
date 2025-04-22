import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64
import re

# Função para exibir imagem centralizada
def exibir_logo():
    st.markdown(
        """
        <div style="text-align: center;">
            <img src="data:image/png;base64,{}" width="100"/>
            <p style="font-size: 14px; color: gray;">Engenharia</p>
        </div>
        """.format(
            base64.b64encode(open("logo_engenharia.png", "rb").read()).decode()
        ),
        unsafe_allow_html=True,
    )

# Pré-processamento leve
def preprocessar(texto):
    texto = str(texto).lower()
    texto = re.sub(r"[^\w\s]", "", texto)
    palavras = texto.split()
    return " ".join([p for p in palavras if len(p) > 2])

# Carregamento e preparação da base
df = pd.read_csv("base_normas_com_recomendacoes_consultas.csv")
df["trecho_processado"] = df["trecho"].apply(preprocessar)

# Vetorização dos trechos
vetorizador = TfidfVectorizer()
matriz_tfidf = vetorizador.fit_transform(df["trecho_processado"])

# Interface Streamlit
st.set_page_config(page_title="Diagnóstico por Manifestação Patológica", layout="centered")
exibir_logo()

st.markdown("<h1 style='text-align: center;'>🔎 Diagnóstico por Manifestação Patológica</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Digite abaixo a manifestação observada (ex: fissura em viga, infiltração na parede, manchas em fachada...)</p>", unsafe_allow_html=True)

entrada = st.text_input("Descreva o problema:")

def buscar_normas(consulta):
    consulta_proc = preprocessar(consulta)
    if not consulta_proc.strip():
        return pd.DataFrame()

    consulta_vec = vetorizador.transform([consulta_proc])
    similaridades = cosine_similarity(consulta_vec, matriz_tfidf).flatten()

    top_indices = similaridades.argsort()[::-1]
    resultados = df.iloc[top_indices]
    resultados = resultados[similaridades[top_indices] > 0.1]

    return resultados.head(3)

def formatar_resultado(linha):
    return f"""
🔎 <b>Manifestação:</b> {linha['manifestacao']}<br>
📘 <b>Segundo a {linha['norma']}, seção {linha['secao']}:</b><br>
{linha['trecho']}<br><br>
✅ <b>Recomendações:</b> {linha['recomendacoes']}<br><br>
💬 <b>Consultas relacionadas:</b> {linha['consultas_relacionadas']}<hr>
    """

if entrada:
    resultados = buscar_normas(entrada)
    if not resultados.empty:
        st.success("Resultados encontrados:")
        for _, linha in resultados.iterrows():
            st.markdown(formatar_resultado(linha), unsafe_allow_html=True)
    else:
        st.warning("Nenhum resultado encontrado para essa manifestação.")

# Rodapé
st.markdown("<p style='text-align: center; font-size: 13px;'>Desenvolvido por Gézica Hemann | Engenharia Civil</p>", unsafe_allow_html=True)
