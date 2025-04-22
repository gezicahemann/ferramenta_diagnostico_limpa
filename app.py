
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Carregar base de dados
df = pd.read_csv("base_normas_com_recomendacoes_consultas.csv")

# Pré-processamento leve, sem spaCy
def preprocessar(texto):
    texto = str(texto).lower()
    texto = re.sub(r"[^\w\s]", "", texto)
    palavras = texto.split()
    return " ".join([p for p in palavras if len(p) > 2])  # remove palavras curtas

df["trecho_processado"] = df["trecho"].apply(preprocessar)

# Verificação se base é válida
if df["trecho_processado"].isnull().all():
    st.error("A base de dados está vazia após o pré-processamento. Verifique se há textos válidos no campo 'trecho'.")
    st.stop()

# Vetorização
vetorizador = TfidfVectorizer()
matriz_tfidf = vetorizador.fit_transform(df["trecho_processado"])

# Interface
st.markdown(
    f"<div style='text-align: center;'><img src='logo_engenharia.png' width='100'/></div>",
    unsafe_allow_html=True,
)

st.markdown("<h1 style='text-align: center;'>🧱 Diagnóstico por Manifestação Patológica</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Digite abaixo a manifestação observada (ex: fissura em viga, infiltração na parede, manchas em fachada...)</p>", unsafe_allow_html=True)

entrada = st.text_input("Descreva o problema:")

def buscar_normas(consulta):
    consulta_proc = preprocessar(consulta)
    if not consulta_proc.strip():
        return pd.DataFrame()

    consulta_vec = vetorizador.transform([consulta_proc])
    similaridades = cosine_similarity(consulta_vec, matriz_tfidf).flatten()

    top_indices = similaridades.argsort()[::-1]
    top_resultados = df.iloc[top_indices]
    top_resultados = top_resultados[["manifestacao", "norma", "secao", "trecho", "recomendacoes", "consultas_relacionadas"]]
    top_resultados = top_resultados[similaridades[top_indices] > 0.1]  # Limite de relevância

    return top_resultados

if entrada:
    resultados = buscar_normas(entrada)
    if not resultados.empty:
        st.success("Resultados encontrados:")
        st.dataframe(resultados)
    else:
        st.warning("Nenhum resultado encontrado para essa manifestação.")

# Rodapé
st.markdown('<div class="rodape">Desenvolvido por Gézica Hemann | Engenharia Civil</div>', unsafe_allow_html=True)
