import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Logo e título
st.image("logo_engenharia.png", width=100)
st.markdown('<p style="text-align: center; font-size: 12px;">Engenharia</p>', unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>🔍 Diagnóstico por Manifestação Patológica</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Digite abaixo a manifestação observada (ex: fissura em viga, infiltração na parede, manchas em fachada...)</p>", unsafe_allow_html=True)

# Carregar base de dados
df = pd.read_csv("base_normas_com_recomendacoes_consultas.csv")

# Pré-processamento leve, sem spaCy
def preprocessar(texto):
    texto = str(texto).lower()
    texto = re.sub(r"[^\w\s]", "", texto)
    palavras = texto.split()
    return " ".join([p for p in palavras if len(p) > 2])

df["trecho_processado"] = df["trecho"].apply(preprocessar)

# Verificação de base válida
if df["trecho_processado"].isnull().all():
    st.error("Erro: A base de dados está vazia após o pré-processamento.")
    st.stop()

# Vetorização
vetorizador = TfidfVectorizer()
matriz_tfidf = vetorizador.fit_transform(df["trecho_processado"])

# Entrada do usuário
entrada = st.text_input("Descreva o problema:")

def buscar_normas(consulta):
    consulta_proc = preprocessar(consulta)
    if not consulta_proc.strip():
        return pd.DataFrame()

    consulta_vec = vetorizador.transform([consulta_proc])
    similaridades = cosine_similarity(consulta_vec, matriz_tfidf).flatten()

    top_indices = similaridades.argsort()[::-1]
    top_resultados = df.iloc[top_indices]
    top_resultados = top_resultados[similaridades[top_indices] > 0.1]  # Limite de relevância

    return top_resultados

if entrada:
    resultados = buscar_normas(entrada)
    if not resultados.empty:
        st.success("Resultados encontrados:")
        for _, linha in resultados.iterrows():
            st.markdown(f"""
<div style="margin-bottom: 20px;">
<b>🔎 Manifestação:</b> {linha['manifestacao']}  
📘 <b>Segundo a {linha['norma']}, seção {linha['secao']}:</b> {linha['trecho']}  
✅ <b>Recomendações:</b> {linha['recomendacoes']}  
🔁 <b>Consultas relacionadas:</b> {linha['consultas_relacionadas']}
</div>
""", unsafe_allow_html=True)
    else:
        st.warning("Nenhum resultado encontrado para essa manifestação.")

# Rodapé
st.markdown("<p style='text-align: center; font-size: 13px;'>Desenvolvido por Gézica Hemann | Engenharia Civil</p>", unsafe_allow_html=True)
