import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from pathlib import Path

# Centraliza a logo com o título
st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://raw.githubusercontent.com/gezicahemann/ferramenta_diagnostico/main/logo_engenharia.png" width="90"/>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<h1 style='text-align: center;'>🔍 Diagnóstico por Manifestação Patológica</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Digite abaixo a manifestação observada (ex: fissura em viga, infiltração na parede, manchas em fachada...)</p>",
    unsafe_allow_html=True
)

# Campo de busca
st.markdown("<label style='color: #333;'>Descreva o problema:</label>", unsafe_allow_html=True)
entrada = st.text_input("", key="entrada")

# Carrega base
df = pd.read_csv("base_normas_com_recomendacoes_consultas.csv")

# Pré-processamento
def preprocessar(texto):
    texto = str(texto).lower()
    texto = re.sub(r"[^\w\s]", "", texto)
    palavras = texto.split()
    return " ".join([p for p in palavras if len(p) > 2])

df["trecho_processado"] = df["trecho"].apply(preprocessar)

# Vetorização
vetorizador = TfidfVectorizer()
matriz_tfidf = vetorizador.fit_transform(df["trecho_processado"])

# Busca
def buscar(consulta):
    consulta_proc = preprocessar(consulta)
    if not consulta_proc.strip():
        return []

    consulta_vec = vetorizador.transform([consulta_proc])
    similaridades = cosine_similarity(consulta_vec, matriz_tfidf).flatten()
    top_indices = similaridades.argsort()[::-1]
    top_resultados = df.iloc[top_indices]
    top_resultados = top_resultados[similaridades[top_indices] > 0.1]
    return top_resultados

# Resultado
if entrada:
    resultados = buscar(entrada)
    if not resultados.empty:
        st.success("Resultados encontrados:")
        for _, row in resultados.iterrows():
            st.markdown(f"""
🔍 **Manifestação:** {row['manifestacao']}  
📘 **Segundo a {row['norma']}, seção {row['secao']}:**  
{row['trecho']}  

✅ **Recomendações:**  
{row['recomendacoes']}  

🔁 **Consultas relacionadas:**  
{row['consultas_relacionadas']}  
            """)
    else:
        st.warning("Nenhum resultado encontrado para essa manifestação.")

# Rodapé
st.markdown(
    "<div style='text-align: center; margin-top: 50px;'>Desenvolvido por Gézica Hemann | Engenharia Civil</div>",
    unsafe_allow_html=True
)
