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

# Aplicar pré-processamento à base
df["trecho_processado"] = df["trecho"].apply(preprocessar)

# Verificação da base válida
if df["trecho_processado"].isnull().all() or df["trecho_processado"].str.strip().eq("").all():
    st.error("A base de dados está vazia após o pré-processamento. Verifique se há textos válidos no campo 'trecho'.")
    st.stop()

# Vetorização
vetorizador = TfidfVectorizer()
matriz_tfidf = vetorizador.fit_transform(df["trecho_processado"])

# Interface
st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://raw.githubusercontent.com/gezicahemann/ferramenta_diagnostico/main/logo_engenharia.png" width="120">
        <h1>🧱 Diagnóstico por Manifestação Patológica</h1>
        <p>Digite abaixo a manifestação observada (ex: fissura em viga, infiltração na parede, manchas em fachada...)</p>
    </div>
    """,
    unsafe_allow_html=True
)

entrada = st.text_input("Descreva o problema:")

# Função de busca
def buscar_normas(consulta):
    consulta_proc = preprocessar(consulta)
    if not consulta_proc.strip():
        return pd.DataFrame()

    consulta_vec = vetorizador.transform([consulta_proc])
    similaridades = cosine_similarity(consulta_vec, matriz_tfidf).flatten()

    top_indices = similaridades.argsort()[::-1]
    top_resultados = df.iloc[top_indices]
    top_resultados = top_resultados[
        ["manifestacao", "norma", "secao", "trecho", "recomendacoes", "consultas_relacionadas"]
    ]
    top_resultados = top_resultados[similaridades[top_indices] > 0.1]  # Limite de relevância
    return top_resultados

# Exibição dos resultados
if entrada:
    resultados = buscar_normas(entrada)
    if not resultados.empty:
        st.success("Resultados encontrados:")
        st.dataframe(resultados)
    else:
        st.warning("Nenhum resultado encontrado para essa manifestação.")

# Rodapé
st.markdown(
    '<div class="rodape" style="text-align:center; margin-top: 50px;">Desenvolvido por Gézica Hemann | Engenharia Civil</div>',
    unsafe_allow_html=True
)
