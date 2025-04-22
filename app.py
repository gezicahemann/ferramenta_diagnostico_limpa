import streamlit as st
import pandas as pd
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Centralizar layout e definir largura
st.set_page_config(layout="centered")

# Logo centralizada
st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://raw.githubusercontent.com/gezicahemann/ferramenta_diagnostico/main/logo_engenharia.png" width="80">
        <div style="margin-top: 5px; font-size: 12px;">Engenharia</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Título e descrição
st.markdown("<h1 style='text-align: center;'>🔍 Diagnóstico por Manifestação Patológica</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>Digite abaixo a manifestação observada (ex: fissura em viga, infiltração na parede, manchas em fachada...)</p>",
    unsafe_allow_html=True
)

# Campo de entrada
entrada = st.text_input("Descreva o problema:")

# Carregar a base de dados
df = pd.read_csv("base_normas_com_recomendacoes_consultas.csv")

# Função de normalização para ignorar acentos e caixa
def normalizar(texto):
    texto = str(texto).lower()
    return unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')

# Criar coluna de texto pré-processado
df["trecho_processado"] = df["trecho"].apply(normalizar)

# Vetorização
vetorizador = TfidfVectorizer()
matriz_tfidf = vetorizador.fit_transform(df["trecho_processado"])

# Função de busca
def buscar_normas(consulta):
    consulta_proc = normalizar(consulta)
    if not consulta_proc.strip():
        return pd.DataFrame()
    
    consulta_vec = vetorizador.transform([consulta_proc])
    similaridades = cosine_similarity(consulta_vec, matriz_tfidf).flatten()

    top_indices = similaridades.argsort()[::-1]
    top_similares = similaridades[top_indices]
    top_relevantes = [i for i, score in zip(top_indices, top_similares) if score > 0.15]

    return df.loc[top_relevantes]

# Exibir resultados
if entrada:
    resultados = buscar_normas(entrada)
    if not resultados.empty:
        st.success("Resultados encontrados:")
        for _, linha in resultados.iterrows():
            st.markdown(
                f"""
                <div style='margin-bottom: 20px;'>
                    <p>🔎 <b>Manifestação:</b> {linha['manifestacao']}</p>
                    <p>📘 <b>Segundo a {linha['norma']}, seção {linha['secao']}:</b><br>{linha['trecho']}</p>
                    <p>✅ <b>Recomendações:</b> {linha['recomendacoes']}</p>
                    <p>🔁 <b>Consultas relacionadas:</b> {linha['consultas_relacionadas']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.warning("Nenhum resultado encontrado para essa manifestação.")

# Rodapé
st.markdown(
    "<div style='text-align: center; margin-top: 40px;'>Desenvolvido por Gézica Hemann | Engenharia Civil</div>",
    unsafe_allow_html=True
)
