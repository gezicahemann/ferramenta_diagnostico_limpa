import re
import unicodedata
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Função de pré-processamento aprimorada
def preprocessar(texto):
    texto = str(texto).lower()
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')  # remove acentos
    texto = re.sub(r"[^a-z\s]", "", texto)  # remove pontuação e números
    return texto

# Carregar base de dados
df = pd.read_csv("base_normas_com_recomendacoes_consultas.csv")
df["trecho_processado"] = df["trecho"].apply(preprocessar)

# Vetorização
vetorizador = TfidfVectorizer()
matriz_tfidf = vetorizador.fit_transform(df["trecho_processado"])

# Interface Streamlit
st.set_page_config(page_title="Diagnóstico por Manifestação Patológica", layout="centered")

# Logo e título
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("logo_engenharia.png", use_column_width=False, width=100)

st.markdown("<h1 style='text-align: center;'>🔍 Diagnóstico por Manifestação Patológica</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Digite abaixo a manifestação observada (ex: fissura em viga, infiltração na parede, manchas em fachada...)</p>", unsafe_allow_html=True)

entrada = st.text_input("Descreva o problema:")

def buscar_normas(consulta):
    consulta_proc = preprocessar(consulta)
    if not consulta_proc.strip():
        return []

    consulta_vec = vetorizador.transform([consulta_proc])
    similaridades = cosine_similarity(consulta_vec, matriz_tfidf).flatten()

    # Ajuste do threshold para permitir aproximações maiores
    limite = 0.1
    top_indices = [i for i, score in enumerate(similaridades) if score > limite]

    resultados = []
    for i in top_indices:
        linha = df.iloc[i]
        texto_formatado = f"""
<br>
🔍 <b>Manifestação:</b> {linha['manifestacao']}  
📘 <b>Segundo a {linha['norma']}, seção {linha['secao']}:</b> {linha['trecho']}  
✅ <b>Recomendações:</b> {linha['recomendacoes']}  
🧱 <b>Consultas relacionadas:</b> {linha['consultas_relacionadas']}
"""
        resultados.append(texto_formatado)

    return resultados

# Exibir resultados
if entrada:
    resultados = buscar_normas(entrada)
    if resultados:
        st.success("Resultados encontrados:")
        for texto in resultados:
            st.markdown(texto, unsafe_allow_html=True)
    else:
        st.warning("Nenhum resultado encontrado para essa manifestação.")

# Rodapé
st.markdown("<p style='text-align: center;'>Desenvolvido por Gézica Hemann | Engenharia Civil</p>", unsafe_allow_html=True)
