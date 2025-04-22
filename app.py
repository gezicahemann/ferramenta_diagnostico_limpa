import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Carregar base de dados
df = pd.read_csv("base_normas_com_recomendacoes_consultas.csv")

# Pré-processamento leve (sem spaCy)
def preprocessar(texto):
    texto = str(texto).lower()
    texto = re.sub(r"[^\w\s]", "", texto)
    palavras = texto.split()
    return " ".join([p for p in palavras if len(p) > 2])  # remove palavras curtas

# Criar coluna processada
df["trecho_processado"] = df["trecho"].apply(preprocessar)

# Vetorização com TF-IDF
vetorizador = TfidfVectorizer()
matriz_tfidf = vetorizador.fit_transform(df["trecho_processado"])

# Função de busca com refinamento
def buscar_normas(consulta):
    consulta_proc = preprocessar(consulta)
    if not consulta_proc.strip():
        return pd.DataFrame()

    # Primeiro: busca direta por nome da manifestação
    resultados_exatos = df[df["manifestacao"].str.lower().str.contains(consulta_proc)]
    if not resultados_exatos.empty:
        return resultados_exatos[
            ["manifestacao", "norma", "secao", "trecho", "recomendacoes", "consultas_relacionadas"]
        ]

    # Caso não encontre correspondência exata, aplica IA com TF-IDF
    consulta_vec = vetorizador.transform([consulta_proc])
    similaridades = cosine_similarity(consulta_vec, matriz_tfidf).flatten()

    top_indices = similaridades.argsort()[::-1]
    top_resultados = df.iloc[top_indices]
    top_resultados = top_resultados[
        ["manifestacao", "norma", "secao", "trecho", "recomendacoes", "consultas_relacionadas"]
    ]
    top_resultados = top_resultados[similaridades[top_indices] > 0.25]

    return top_resultados

# Interface
st.set_page_config(page_title="Diagnóstico Patológico", layout="centered")

# Estilo customizado
st.markdown("""
    <style>
        body { background-color: #111; color: #fff; }
        .stTextInput label { color: #ccc; font-weight: 500; }
        .rodape { text-align: center; margin-top: 4rem; font-size: 0.85rem; color: #aaa; }
        .stApp { padding-top: 2rem; }
        img { display: block; margin: auto; width: 100px; }
    </style>
""", unsafe_allow_html=True)

# Logo
st.image("logo_engenharia.png")

# Título e descrição
st.markdown("### 🧱 Diagnóstico por Manifestação Patológica")
st.write("Digite abaixo a manifestação observada (ex: fissura em viga, infiltração na parede, manchas em fachada...)")

# Campo de entrada
entrada = st.text_input("Descreva o problema:")

# Resultado
if entrada:
    resultados = buscar_normas(entrada)
    if not resultados.empty:
        st.success("Resultados encontrados:")
        st.dataframe(resultados, use_container_width=True)
    else:
        st.warning("Nenhum resultado encontrado para essa manifestação.")

# Rodapé
st.markdown('<div class="rodape">Desenvolvido por Gézica Hemann | Engenharia Civil</div>', unsafe_allow_html=True)
