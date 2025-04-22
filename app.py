import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuração da página
st.set_page_config(page_title="Diagnóstico Patológico", layout="centered")

# Estilização customizada
st.markdown(
    """
    <style>
        .logo-container {
            display: flex;
            justify-content: center;
            margin-bottom: 10px;
        }
        .logo-container img {
            width: 100px;
        }
        .rodape {
            text-align: center;
            margin-top: 50px;
            font-size: 0.9em;
            color: #888;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Logo centralizada
st.markdown('<div class="logo-container"><img src="logo_engenharia.png" alt="Logo Engenharia"></div>', unsafe_allow_html=True)

# Título
st.markdown("## 🔎 Diagnóstico por Manifestação Patológica")
st.write("Digite abaixo a manifestação observada (ex: fissura em viga, infiltração na parede, manchas em fachada...)")

# Pré-processamento leve
def preprocessar(texto):
    texto = str(texto).lower()
    texto = re.sub(r"[^\w\s]", "", texto)
    palavras = texto.split()
    return " ".join(p for p in palavras if len(p) > 2)

# Carregar base
df = pd.read_csv("base_normas_com_recomendacoes_consultas.csv")
df["trecho_processado"] = df["trecho"].apply(preprocessar)

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

# Mostrar resultados
if entrada:
    resultados = buscar_normas(entrada)

    if not resultados.empty:
        st.success("Resultados encontrados:")

        for i, linha in resultados.iterrows():
            st.markdown(f"""
🔎 **Manifestação:** {linha['manifestacao']}
📘 **Segundo a {linha['norma']}, seção {linha['secao']}:** {linha['trecho']}
✅ **Recomendações:** {linha['recomendacoes']}
🔁 **Consultas relacionadas:** {linha['consultas_relacionadas']}
---
            """)
    else:
        st.warning("Nenhum resultado encontrado para essa manifestação.")

# Rodapé
st.markdown('<div class="rodape">Desenvolvido por Gézica Hemann | Engenharia Civil</div>', unsafe_allow_html=True)
