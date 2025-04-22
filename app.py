import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Diagnóstico Patológico", layout="centered")

# Estilo customizado
st.markdown(
    """
    <style>
        .logo-container {
            display: flex;
            justify-content: center;
            margin-top: 10px;
            margin-bottom: -15px;
        }
        .logo-container img {
            width: 80px;
        }
        .rodape {
            text-align: center;
            margin-top: 50px;
            font-size: 0.9em;
            color: #888;
        }
        .resultado {
            font-size: 0.95rem;
            line-height: 1.6;
            margin-bottom: 20px;
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

# Função de pré-processamento simples
def preprocessar(texto):
    texto = str(texto).lower()
    texto = re.sub(r"[^\w\s]", "", texto)
    palavras = texto.split()
    return " ".join(p for p in palavras if len(p) > 2)

# Carregamento da base
df = pd.read_csv("base_normas_com_recomendacoes_consultas.csv")

# Coluna combinada para busca por similaridade
df["texto_completo"] = (
    df["manifestacao"].fillna('') + " "
    + df["trecho"].fillna('') + " "
    + df["recomendacoes"].fillna('') + " "
    + df["consultas_relacionadas"].fillna('')
)

# Pré-processamento da base
df["texto_processado"] = df["texto_completo"].apply(preprocessar)

# Vetorização
vetorizador = TfidfVectorizer()
matriz_tfidf = vetorizador.fit_transform(df["texto_processado"])

# Campo de entrada
entrada = st.text_input("Descreva o problema:")

def buscar_normas(consulta):
    consulta_proc = preprocessar(consulta)
    if not consulta_proc.strip():
        return pd.DataFrame()

    consulta_vec = vetorizador.transform([consulta_proc])
    similaridades = cosine_similarity(consulta_vec, matriz_tfidf).flatten()
    top_indices = similaridades.argsort()[::-1]
    top_resultados = df.iloc[top_indices]
    top_resultados = top_resultados[similaridades[top_indices] > 0.1]
    return top_resultados

# Exibição dos resultados
if entrada:
    resultados = buscar_normas(entrada)

    if not resultados.empty:
        st.success("Resultados encontrados:")

        for _, linha in resultados.iterrows():
            st.markdown(f"""
<div class="resultado">
🔎 **Manifestação:** {linha['manifestacao']}<br>
📘 **Segundo a {linha['norma']}, seção {linha['secao']}:**<br>
{linha['trecho']}<br><br>
✅ **Recomendações:**<br>
{linha['recomendacoes']}<br><br>
🔁 **Consultas relacionadas:** {linha['consultas_relacionadas']}
</div>
""", unsafe_allow_html=True)

    else:
        st.warning("Nenhum resultado encontrado para essa manifestação.")

# Rodapé
st.markdown('<div class="rodape">Desenvolvido por Gézica Hemann | Engenharia Civil</div>', unsafe_allow_html=True)
