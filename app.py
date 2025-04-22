import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === CONFIGURAÇÃO DA PÁGINA e ESTILO ===
st.set_page_config(page_title="Diagnóstico Patológico", layout="centered")

st.markdown("""
    <style>
        /* Centraliza e dimensiona a logo */
        .logo-container {
            display: flex;
            justify-content: center;
            margin-bottom: 20px;
        }
        .logo-container img {
            width: 80px;
            height: auto;
        }

        /* Remove o texto “Logo Engenharia” quebrado */
        .logo-alt { display: none; }

        /* Título principal com destaque */
        .titulo {
            text-align: center;
            font-size: 2rem;
            margin-bottom: 5px;
        }

        /* Input label mais escuro */
        .stTextInput > label {
            color: #333 !important;
        }

        /* Área de resultado */
        .resultado {
            font-size: 0.95em;
            line-height: 1.5em;
            margin-bottom: 1.5em;
        }

        /* Rodapé */
        .rodape {
            text-align: center;
            margin-top: 50px;
            font-size: 0.9em;
            color: #888;
        }
    </style>
""", unsafe_allow_html=True)

# === LOGO ===
st.markdown('<div class="logo-container"><img src="logo_engenharia.png" /></div>', unsafe_allow_html=True)

# === TÍTULO E SUBTÍTULO ===
st.markdown('<div class="titulo">🔎 Diagnóstico por Manifestação Patológica</div>', unsafe_allow_html=True)
st.write("Digite abaixo a manifestação observada (ex: fissura em viga, infiltração na parede, manchas em fachada...)")

# === PRÉ-PROCESSAMENTO (mantido) ===
def preprocessar(texto):
    texto = str(texto).lower()
    texto = re.sub(r"[^\w\s]", "", texto)
    tokens = texto.split()
    # remove palavras muito curtas, mantém raiz de "fissur"
    return " ".join(tok for tok in tokens if len(tok) > 2)

# === CARREGA BASE DE DADOS (verifique o nome exato do CSV) ===
df = pd.read_csv("base_normas_com_recomendacoes_consultas.csv")
df["trecho_processado"] = df["trecho"].apply(preprocessar)

# === VETORIZAÇÃO TF-IDF (mantido) ===
vet = TfidfVectorizer()
matriz = vet.fit_transform(df["trecho_processado"])

# === FUNÇÃO DE BUSCA (mantida) ===
def buscar_normas(consulta):
    proc = preprocessar(consulta)
    if not proc.strip():
        return pd.DataFrame()
    vec = vet.transform([proc])
    sims = cosine_similarity(vec, matriz).flatten()
    idxs = sims.argsort()[::-1]
    resultados = df.iloc[idxs][sims[idxs] > 0.1]
    return resultados

# === INTERAÇÃO ===
entrada = st.text_input("Descreva o problema:")

if entrada:
    resultados = buscar_normas(entrada)
    if not resultados.empty:
        st.success("Resultados encontrados:")
        for _, row in resultados.iterrows():
            st.markdown(f"""
<div class="resultado">
**🔎 Manifestação:** {row['manifestacao']}  
**📘 Segundo a {row['norma']}, seção {row['secao']}:**  
{row['trecho']}  

**✅ Recomendações:**  
{row['recomendacoes']}  

**🔁 Consultas relacionadas:** {row['consultas_relacionadas']}
</div>
""", unsafe_allow_html=True)
    else:
        st.warning("Nenhum resultado encontrado para essa manifestação.")

# === RODAPÉ ===
st.markdown('<div class="rodape">Desenvolvido por Gézica Hemann | Engenharia Civil</div>', unsafe_allow_html=True)
