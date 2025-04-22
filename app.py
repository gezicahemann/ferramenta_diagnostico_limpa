import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === PÁGINA & CONFIGURAÇÃO ===
st.set_page_config(page_title="Diagnóstico Patológico", layout="centered")

# === CSS PERSONALIZADO ===
st.markdown("""
<style>
  /* Título e subtítulo */
  .titulo {
    text-align: center;
    font-size: 2.5rem;
    margin-bottom: 0.2rem;
  }
  .subtitulo {
    text-align: center;
    margin-bottom: 1.5rem;
    color: #555;
  }
  /* Resultado formatado */
  .resultado {
    font-size: 0.95em;
    line-height: 1.5em;
    margin-bottom: 2rem;
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

# === LOGO (centralizada via colunas) ===
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("logo_engenharia.png", width=80)

# === TÍTULO & SUBTÍTULO ===
st.markdown('<div class="titulo">🔎 Diagnóstico por Manifestação Patológica</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitulo">'
    'Digite abaixo a manifestação observada '
    '(ex: fissura em viga, infiltração na parede, manchas em fachada...)'
    '</div>',
    unsafe_allow_html=True
)

# === FUNÇÃO DE PRÉ‑PROCESSAMENTO ===
def preprocessar(texto: str) -> str:
    txt = texto.lower()
    txt = re.sub(r"[^\w\s]", "", txt)      # remove pontuação
    toks = txt.split()
    return " ".join(t for t in toks if len(t) > 2)

# === CARREGA A BASE E PREPROCESSA ===
df = pd.read_csv("base_normas_com_recomendacoes_consultas.csv")
df["trecho_proc"] = df["trecho"].apply(preprocessar)

# === VETORIZAÇÃO CHAR N‑GRAM (3 a 5) ===
vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3,5),
    lowercase=True
)
tfidf_matrix = vectorizer.fit_transform(df["trecho_proc"])

# === FUNÇÃO DE BUSCA ===
def buscar(consulta: str) -> pd.DataFrame:
    proc = preprocessar(consulta)
    if not proc:
        return pd.DataFrame()
    # busca por similaridade
    vec = vectorizer.transform([proc])
    sims = cosine_similarity(vec, tfidf_matrix).flatten()
    idxs = sims.argsort()[::-1]
    encontrados = df.iloc[idxs].loc[sims[idxs] > 0.1].copy()
    # fallback: busca direta por substring em 'manifestacao'
    if encontrados.empty:
        mask = df["manifestacao"].str.contains(proc, case=False, na=False)
        encontrados = df[mask].copy()
    return encontrados

# === INPUT E EXIBIÇÃO DE RESULTADOS ===
entrada = st.text_input("Descreva o problema:")

if entrada:
    resultados = buscar(entrada)
    if not resultados.empty:
        st.success("Resultados encontrados:")
        for _, row in resultados.iterrows():
            st.markdown(f"""
<div class="resultado">
<strong>🔎 Manifestação:</strong> {row['manifestacao']}<br><br>
<strong>📘 Segundo a {row['norma']}, seção {row['secao']}:</strong><br>
{row['trecho']}<br><br>
<strong>✅ Recomendações:</strong><br>
{row['recomendacoes']}<br><br>
<strong>🔁 Consultas relacionadas:</strong> {row['consultas_relacionadas']}
</div>
""", unsafe_allow_html=True)
    else:
        st.warning("Nenhum resultado encontrado para essa manifestação.")

# === RODAPÉ ===
st.markdown('<div class="rodape">Desenvolvido por Gézica Hemann | Engenharia Civil</div>', unsafe_allow_html=True)
