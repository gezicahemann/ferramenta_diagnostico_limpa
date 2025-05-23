import streamlit as st
import pandas as pd
import re
import base64
from io import BytesIO
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === CONFIG DA PÁGINA ===
st.set_page_config(page_title="Diagnóstico Patológico", layout="centered")

# === CSS ===
st.markdown("""
<style>
  .titulo {
    text-align: center; font-size: 2.5rem; margin-bottom: 0.2rem;
  }
  .subtitulo {
    text-align: center; color: #555; margin-bottom: 1.5rem;
  }
  .resultado {
    font-size: 0.95em; line-height: 1.5em; margin-bottom: 2rem;
  }
  .rodape {
    text-align: center; margin-top: 50px; font-size: 0.9em; color: #888;
  }
</style>
""", unsafe_allow_html=True)

# === LOGO EMBED & CENTRALIZAÇÃO ===
def load_logo(path: str, width: int = 80):
    img = Image.open(path)
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{b64}" width="{width}" style="display:block; margin:0 auto;" />'

st.markdown(load_logo("logo_engenharia.png", width=80), unsafe_allow_html=True)

# === TÍTULO & SUBTÍTULO ===
st.markdown('<div class="titulo">🔎 Manifestação Patológica</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitulo">'
    'Digite abaixo a manifestação observada '
    '(ex: fissura em viga, infiltração na parede, manchas em fachada...)'
    '</div>',
    unsafe_allow_html=True
)

# === PRÉ‑PROCESSAMENTO ===
def preprocessar(texto: str) -> str:
    txt = texto.lower()
    txt = re.sub(r"[^\w\s]", "", txt)
    toks = txt.split()
    return " ".join(t for t in toks if len(t) > 2)

# === BASE & VETORIZAÇÃO ===
df = pd.read_csv("base_normas_com_recomendacoes_consultas.csv")
df["trecho_proc"] = df["trecho"].apply(preprocessar)
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), lowercase=True)
tfidf_matrix = vectorizer.fit_transform(df["trecho_proc"])

# === BUSCA ===
def buscar(consulta: str) -> pd.DataFrame:
    proc = preprocessar(consulta)
    if not proc:
        return pd.DataFrame()
    vec = vectorizer.transform([proc])
    sims = cosine_similarity(vec, tfidf_matrix).flatten()
    idxs = sims.argsort()[::-1]
    encontrados = df.iloc[idxs].loc[sims[idxs] > 0.1].copy()
    if encontrados.empty:
        mask = df["manifestacao"].str.contains(proc, case=False, na=False)
        encontrados = df[mask].copy()
    return encontrados

# === INPUT & OUTPUT ===
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
