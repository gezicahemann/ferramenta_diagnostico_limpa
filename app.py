import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === PÁGINA & CSS ===
st.set_page_config(page_title="Diagnóstico Patológico", layout="centered")
st.markdown("""
<style>
  .resultado {
    font-size: 0.95em;
    line-height: 1.4em;
    margin-bottom: 2em;
  }
  .resultado p {
    margin: 0.3em 0;
  }
  .rodape {
    text-align: center; margin-top: 50px; font-size: 0.9em; color: #888;
  }
</style>
""", unsafe_allow_html=True)

# === LOGO CENTRALIZADO via st.image ===
with st.container():
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("logo_engenharia.png", width=80)

# === TÍTULO & SUBTÍTULO ===
st.markdown("## 🔎 Diagnóstico por Manifestação Patológica")
st.write("Digite abaixo a manifestação observada (ex: fissura em viga, infiltração na parede, manchas em fachada...)")

# === PREPROCESSAMENTO LEVE ===
def preprocessar(texto: str) -> str:
    txt = texto.lower()
    txt = re.sub(r"[^\w\s]", "", txt)
    toks = txt.split()
    return " ".join(t for t in toks if len(t) > 2)

# === CARREGA A BASE ===
df = pd.read_csv("base_normas_com_recomendacoes_consultas.csv")
df["trecho_proc"] = df["trecho"].apply(preprocessar)

# === VETORIZADOR ===
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), lowercase=True)
tfidf_matrix = vectorizer.fit_transform(df["trecho_proc"])

def buscar(consulta: str) -> pd.DataFrame:
    proc = preprocessar(consulta)
    if not proc:
        return pd.DataFrame()
    vec = vectorizer.transform([proc])
    sims = cosine_similarity(vec, tfidf_matrix).flatten()
    idxs = sims.argsort()[::-1]
    encontrados = df.iloc[idxs][sims[idxs] > 0.1].copy()
    if encontrados.empty:
        mask = df["manifestacao"].str.contains(proc, case=False, na=False)
        encontrados = df[mask]
    return encontrados

# === INPUT & SAÍDA ===
entrada = st.text_input("Descreva o problema:")

if entrada:
    res = buscar(entrada)
    if not res.empty:
        st.success("Resultados encontrados:")
        for _, row in res.iterrows():
            st.markdown(f"""
<div class="resultado">
  <p><strong>🔎 Manifestação:</strong> {row['manifestacao']}</p>
  <p><strong>📘 Segundo a {row['norma']}, seção {row['secao']}:</strong><br>
  {row['trecho']}</p>
  <p><strong>✅ Recomendações:</strong><br>
  {row['recomendacoes']}</p>
  <p><strong>🔁 Consultas relacionadas:</strong> {row['consultas_relacionadas']}</p>
</div>
""", unsafe_allow_html=True)
    else:
        st.warning("Nenhum resultado encontrado para essa manifestação.")

# === RODAPÉ ===
st.markdown('<div class="rodape">Desenvolvido por Gézica Hemann | Engenharia Civil</div>', unsafe_allow_html=True)
