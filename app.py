import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === CONFIGURAÇÃO DA PÁGINA e ESTILO ===
st.set_page_config(page_title="Diagnóstico Patológico", layout="centered")

st.markdown("""
    <style>
        /* Centraliza logo */
        .logo-container { text-align: center; margin: 20px 0; }
        .logo-container img { width: 80px; }

        /* Resultado: fonte um pouco menor e espaçamento */
        .resultado { font-size: 0.95em; line-height: 1.4em; margin-bottom: 1.5em; }

        /* Label do input mais escuro */
        label[for="textarea"] { color: #333 !important; }

        /* Rodapé */
        .rodape { text-align: center; margin-top: 50px; font-size: 0.9em; color: #888; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="logo-container"><img src="logo_engenharia.png" alt="Logo Engenharia"></div>', unsafe_allow_html=True)
st.markdown("## 🔎 Diagnóstico por Manifestação Patológica")
st.write("Digite abaixo a manifestação observada (ex: fissura em viga, infiltração na parede, manchas em fachada...)")

# === FUNÇÃO DE PRÉ-PROCESSAMENTO (SEM ALTERAR) ===
def preprocessar(texto):
    texto = str(texto).lower()
    texto = re.sub(r"[^\w\s]", "", texto)
    palavras = texto.split()
    # opcional: tratar 'fissur' como raíz
    return " ".join(p for p in palavras if len(p) > 2)

# === CARREGA E PREPARA A BASE (SEM ALTERAR) ===
df = pd.read_csv("base_normas_com_recomendacoes_consultas.csv")
df["trecho_processado"] = df["trecho"].apply(preprocessar)

# === VETORIZADOR E MATRIZ TF‑IDF (SEM ALTERAR) ===
vet = TfidfVectorizer()
matriz = vet.fit_transform(df["trecho_processado"])

# === FUNÇÃO DE BUSCA “TRAVADA” ===
def buscar_normas(consulta):
    proc = preprocessar(consulta)
    if not proc:
        return pd.DataFrame()

    vec = vet.transform([proc])
    sims = cosine_similarity(vec, matriz).flatten()
    idxs = sims.argsort()[::-1]
    resultados = df.iloc[idxs][sims[idxs] > 0.1]  # limiar de relevância
    return resultados

# === INTERAÇÃO COM USUÁRIO ===
entrada = st.text_input("Descreva o problema:")

if entrada:
    res = buscar_normas(entrada)
    if not res.empty:
        st.success("Resultados encontrados:")
        for _, row in res.iterrows():
            st.markdown(f"""
<div class="resultado">
🔎 **Manifestação:** {row['manifestacao']}  
📘 **Segundo a {row['norma']}, seção {row['secao']}:**  
{row['trecho']}  

✅ **Recomendações:** {row['recomendacoes']}  

🔁 **Consultas relacionadas:** {row['consultas_relacionadas']}
</div>
""", unsafe_allow_html=True)
    else:
        st.warning("Nenhum resultado encontrado para essa manifestação.")

# === RODAPÉ ===
st.markdown('<div class="rodape">Desenvolvido por Gézica Hemann | Engenharia Civil</div>', unsafe_allow_html=True)
