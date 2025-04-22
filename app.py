import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Carregar base de dados
df = pd.read_csv("base_normas_com_recomendacoes_consultas.csv")

# Preprocessamento leve
def preprocessar(texto):
    texto = str(texto).lower()
    texto = re.sub(r"[^\w\s]", "", texto)
    palavras = texto.split()
    return " ".join([p for p in palavras if len(p) > 2])

# Aplicar pré-processamento
df["trecho_processado"] = df["trecho"].apply(preprocessar)

# Verifica se a base foi processada corretamente
if df["trecho_processado"].isnull().all() or df.empty:
    st.error("Erro: a base de dados está vazia ou mal formatada. Verifique o arquivo CSV.")
else:
    # Vetoriza os trechos das normas
    vetorizar = TfidfVectorizer()
    matriz_tfidf = vetorizar.fit_transform(df["trecho_processado"])

    # Interface do app
    st.markdown("""
    <div style='text-align: center;'>
        <img src='https://raw.githubusercontent.com/gezicahemann/ferramenta_diagnostico/main/logo_engenharia.png' width='120'>
        <h1>🔎 Diagnóstico por Manifestação Patológica</h1>
        <p>Digite abaixo a manifestação observada (ex: fissura em viga, infiltração na parede, manchas em fachada...)</p>
    </div>
    """, unsafe_allow_html=True)

    consulta = st.text_input("Descreva o problema:")

    def buscar_normas(consulta):
        consulta_proc = preprocessar(consulta)
        if not consulta_proc.strip():
            return pd.DataFrame()

        consulta_vec = vetorizar.transform([consulta_proc])
        similaridades = cosine_similarity(consulta_vec, matriz_tfidf).flatten()
        top_indices = similaridades.argsort()[::-1]
        top_resultados = df.iloc[top_indices]
        top_resultados = top_resultados[similaridades[top_indices] > 0.2]  # Limite de relevância
        return top_resultados

    if consulta:
        resultados = buscar_normas(consulta)

        if not resultados.empty:
            st.success("Resultados encontrados:")
            for _, linha in resultados.iterrows():
                st.markdown(f"""
                <div style='background-color: #f9f9f9; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
                <b>🔎 Manifestação:</b> {linha['manifestacao']}<br>
                <b>📘 Segundo a {linha['norma']}, seções {linha['secao']},</b> {linha['trecho']}<br>
                <b>✅ Recomendações:</b> {linha['recomendacoes']}<br>
                <b>🔁 Consultas relacionadas:</b> {linha['consultas_relacionadas']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Nenhum resultado encontrado para essa manifestação.")

    # Rodapé
    st.markdown("""
    <div style='text-align: center; font-size: 13px; margin-top: 40px;'>
        Desenvolvido por Gézica Hemann | Engenharia Civil
    </div>
    """, unsafe_allow_html=True)
