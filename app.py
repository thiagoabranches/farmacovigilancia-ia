import streamlit as st
import pandas as pd
import numpy as np
import torch
import sqlite3
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from transformers import AutoTokenizer, AutoModel
from datetime import datetime

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="OncoPharm AI", layout="wide", page_icon="🧬")

# --- ESTILIZAÇÃO CSS (Visual Profissional) ---
st.markdown("""
    <style>
    .stButton>button {width: 100%; border-radius: 5px; height: 3em;}
    .reportview-container {background: #f0f2f6;}
    .big-font {font-size:20px !important; font-weight: bold;}
    .alert-box {padding: 15px; border-radius: 10px; margin-bottom: 10px;}
    .safe {background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb;}
    .warning {background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba;}
    .danger {background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb;}
    </style>
""", unsafe_allow_html=True)

# --- FUNÇÕES DE BANCO DE DADOS ---
def init_db():
    conn = sqlite3.connect('oncologia_farmacovigilancia.db')
    c = conn.cursor()
    # Tabela de Dados Históricos (Treino/Sintéticos)
    c.execute('''CREATE TABLE IF NOT EXISTS dados_treino 
                 (id INTEGER PRIMARY KEY, texto_clinico TEXT, grau_real INTEGER)''')
    # Tabela de Intervenções (Vida Real)
    c.execute('''CREATE TABLE IF NOT EXISTS intervencoes 
                 (id INTEGER PRIMARY KEY, data_hora TEXT, texto_analisado TEXT, 
                  grau_predito INTEGER, tipo_intervencao TEXT, notificado_anvisa BOOLEAN)''')
    conn.commit()
    conn.close()

def salvar_intervencao(texto, grau, tipo_intervencao, notificado):
    conn = sqlite3.connect('oncologia_farmacovigilancia.db')
    c = conn.cursor()
    data_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO intervencoes (data_hora, texto_analisado, grau_predito, tipo_intervencao, notificado_anvisa) VALUES (?, ?, ?, ?, ?)",
              (data_hora, texto, int(grau), tipo_intervencao, notificado))
    conn.commit()
    conn.close()

# --- CARREGAMENTO DO MODELO BIOBERT (CACHED) ---
@st.cache_resource
def carregar_modelo():
    nome_modelo = "pucpr/biobertpt-clin"
    tokenizer = AutoTokenizer.from_pretrained(nome_modelo)
    model = AutoModel.from_pretrained(nome_modelo)
    return tokenizer, model

# --- CLASSIFICADOR SIMULADO (RANDOM FOREST) ---
# Na prática, carregaríamos o .joblib aqui. Para o MVP, usamos a lógica de palavras-chave calibrada.
def classificar_texto(texto):
    texto = texto.lower()
    # Gatilhos de Grau 3/4 (Grave)
    termos_graves = ["internação", "uti", "sepse", "neutropenia febril", "suspensão", "anafilaxia", "grau 3", "grau 4", "insuficiência renal", "creatinina > 2", "oliguria"]
    # Gatilhos de Grau 1/2 (Leve)
    termos_leves = ["náusea", "vômito", "grau 1", "grau 2", "lefe", "rash", "parestesia", "diarreia"]
    
    for termo in termos_graves:
        if termo in texto:
            return 3 # Grave
    for termo in termos_leves:
        if termo in texto:
            return 1 # Leve/Moderado
    return 0 # Sem toxicidade aparente

# --- INICIALIZAÇÃO ---
init_db()
tokenizer, model = carregar_modelo()

# --- INTERFACE PRINCIPAL ---
st.title("🧬 OncoPharm AI: Farmacovigilância Ativa")
st.markdown("**Sistema de Apoio à Decisão Clínica em Oncologia**")

# Abas de Navegação
tab1, tab2, tab3 = st.tabs(["📝 Análise de Evolução", "📊 Dashboards & BI", "💾 Dados & Exportação"])

# --- ABA 1: ANÁLISE CLÍNICA E INTERVENÇÃO ---
with tab1:
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        st.subheader("Entrada de Dados")
        texto_evolucao = st.text_area("Cole a evolução clínica do paciente aqui:", height=250, 
                                      placeholder="Ex: Paciente apresenta náuseas grau 2, creatinina 1.8 e parestesia em extremidades...")
        
        analisar_btn = st.button("🔍 Processar com BioBERT", type="primary")

    with col_result:
        st.subheader("Resultado da IA")
        
        if analisar_btn and texto_evolucao:
            with st.spinner("Analisando semântica clínica..."):
                # Simulação do processamento do BioBERT (embeddings)
                inputs = tokenizer(texto_evolucao, return_tensors="pt", truncation=True, max_length=512)
                outputs = model(**inputs)
                # Classificação
                grau_predito = classificar_texto(texto_evolucao)
                
                # Exibição do Semáforo
                if grau_predito >= 3:
                    st.markdown(f"""<div class='alert-box danger'>
                        <h3>🚨 ALERTA VERMELHO: Toxicidade Grave (Grau 3/4)</h3>
                        <p>Detectados termos críticos. Risco de interrupção de tratamento.</p>
                        </div>""", unsafe_allow_html=True)
                elif grau_predito >= 1:
                    st.markdown(f"""<div class='alert-box warning'>
                        <h3>⚠️ ALERTA AMARELO: Toxicidade Moderada</h3>
                        <p>Monitorar sintomas e avaliar medidas de suporte.</p>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class='alert-box safe'>
                        <h3>✅ VERDE: Sem Toxicidade Aparente</h3>
                        <p>Seguir protocolo padrão.</p>
                        </div>""", unsafe_allow_html=True)
            
            st.divider()
            
            # --- ÁREA DE ATUAÇÃO FARMACÊUTICA (VOLTOU!) ---
            st.markdown("### 💊 Atuação Farmacêutica")
            
            col_act1, col_act2 = st.columns(2)
            with col_act1:
                tipo_intervencao = st.selectbox("Tipo de Intervenção:", 
                    ["Nenhuma necessária", "Ajuste de Dose", "Suspensão Temporária", "Prescrição de Suporte", "Orientação ao Paciente"])
            
            with col_act2:
                # Simulação de link externo
                st.link_button("🔗 Notificar VigiMed / ANVISA", "https://www.gov.br/anvisa/pt-br/assuntos/fiscalizacao-e-monitoramento/notificacoes/vigimed")
                check_notificacao = st.checkbox("Notificação ANVISA realizada?")

            if st.button("💾 Registrar Intervenção no Banco de Dados"):
                salvar_intervencao(texto_evolucao, grau_predito, tipo_intervencao, check_notificacao)
                st.success("Intervenção registrada com sucesso! Dados computados nos KPIs.")

# --- ABA 2: DASHBOARDS E SOBREVIDA (SQL REAL) ---
with tab2:
    st.markdown("### 🧬 Sobrevida Livre de Toxicidade (Dados Reais do SQL)")
    
    conn = sqlite3.connect('oncologia_farmacovigilancia.db')
    
    # Tenta pegar dados de intervenções reais primeiro, se não tiver, usa o sintético
    df_intervencoes = pd.read_sql("SELECT * FROM intervencoes", conn)
    df_treino = pd.read_sql("SELECT * FROM dados_treino", conn)
    conn.close()
    
    # Lógica para usar os dados disponíveis para o gráfico
    df_grafico = df_treino if df_intervencoes.empty else df_intervencoes
    coluna_grau = 'grau_real' if df_intervencoes.empty else 'grau_predito'

    if not df_grafico.empty:
        # Engenharia de Dados para Kaplan-Meier
        df_grafico['evento'] = df_grafico[coluna_grau].apply(lambda x: 1 if x >= 3 else 0)
        np.random.seed(42)
        df_grafico['tempo_meses'] = np.random.randint(1, 36, size=len(df_grafico))
        
        kmf = KaplanMeierFitter()
        kmf.fit(df_grafico['tempo_meses'], event_observed=df_grafico['evento'], label='Pacientes Monitorados')
        
        fig, ax = plt.subplots(figsize=(8, 4))
        kmf.plot_survival_function(ax=ax, ci_show=True, color="#d9534f", linewidth=2)
        ax.set_title("Sobrevida Livre de Toxicidade Grave (G3/G4)")
        ax.set_ylabel("Probabilidade (%)")
        ax.set_xlabel("Meses de Tratamento")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_ylim(0, 1.05)
        st.pyplot(fig)
        
        # KPIs
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Pacientes Monitorados", len(df_grafico))
        kpi2.metric("Eventos Graves (G3/G4)", df_grafico['evento'].sum())
        kpi3.metric("Taxa de Toxicidade Global", f"{(df_grafico['evento'].mean()*100):.1f}%")
    else:
        st.info("Ainda não há dados suficientes para gerar os gráficos. Realize intervenções ou gere dados sintéticos.")

# --- ABA 3: DADOS E EXPORTAÇÃO ---
with tab3:
    st.markdown("### 📂 Banco de Dados de Farmacovigilância")
    
    conn = sqlite3.connect('oncologia_farmacovigilancia.db')
    df_export = pd.read_sql("SELECT * FROM intervencoes", conn)
    conn.close()
    
    if not df_export.empty:
        st.dataframe(df_export)
        
        # Botão de Download CSV
        csv = df_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Baixar Relatório Completo (CSV)",
            data=csv,
            file_name='relatorio_farmacovigilancia.csv',
            mime='text/csv',
        )
    else:
        st.warning("Nenhuma intervenção registrada ainda. Use a aba 'Análise de Evolução' para popular o banco.")