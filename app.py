import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModel
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="OncoPharmacovigilance AI", layout="wide")

# --- CARREGAMENTO DO CÉREBRO (MODELO) ---
@st.cache_resource
def carregar_modelo():
    model_path = "models/classificador_ram_v1.pkl"
    bert_path = "pucpr/biobertpt-clin"
    
    try:
        clf = joblib.load(model_path)
        tokenizer = AutoTokenizer.from_pretrained(bert_path)
        bert_model = AutoModel.from_pretrained(bert_path)
        return clf, tokenizer, bert_model
    except Exception as e:
        st.error(f"Erro ao carregar modelos: {e}")
        return None, None, None

clf, tokenizer, bert_model = carregar_modelo()

def classificar_texto(texto):
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    vetor = outputs.last_hidden_state[:, 0, :].numpy()[0]
    return clf.predict(vetor.reshape(1, -1))[0]

# --- INTERFACE VISUAL ---
st.title("🛡️ Sistema de Farmacovigilância Ativa em Oncologia")
st.markdown("---")

# --- KPI DASHBOARD (PAINEL DE GESTÃO) ---
# Aqui simulamos métricas estratégicas para a Farmácia Clínica
st.markdown("### 📈 Indicadores de Desempenho (KPIs)")

# Layout de 4 colunas para os números ficarem lado a lado
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        label="RAMs Graves Detectadas", 
        value="127", 
        delta="+14% este mês",
        help="Total de eventos Grau 3 ou 4 identificados pela IA nos últimos 30 dias."
    )

with kpi2:
    st.metric(
        label="Tempo Médio de Intervenção", 
        value="45 min", 
        delta="-30% (Meta Atingida)",
        delta_color="normal",
        help="Tempo entre a prescrição e o alerta farmacêutico."
    )

with kpi3:
    st.metric(
        label="Acurácia do Modelo", 
        value="92%", 
        delta="Estável",
        help="Confiabilidade da IA em distinguir casos Graves de Leves."
    )

with kpi4:
    st.metric(
        label="Custo Evitado (Estimado)", 
        value="R$ 42.000", 
        delta="Internações Prevenidas",
        delta_color="inverse", # Fica verde se o número for positivo
        help="Cálculo baseado no custo médio de internação por toxicidade x RAMs graves interceptadas."
    )

st.markdown("---") # Uma linha divisória para separar os KPIs da ferramenta de texto

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Análise de Prontuário Individual")
    texto_input = st.text_area("Cole a evolução clínica aqui:", height=150, 
                              placeholder="Ex: Paciente relata diarreia grau 3...")
    
    if st.button("Analisar com IA"):
        if texto_input:
            with st.spinner("O BioBERT está lendo o prontuário..."):
                grau = classificar_texto(texto_input)
            
            # Lógica do "Semáforo"
            if grau >= 3:
                st.error(f"🚨 ALERTA VERMELHO: Reação Grave Detectada (Grau {grau})")
                st.info("Recomendação: Notificação imediata e revisão de protocolo.")
            elif grau > 0:
                st.warning(f"⚠️ ALERTA AMARELO: Reação Moderada (Grau {grau})")
                st.info("Recomendação: Monitoramento sintomático.")
            else:
                st.success("✅ Sem RAM detectada.")
        else:
            st.warning("Por favor, insira um texto.")

with col2:
    st.subheader("📊 Indicadores do Setor")
    st.write("Dados simulados de toxicidade do serviço")
    
    # --- O GRÁFICO KAPLAN-MEIER ---
    # Simulando dados: "Tempo até a primeira RAM Grave"
    # T = Tempo em meses, E = Evento (1=Teve RAM Grave, 0=Censurado/Sem RAM)
    np.random.seed(42)
    T = np.random.exponential(8, size=100) # Média de 8 meses
    E = np.random.binomial(1, 0.6, size=100) # 60% tiveram evento
    
    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=E, label='Sobrevida Livre de RAM Grave')
    
    fig, ax = plt.subplots()
    kmf.plot_survival_function(ax=ax, ci_show=True, color="#d9534f")
    ax.set_title("Tempo até Toxicidade Limitante (Kaplan-Meier)")
    ax.set_xlabel("Meses de Tratamento")
    ax.set_ylabel("Probabilidade Livre de Evento")
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    st.metric(label="Mediana Livre de Toxicidade", value=f"{kmf.median_survival_time_:.1f} Meses")

# --- RODAPÉ ---
st.markdown("---")
st.caption("Desenvolvido por Thiago Abranches | BioBERT + Random Forest | v2.0")
