import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import sqlite3
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
    
    # CRIANDO AS ABAS (Botoes de Navegação)
    tab_graficos, tab_tabelas = st.tabs(["📈 Gráficos Visuais", "📋 Dados Tabulares"])
    
    # --- SIMULAÇÃO DE DADOS PARA O DASHBOARD ---
    # Criamos um DataFrame fictício para alimentar os gráficos e a tabela
    np.random.seed(42)
    dados_dashboard = pd.DataFrame({
        'ID Paciente': [f'#{x}' for x in range(1040, 1090)],
        'Protocolo': np.random.choice(['AC-T', 'FOLFOX', 'FOLFIRI', 'Carbo/Taxol'], 50),
        'RAM Detectada': np.random.choice(['Neutropenia', 'Diarreia', 'Neuropatia', 'Rash', 'Náusea'], 50),
        'Grau CTCAE': np.random.choice([1, 2, 3, 4], 50, p=[0.4, 0.3, 0.2, 0.1]),
        'Status': np.random.choice(['Resolvido', 'Em Monitoramento', 'Intervenção Farmacêutica'], 50)
    })

    # --- ABA 1: VISÃO GRÁFICA (CONECTADA AO SQL) ---
    with tab_graficos:
        st.markdown("### 🧬 Sobrevida Livre de Toxicidade (Dados Reais do SQL)")
        
        # 1. Conexão com o Banco de Dados Real
        conn = sqlite3.connect('oncologia_farmacovigilancia.db')
        
        # Puxamos apenas o 'grau_real' da tabela de treino/histórico
        df_sql = pd.read_sql("SELECT grau_real FROM dados_treino", conn)
        conn.close()
        
        if not df_sql.empty:
            # 2. Engenharia de Dados para o Kaplan-Meier
            # Definição de Evento: Grau 3 ou 4 (Toxicidade Limitante)
            # Se grau >= 3, evento = 1. Se grau < 3, evento = 0 (Censurado)
            df_sql['evento'] = df_sql['grau_real'].apply(lambda x: 1 if x >= 3 else 0)
            
            # Simulação do Tempo (Eixo X)
            # Como nosso gerador sintético não criou datas, atribuímos tempos aleatórios (1 a 36 meses)
            # Num cenário real hospitalar, faríamos: Data_Evento - Data_Inicio_Tratamento
            np.random.seed(42) 
            df_sql['tempo_meses'] = np.random.randint(1, 36, size=len(df_sql))
            
            # 3. Plotagem
            kmf = KaplanMeierFitter()
            kmf.fit(df_sql['tempo_meses'], event_observed=df_sql['evento'], label='Protocolos da Instituição')
            
            fig, ax = plt.subplots(figsize=(8, 5))
            kmf.plot_survival_function(ax=ax, ci_show=True, color="#d9534f", linewidth=2)
            
            # Formatação Clínica
            ax.set_title(f"Análise de Sobrevida (N = {len(df_sql)} Pacientes)", fontsize=12)
            ax.set_xlabel("Meses de Tratamento", fontsize=10)
            ax.set_ylabel("Probabilidade de Permanecer sem Toxicidade Grave", fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_ylim(0, 1.05) # Eixo Y de 0 a 100%
            
            # Adiciona linha de corte de 50% (Mediana)
            if kmf.median_survival_time_ < float('inf'):
                ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
                ax.text(0, 0.51, f' Mediana: {kmf.median_survival_time_:.1f} meses', color='gray', fontsize=9)

            st.pyplot(fig)
            
            # Métricas abaixo do gráfico
            c1, c2, c3 = st.columns(3)
            c1.metric("Total de Pacientes Analisados", len(df_sql))
            c2.metric("Eventos Graves (G3/G4)", df_sql['evento'].sum())
            c3.metric("Taxa de Toxicidade Global", f"{(df_sql['evento'].mean()*100):.1f}%")
            
        else:
            st.warning("⚠️ O banco de dados está vazio. Gere dados sintéticos primeiro.")

        st.write("---")
        st.write("**Distribuição dos Graus CTCAE no Banco**")
        # Gráfico de barras simples usando os dados do SQL
        st.bar_chart(df_sql['grau_real'].value_counts().sort_index(), color="#2E86C1")

    # --- ABA 2: VISÃO DE TABELA ---
    with tab_tabelas:
        st.write("**Histórico Recente de Alertas**")
        
        # Filtro interativo (Bônus)
        filtro_grau = st.multiselect(
            "Filtrar por Gravidade:", 
            options=[1, 2, 3, 4],
            default=[3, 4] # Já vem marcado os graves por padrão
        )
        
        # Aplica o filtro na tabela
        if filtro_grau:
            df_filtrado = dados_dashboard[dados_dashboard['Grau CTCAE'].isin(filtro_grau)]
        else:
            df_filtrado = dados_dashboard
            
        # Mostra a tabela interativa (dá para ordenar clicando na coluna)
        st.dataframe(
            df_filtrado, 
            hide_index=True,
            column_config={
                "Grau CTCAE": st.column_config.NumberColumn(
                    "Grau",
                    help="Classificação CTCAE v6.0",
                    format="%d ⭐" # Formatação visual bonitinha
                ),
                "Status": st.column_config.SelectboxColumn(
                    "Status Clínico",
                    options=["Resolvido", "Em Monitoramento", "Intervenção Farmacêutica"],
                    required=True
                )
            }
        )
        
        # Botão de Download (Muito útil para gestão)
        csv = df_filtrado.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Baixar Relatório em Excel (CSV)",
            data=csv,
            file_name='relatorio_rams_oncologia.csv',
            mime='text/csv',
        )

# --- RODAPÉ ---
st.markdown("---")
st.caption("Desenvolvido por Thiago Abranches | BioBERT + Random Forest | v3.1")
