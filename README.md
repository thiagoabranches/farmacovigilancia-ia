\# 🛡️ Sistema de Farmacovigilância Ativa em Oncologia (IA + RWD)



Este projeto é uma Solução de Saúde Digital desenvolvida para automatizar a detecção de Reações Adversas a Medicamentos (RAMs) em prontuários oncológicos não estruturados.



\## 🎯 O Problema

Apenas 12% das RAMs são notificadas no Brasil. A maior parte da informação valiosa está "escondida" em textos livres (evoluções médicas), dificultando a ação proativa do farmacêutico clínico.



\## 💡 A Solução

Um pipeline de Inteligência Artificial que:

1\. \*\*Lê\*\* evoluções médicas usando \*\*BioBERT\*\* (Processamento de Linguagem Natural treinado em textos clínicos em PT-BR).

2\. \*\*Classifica\*\* a gravidade da reação (Grau 0 a 4 do CTCAE) usando \*\*Random Forest\*\*.

3\. \*\*Gera Alertas\*\* em tempo real via Dashboard.

4\. \*\*Analisa Sobrevida\*\* (Kaplan-Meier) livre de toxicidade.



\## 🛠️ Tecnologias Utilizadas

\* \*\*Linguagem:\*\* Python 3.11

\* \*\*Banco de Dados:\*\* SQL (SQLite)

\* \*\*IA/NLP:\*\* Transformers (Hugging Face), BioBERTpt-clin, Scikit-Learn

\* \*\*Visualização:\*\* Streamlit, Lifelines, Matplotlib

\* \*\*Versionamento:\*\* Git \& Git Bash



\## 🚀 Como Executar

1\. Clone o repositório.

2\. Instale as dependências: `pip install -r requirements.txt`

3\. Gere dados sintéticos (opcional): `python gerar\_sinteticos.py`

4\. Treine o modelo: `python treinar\_modelo.py`

5\. Inicie o dashboard: `streamlit run app.py`



\## 📊 Resultados Preliminares

O modelo demonstrou alta capacidade de generalização, identificando corretamente:

\* ✅ Abreviações médicas (`PTX`, `AC-T`, `MMII`)

\* ✅ Contexto de gravidade (`Internação`, `Suspensão de dose`)

\* ✅ Negação de sintomas (`Assintomático`, `Nega queixas`)



---

\*\*Desenvolvedor:\*\* Thiago Abranches | Farmacêutico Clínico em Oncologia

