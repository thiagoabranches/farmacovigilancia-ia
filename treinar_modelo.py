import sqlite3
import pandas as pd
import numpy as np
import torch
import joblib 
from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- CONFIGURAÇÕES ---
NOME_MODELO = "pucpr/biobertpt-clin"
CAMINHO_SALVAR = "models/classificador_ram_v1.pkl"

print(">>> Inicializando BioBERT...")
tokenizer = AutoTokenizer.from_pretrained(NOME_MODELO)
model = AutoModel.from_pretrained(NOME_MODELO)

# --- GABARITO (SIMULAÇÃO) ---
# Como nossos dados no banco ainda não têm a classificação "oficial", 
# criamos esse dicionário simulando que um humano classificou os 12 casos.
# ID do Prontuário : Grau da RAM (0 a 4)
gabarito_reais = {
    1: 3, # Vômito G3
    2: 4, # Nefrotoxicidade G4
    3: 2, # Neuropatia G2
    4: 1, # Rash G1
    5: 3, # Neutropenia G3
    6: 2, # Rash G2
    7: 2, # Cardiotoxicidade G2
    8: 2, # Mucosite G2
    9: 3, # Colite G3
    10: 0, # Sem RAM
    11: 2, # Reação Infusional G2
    12: 1  # Fogachos G1
}

def gerar_embedding(texto):
    """Transforma texto em vetor numérico"""
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()[0]

def treinar():
    # 1. Busca dados no SQL
    print(">>> Buscando dados no SQL...")
    conn = sqlite3.connect('oncologia_farmacovigilancia.db')
    df = pd.read_sql("SELECT id, texto_clinico FROM prontuarios", conn)
    conn.close()

    print(f">>> Processando {len(df)} prontuários (Transformando texto em números)...")
    
    X = [] # Dados (Vetores)
    y = [] # Respostas (Gabarito)

    for index, row in df.iterrows():
        # Gera o vetor do texto
        vetor = gerar_embedding(row['texto_clinico'])
        X.append(vetor)
        
        # Pega a resposta correta no gabarito
        gravidade = gabarito_reais.get(row['id'], 0)
        y.append(gravidade)

    # Converte para formato numérico do Numpy
    X = np.array(X)
    y = np.array(y)

    # 2. Treina o Modelo
    print(">>> Treinando o Random Forest...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    # 3. Avalia
    print("\n--- RESULTADOS PRELIMINARES ---")
    preds = clf.predict(X)
    acc = accuracy_score(y, preds)
    print(f"Acurácia no treino: {acc * 100:.1f}%")
    print("(Nota: Acurácia alta é esperada pois estamos testando com os mesmos dados de treino)")

    # 4. Salva o modelo treinado
    joblib.dump(clf, CAMINHO_SALVAR)
    print(f"\n>>> SUCESSO! Modelo salvo em: {CAMINHO_SALVAR}")
    print("Agora seu projeto tem um 'cérebro' capaz de classificar novos casos.")

if __name__ == "__main__":
    treinar()