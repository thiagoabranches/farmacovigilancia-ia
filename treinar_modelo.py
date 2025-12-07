import sqlite3
import pandas as pd
import numpy as np
import torch
import joblib 
from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# --- CONFIGURAÇÕES ---
NOME_MODELO = "pucpr/biobertpt-clin"
CAMINHO_SALVAR = "models/classificador_ram_v1.pkl"

print(">>> Inicializando BioBERT...")
tokenizer = AutoTokenizer.from_pretrained(NOME_MODELO)
model = AutoModel.from_pretrained(NOME_MODELO)

def gerar_embedding(texto):
    # Diminuí max_length para 128 para ser mais rápido no treino massivo
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()[0]

def treinar():
    # 1. Busca dados na tabela NOVA (dados_treino)
    print(">>> Lendo dados gerados sinteticamente...")
    conn = sqlite3.connect('oncologia_farmacovigilancia.db')
    
    # ATENÇÃO: Aqui pegamos da tabela 'dados_treino', não 'prontuarios'
    df = pd.read_sql("SELECT texto, grau_real FROM dados_treino", conn)
    conn.close()

    if len(df) == 0:
        print("ERRO: Tabela vazia. Rode 'python gerar_sinteticos.py' primeiro.")
        return

    print(f">>> Processando {len(df)} exemplos para ensinar a IA... (Isso vai levar uns 20-30 seg)")
    
    X = []
    y = []

    # Gera os vetores para os 200 casos
    for i, texto in enumerate(df['texto']):
        if i % 50 == 0: print(f"   ... processado {i} de {len(df)}")
        X.append(gerar_embedding(texto))
    
    X = np.array(X)
    y = df['grau_real'].values

    # 2. Separa 20% para prova final
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Treina o Random Forest
    print(">>> Treinando modelo...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # 4. Avalia
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print(f"\n--- RESULTADO APÓS DATA AUGMENTATION ---")
    print(f"Acurácia em dados novos: {acc * 100:.1f}%")
    
    # 5. Salva
    joblib.dump(clf, CAMINHO_SALVAR)
    print(f">>> Modelo RE-TREINADO salvo em: {CAMINHO_SALVAR}")

if __name__ == "__main__":
    treinar()