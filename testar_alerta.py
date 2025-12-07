import joblib
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# --- CONFIGURAÇÃO ---
CAMINHO_MODELO = "models/classificador_ram_v1.pkl"
NOME_BERT = "pucpr/biobertpt-clin"

print(">>> Inicializando sistema de Alerta...")

# 1. Carrega o 'Cérebro' treinado
try:
    clf = joblib.load(CAMINHO_MODELO)
    print("   [OK] Modelo Random Forest carregado.")
except:
    print("   [ERRO] Não encontrei o arquivo em 'models/'. Rode o treino primeiro.")
    exit()

# 2. Carrega o BioBERT (apenas para traduzir o texto, não precisa treinar)
print("   [OK] Carregando BioBERT (pode levar alguns segundos)...")
tokenizer = AutoTokenizer.from_pretrained(NOME_BERT)
model = AutoModel.from_pretrained(NOME_BERT)

def classificar_novo_caso(texto_medico):
    print(f"\nANÁLISE DE NOVO CASO:\n'{texto_medico}'")
    
    # 1. Transforma texto em números (Vetorização)
    inputs = tokenizer(texto_medico, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    vetor = outputs.last_hidden_state[:, 0, :].numpy()[0]
    
    # 2. A IA faz a previsão
    # O reshape(1, -1) é necessário porque é um caso só
    previsao = clf.predict(vetor.reshape(1, -1))
    
    # 3. Resultado
    grau = previsao[0]
    
    if grau >= 3:
        print(f"🚨 ALERTA VERMELHO: Reação Grave Detectada (Grau {grau})")
        print("-> Ação Sugerida: Notificar médico prescritor imediatamente.")
    elif grau > 0:
        print(f"⚠️ ALERTA AMARELO: Reação Leve/Moderada (Grau {grau})")
        print("-> Ação Sugerida: Monitorar sintomas no próximo ciclo.")
    else:
        print("✅ NENHUMA reação adversa grave detectada.")

# --- SIMULAÇÃO ---
# Caso fictício novo (não estava no banco de dados)
caso_novo = "Paciente em imunoterapia relata aumento do numero de evacuacoes (7x ao dia) e dor abdominal intensa. Necessitou internacao para hidratacao."

classificar_novo_caso(caso_novo)