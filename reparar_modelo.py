from transformers import AutoTokenizer, AutoModel
import os

# Nome do modelo que deu erro
NOME_MODELO = "pucpr/biobertpt-clin"

print(f"--- INICIANDO REPARO DO MODELO ---")
print(f"Detectado arquivo corrompido para: {NOME_MODELO}")
print("Baixando novamente os arquivos (Isso pode levar alguns minutos)...")

try:
    # 1. Baixa o Tokenizer novamente
    print("-> Baixando Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(NOME_MODELO, force_download=True)
    
    # 2. Baixa o Modelo novamente (O arquivo pesado)
    print("-> Baixando Modelo (Pesos)...")
    model = AutoModel.from_pretrained(NOME_MODELO, force_download=True)
    
    print("\n✅ SUCESSO! O modelo foi reparado e carregado na memória.")
    print("Agora você pode rodar o 'streamlit run app.py' que vai funcionar.")

except Exception as e:
    print(f"\n❌ Erro durante o reparo: {e}")
    print("Tente verificar sua conexão com a internet.")