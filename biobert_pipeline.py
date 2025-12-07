import sqlite3
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel

# --- CONFIGURAÇÃO DO MODELO ---
# Usando BioBERTpt (Clinical) - Especialista em termos médicos em PT
NOME_MODELO = "pucpr/biobertpt-clin"

print(f"--- INICIANDO SISTEMA DE IA ---")
print(f"Carregando o modelo {NOME_MODELO}...")
print("(A primeira vez demora alguns minutos pois fará o download de ~400MB)")

# Carrega o tokenizador e o modelo
tokenizer = AutoTokenizer.from_pretrained(NOME_MODELO)
model = AutoModel.from_pretrained(NOME_MODELO)

# --- FUNÇÕES ---

def obter_prontuarios():
    """Conecta no SQLite e puxa os textos"""
    conn = sqlite3.connect('oncologia_farmacovigilancia.db')
    query = "SELECT id, texto_clinico FROM prontuarios"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def processar_texto_biobert(texto):
    """Gera a representação vetorial (embedding) do texto"""
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()[0]

# --- FLUXO PRINCIPAL ---

def executar():
    try:
        df = obter_prontuarios()
    except Exception as e:
        print(f"Erro ao ler banco de dados: {e}")
        return

    print(f"\n>>> Sucesso! Encontrados {len(df)} prontuários para análise.\n")
    
    for i, row in df.iterrows():
        id_pct = row['id']
        texto = row['texto_clinico']
        
        print(f"Processando Prontuário #{id_pct}...")
        
        # 1. O BioBERT lê o texto
        vetor = processar_texto_biobert(texto)
        
        # 2. Mostra o que ele 'entendeu' (tokenização)
        tokens = tokenizer.tokenize(texto)
        # Filtra termos que começam com '##' (sufixos) ou termos chaves
        termos_chave = [t for t in tokens if t.startswith('##') or 'cisp' in t or 'rubi' in t]
        
        print(f"   -> Vetor gerado (tamanho): {len(vetor)}")
        print(f"   -> Tokens identificados: {termos_chave[:5]}...") 
        print("-" * 50)

if __name__ == "__main__":
    executar()