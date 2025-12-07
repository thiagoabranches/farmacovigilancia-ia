from transformers import AutoTokenizer, AutoModel
import sys

# Nome do modelo
NOME_MODELO = "pucpr/biobertpt-clin"

print(f"--- INICIANDO DIAGNÓSTICO DE DOWNLOAD ---")
print(f"Modelo alvo: {NOME_MODELO}")
print("1. Tentando baixar o Tokenizer (Leve)...")

try:
    tokenizer = AutoTokenizer.from_pretrained(NOME_MODELO)
    print("   [OK] Tokenizer baixado com sucesso!")
except Exception as e:
    print(f"   [ERRO] Falha no Tokenizer: {e}")
    sys.exit()

print("\n2. Tentando baixar o Modelo (Pesado - aprox 440MB)...")
print("   Aguarde... se demorar mais que 1 min sem mensagem, sua conexão pode estar instável.")

try:
    # O force_download ajuda se o arquivo anterior ficou corrompido quando travou
    model = AutoModel.from_pretrained(NOME_MODELO, force_download=True)
    print("   [OK] Modelo baixado e carregado na memória!")
except Exception as e:
    print(f"   [ERRO] Falha no Modelo: {e}")
    sys.exit()

print("\n--- SUCESSO TOTAL ---")
print("O modelo está salvo no seu computador. Agora o outro script vai rodar instantaneamente.")