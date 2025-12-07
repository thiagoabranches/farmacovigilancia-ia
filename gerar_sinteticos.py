import sqlite3
import random

# --- CONFIGURAÇÃO ---
QTD_CASOS = 200  # Vamos gerar 200 casos para a IA ficar esperta
ARQUIVO_DB = 'oncologia_farmacovigilancia.db'

# --- TEMPLATES DE TEXTO (O Segredo do Data Augmentation) ---

# Casos GRAVES (Grau 3 e 4) - O que a IA precisa aprender a temer
templates_graves = [
    "Paciente apresentou {sintoma} severa, necessitando de {acao}. Suspensao do protocolo.",
    "Admitido na emergencia com {sintoma}. Realizado {acao}. Quadro grave.",
    "Relato de {sintoma} grau 3 apos infusao. Indicada {acao} imediata.",
    "Toxicidade limitante de dose: {sintoma} incontrolavel. Paciente encaminhado para {acao}.",
    "Choque anafilatico e {sintoma} durante a medicacao. Feito {acao}."
]
sintomas_graves = ["diarreia liquida (>7x/dia)", "neutropenia febril", "reacao anafilatica", "insuficiencia renal aguda", "sangramento digestivo"]
acoes_graves = ["internacao hospitalar", "hidratacao venosa intensiva", "transferencia para UTI", "transfusao de sangue"]

# Casos LEVES/MODERADOS (Grau 1 e 2)
templates_leves = [
    "Paciente refere {sintoma} leve. Orientado uso de {med_suporte}.",
    "Queixa de {sintoma} grau 1. Mantida conduta e prescrito {med_suporte}.",
    "Exames mostram {sintoma} discreta. Sem repercussao clinica. Segue tratamento.",
    "Apresentou {sintoma} moderada, controlada com {med_suporte} oral."
]
sintomas_leves = ["nauseas esporadicas", "rash cutaneo", "formigamento nas maos", "mucosite oral", "fadiga"]
meds_suporte = ["analgesico", "hidratante", "ondansetrona", "loperamida se necessario"]

# Casos SEM RAM (Grau 0)
templates_normais = [
    "Paciente assintomatico. Exames normais. Liberado para quimio.",
    "Nega queixas. Boa tolerancia ao tratamento.",
    "Retorno de seguimento, sem intercorrencias.",
    "Avaliacao pre-quimio: tudo dentro da normalidade."
]

def gerar_dataset():
    conn = sqlite3.connect(ARQUIVO_DB)
    cursor = conn.cursor()
    
    # 1. Cria uma tabela ESPECÍFICA para treino (Texto + Resposta Correta)
    # Diferente da tabela 'prontuarios' que é dado bruto, essa aqui já tem o gabarito.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dados_treino (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            texto TEXT,
            grau_real INTEGER
        )
    """)
    
    # Limpa dados antigos para não duplicar se rodar 2x
    cursor.execute("DELETE FROM dados_treino")
    
    dados = []
    
    print(f"Gerando {QTD_CASOS} prontuários sintéticos...")
    
    for _ in range(QTD_CASOS):
        dado_random = random.random()
        
        # 30% de chance de ser GRAVE (Para a IA ver bastante caso ruim)
        if dado_random < 0.3:
            txt = random.choice(templates_graves).format(
                sintoma=random.choice(sintomas_graves),
                acao=random.choice(acoes_graves)
            )
            grau = random.choice([3, 4]) # Grau 3 ou 4
            
        # 40% de chance de ser LEVE
        elif dado_random < 0.7:
            txt = random.choice(templates_leves).format(
                sintoma=random.choice(sintomas_leves),
                med_suporte=random.choice(meds_suporte)
            )
            grau = random.choice([1, 2]) # Grau 1 ou 2
            
        # 30% de chance de ser NORMAL
        else:
            txt = random.choice(templates_normais)
            grau = 0
            
        dados.append((txt, grau))
    
    cursor.executemany("INSERT INTO dados_treino (texto, grau_real) VALUES (?, ?)", dados)
    conn.commit()
    conn.close()
    print(" Sucesso! Base de dados de treino criada.")

if __name__ == "__main__":
    gerar_dataset()