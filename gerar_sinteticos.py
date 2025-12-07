import sqlite3
import random

# --- CONFIGURAÇÃO ---
QTD_CASOS = 300  # Aumentei para 300 para ter mais exemplos
ARQUIVO_DB = 'oncologia_farmacovigilancia.db'

# --- TEMPLATES (O Segredo da Calibragem) ---

# GRAVES (Grau 3 e 4)
templates_graves = [
    "Paciente apresentou {sintoma} severa, necessitando de {acao}. Suspensao do protocolo.",
    "Admitido na emergencia com {sintoma}. Realizado {acao}. Quadro grave.",
    "Relato de {sintoma} grau 3 apos infusao. Indicada {acao} imediata.",
    "Toxicidade limitante de dose: {sintoma} incontrolavel. Paciente encaminhado para {acao}.",
    "Choque anafilatico e {sintoma} durante a medicacao. Feito {acao}.",
    "Paciente internado devido a {sintoma} com desidratacao grave."
]
sintomas_graves = ["diarreia liquida (>7x/dia)", "neutropenia febril", "reacao anafilatica", "insuficiencia renal aguda", "sangramento digestivo", "dispneia em repouso"]
acoes_graves = ["internacao hospitalar", "hidratacao venosa intensiva", "transferencia para UTI", "transfusao de sangue"]

# LEVES (Grau 1 e 2) - Foco em sintomas que NÃO impedem a vida diária
templates_leves = [
    "Paciente refere {sintoma} leve. Orientado uso de {med_suporte}.",
    "Queixa de {sintoma} grau 1. Mantida conduta e prescrito {med_suporte}.",
    "Exames mostram {sintoma} discreta. Sem repercussao clinica. Segue tratamento.",
    "Apresentou {sintoma} moderada, controlada com {med_suporte} oral.",
    "Leve desconforto: {sintoma}, sem necessidade de hospitalizacao."
]
sintomas_leves = ["nauseas esporadicas", "rash cutaneo", "formigamento nas maos", "mucosite oral leve", "fadiga", "tontura leve"]
meds_suporte = ["analgesico", "hidratante", "ondansetrona", "loperamida se necessario", "bochecho com nistatina"]

# NORMAIS (Grau 0) - Aqui vamos reforçar o "Sem Queixas"
templates_normais = [
    "Paciente assintomatico. Exames normais. Liberado para quimio.",
    "Nega queixas. Boa tolerancia ao tratamento.",
    "Retorno de seguimento, sem intercorrencias clinicas.",
    "Avaliacao pre-quimio: tudo dentro da normalidade.",
    "Paciente em otimo estado geral (ECOG 0). Nega toxicidades.",
    "Sem relato de reacoes adversas no ultimo ciclo.",
    "Exames laboratoriais sem alteracoes significativas. Segue conduta.",
    "Paciente retorna para seguimento assintomatico.",
    "Nega nauseas, nega vomitos, nega febre. Apto para infusao."
]

def gerar_dataset():
    conn = sqlite3.connect(ARQUIVO_DB)
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM dados_treino")
    
    dados = []
    print(f"Gerando {QTD_CASOS} novos prontuários calibrados...")
    
    for _ in range(QTD_CASOS):
        dado_random = random.random()
        
        # 33% Graves
        if dado_random < 0.33:
            txt = random.choice(templates_graves).format(
                sintoma=random.choice(sintomas_graves),
                acao=random.choice(acoes_graves)
            )
            grau = random.choice([3, 4])
            
        # 33% Leves
        elif dado_random < 0.66:
            txt = random.choice(templates_leves).format(
                sintoma=random.choice(sintomas_leves),
                med_suporte=random.choice(meds_suporte)
            )
            grau = random.choice([1, 2])
            
        # 33% Normais (Garante que ele veja bastante caso "limpo")
        else:
            txt = random.choice(templates_normais)
            grau = 0 # ZERO absoluto
            
        dados.append((txt, grau))
    
    cursor.executemany("INSERT INTO dados_treino (texto, grau_real) VALUES (?, ?)", dados)
    conn.commit()
    conn.close()
    print(">>> Sucesso! Base de dados recalibrada.")

if __name__ == "__main__":
    gerar_dataset()