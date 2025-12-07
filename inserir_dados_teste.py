import sqlite3
from datetime import datetime

# Lista de 12 prontuários fictícios simulando anotações médicas reais (Desestruturadas)
# Cenários variados para desafiar o BioBERT depois
dados_ficticios = [
    # Caso 1: Ca Mama - AC (Doxorrubicina + Ciclofosfamida) - RAM Grave (Vômito G3)
    ("PT_ONCO_001", "Paciente 45a, Ca Mama EC IIB. Retorna para C3 de AC. Relata que 2 dias após a última infusão apresentou 7 episódios de vômito em 24h, necessitando hidratação venosa. Nega febre.", "2023-10-01"),
    
    # Caso 2: Ca Pulmão - Cisplatina - RAM Grave (Nefrotoxicidade)
    ("PT_ONCO_002", "QP: Diminuição do volume urinário. Em uso de Cisplatina/Pemetrexed. Exames trazem Creatinina 2.1 (basal 0.9). Suspenso ciclo atual. Encaminhado para nefrologia.", "2023-10-02"),
    
    # Caso 3: Ca Colorretal - FOLFOX - RAM Moderada (Neuropatia Periférica)
    ("PT_ONCO_003", "Paciente em D1 de C6 esquema FOLFOX. Queixa de parestesia em extremidades (mãos e pés) desencadeada pelo frio, dificultando abotoar camisas. Mantém atividades diárias. Oxialiplatina reduzida em 20%.", "2023-10-03"),
    
    # Caso 4: Ca Mama - Paclitaxel - RAM Leve (Rash Cutâneo)
    ("PT_ONCO_004", "Seguimento semanal de Paclitaxel. Paciente refere surgimento de erupção cutânea leve em face e tronco, sem prurido. Sem sinais de infecção. Orientado uso de hidratante.", "2023-10-04"),
    
    # Caso 5: Linfoma - R-CHOP - RAM Grave (Neutropenia Febril)
    ("PT_ONCO_005", "Admitido no PS com febre de 38.5ºC. D10 pós R-CHOP. Hemograma: Neutrófilos 350/mm3. Iniciado protocolo de neutropenia febril com Cefepima. Estado geral regular.", "2023-10-05"),
    
    # Caso 6: Ca Cabeça e Pescoço - Cetuximabe - RAM Esperada (Rash Acneiforme)
    ("PT_ONCO_006", "Retorno ambulatorial. Apresenta rash acneiforme em face e região superior do tórax, grau 2. Incômodo estético, mas sem infecção secundária. Prescrito Doxiciclina profilática.", "2023-10-06"),
    
    # Caso 7: Ca Mama HER2+ - Trastuzumabe - RAM Assintomática (Queda FEVE)
    ("PT_ONCO_007", "Assintomática. Ecocardiograma de controle mostra Fração de Ejeção do Ventrículo Esquerdo (FEVE) de 45% (Queda > 10% do basal). Suspensão temporária do Trastuzumabe conforme protocolo.", "2023-10-07"),
    
    # Caso 8: Ca Gástrico - 5-FU - RAM Moderada (Mucosite)
    ("PT_ONCO_008", "Paciente refere dor ao engolir e feridas na boca. Ao exame: eritema e úlceras em mucosa jugal, consegue ingerir dieta pastosa. Mucosite Grau 2. Prescrito laserterapia.", "2023-10-08"),
    
    # Caso 9: Imunoterapia (Melanoma) - Ipilimumabe - RAM Autoimune (Colite)
    ("PT_ONCO_009", "Paciente relata diarreia líquida (5 episódios/dia) e dor abdominal cólica. Suspeita de colite imuno-mediada. Iniciado corticoide oral 1mg/kg. Aguarda colono.", "2023-10-09"),
    
    # Caso 10: Sem RAM (Controle)
    ("PT_ONCO_010", "Paciente retorna para seguimento. Nega queixas álgicas, nega náuseas ou vômitos. Exames laboratoriais dentro da normalidade. Liberado para próximo ciclo.", "2023-10-10"),
    
    # Caso 11: Ca Ovário - Carboplatina - Reação Infusional
    ("PT_ONCO_011", "Durante infusão de Carboplatina, paciente apresentou flushing facial, dispneia leve e dor lombar. Infusão interrompida, administrado hidrocortisona e anti-histamínico. Melhora total.", "2023-10-11"),
    
    # Caso 12: Ca Mama - Tamoxifeno - Efeito Leve
    ("PT_ONCO_012", "Em uso de hormonoterapia adjuvante. Queixa principal: fogachos noturnos que atrapalham o sono esporadicamente. Sem sangramento vaginal. Mantida conduta.", "2023-10-12")
]

conn = sqlite3.connect('oncologia_farmacovigilancia.db')
cursor = conn.cursor()

# Comando SQL de inserção
sql_insert = "INSERT INTO prontuarios (paciente_hash, texto_clinico, data_importacao) VALUES (?, ?, ?)"

print(f"Inserindo {len(dados_ficticios)} registros de teste...")

try:
    cursor.executemany(sql_insert, dados_ficticios)
    conn.commit()
    print("✅ Sucesso! 12 prontuários foram inseridos no banco de dados.")
except sqlite3.Error as e:
    print(f"❌ Erro ao inserir dados: {e}")
finally:
    conn.close()