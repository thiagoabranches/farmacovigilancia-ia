import sqlite3

# Conecta ao banco (se não existir, ele cria automaticamente)
conexao = sqlite3.connect('oncologia_farmacovigilancia.db')
cursor = conexao.cursor()

# --- TABELA 1: Prontuários (Dados Brutos) ---
# Fonte primária conforme o PDF [cite: 6, 15]
sql_prontuarios = """
CREATE TABLE IF NOT EXISTS prontuarios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_hash VARCHAR(100),  -- Anonimizado
    texto_clinico TEXT NOT NULL, -- O texto não estruturado para o BioBERT
    data_importacao DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

# --- TABELA 2: Reações Adversas (Dados Processados pela IA) ---
# Campos baseados na metodologia do projeto [cite: 19, 20, 23]
sql_rams = """
CREATE TABLE IF NOT EXISTS alertas_ram (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prontuario_id INTEGER,
    medicamento VARCHAR(100),       -- Extraído (WhoDRUG)
    reacao_adversa VARCHAR(100),    -- Extraído (MedDRA)
    gravidade_ctcae INTEGER,        -- Classificado (1-5) via Random Forest
    confianca_ia FLOAT,             -- Probabilidade do modelo
    validado_farmaceutico BOOLEAN DEFAULT 0, -- Para validação humana posterior
    FOREIGN KEY(prontuario_id) REFERENCES prontuarios(id)
);
"""

# Executando os comandos
print("Criando tabelas...")
cursor.execute(sql_prontuarios)
cursor.execute(sql_rams)

# Salvando as alterações
conexao.commit()
conexao.close()
print("Banco de dados 'oncologia_farmacovigilancia.db' criado com sucesso!")