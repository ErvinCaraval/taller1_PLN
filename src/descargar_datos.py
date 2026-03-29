import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# ==========================================
# CREACIÓN DE DIRECTORIOS (ORDEN DEL PROYECTO)
# ==========================================
# Definir rutas para las carpetas
BASE_DIR = 'data'
ANCORA_DIR = os.path.join(BASE_DIR, 'ancora')
CONLL_DIR = os.path.join(BASE_DIR, 'conll2002')

# Crear las carpetas si no existen
os.makedirs(ANCORA_DIR, exist_ok=True)
os.makedirs(CONLL_DIR, exist_ok=True)

print(f"Carpetas estructuradas correctamente en: ./{BASE_DIR}/")

# ==========================================
# DESCARGA DE DATOS
# ==========================================
print("\nDescargando corpus desde NLTK (si ya están descargados, este paso será rápido)...")
nltk.download('cess_esp', quiet=True)   # Corpus equivalente a Ancora
nltk.download('conll2002', quiet=True)  # Corpus CoNLL2002 en español

# ==========================================
# PROCESAMIENTO DE ANCORA
# ==========================================
print("\nProcesando dataset Ancora...")
sentences_ancora = nltk.corpus.cess_esp.tagged_sents()

data_ancora = []
for i, sent in enumerate(sentences_ancora):
    for word, pos in sent:
        data_ancora.append((f"Sentence: {i+1}", word, pos))

df_ancora = pd.DataFrame(data_ancora, columns=['Sentence #', 'Word', 'POS'])

# Guardar en la carpeta de Ancora
ruta_ancora_completo = os.path.join(ANCORA_DIR, 'ancora_corpus_pos.csv')
df_ancora.to_csv(ruta_ancora_completo, index=False)
print(f"-> Generado: {ruta_ancora_completo}")

# Crear las divisiones
sent_ids = df_ancora['Sentence #'].unique()
train_ids, temp_ids = train_test_split(sent_ids, test_size=0.30, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=42)

# Guardar los splits en la carpeta de Ancora
df_ancora[df_ancora['Sentence #'].isin(train_ids)].to_csv(os.path.join(ANCORA_DIR, 'ancora_train.csv'), index=False)
df_ancora[df_ancora['Sentence #'].isin(val_ids)].to_csv(os.path.join(ANCORA_DIR, 'ancora_val.csv'), index=False)
df_ancora[df_ancora['Sentence #'].isin(test_ids)].to_csv(os.path.join(ANCORA_DIR, 'ancora_test.csv'), index=False)
print(f"-> Generados splits de Ancora en: ./{ANCORA_DIR}/")


# ==========================================
# PROCESAMIENTO DE CONLL2002
# ==========================================
print("\nProcesando dataset CoNLL2002...")

def guardar_conll_texto(nombre_archivo, particion_nltk):
    ruta_salida = os.path.join(CONLL_DIR, nombre_archivo)
    sents = nltk.corpus.conll2002.iob_sents(particion_nltk)
    
    with open(ruta_salida, 'w', encoding='utf-8') as f:
        for sent in sents:
            for word, pos, ner in sent:
                f.write(f"{word} {pos} {ner}\n")
            f.write("\n")

guardar_conll_texto('esp.train', 'esp.train')
guardar_conll_texto('esp.val', 'esp.testa') 
guardar_conll_texto('esp.test', 'esp.testb')
print(f"-> Generados archivos CoNLL2002 en: ./{CONLL_DIR}/")

print("\n¡Proceso finalizado con éxito! Revisa la carpeta 'data' en tu proyecto.")