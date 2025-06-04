import os
import pandas as pd
import pyodbc
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Cargar modelo FinBERT
model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
finbert = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Ruta a la base de datos Access
script_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(script_dir, 'Base_de_datos', 'Articulos_sentimiento_bolsa.accdb')
conn_str = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    rf'DBQ={db_path};'
)

# Función para analizar sentimiento y guardar en tabla
def analizar_y_guardar(nombre_tabla_origen, nombre_tabla_destino):
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    # Leer datos de la tabla origen
    query = f"SELECT article_date, title FROM {nombre_tabla_origen}"
    df = pd.read_sql(query, conn)
    df = df.sort_values(by='article_date', ascending=False)

    # Análisis de sentimiento
    sentiments = []
    scores = []

    for title in df['title']:
        result = finbert(title)[0]
        sentiments.append(result['label'])
        scores.append(result['score'])

    df['top_sentiment'] = sentiments
    df['sentiment_score'] = scores

    # Crear tabla destino si no existe
    if not cursor.tables(table=nombre_tabla_destino, tableType='TABLE').fetchone():
        cursor.execute(f"""
            CREATE TABLE {nombre_tabla_destino} (
                article_date DATETIME,
                title TEXT,
                top_sentiment TEXT,
                sentiment_score DOUBLE
            )
        """)
        conn.commit()

    # Insertar resultados en la tabla destino
    for _, row in df.iterrows():
        cursor.execute(f"""
            INSERT INTO {nombre_tabla_destino} (article_date, title, top_sentiment, sentiment_score)
            VALUES (?, ?, ?, ?)
        """, row['article_date'], row['title'], row['top_sentiment'], row['sentiment_score'])

    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ Análisis completado e insertado en la tabla '{nombre_tabla_destino}'.")

# Ejecutar para ambas tablas
analizar_y_guardar("apple", "AnalisisSentimientoApple")
analizar_y_guardar("entorno", "AnalisisSentimientoEntorno")
