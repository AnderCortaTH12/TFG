import os
import pandas as pd
import pyodbc

# Ruta a la base de datos Access
script_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(script_dir, 'Base_de_datos', 'Articulos_sentimiento_bolsa.accdb')
conn_str = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    rf'DBQ={db_path};'
)

# Función para agrupar y guardar
def agrupar_sentimientos(nombre_tabla_origen, nombre_tabla_destino):
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    # Leer datos
    df = pd.read_sql(f"SELECT article_date, top_sentiment FROM {nombre_tabla_origen}", conn)
    df['article_date'] = pd.to_datetime(df['article_date']).dt.date

    # Agrupar por fecha y sentimiento
    grouped = df.groupby(['article_date', 'top_sentiment']).size().unstack(fill_value=0)

    # Asegurar columnas consistentes
    for col in ['Positive', 'Negative', 'Neutral']:
        if col not in grouped.columns:
            grouped[col] = 0

    grouped = grouped[['Positive', 'Negative', 'Neutral']].reset_index()

    # Calcular sentimiento del día
    grouped['sentimiento_del_dia'] = (
        grouped['Positive'] * 1.5 -
        grouped['Negative'] * 2.5
    )

    # Ordenar por fecha ascendente
    grouped = grouped.sort_values(by='article_date')

    # Eliminar la tabla destino si ya existe
    if cursor.tables(table=nombre_tabla_destino, tableType='TABLE').fetchone():
        cursor.execute(f"DROP TABLE {nombre_tabla_destino}")
        conn.commit()

    # Crear tabla destino sin la columna 'acumulado'
    cursor.execute(f"""
        CREATE TABLE {nombre_tabla_destino} (
            article_date DATE,
            Positive INT,
            Negative INT,
            Neutral INT,
            sentimiento_del_dia DOUBLE
        )
    """)
    conn.commit()

    # Insertar datos
    for _, row in grouped.iterrows():
        cursor.execute(f"""
            INSERT INTO {nombre_tabla_destino} (article_date, Positive, Negative, Neutral, sentimiento_del_dia)
            VALUES (?, ?, ?, ?, ?)
        """, row['article_date'], int(row['Positive']), int(row['Negative']), int(row['Neutral']),
             float(row['sentimiento_del_dia']))

    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ Datos agrupados insertados en '{nombre_tabla_destino}' con 'sentimiento_del_dia'.")

# Ejecutar para ambas tablas
agrupar_sentimientos("AnalisisSentimientoApple", "AppleAgrupadas")
agrupar_sentimientos("AnalisisSentimientoEntorno", "EntornoAgrupadas")
