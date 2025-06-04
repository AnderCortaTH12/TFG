import os
import pyodbc
import pandas as pd

# === Configuración de conexión ===
script_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(script_dir, 'Base_de_datos', 'Articulos_sentimiento_bolsa.accdb')
conn_str = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    rf'DBQ={db_path};'
)

# Tablas
tabla_cierre = "AAPL_cierre_filtrado"
tabla_sentimiento_apple = "AppleAgrupadas"
tabla_sentimiento_entorno = "EntornoAgrupadas"

try:
    # === Cargar datos desde Access ===
    conn = pyodbc.connect(conn_str)

    df_cierre = pd.read_sql(f"SELECT * FROM {tabla_cierre}", conn)
    df_cierre['Fecha'] = pd.to_datetime(df_cierre['Fecha']).dt.date

    df_apple = pd.read_sql(f"SELECT article_date, sentimiento_del_dia FROM {tabla_sentimiento_apple}", conn)
    df_apple['article_date'] = pd.to_datetime(df_apple['article_date']).dt.date
    df_apple.rename(columns={'article_date': 'Fecha', 'sentimiento_del_dia': 'Sentimiento_apple'}, inplace=True)

    df_entorno = pd.read_sql(f"SELECT article_date, sentimiento_del_dia FROM {tabla_sentimiento_entorno}", conn)
    df_entorno['article_date'] = pd.to_datetime(df_entorno['article_date']).dt.date
    df_entorno.rename(columns={'article_date': 'Fecha', 'sentimiento_del_dia': 'Sentimiento_entorno'}, inplace=True)

    conn.close()

    # === Fusionar y procesar ===
    df_final = pd.merge(df_cierre, df_apple, on='Fecha', how='left')
    df_final = pd.merge(df_final, df_entorno, on='Fecha', how='left')

    df_final = df_final.sort_values(by='Fecha').reset_index(drop=True)

    # Calcular opinión generalizada
    df_final['Opinion_generalizada'] = df_final['Sentimiento_apple'].fillna(0) + 0.5 * df_final['Sentimiento_entorno'].fillna(0)

    # Medias móviles exponenciales con LAG (shift) aplicado para evitar data leakage
    df_final['media_movil_5dias'] = df_final['Opinion_generalizada'].ewm(span=5, adjust=False).mean().shift(1)
    df_final['media_movil_20dias'] = df_final['Opinion_generalizada'].ewm(span=20, adjust=False).mean().shift(1)

    # Calcular sentimiento acumulado solo con datos reales
    df_final['sentimiento_acumulado'] = df_final['Opinion_generalizada'].where(
        ~df_final['Sentimiento_apple'].isna(), 0
    ).cumsum()

    # Asegurar columnas necesarias
    if 'MediaMovil20' not in df_final.columns:
        df_final['MediaMovil20'] = None
    if 'MediaMovil60' not in df_final.columns:
        df_final['MediaMovil60'] = None
    if 'Diferencial' not in df_final.columns:
        df_final['Diferencial'] = None

    # === Escribir nueva tabla Access ===
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    if cursor.tables(table=tabla_cierre, tableType='TABLE').fetchone():
        cursor.execute(f"DROP TABLE {tabla_cierre}")
        conn.commit()

    cursor.execute(f"""
        CREATE TABLE {tabla_cierre} (
            Fecha DATE,
            Close DOUBLE,
            Diferencial DOUBLE,
            MediaMovil20 DOUBLE,
            MediaMovil60 DOUBLE,
            Sentimiento_apple DOUBLE,
            Sentimiento_entorno DOUBLE,
            Opinion_generalizada DOUBLE,
            media_movil_5dias DOUBLE,
            media_movil_20dias DOUBLE,
            sentimiento_acumulado DOUBLE
        )
    """)
    conn.commit()

    for _, row in df_final.iterrows():
        cursor.execute(f"""
            INSERT INTO {tabla_cierre}
            (Fecha, Close, Diferencial, MediaMovil20, MediaMovil60,
             Sentimiento_apple, Sentimiento_entorno, Opinion_generalizada,
             media_movil_5dias, media_movil_20dias, sentimiento_acumulado)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row['Fecha'],
            row['Close'],
            None if pd.isna(row['Diferencial']) else float(row['Diferencial']),
            None if pd.isna(row['MediaMovil20']) else float(row['MediaMovil20']),
            None if pd.isna(row['MediaMovil60']) else float(row['MediaMovil60']),
            None if pd.isna(row['Sentimiento_apple']) else float(row['Sentimiento_apple']),
            None if pd.isna(row['Sentimiento_entorno']) else float(row['Sentimiento_entorno']),
            float(row['Opinion_generalizada']),
            None if pd.isna(row['media_movil_5dias']) else float(row['media_movil_5dias']),
            None if pd.isna(row['media_movil_20dias']) else float(row['media_movil_20dias']),
            float(row['sentimiento_acumulado'])
        ))

    conn.commit()
    cursor.close()
    conn.close()

    print(f"✅ Tabla '{tabla_cierre}' actualizada correctamente con medias móviles exponenciales y acumulado corregido.")

except Exception as e:
    print(f"❌ Error en la ejecución: {e}")
