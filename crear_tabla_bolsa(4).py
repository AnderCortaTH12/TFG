import os
import pyodbc
import pandas as pd

# Ruta al directorio del script
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- BASE DE DATOS ACCESS ---
db_path = os.path.join(script_dir, 'Base_de_datos', 'Articulos_sentimiento_bolsa.accdb')
conn_str = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    rf'DBQ={db_path};'
)

# Consulta para obtener la fecha más antigua
query_fecha = """
    SELECT MIN(article_date) AS fecha_mas_antigua FROM (
        SELECT article_date FROM apple
        UNION ALL
        SELECT article_date FROM entorno
    ) AS todas_las_fechas
"""

try:
    conn = pyodbc.connect(conn_str)
    df_fecha = pd.read_sql(query_fecha, conn)
    conn.close()

    fecha_mas_antigua = pd.to_datetime(df_fecha.iloc[0]['fecha_mas_antigua']).date()
    print(f"📅 La fecha más antigua encontrada es: {fecha_mas_antigua}")
except Exception as e:
    print(f"❌ Error al acceder a la base de datos: {e}")
    fecha_mas_antigua = None

# --- LECTURA Y FILTRADO DEL CSV ---
csv_path = os.path.join(script_dir, 'Base_de_datos', 'AAPL.csv')

try:
    df_csv = pd.read_csv(csv_path)
    df_csv['Unnamed: 0'] = pd.to_datetime(df_csv['Unnamed: 0'], errors='coerce').dt.date
    df_csv = df_csv.dropna(subset=['Unnamed: 0'])

    # Obtener la fecha más reciente del CSV
    fecha_mas_reciente = df_csv['Unnamed: 0'].max()
    print(f"📅 La fecha más reciente en el CSV es: {fecha_mas_reciente}")

    # Filtrar por fecha más antigua
    if fecha_mas_antigua:
        df_filtrado = df_csv[df_csv['Unnamed: 0'] >= fecha_mas_antigua].copy()
        print("✅ Datos filtrados desde la fecha más antigua:")
    else:
        print("⚠️ No se aplicó el filtro por fecha.")
        df_filtrado = df_csv.copy()

    # Calcular columnas adicionales
    df_filtrado['Diferencial'] = df_filtrado['Close'].pct_change() * 100
    df_filtrado['MediaMovil20'] = df_filtrado['Close'].rolling(window=20).mean()
    df_filtrado['MediaMovil60'] = df_filtrado['Close'].rolling(window=60).mean()

    print(df_filtrado[['Unnamed: 0', 'Close', 'Diferencial', 'MediaMovil20', 'MediaMovil60']].head())

    # --- GUARDAR EN ACCESS ---
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    table_name = "AAPL_cierre_filtrado"

    # Eliminar tabla si existe
    if cursor.tables(table=table_name, tableType='TABLE').fetchone():
        cursor.execute(f"DROP TABLE {table_name}")
        conn.commit()

    # Crear nueva tabla
    cursor.execute(f"""
        CREATE TABLE {table_name} (
            Fecha DATE,
            Close DOUBLE,
            Diferencial DOUBLE,
            MediaMovil20 DOUBLE,
            MediaMovil60 DOUBLE
        )
    """)
    conn.commit()

    # Insertar datos
    for _, row in df_filtrado.iterrows():
        cursor.execute(f"""
            INSERT INTO {table_name} (Fecha, Close, Diferencial, MediaMovil20, MediaMovil60)
            VALUES (?, ?, ?, ?, ?)
        """,
        row['Unnamed: 0'],
        row['Close'],
        None if pd.isna(row['Diferencial']) else float(row['Diferencial']),
        None if pd.isna(row['MediaMovil20']) else float(row['MediaMovil20']),
        None if pd.isna(row['MediaMovil60']) else float(row['MediaMovil60']))

    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ Tabla '{table_name}' actualizada con medias móviles de 20 y 60 días.")

except FileNotFoundError:
    print(f"❌ No se encontró el archivo: {csv_path}")
except Exception as e:
    print(f"⚠️ Error al leer o guardar el CSV: {e}")
