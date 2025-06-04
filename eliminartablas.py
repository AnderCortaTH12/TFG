import os
import pyodbc

# Ruta al directorio del script
script_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(script_dir, 'Base_de_datos', 'Articulos_sentimiento_bolsa.accdb')
conn_str = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    rf'DBQ={db_path};'
)

tablas_a_eliminar = ["AAPL_cierre_filtrado", "AppleAgrupadas", "EntornoAgrupadas"]

try:
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    for tabla in tablas_a_eliminar:
        if cursor.tables(table=tabla, tableType='TABLE').fetchone():
            cursor.execute(f"DROP TABLE {tabla}")
            print(f"🗑️  Tabla eliminada: {tabla}")
        else:
            print(f"⚠️  La tabla '{tabla}' no existe o ya fue eliminada.")

    conn.commit()
    cursor.close()
    conn.close()
    print("✅ Proceso completado.")

except Exception as e:
    print(f"❌ Error eliminando tablas: {e}")
