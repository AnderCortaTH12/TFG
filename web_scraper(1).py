import os
from bs4 import BeautifulSoup
import requests
import pandas as pd
import pyodbc

# Palabras clave
apple_keywords = ['apple', 'aapl', 'iphone']
entorno_keywords = ['technology', 'tech', 'google', 'microsoft', 'samsung', 'amazon', 'intel', 'semiconductor', 'software']

# Crear DataFrames
df_apple = []
df_entorno = []

# Scrapeo
for page in range(1, 250):
    url = f'https://markets.businessinsider.com/news/aapl-stock?p={page}'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')

    articles = soup.find_all('div', class_="latest-news__story")
    for article in articles:
        article_date = article.find('time', class_='latest-news__date').get('datetime')
        title = article.find('a', class_='news-link').text.strip()
        source = article.find('span', class_='latest-news__source').text.strip()
        link = article.find('a', class_='news-link').get('href')
        title_lower = title.lower()

        if any(keyword in title_lower for keyword in apple_keywords):
            df_apple.append((article_date, title, source, link))
        elif any(keyword in title_lower for keyword in entorno_keywords):
            df_entorno.append((article_date, title, source, link))

# Conexión a Access (base de datos real)
script_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(script_dir, 'Base_de_datos', 'Articulos_sentimiento_bolsa.accdb')
conn_str = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    rf'DBQ={db_path};'
)
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

# Crear tabla 'apple' si no existe
if not cursor.tables(table='apple', tableType='TABLE').fetchone():
    cursor.execute("""
        CREATE TABLE apple (
            article_date DATETIME,
            title TEXT,
            source TEXT,
            link TEXT
        )
    """)
    conn.commit()

# Crear tabla 'entorno' si no existe
if not cursor.tables(table='entorno', tableType='TABLE').fetchone():
    cursor.execute("""
        CREATE TABLE entorno (
            article_date DATETIME,
            title TEXT,
            source TEXT,
            link TEXT
        )
    """)
    conn.commit()

# Insertar en tabla 'apple'
for row in df_apple:
    cursor.execute("""
        INSERT INTO apple (article_date, title, source, link)
        VALUES (?, ?, ?, ?)
    """, *row)

# Insertar en tabla 'entorno'
for row in df_entorno:
    cursor.execute("""
        INSERT INTO entorno (article_date, title, source, link)
        VALUES (?, ?, ?, ?)
    """, *row)

conn.commit()
cursor.close()
conn.close()

print("✅ Noticias clasificadas e insertadas en las tablas 'apple' y 'entorno'.")
