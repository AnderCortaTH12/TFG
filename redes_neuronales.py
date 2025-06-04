import pandas as pd
import numpy as np
import pyodbc
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
import tensorflow as tf
import random
import os

# === SEMILLA PARA REPRODUCIBILIDAD ===
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# === PARÁMETROS AJUSTABLES ===
EPOCHS = 80
BATCH_SIZE = 16
WINDOW_SIZE = 10

# === CONEXIÓN A BASE DE DATOS ACCESS ===
db_path = r'Base_de_datos\\Articulos_sentimiento_bolsa.accdb'
conn_str = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    rf'DBQ={db_path};'
)
conn = pyodbc.connect(conn_str)
df = pd.read_sql("SELECT * FROM AAPL_cierre_filtrado", conn)
conn.close()

# === ORDENAR Y LIMPIAR ===
df = df.sort_values(by='Fecha')
print("\n📊 Filas originales en la base de datos:", len(df))
df = df.dropna()
print("🧹 Filas después de eliminar nulos:", len(df))

# === VARIABLES DE LOS DOS MODELOS ===
features_A = ['Close', 'MediaMovil20', 'MediaMovil60']
features_B = features_A + ['media_movil_5dias', 'media_movil_20dias']
target = 'Close'

# === FUNCIÓN PARA PREPARAR DATOS PARA LSTM ===
def prepare_lstm_data(df, features, target, n_steps):
    scaler = MinMaxScaler()
    data = df[features + [target]].copy()
    scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(n_steps, len(scaled)):
        X.append(scaled[i - n_steps:i, :-1])
        y.append(scaled[i, -1])
    return np.array(X), np.array(y), scaler

# === DIVISIÓN PERSONALIZADA ===
def split_data(X, y, train_frac=0.7, val_frac=0.15):
    n = len(X)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    print(f"\nTotal muestras: {n}")
    print(f"Entrenamiento: {train_end}, Validación: {val_end - train_end}, Test: {n - val_end}")
    return (
        X[:train_end], y[:train_end],
        X[train_end:val_end], y[train_end:val_end],
        X[val_end:], y[val_end:]
    )

# === FUNCIONES PARA CONSTRUIR Y EVALUAR EL MODELO ===
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def inverse_transform_predictions(scaler, y_pred, y_real):
    n_total_features = scaler.n_features_in_
    pad_pred = np.zeros((len(y_pred), n_total_features))
    pad_true = np.zeros((len(y_real), n_total_features))
    pad_pred[:, -1] = y_pred.flatten()
    pad_true[:, -1] = y_real.flatten()
    inverse_pred = scaler.inverse_transform(pad_pred)[:, -1]
    inverse_true = scaler.inverse_transform(pad_true)[:, -1]
    return inverse_true, inverse_pred

def plot_predictions(inverse_train, inverse_val, inverse_test, inverse_train_pred, inverse_val_pred, inverse_test_pred, title=""):
    total_len = len(inverse_train) + len(inverse_val) + len(inverse_test)
    x_train = np.arange(0, len(inverse_train))
    x_val = np.arange(len(inverse_train), len(inverse_train) + len(inverse_val))
    x_test = np.arange(len(inverse_train) + len(inverse_val), total_len)

    plt.figure(figsize=(12, 5))
    plt.plot(x_train, inverse_train, label="Train - Real", color="green")
    plt.plot(x_val, inverse_val, label="Validation - Real", color="blue")
    plt.plot(x_test, inverse_test, label="Test - Real", color="black")

    plt.plot(x_train, inverse_train_pred, '--', label="Train - Pred", color="lightgreen")
    plt.plot(x_val, inverse_val_pred, '--', label="Validation - Pred", color="lightblue")
    plt.plot(x_test, inverse_test_pred, '--', label="Test - Pred", color="orange")

    plt.title(title)
    plt.xlabel("Índice temporal")
    plt.ylabel("Precio de cierre")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_test_only(inverse_test, inverse_test_pred, title=""):
    plt.figure(figsize=(10, 4))
    plt.plot(inverse_test, label="Test - Real", color="black")
    plt.plot(inverse_test_pred, '--', label="Test - Pred", color="orange")
    plt.title(f"{title} - Solo Test")
    plt.xlabel("Índice temporal (Test)")
    plt.ylabel("Precio de cierre")
    plt.legend()
    plt.tight_layout()
    plt.show()

def train_and_evaluate_model(name, X_train, y_train, X_val, y_val, X_test, y_test, scaler):
    print(f"\nEntrenando {name}...")
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    inv_train, inv_train_pred = inverse_transform_predictions(scaler, y_train_pred, y_train)
    inv_val, inv_val_pred = inverse_transform_predictions(scaler, y_val_pred, y_val)
    inv_test, inv_test_pred = inverse_transform_predictions(scaler, y_test_pred, y_test)

    print(f"\n📊 Resultados de {name}:")
    print("MAE:", mean_absolute_error(inv_test, inv_test_pred))
    print("RMSE:", np.sqrt(mean_squared_error(inv_test, inv_test_pred)))
    print("R2:", r2_score(inv_test, inv_test_pred))

    # === Gráfico general (train, val, test) ===
    plot_predictions(inv_train, inv_val, inv_test, inv_train_pred, inv_val_pred, inv_test_pred, title=name)

    # === Gráfico solo del test ===
    plot_test_only(inv_test, inv_test_pred, title=name)

    # === Comparación con modelo Naive (y_t = y_{t-1}) ===
    inv_test_naive = inv_test[:-1]            # y_{t-1}
    inv_test_real = inv_test[1:]              # y_{t}
    inv_test_pred_shifted = inv_test_pred[1:] # LSTM predicción correspondiente

    plt.figure(figsize=(8, 4))
    plt.plot(inv_test_real, label="Real", color="black")
    plt.plot(inv_test_naive, '--', label="Naive (y_{t-1})", color="gray")
    plt.plot(inv_test_pred_shifted, label="Modelo (LSTM)", color="orange")
    plt.title(f"{name} - Comparación real vs. naive vs. LSTM")
    plt.xlabel("Índice temporal (Test)")
    plt.ylabel("Precio de cierre")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # === MAE del modelo naive ===
    naive_mae = mean_absolute_error(inv_test_real, inv_test_naive)
    print(f"📏 MAE modelo naive (y_t = y_t-1): {naive_mae:.4f}")


# === ENTRENAMIENTO Y EVALUACIÓN ===
X_a, y_a, scaler_a = prepare_lstm_data(df, features_A, target, WINDOW_SIZE)
X_b, y_b, scaler_b = prepare_lstm_data(df, features_B, target, WINDOW_SIZE)

X_a_train, y_a_train, X_a_val, y_a_val, X_a_test, y_a_test = split_data(X_a, y_a)
X_b_train, y_b_train, X_b_val, y_b_val, X_b_test, y_b_test = split_data(X_b, y_b)

train_and_evaluate_model("Modelo A (sin sentimiento)", X_a_train, y_a_train, X_a_val, y_a_val, X_a_test, y_a_test, scaler_a)
train_and_evaluate_model("Modelo B (con sentimiento)", X_b_train, y_b_train, X_b_val, y_b_val, X_b_test, y_b_test, scaler_b)


