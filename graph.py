# ============================================================================
# ANÁLISIS Y PREDICCIÓN DE ACTIVIDAD SÍSMICA
# ============================================================================
# Este script implementa una interfaz gráfica para analizar datos sísmicos y
# realizar predicciones usando una red bayesiana. Utiliza Gradio para crear
# una interfaz web interactiva.

# Importaciones necesarias
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import os
from data_cleaner import preprocesar_para_bayes, cargar_datos
from model import inference
import io
from PIL import Image

# Configuración para generar gráficos sin interfaz GUI
import matplotlib
matplotlib.use('Agg')

# ============================================================================
# FUNCIONES DE CARGA Y PREPROCESAMIENTO
# ============================================================================

def cargar_datos(archivo):
    """
    Carga y preprocesa los datos del archivo CSV de sismos.
    Args:
        archivo: Archivo CSV con datos sísmicos
    Returns:
        DataFrame con los datos procesados
    """
    df = pd.read_csv(archivo.name)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
    return df

# ============================================================================
# FUNCIONES DE ANÁLISIS Y VISUALIZACIÓN
# ============================================================================

def distribucion_magnitudes(df):
    """
    Genera un histograma de la distribución de magnitudes sísmicas.
    Muestra la frecuencia de diferentes magnitudes para entender
    el patrón de actividad sísmica.
    """
    plt.figure(figsize=(8,4))
    sns.histplot(df['mag'].dropna(), bins=30, kde=True, color='skyblue', edgecolor='black')
    plt.title("Distribución de magnitudes", fontsize=16)
    plt.xlabel("Magnitud", fontsize=12)
    plt.ylabel("Frecuencia", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return Image.open(buf)

def sismos_por_mes(df):
    """
    Analiza la frecuencia de sismos por mes.
    Útil para identificar patrones temporales y estacionalidad
    en la actividad sísmica.
    """
    if 'time' not in df.columns:
        return None
    
    # Convertir a datetime si no lo es
    df['time'] = pd.to_datetime(df['time'])
    
    # Agrupar por mes y año para reducir la saturación
    df['month_year'] = df['time'].dt.to_period('M')
    monthly_counts = df['month_year'].value_counts().sort_index()
    
    # Crear el gráfico con un tamaño más grande
    plt.figure(figsize=(15, 6))
    
    # Usar un color más suave y agregar transparencia
    bars = plt.bar(range(len(monthly_counts)), monthly_counts.values, 
                  color='lightgreen', alpha=0.7, edgecolor='black')
    
    # Configurar el eje x para mostrar solo algunos meses
    plt.xticks(range(len(monthly_counts))[::3],  # Mostrar cada tercer mes
              [str(x) for x in monthly_counts.index[::3]],  # Etiquetas
              rotation=45)
    
    plt.title("Sismos por mes", fontsize=16, pad=20)
    plt.xlabel("Mes", fontsize=12, labelpad=10)
    plt.ylabel("Cantidad de sismos", fontsize=12)
    
    # Agregar una cuadrícula suave
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Ajustar los márgenes
    plt.tight_layout()
    
    # Guardar el gráfico
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    buf.seek(0)
    return Image.open(buf)

def mapa_epicentros(df):
    """
    Crea un mapa interactivo mostrando la ubicación de los epicentros.
    Permite visualizar la distribución geográfica de la actividad sísmica.
    """
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        return None

    mapa = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=5)
    for _, row in df.iterrows():
        if pd.notnull(row['latitude']) and pd.notnull(row['longitude']):
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=2,
                color='red',
                fill=True,
                fill_opacity=0.6
            ).add_to(mapa)
    return mapa

def profundidad_vs_magnitud(df):
    """
    Analiza la relación entre la profundidad y la magnitud de los sismos.
    Ayuda a entender si existe correlación entre estos dos factores.
    """
    if 'depth' not in df.columns or 'mag' not in df.columns:
        return None
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x='depth', y='mag', alpha=0.5, color='purple', edgecolor='black')
    plt.title("Profundidad vs Magnitud", fontsize=16)
    plt.xlabel("Profundidad (km)", fontsize=12)
    plt.ylabel("Magnitud", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return Image.open(buf)


def zonas_sismicas(df):
    """
    Identifica las zonas con mayor actividad sísmica.
    Muestra las 10 ubicaciones con más sismos registrados.
    """
    if 'place' not in df.columns:
        return None
    plt.figure(figsize=(10,5))
    df['place'].value_counts().head(10).plot(kind='bar', color='lightblue', edgecolor='black')
    plt.title("Top 10 zonas más sísmicas", fontsize=16)
    plt.xlabel("Lugar", fontsize=12)
    plt.ylabel("Cantidad", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return Image.open(buf)

# ============================================================================
# FUNCIONES PRINCIPALES DE PROCESAMIENTO
# ============================================================================

def procesar_archivo(archivo):
    """
    Función principal que procesa el archivo de datos y genera todas las visualizaciones.
    Coordina el análisis completo de los datos sísmicos.
    """
    df = cargar_datos(archivo)

    resultados = {
        "Distribución de Magnitudes": distribucion_magnitudes(df),
        "Sismos por Mes": sismos_por_mes(df),
        "Profundidad vs Magnitud": profundidad_vs_magnitud(df),
        "Top Zonas Sísmicas": zonas_sismicas(df),
        "Mapa Epicentros": mapa_epicentros(df)
    }

    return (
        resultados["Distribución de Magnitudes"],
        resultados["Sismos por Mes"],
        resultados["Profundidad vs Magnitud"],
        resultados["Top Zonas Sísmicas"],
        resultados["Mapa Epicentros"]
    )

def predecir_sismo(magnitud, profundidad, tiempo, actividad, patron, intensidad, frecuencia):
    """
    Realiza predicciones de probabilidad de sismo usando la red bayesiana.
    Permite al usuario ingresar diferentes parámetros para obtener
    una predicción personalizada.
    """
    # Crear evidencia para la red bayesiana
    evidencia = {
        "magnitud_historica": magnitud,
        "profundidad_sismica": profundidad,
        "tiempo_ultimo_sismo": tiempo,
        "actividad_falla": actividad,
        "patron_sismico": patron,
        "intensidad_historica": intensidad,
        "frecuencia_mensual": frecuencia
    }
    
    # Realizar inferencia
    resultado = inference.query(variables=["probabilidad_sismo"], evidence=evidencia)
    
    # Visualizar resultados
    plt.figure(figsize=(8, 4))
    probabilidades = resultado.values
    estados = ["Baja", "Media", "Alta"]
    plt.bar(estados, probabilidades, color=['lightgreen', 'lightcoral', 'lightblue'], edgecolor='black')
    plt.title("Probabilidad de Sismo", fontsize=16)
    plt.ylabel("Probabilidad", fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    return Image.open(buf), str(resultado)

# ============================================================================
# INTERFACES DE GRADIO
# ============================================================================

# Interfaz para análisis de datos
demo = gr.Interface(
    fn=procesar_archivo,
    inputs=gr.File(label="Sube tu archivo CSV de sismos"),
    outputs=[
        gr.Image(label="Distribución de magnitudes"),
        gr.Image(label="Sismos por mes"),
        gr.Image(label="Profundidad vs Magnitud"),
        gr.Image(label="Zonas más sísmicas"),
        gr.HTML(label="Mapa de epicentros")
    ],
    title="Análisis de datos sísmicos",
    description="Carga un archivo CSV de datos sísmicos (como los del Servicio Geológico Colombiano o USGS)."
)

# Interfaz para predicción personalizada
demo_prediccion = gr.Interface(
    fn=predecir_sismo,
    inputs=[
        gr.Dropdown(["baja", "media", "alta", "desconocida"], label="Magnitud Histórica"),
        gr.Dropdown(["superficial", "intermedia", "profunda", "desconocida"], label="Profundidad Sísmica"),
        gr.Dropdown(["reciente", "medio", "lejano", "desconocido"], label="Tiempo desde Último Sismo"),
        gr.Dropdown(["baja", "media", "alta"], label="Actividad de Falla"),
        gr.Dropdown(["esporádico", "regular", "frecuente", "desconocido"], label="Patrón Sísmico"),
        gr.Dropdown(["baja", "media", "alta", "desconocida"], label="Intensidad Histórica"),
        gr.Dropdown(["baja", "media", "alta", "desconocida"], label="Frecuencia Mensual")
    ],
    outputs=[
        gr.Image(label="Gráfico de Probabilidades"),
        gr.Textbox(label="Resultado Detallado")
    ],
    title="Predicción de Probabilidad de Sismo",
    description="Ingresa los parámetros para predecir la probabilidad de un sismo."
)

# Lanzar las interfaces
if __name__ == "__main__":
    demo.launch()
    demo_prediccion.launch()
