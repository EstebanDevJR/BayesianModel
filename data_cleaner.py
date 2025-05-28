import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cargar_datos(ruta_archivo=None):
    """
    Carga los datos del archivo CSV.
    Si no se proporciona una ruta, busca en el directorio actual y luego en 'data/'.
    """
    try:
        if ruta_archivo is None:
            # Buscar el archivo en el directorio actual
            archivos_csv = [f for f in os.listdir('.') if f.endswith('.csv')]
            if archivos_csv:
                ruta_archivo = archivos_csv[0]
                logger.info(f"Usando archivo encontrado: {ruta_archivo}")
            else:
                # Buscar en la subcarpeta 'data/'
                data_dir = os.path.join('.', 'data')
                if os.path.isdir(data_dir):
                    archivos_csv_data = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
                    if archivos_csv_data:
                        ruta_archivo = os.path.join(data_dir, archivos_csv_data[0])
                        logger.info(f"Usando archivo encontrado en 'data/': {ruta_archivo}")
                    else:
                        raise FileNotFoundError("No se encontró ningún archivo CSV en el directorio actual ni en 'data/'")
                else:
                    raise FileNotFoundError("No se encontró ningún archivo CSV en el directorio actual ni en 'data/'")

        # Leer el CSV
        df = pd.read_csv(ruta_archivo)
        logger.info(f"Datos cargados exitosamente desde {ruta_archivo}")
        return df
    except Exception as e:
        logger.error(f"Error al cargar los datos: {str(e)}")
        raise

def generar_visualizaciones(df):
    """
    Genera todas las visualizaciones del dataset.
    Esta función debe ser llamada explícitamente cuando se deseen ver las gráficas.
    """
    # ================================
    # 1. DISTRIBUCIÓN DE MAGNITUDES
    # ================================
    plt.figure(figsize=(10,5))
    sns.histplot(df['mag'].dropna(), bins=30, kde=True)
    plt.title("Distribución de magnitudes sísmicas")
    plt.xlabel("Magnitud")
    plt.ylabel("Frecuencia")
    plt.grid(True)
    plt.show()

    # ================================
    # 2. SISMOS POR MES
    # ================================
    if 'time' in df.columns:
        df['month'] = df['time'].dt.to_period('M')
        plt.figure(figsize=(14,6))
        df['month'].value_counts().sort_index().plot(kind='bar')
        plt.title("Cantidad de sismos por mes")
        plt.xlabel("Mes")
        plt.ylabel("Número de sismos")
        plt.xticks(rotation=90)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # ================================
    # 3. MAPA DE EPICENTROS
    # ================================
    if 'latitude' in df.columns and 'longitude' in df.columns:
        map_center = [df['latitude'].mean(), df['longitude'].mean()]
        mapa = folium.Map(location=map_center, zoom_start=5)

        for _, row in df.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=2,
                color='red',
                fill=True,
                fill_opacity=0.6
            ).add_to(mapa)

        mapa.save("mapa_sismos.html")
        print("🌍 Mapa guardado como mapa_sismos.html")

    # ================================
    # 4. PROFUNDIDAD VS MAGNITUD
    # ================================
    if 'depth' in df.columns and 'mag' in df.columns:
        plt.figure(figsize=(8,6))
        sns.scatterplot(data=df, x='depth', y='mag', alpha=0.5)
        plt.title("Relación entre profundidad y magnitud")
        plt.xlabel("Profundidad (km)")
        plt.ylabel("Magnitud")
        plt.grid(True)
        plt.show()

    # ================================
    # 5. FRECUENCIA POR TIPO DE MAGNITUD
    # ================================
    if 'magType' in df.columns:
        plt.figure(figsize=(10,6))
        df['magType'].value_counts().plot(kind='bar')
        plt.title("Frecuencia de tipos de magnitud (magType)")
        plt.xlabel("Tipo de magnitud")
        plt.ylabel("Número de sismos")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # ================================
    # 6. ZONAS MÁS SÍSMICAS
    # ================================
    if 'place' in df.columns:
        top_places = df['place'].value_counts().head(10)
        plt.figure(figsize=(10,6))
        top_places.plot(kind='bar')
        plt.title("Top 10 zonas más sísmicas")
        plt.xlabel("Lugar")
        plt.ylabel("Cantidad de sismos")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def preprocesar_para_bayes(df):
    """
    Preprocesa los datos para el modelo bayesiano.
    """
    import numpy as np
    import pandas as pd
    from datetime import datetime
    import logging

    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # Validación de datos de entrada
        required_columns = ['mag', 'depth', 'time']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Columnas requeridas faltantes: {missing_columns}")

        # 1. magnitud_historica (mejorado)
        def categorizar_magnitud(mag):
            if pd.isna(mag):
                return 'desconocida'
            if mag < 4:
                return 'baja'
            elif 4 <= mag <= 6:
                return 'media'
            else:
                return 'alta'
        df['magnitud_historica'] = df['mag'].apply(categorizar_magnitud)

        # 2. profundidad_sismica (mejorado)
        def categorizar_profundidad(depth):
            if pd.isna(depth):
                return 'desconocida'
            if depth < 70:
                return 'superficial'
            elif 70 <= depth <= 300:
                return 'intermedia'
            else:
                return 'profunda'
        df['profundidad_sismica'] = df['depth'].apply(categorizar_profundidad)

        # 3. tiempo_ultimo_sismo (mejorado)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            sismos_fuertes = df[df['mag'] >= 6].sort_values('time', ascending=False)
            if not sismos_fuertes.empty:
                ultimo_sismo = sismos_fuertes.iloc[0]['time']
                ahora = df['time'].max() if not pd.isnull(df['time'].max()) else datetime.now()
                diff_years = (ahora - ultimo_sismo).days / 365.25
                if diff_years < 1:
                    tiempo_cat = 'reciente'
                elif diff_years <= 10:
                    tiempo_cat = 'medio'
                else:
                    tiempo_cat = 'lejano'
            else:
                tiempo_cat = 'lejano'
            df['tiempo_ultimo_sismo'] = tiempo_cat
        else:
            df['tiempo_ultimo_sismo'] = 'desconocido'

        # 4. actividad_falla (mejorado)
        def calcular_actividad_falla(df):
            total_sismos = len(df)
            if total_sismos < 50:
                return 'baja'
            elif total_sismos < 200:
                return 'media'
            else:
                return 'alta'
        df['actividad_falla'] = calcular_actividad_falla(df)

        # 5. patron_sismico (mejorado)
        if 'time' in df.columns:
            ahora = df['time'].max() if not pd.isnull(df['time'].max()) else datetime.now()
            cinco_anos = ahora - pd.DateOffset(years=5)
            recientes = df[df['time'] >= cinco_anos]
            n_recientes = len(recientes)
            if n_recientes < 10:
                patron = 'esporádico'
            elif n_recientes < 50:
                patron = 'regular'
            else:
                patron = 'frecuente'
        else:
            patron = 'desconocido'
        df['patron_sismico'] = patron

        # 6. Nueva característica: intensidad_historica
        def calcular_intensidad_historica(df):
            if 'mag' in df.columns:
                media_magnitud = df['mag'].mean()
                if pd.isna(media_magnitud):
                    return 'desconocida'
                if media_magnitud < 4:
                    return 'baja'
                elif media_magnitud <= 5:
                    return 'media'
                else:
                    return 'alta'
            return 'desconocida'
        df['intensidad_historica'] = calcular_intensidad_historica(df)

        # 7. Nueva característica: frecuencia_mensual
        if 'time' in df.columns:
            df['month'] = df['time'].dt.to_period('M')
            freq_mensual = df.groupby('month').size().mean()
            if freq_mensual < 5:
                df['frecuencia_mensual'] = 'baja'
            elif freq_mensual < 20:
                df['frecuencia_mensual'] = 'media'
            else:
                df['frecuencia_mensual'] = 'alta'
        else:
            df['frecuencia_mensual'] = 'desconocida'

        # Verificar valores nulos
        null_counts = df.isnull().sum()
        if null_counts.any():
            logger.warning(f"Valores nulos encontrados: {null_counts[null_counts > 0]}")

        # Devolver solo las columnas necesarias
        return df[[
            'magnitud_historica',
            'profundidad_sismica',
            'tiempo_ultimo_sismo',
            'actividad_falla',
            'patron_sismico',
            'intensidad_historica',
            'frecuencia_mensual'
        ]]

    except Exception as e:
        logger.error(f"Error en el preprocesamiento: {str(e)}")
        raise
