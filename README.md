# Sistema de Análisis y Predicción de Actividad Sísmica

## Descripción General
Este sistema implementa un análisis completo de datos sísmicos y predicciones de probabilidad utilizando redes bayesianas. Está diseñado para ayudar en la comprensión de patrones sísmicos y la evaluación de riesgos.

## Componentes del Sistema

### 1. Análisis de Datos Sísmicos
El sistema incluye varias visualizaciones y análisis:

#### 1.1 Distribución de Magnitudes
- Genera un histograma de las magnitudes sísmicas
- Muestra la frecuencia de diferentes magnitudes
- Ayuda a entender los patrones de actividad sísmica

#### 1.2 Análisis Temporal
- Muestra la frecuencia de sismos por mes
- Permite identificar patrones estacionales
- Visualiza tendencias temporales

#### 1.3 Mapa de Epicentros
- Crea un mapa interactivo con la ubicación de los sismos
- Permite visualizar la distribución geográfica
- Ayuda a identificar zonas de mayor actividad

#### 1.4 Análisis de Profundidad
- Relaciona la profundidad con la magnitud de los sismos
- Ayuda a entender la correlación entre estos factores
- Proporciona insights sobre el comportamiento sísmico

#### 1.5 Análisis de Tipos de Magnitud
- Muestra la distribución de diferentes métodos de medición
- Ayuda a entender qué tipos de medición son más comunes
- Proporciona contexto sobre los métodos de registro

#### 1.6 Zonas de Mayor Actividad
- Identifica las 10 zonas con mayor actividad sísmica
- Ayuda a entender la distribución geográfica del riesgo
- Útil para planificación y prevención

### 2. Sistema de Predicción

#### 2.1 Red Bayesiana
El sistema utiliza una red bayesiana para realizar predicciones, considerando:
- Magnitud histórica
- Profundidad sísmica
- Tiempo desde el último sismo
- Actividad de falla
- Patrón sísmico
- Intensidad histórica
- Frecuencia mensual

#### 2.2 Interfaz de Predicción
Permite al usuario ingresar:
- Parámetros específicos de la zona
- Condiciones actuales
- Datos históricos relevantes

#### 2.3 Resultados
Proporciona:
- Probabilidades de sismo (baja, media, alta)
- Visualización gráfica de las probabilidades
- Resultados detallados en formato texto

## Cómo Usar el Sistema

### 1. Análisis de Datos
1. Sube un archivo CSV con datos sísmicos
2. El sistema generará automáticamente:
   - Gráficos de distribución
   - Mapa de epicentros
   - Análisis temporales
   - Estadísticas de zonas

### 2. Predicción de Sismos
1. Accede a la interfaz de predicción
2. Ingresa los parámetros relevantes:
   - Magnitud histórica
   - Profundidad
   - Tiempo desde último sismo
   - Actividad de falla
   - Patrón sísmico
   - Intensidad
   - Frecuencia
3. Obtén la predicción de probabilidad

## Requisitos Técnicos

### Dependencias
- Python 3.8+
- pandas
- matplotlib
- seaborn
- folium
- gradio
- pgmpy
- PIL

### Estructura de Datos
El archivo CSV debe contener las siguientes columnas:
- time: Fecha y hora del sismo
- latitude: Latitud del epicentro
- longitude: Longitud del epicentro
- depth: Profundidad del sismo
- mag: Magnitud
- magType: Tipo de magnitud
- place: Ubicación

## Limitaciones y Consideraciones

### Limitaciones
- Las predicciones son probabilísticas
- La precisión depende de la calidad de los datos
- No predice el momento exacto de un sismo

### Consideraciones
- Usar datos actualizados
- Considerar el contexto geológico
- Interpretar resultados con precaución

## Contribuciones
Se aceptan contribuciones para mejorar:
- Modelos de predicción
- Visualizaciones
- Interfaz de usuario
- Documentación

## Licencia
Este proyecto está bajo la Licencia MIT.

## Contacto
Para preguntas o sugerencias, por favor crear un issue en el repositorio. 