import pandas as pd
import logging
import os
from data_cleaner import preprocesar_para_bayes, cargar_datos
from model import inference

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # 1. Cargar y preprocesar
        logger.info("Cargando datos...")
        df = cargar_datos()
        df_bayes = preprocesar_para_bayes(df)
        logger.info("Datos preprocesados exitosamente")

        # 2. Seleccionar una evidencia (puedes cambiar el índice)
        evidencia = df_bayes.iloc[-1].to_dict()
        logger.info("Evidencia seleccionada:")
        for var, val in evidencia.items():
            logger.info(f"{var}: {val}")

        # 3. Inferencia
        logger.info("Realizando inferencia...")
        resultado = inference.query(variables=["probabilidad_sismo"], evidence=evidencia)
        
        # 4. Mostrar resultados
        logger.info("\nResultados de la inferencia:")
        logger.info(resultado)

        # 5. Interpretación de resultados
        prob_baja = resultado.values[0]
        prob_media = resultado.values[1]
        prob_alta = resultado.values[2]

        logger.info("\nInterpretación:")
        logger.info(f"Probabilidad de sismo bajo: {prob_baja:.2%}")
        logger.info(f"Probabilidad de sismo medio: {prob_media:.2%}")
        logger.info(f"Probabilidad de sismo alto: {prob_alta:.2%}")

        # 6. Recomendaciones basadas en la probabilidad
        if prob_alta > 0.5:
            logger.info("\nRecomendación: ALERTA - Alta probabilidad de sismo")
        elif prob_media > 0.5:
            logger.info("\nRecomendación: PRECAUCIÓN - Probabilidad media de sismo")
        else:
            logger.info("\nRecomendación: NORMAL - Baja probabilidad de sismo")

    except FileNotFoundError as e:
        logger.error(f"Error: {str(e)}")
        logger.info("Por favor, asegúrate de que el archivo CSV esté en el directorio del proyecto.")
    except Exception as e:
        logger.error(f"Error durante la ejecución: {str(e)}")

if __name__ == "__main__":
    main()