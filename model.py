from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import logging
from itertools import product

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Definir el modelo con estructura mejorada
    model = DiscreteBayesianNetwork()
    model.add_nodes_from([
        "magnitud_historica",
        "profundidad_sismica",
        "tiempo_ultimo_sismo",
        "actividad_falla",
        "patron_sismico",
        "intensidad_historica",
        "frecuencia_mensual",
        "probabilidad_sismo"
    ])

    # Estructura más realista con relaciones intermedias
    model.add_edges_from([
        ("magnitud_historica", "intensidad_historica"),
        ("profundidad_sismica", "intensidad_historica"),
        ("actividad_falla", "patron_sismico"),
        ("actividad_falla", "frecuencia_mensual"),
        ("magnitud_historica", "probabilidad_sismo"),
        ("profundidad_sismica", "probabilidad_sismo"),
        ("tiempo_ultimo_sismo", "probabilidad_sismo"),
        ("actividad_falla", "probabilidad_sismo"),
        ("patron_sismico", "probabilidad_sismo"),
        ("intensidad_historica", "probabilidad_sismo"),
        ("frecuencia_mensual", "probabilidad_sismo")
    ])

    # Función para calcular probabilidades de manera más precisa
    def calcular_probabilidades(magnitud, profundidad, tiempo, falla, patron, intensidad, frecuencia):
        """
        Calcula probabilidades basadas en evidencia específica.
        Retorna [prob_baja, prob_media, prob_alta]
        """
        # Base inicial de probabilidad
        prob_base = [0.6, 0.3, 0.1]  # [baja, media, alta]
        
        # Factores de ajuste para cada variable
        factores = {
            "magnitud": {"baja": [0.1, -0.05, -0.05], "media": [0, 0, 0], "alta": [-0.2, 0, 0.2], "desconocida": [0, 0, 0]},
            "profundidad": {"superficial": [-0.1, 0, 0.1], "intermedia": [0, 0, 0], "profunda": [0.1, 0, -0.1], "desconocida": [0, 0, 0]},
            "tiempo": {"reciente": [-0.1, 0, 0.1], "medio": [0, 0, 0], "lejano": [0.1, 0, -0.1], "desconocido": [0, 0, 0]},
            "falla": {"baja": [0.1, 0, -0.1], "media": [0, 0, 0], "alta": [-0.15, 0, 0.15]},
            "patron": {"esporádico": [0.1, 0, -0.1], "regular": [0, 0, 0], "frecuente": [-0.15, 0, 0.15], "desconocido": [0, 0, 0]},
            "intensidad": {"baja": [0.1, 0, -0.1], "media": [0, 0, 0], "alta": [-0.15, 0, 0.15], "desconocida": [0, 0, 0]},
            "frecuencia": {"baja": [0.1, 0, -0.1], "media": [0, 0, 0], "alta": [-0.15, 0, 0.15], "desconocida": [0, 0, 0]},
        }
        
        # Aplicar factores de ajuste
        ajuste = [0, 0, 0]
        ajuste = [x + y for x, y in zip(ajuste, factores["magnitud"][magnitud])]
        ajuste = [x + y for x, y in zip(ajuste, factores["profundidad"][profundidad])]
        ajuste = [x + y for x, y in zip(ajuste, factores["tiempo"][tiempo])]
        ajuste = [x + y for x, y in zip(ajuste, factores["falla"][falla])]
        ajuste = [x + y for x, y in zip(ajuste, factores["patron"][patron])]
        ajuste = [x + y for x, y in zip(ajuste, factores["intensidad"][intensidad])]
        ajuste = [x + y for x, y in zip(ajuste, factores["frecuencia"][frecuencia])]
        
        # Calcular probabilidades finales
        probs = [max(0, min(1, p + a)) for p, a in zip(prob_base, ajuste)]
        
        # Normalizar para asegurar que sumen 1
        suma = sum(probs)
        probs = [p / suma for p in probs]
        
        return probs

    # Definir estados posibles
    magnitudes = ["baja", "media", "alta", "desconocida"]
    profundidades = ["superficial", "intermedia", "profunda", "desconocida"]
    tiempos = ["reciente", "medio", "lejano", "desconocido"]
    fallas = ["baja", "media", "alta"]
    patrones = ["esporádico", "regular", "frecuente", "desconocido"]
    intensidades = ["baja", "media", "alta", "desconocida"]
    frecuencias = ["baja", "media", "alta", "desconocida"]

    # Generar todas las combinaciones y sus probabilidades
    full_probs = []
    for comb in product(magnitudes, profundidades, tiempos, fallas, patrones, intensidades, frecuencias):
        probs = calcular_probabilidades(*comb)
        full_probs.append(probs)

    # Transponer para formato de pgmpy
    full_probs_transposed = list(map(list, zip(*full_probs)))

    # CPDs para las variables
    cpd_magnitud = TabularCPD(
        variable="magnitud_historica", variable_card=4,
        values=[[0.3], [0.4], [0.2], [0.1]],
        state_names={"magnitud_historica": magnitudes}
    )

    cpd_profundidad = TabularCPD(
        variable="profundidad_sismica", variable_card=4,
        values=[[0.35], [0.35], [0.2], [0.1]],
        state_names={"profundidad_sismica": profundidades}
    )

    cpd_tiempo = TabularCPD(
        variable="tiempo_ultimo_sismo", variable_card=4,
        values=[[0.3], [0.4], [0.2], [0.1]],
        state_names={"tiempo_ultimo_sismo": tiempos}
    )

    cpd_falla = TabularCPD(
        variable="actividad_falla", variable_card=3,
        values=[[0.2], [0.5], [0.3]],
        state_names={"actividad_falla": fallas}
    )

    # CPD para patron_sismico (depende de actividad_falla)
    cpd_patron = TabularCPD(
        variable="patron_sismico", variable_card=4,
        values=[
            [0.6, 0.3, 0.1],  # esporádico
            [0.3, 0.4, 0.2],  # regular
            [0.1, 0.2, 0.4],  # frecuente
            [0.0, 0.1, 0.3]   # desconocido
        ],
        evidence=["actividad_falla"],
        evidence_card=[3],
        state_names={
            "patron_sismico": patrones,
            "actividad_falla": fallas
        }
    )

    # CPD para intensidad_historica (depende de magnitud_historica y profundidad_sismica)
    cpd_intensidad = TabularCPD(
        variable="intensidad_historica", variable_card=4,
        values=[
            [0.7, 0.6, 0.5, 0.4, 0.5, 0.4, 0.3, 0.2, 0.3, 0.2, 0.1, 0.1, 0.4, 0.3, 0.2, 0.1],  # baja
            [0.2, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.3, 0.4, 0.4, 0.3, 0.2, 0.3, 0.4, 0.3, 0.2],  # media
            [0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.5, 0.4, 0.2, 0.2, 0.3, 0.3],  # alta
            [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.1, 0.3, 0.1, 0.1, 0.2, 0.4]   # desconocida
        ],
        evidence=["magnitud_historica", "profundidad_sismica"],
        evidence_card=[4, 4],
        state_names={
            "intensidad_historica": intensidades,
            "magnitud_historica": magnitudes,
            "profundidad_sismica": profundidades
        }
    )

    # CPD para frecuencia_mensual (depende de actividad_falla)
    cpd_frecuencia = TabularCPD(
        variable="frecuencia_mensual", variable_card=4,
        values=[
            [0.7, 0.3, 0.1],  # baja
            [0.2, 0.4, 0.2],  # media
            [0.1, 0.2, 0.4],  # alta
            [0.0, 0.1, 0.3]   # desconocida
        ],
        evidence=["actividad_falla"],
        evidence_card=[3],
        state_names={
            "frecuencia_mensual": frecuencias,
            "actividad_falla": fallas
        }
    )

    # CPD para probabilidad_sismo usando las probabilidades calculadas
    cpd_probabilidad = TabularCPD(
        variable="probabilidad_sismo", variable_card=3,
        values=full_probs_transposed,
        evidence=["magnitud_historica", "profundidad_sismica", "tiempo_ultimo_sismo", "actividad_falla", "patron_sismico", "intensidad_historica", "frecuencia_mensual"],
        evidence_card=[4, 4, 4, 3, 4, 4, 4],
        state_names={
            "probabilidad_sismo": ["baja", "media", "alta"],
            "magnitud_historica": magnitudes,
            "profundidad_sismica": profundidades,
            "tiempo_ultimo_sismo": tiempos,
            "actividad_falla": fallas,
            "patron_sismico": patrones,
            "intensidad_historica": intensidades,
            "frecuencia_mensual": frecuencias
        }
    )

    # Agregar CPDs al modelo
    model.add_cpds(
        cpd_magnitud, cpd_profundidad, cpd_tiempo, cpd_falla, 
        cpd_patron, cpd_intensidad, cpd_frecuencia, cpd_probabilidad
    )

    # Verificar que el modelo es válido
    if not model.check_model():
        raise ValueError("El modelo no es válido")

    # Inferencia
    inference = VariableElimination(model)

    # Ejemplo de inferencia con evidencia parcial
    evidence = {
        "magnitud_historica": "alta",
        "profundidad_sismica": "superficial",
        "tiempo_ultimo_sismo": "lejano",
        "actividad_falla": "alta",
        "patron_sismico": "frecuente",
        "intensidad_historica": "alta",
        "frecuencia_mensual": "alta"
    }

    result = inference.query(variables=["probabilidad_sismo"], evidence=evidence)
    logger.info("\nProbabilidad de sismo dado:")
    for var, val in evidence.items():
        logger.info(f"{var}: {val}")
    logger.info("\nResultado:")
    logger.info(result)

except Exception as e:
    logger.error(f"Error en el modelo: {str(e)}")
    raise
