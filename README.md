# ChatEV — Replicación del paper

**Trabajo Final del Máster de Ciencia de Datos para la asignatura de Modelado Computacional, Simulación y Optimización**
Replicación de: *"ChatEV: Predicting electric vehicle charging demand as natural language processing"*
Qu et al., Transportation Research Part D 136 (2024) 104470.

---

## ¿Qué hace este proyecto?

Replica el sistema ChatEV: un modelo T5-small entrenado con meta-aprendizaje (Reptile) que predice la demanda de recarga de vehículos eléctricos en lenguaje natural. Dado el historial de ocupación de una zona de carga, el modelo genera una predicción numérica de ocupación para los próximos 30 minutos.

---

## Estructura del repositorio

Trabajo-final-modelado/
│
├── ChatEV_nuevo.ipynb          # Notebook principal: entrenamiento completo
├── chat_interface.ipynb        # Interfaz standalone (no requiere reentrenar)
│
├── datasets/                   # Datos ST-EVCDP (Shenzhen 2022, 247 zonas)
│   ├── information.csv         # Características estáticas por zona
│   ├── occupancy.csv           # Tasa de ocupación (247 × 8640)
│   ├── price.csv               # Precio de carga (247 × 8640)
│   ├── duration.csv            # Duración de carga (247 × 8640)
│   ├── volume.csv              # Volumen energético (247 × 8640)
│   ├── adj.csv                 # Matriz de adyacencia (247 × 247)
│   ├── distance.csv            # Matriz de distancias (247 × 247)
│   ├── time.csv                # Timestamps (8640 pasos, 5 min)
│   ├── weather_shenzhen.csv    # Temperatura, humedad, precipitación
│   ├── addresses_shenzhen.csv  # Dirección postal por zona
│   ├── calendar_features.csv   # Festivos chinos, hora punta eléctrica
│   └── SZ_districts/           # Shapefile geográfico de Shenzhen
│
├── chatev_model/               # Modelo entrenado (excluido de git, en Drive)
│   ├── weights/                # Pesos T5 + tokenizer (model.safetensors, ~242 MB)
│   ├── dataset_arrays.npz      # Arrays numpy preprocesados
│   └── dataset_meta.json       # Hiperparámetros y metadatos del entrenamiento
│
├── fetch_weather.ipynb         # Descarga datos meteorológicos (Open-Meteo API)
├── fetch_address.ipynb         # Geocodificación inversa por zona
├── fetch_calendar.ipynb        # Generación de features de calendario chino
├── fetch_spatial_features.ipynb # Road length y POI count por zona
├── visualizacion_espacial.ipynb # Mapas coropléticos de ocupación
│
├── ChatEV_memoria.docx         # Versión Word de la memoria
└── 1-s2.0-S1361920924004279-main.pdf  # Paper original

---

## Cómo ejecutar

### Opción A — Ver la interfaz sin reentrenar (recomendado para evaluación)

1. Abrir [`chat_interface.ipynb`](chat_interface.ipynb) en Google Colab
2. Ejecutar todas las celdas (Entorno de ejecución → Ejecutar todo)
3. Primera ejecución descarga ~250 MB automáticamente desde Drive público:
   `https://drive.google.com/drive/folders/1eedMIiny2QJVYhbLFUPWUqEnjz8kuj2q`
4. Al final aparece un link `https://xxxx.gradio.live` — funciona para cualquiera sin login

### Opción B — Entrenamiento completo

1. Abrir [`ChatEV_nuevo.ipynb`](ChatEV_nuevo.ipynb) en Google Colab con GPU (T4 o superior)
2. Montar Google Drive con la carpeta `datasets/` en `MyDrive/datasets/`
3. Ejecutar todas las celdas en orden
4. El entrenamiento tarda ~2-4 horas con GPU T4

### Notebooks auxiliares (opcionales)

Generan los CSVs de datos externos. Ya están generados en `datasets/`, solo necesario re-ejecutar si se quieren actualizar:

| Notebook                         | Genera                            |
| -------------------------------- | --------------------------------- |
| `fetch_weather.ipynb`          | `weather_shenzhen.csv`          |
| `fetch_address.ipynb`          | `addresses_shenzhen.csv`        |
| `fetch_calendar.ipynb`         | `calendar_features.csv`         |
| `fetch_spatial_features.ipynb` | `spatial_features_shenzhen.csv` |

---

## Dataset

**ST-EVCDP** — Spatio-Temporal Electric Vehicle Charging Demand Prediction

- 247 zonas de tráfico (TAZ) en Shenzhen, China
- 8.640 pasos temporales (5 min/paso, 30 días: 19 jun – 18 jul 2022)
- Fuente: [github.com/IntelligentSystemsLab/ST-EVCDP](https://github.com/IntelligentSystemsLab/ST-EVCDP)

---

## Resultados obtenidos

| Métrica      | Full-shot      | Zero-shot      |
| ------------- | -------------- | -------------- |
| MAE (nuestro) | 7.27 × 10⁻² | 7.30 × 10⁻² |
| MAE (paper)   | 3.29 × 10⁻² | 3.61 × 10⁻² |

La diferencia se debe a limitaciones de hardware (GPU T4 vs servidores industriales) y uso de T5-small en lugar del modelo completo Sentence-T5.

---

## Referencia

Qu, H., Li, X., Guo, Z., Zhang, J., & Huang, H. (2024). ChatEV: Predicting electric vehicle charging demand as natural language processing. *Transportation Research Part D*, 136, 104470.
