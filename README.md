# ChatEV — Replicación del paper

**Trabajo Final del Máster de Ciencia de Datos para la asignatura de Modelado Computacional, Simulación y Optimización**
Autoras: Jimena Milla Moreno, Itsaso Ariztimuño Cenoz

Replicación de: *"ChatEV: Predicting electric vehicle charging demand as natural language processing"*
Qu et al., Transportation Research Part D 136 (2024) 104470.

---

## ¿Qué hace este proyecto?

Replica el sistema ChatEV: un modelo T5-small entrenado con meta-aprendizaje (Reptile) que predice la demanda de recarga de vehículos eléctricos en lenguaje natural. Dado el historial de ocupación de una zona de carga, el modelo genera una predicción numérica de ocupación para los próximos 30 minutos.

---

## Estructura del repositorio

```
Trabajo-final-modelado/
│
├── ChatEV_nuevo.ipynb             # Notebook principal: entrenamiento completo
├── chat_interface.ipynb           # Interfaz standalone Gradio (no requiere reentrenar)
├── visualizacion_espacial.ipynb   # Mapas y EDA geoespacial (geopandas + folium)
│
├── datasets/                      # Datos ST-EVCDP (Shenzhen 2022, 247 zonas)
│   ├── information.csv            # Características estáticas por zona
│   ├── occupancy.csv              # Tasa de ocupación (247 × 8640)
│   ├── price.csv                  # Precio de carga (247 × 8640)
│   ├── duration.csv               # Duración de carga (247 × 8640)
│   ├── volume.csv                 # Volumen energético kWh (247 × 8640)
│   ├── adj.csv                    # Matriz de adyacencia (247 × 247, 1006 aristas dirigidas)
│   ├── distance.csv               # Matriz de distancias (247 × 247)
│   ├── time.csv                   # Timestamps (8640 pasos, 5 min)
│   ├── stations.csv               # 1.706 estaciones individuales con lat/lon
│   ├── weather_shenzhen.csv       # Temperatura, humedad, precipitación (Open-Meteo)
│   ├── addresses_shenzhen.csv     # Dirección postal por zona (Nominatim)
│   ├── calendar_features.csv      # Festivos chinos, hora punta, codificación cíclica
│   └── SZ_districts/              # Shapefile geográfico de Shenzhen (EPSG:3857)
│       ├── SZ_districts.shp
│       ├── SZ_districts.dbf
│       ├── SZ_districts.prj
│       └── SZ_districts.shx
│   └── Shenzhen.qgz               # Proyecto QGIS con simbología de distritos
│
├── chatev_model/                  # Modelo entrenado (en Drive si no está local)
│   ├── weights/                   # Pesos T5-small + tokenizer (~242 MB)
│   ├── dataset_arrays.npz         # Arrays numpy preprocesados
│   └── dataset_meta.json          # Hiperparámetros y metadatos del entrenamiento
│
├── fetch_weather.ipynb            # Descarga datos meteorológicos (Open-Meteo API)
├── fetch_address.ipynb            # Geocodificación inversa por zona (Nominatim)
├── fetch_calendar.ipynb           # Generación de features de calendario chino
│
├── figures/                       # Imágenes embebidas en la memoria técnica
│   ├── distribucion_estaciones_shenzhen.png
│   └── estadisticas_dataset.png
│
├── mapa_interactivo.html          # Mapa folium interactivo generado por visualizacion_espacial.ipynb
│
├── memoria_tecnica_ChatEV.pdf     # Memoria técnica completa del proyecto
├── ChatEV.pdf                     # Presentación exportada a PDF
├── ChatEV.pptx                    # Presentación PowerPoint
└── paper_original.pdf             # Artículo original (Qu et al., 2024)
```

**Nota:** `spatial_features_shenzhen.csv` (road length y POI density por zona) no fue generado — el notebook auxiliar correspondiente requería la API Overpass de OpenStreetMap, que no respondió durante la ejecución (timeouts en los 3 endpoints disponibles).

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
4. El entrenamiento tarda ~2–4 horas con GPU T4

### Notebooks auxiliares (opcionales)

Generan los CSVs de datos externos. Ya están generados en `datasets/`, solo re-ejecutar si se quieren actualizar:

| Notebook | Genera |
|---|---|
| `fetch_weather.ipynb` | `weather_shenzhen.csv` |
| `fetch_address.ipynb` | `addresses_shenzhen.csv` |
| `fetch_calendar.ipynb` | `calendar_features.csv` |

---

## Dataset

**ST-EVCDP** — Spatio-Temporal Electric Vehicle Charging Demand Prediction

- 247 zonas de tráfico (TAZ) en Shenzhen, China
- 8.640 pasos temporales (5 min/paso, 30 días: 19 jun – 18 jul 2022)
- 18.061 puntos de carga públicos, 1.706 estaciones
- Fuente: [github.com/IntelligentSystemsLab/ST-EVCDP](https://github.com/IntelligentSystemsLab/ST-EVCDP)

---

## Resultados obtenidos

| Escenario | MAE replicación (×10⁻²) | MAE paper (×10⁻²) | RMSE replicación (×10⁻²) | RMSE paper (×10⁻²) |
|---|---|---|---|---|
| Full-shot | 5,31 | 3,29 | 10,87 | 5,40 |
| Zero-shot | 11,32 | 3,61 | 18,17 | 5,91 |
| Few-shot (prom.) | 2,38 | 3,49 | 5,47 | 5,68 |

La diferencia en full/zero-shot se debe al uso de T5-small (60M parámetros) frente a Sentence-T5 del paper, y a un máximo de 30 épocas Reptile frente a hasta 200 en el paper. El resultado few-shot supera al paper probablemente por evaluación sobre subconjunto reducido (MAX\_EVAL=300).

---

## Referencia

Qu, H., Li, X., Guo, Z., Zhang, J., & Huang, H. (2024). ChatEV: Predicting electric vehicle charging demand as natural language processing. *Transportation Research Part D*, 136, 104470.
