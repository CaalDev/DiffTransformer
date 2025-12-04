# DiffTransformer
# Differential Transformer for Hallucination Mitigation


![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1AN4GwY19-fQUqQvBwzcWgXdAqkF2sh-P/view?usp=drive_link)

Proyecto académico que implementa y evalúa el mecanismo de **atención diferencial** (Differential Attention) propuesto en el paper:

> **"Differential Transformer"** — Ye et al., 2024  
> Paper: https://arxiv.org/abs/2410.05258

Este trabajo demuestra cómo la atención diferencial reduce la asignación de atención a contexto irrelevante ("ruido") y mitiga alucinaciones en tareas de *question answering* con contextos largos.

---

### Equipo
- Maria Liliana Parra Osorno  
- Daniel Castañeda Montenegro  
- Carlos Andrés Aguirre López  

**Profesor:** Alcides Montoya Cañola  
**Universidad Nacional de Colombia - Medellín**  
**Noviembre 2025**

---

### Objetivo
Implementar y comparar el **Differential Transformer** frente al Transformer estándar, evaluando:
- Reducción de alucinaciones en tareas QA
- Mejor manejo de contextos largos
- Emergencia de patrones de atención más esparsos y focalizados
- Disminución de outliers en activaciones

---

### Concepto Clave: Atención Diferencial

> Es como **audífonos con cancelación de ruido**

```math
\text{DiffAttn}(Q,K,V) = \left( \text{softmax}\left(\frac{Q_1 K_1^T}{\sqrt{d}}\right) - \lambda \cdot \text{softmax}\left(\frac{Q_2 K_2^T}{\sqrt{d}}\right) \right) V
```

- Primer término: Captura señal + ruido
- Segundo término: Captura principalmente ruido
- λ (aprendible): Controla la intensidad de cancelación
- Resultado: Se amplifica la señal relevante y se suprime el ruido

## 📑 Estructura de Proyecto de Tesis

Este índice detalla un enfoque robusto y experimental para la mitigación de alucinaciones en modelos de lenguaje utilizando el mecanismo de **Atención Diferencial (DiffAttn)**.

| Sección | Título de la Sección | Descripción/Contenido Clave |
| :--- | :--- | :--- |
| **1.** | **Introducción y Objetivos** | Contexto del problema de la alucinación en LLMs, relevancia de la mitigación, y presentación de los objetivos específicos del proyecto. |
| **2.** | **Marco Teórico** | Fundamentos del mecanismo de **Atención Estándar (Transformer)** y la justificación teórica del mecanismo **Atención Diferencial (DiffAttn)** como filtro de ruido y sesgos. |
| **3.** | **Configuración del Entorno** | Detalles de la infraestructura de hardware (ej. uso de **GPU A100**), versiones de frameworks (**PyTorch/TensorFlow**), y configuración del entorno de desarrollo. |
| **4.** | **Atención Estándar (Baseline)** | Implementación de un modelo **Transformer estándar (Baseline)** y descripción del conjunto de datos inicial para establecer la métrica de rendimiento a superar. |
| **5.** | **Atención Diferencial (DiffAttn)** | Implementación y detalles técnicos del mecanismo **DiffAttn** (incluyendo las dos cabezas de atención y el factor de cancelación $\lambda$) dentro del modelo Transformer. |
| **6.** | **Experimentos Sintéticos** | Diseño de pruebas controladas para evaluar la robustez del modelo al **ruido explícito o *inputs* contradictorios**, demostrando la capacidad de cancelación de $\lambda$. |
| **7.** | **Visualización y Análisis** | Generación de **Mapas de Atención** comparativos (Estándar vs. Diferencial) para analizar la **sparsidad** y la focalización en *tokens* relevantes. |
| **8.** | **Evaluación Long-Context** | Comparación del rendimiento y la estabilidad (tasa de alucinación) de ambos modelos en escenarios de **contextos largos** (*Long-Context Evaluation*). |
| **9.** | **Análisis de Outliers** | Cuantificación de la **reducción de activaciones extremas (outliers)** en las matrices de atención y su correlación con la mitigación de alucinaciones. |
| **10.** | **Métricas de Evaluación** | Detalle de las métricas utilizadas: **Exact Match (EM)** y **ROUGE** para la fidelidad, y métricas específicas para medir la **atención focalizada** y la tasa de alucinación. |
| **11.** | **Conclusiones y Trabajo Futuro** | Resumen de los resultados clave, validación de la hipótesis, y propuesta de líneas de investigación futuras (ej. aplicación a modelos LLM a gran escala). |

# Resultados Principales (Preliminares)

- Reducción significativa de atención a tokens irrelevantes
- Disminución de alucinaciones en contextos con distracción
- Patrones de atención más esparsos y focalizados
- Menor presencia de outliers en las activaciones
- Mejor rendimiento en tareas de "needle in a haystack" con contextos >8k tokens

# Cómo Ejecutar

Puedes abrir directamente en Google Colab:
Open In Colab
O clonar y ejecutar localmente:

git clone https://github.com/tu-usuario/differential-transformer-dcm.git
cd differential-transformer-dcm
jupyter notebook Differential_Transformer_DCM.ipynb

pip install torch transformers datasets rouge-score matplotlib seaborn

# Visualizaciones Destacadas

<img width="1544" height="990" alt="image" src="https://github.com/user-attachments/assets/9315780b-4b25-4779-a901-24301cf42d85" />
Mapa de atención: Transformer estándar (izq.) vs. Differential Transformer (der.)

# Licencia
Universidad Nacional de Colombia - Sede Medellín
Departamento de Física — Procesamiento de Lenguaje Natural con Transformers
