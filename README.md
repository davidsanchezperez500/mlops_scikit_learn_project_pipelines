# Vertex AI Training Pipelines

Este repositorio contiene pipelines para el entrenamiento de modelos de machine learning utilizando Vertex AI en Google Cloud Platform (GCP).

## Descripción
Aquí se definen y gestionan los flujos de trabajo (pipelines) que automatizan el proceso de entrenamiento, validación y despliegue de modelos, aprovechando las capacidades de Vertex AI para orquestar tareas de ML a escala.

## Estructura del repositorio
- `README.md`: Documentación del repositorio.
- Otros archivos y carpetas: Scripts y definiciones de los pipelines.

## ¿Qué es Vertex AI?
Vertex AI es la plataforma de Google Cloud para el desarrollo, entrenamiento, despliegue y gestión de modelos de machine learning de manera escalable y gestionada.

## ¿Qué es un pipeline de entrenamiento?
Un pipeline de entrenamiento es una secuencia de pasos automatizados que incluyen la preparación de datos, entrenamiento, evaluación y, opcionalmente, el despliegue de modelos.

## ¿Cómo usar este repositorio?
1. Clona el repositorio:
   ```bash
   git clone <url-del-repo>
   ```
2. Configura tu entorno de GCP y autentícate:
   ```bash
   gcloud auth login
   gcloud config set project <tu-proyecto>
   ```
3. Personaliza y ejecuta los pipelines según tus necesidades.

## Requisitos
- Tener una cuenta en Google Cloud Platform.
- Acceso a Vertex AI y permisos para crear recursos.
- Python 3.7+
- Instalar dependencias:
  ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
  ```

## Recursos útiles
- [Documentación oficial Vertex AI](https://cloud.google.com/vertex-ai/docs)
- [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/)


