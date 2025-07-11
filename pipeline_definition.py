# pipeline_definition.py

import kfp
from kfp.v2 import compiler
from kfp.v2.dsl import pipeline, component, Output, Metrics, Input
from google.cloud.aiplatform import pipeline_jobs

import os
import time

# ==============================================================================
# 1. Definir las variables de configuración del proyecto y recursos
#    Estas variables se obtendrán del entorno.
# ==============================================================================
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "mlops-training-462812")
REGION = os.environ.get("GCP_REGION", "us-central1")
GCP_VERTEX_BUCKET = os.environ.get("GCP_VERTEX_BUCKET", "mlops-training-models-for-training-46281")
GCP_ARTIFACT_REGISTRY_REPO = os.environ.get("GCP_ARTIFACT_REGISTRY_REPO", "docker-repository")

TRAINING_IMAGE_URI = os.environ.get("TRAINING_IMAGE_URI", f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{GCP_ARTIFACT_REGISTRY_REPO}/house-price-trainer:latest")
GCS_DATA_PATH = os.environ.get("GCS_DATA_PATH", f"gs://{GCP_VERTEX_BUCKET}/data/housing.csv")
GCS_MODEL_OUTPUT_DIR = os.environ.get("GCS_MODEL_OUTPUT_DIR", f"gs://{GCP_VERTEX_BUCKET}/models/house-price-model")
GCS_TEST_DATA_PATH = os.environ.get("GCS_TEST_DATA_PATH", f"gs://{GCP_VERTEX_BUCKET}/data/housing_test.csv")

TRAINING_JOB_SERVICE_ACCOUNT = "808452778180-compute@developer.gserviceaccount.com"
KFP_COMPONENTS_IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{GCP_ARTIFACT_REGISTRY_REPO}/house-price-base:latest"


# ==============================================================================
# 2. Definir componentes del pipeline
# ==============================================================================

@component(
    base_image="gcr.io/ml-pipeline/google-cloud-pipeline-components:1.0.4",
)
def train_model_component_op(
    project: str,
    location: str,
    training_image_uri: str,
    gcs_data_path: str,
    gcs_model_output_dir: str,
    service_account: str,
    job_display_name: str,
    gcp_vertex_bucket: str,
):
    """
    Componente para lanzar el trabajo de entrenamiento personalizado en Vertex AI.
    """
    from google.cloud import aiplatform
    import os

    print(f"Iniciando el componente de entrenamiento para el job: {job_display_name}")
    print(f"Imagen de entrenamiento: {training_image_uri}")
    print(f"Ruta de datos: {gcs_data_path}")
    print(f"Directorio de salida del modelo: {gcs_model_output_dir}")
    print(f"Cuenta de servicio para el job: {service_account}")

    aiplatform.init(
        project=project,
        location=location,
        staging_bucket=f"gs://{gcp_vertex_bucket}/pipeline_staging"
    )
    job = aiplatform.CustomContainerTrainingJob(
        display_name=job_display_name,
        container_uri=training_image_uri,
    )
    job.run(
        args=[
            f"--data-path={gcs_data_path}",
            f"--model-dir={gcs_model_output_dir}"
        ],
        replica_count=1,
        machine_type="n1-standard-4",
        service_account=service_account,
        base_output_dir=gcs_model_output_dir,
        sync=True
    )
    print(f"Componente de entrenamiento {job_display_name} completado (o lanzado).")


@component(
    base_image=KFP_COMPONENTS_IMAGE_URI,
)
def evaluate_and_upload_model_component_op(
    project: str,
    location: str,
    model_gcs_path: str,
    test_data_gcs_path: str,
    model_display_name: str,
    serving_container_image_uri: str,
    metrics: Output[Metrics],
    model_resource_name: Output[str],
):
    """
    Componente para evaluar el modelo y registrarlo en Vertex AI Model Registry.
    Calcula el RMSE y lo asocia al modelo.
    """
    from google.cloud import aiplatform, storage
    import pandas as pd
    import joblib
    from sklearn.metrics import mean_squared_error
    import numpy as np
    import os

    print(f"Iniciando evaluación y registro del modelo: {model_display_name}")
    print(f"Ruta del modelo en GCS: {model_gcs_path}")
    print(f"Ruta de los datos de prueba en GCS: {test_data_gcs_path}")

    # Descargar modelo desde GCS
    client = storage.Client(project=project)
    bucket_name = model_gcs_path.split('/')[2]
    
    blob_path_prefix = '/'.join(model_gcs_path.split('/')[3:])
    blob_path = os.path.join(blob_path_prefix, 'model.joblib')

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    model_file_path = "model.joblib"
    blob.download_to_filename(model_file_path)
    model = joblib.load(model_file_path)
    print("Modelo descargado y cargado exitosamente.")

    # Descargar datos de prueba desde GCS
    test_bucket_name = test_data_gcs_path.split('/')[2]
    test_blob_path = '/'.join(test_data_gcs_path.split('/')[3:])
    test_bucket = client.bucket(test_bucket_name)
    test_blob = test_bucket.blob(test_blob_path)
    test_file_path = "housing_test.csv"
    test_blob.download_to_filename(test_file_path)
    test_df = pd.read_csv(test_file_path)
    print("Datos de prueba descargados y cargados exitosamente.")

    # Preparar datos para evaluación (asumiendo las mismas características)
    EXPECTED_FEATURES = ['bedrooms', 'bathrooms', 'sq_footage']
    target_feature = 'price'

    X_test = test_df[EXPECTED_FEATURES]
    y_test = test_df[target_feature]

    # Realizar predicciones
    predictions = model.predict(X_test)

    # Calcular RMSE
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"RMSE del modelo: {rmse}")

    # Registrar métricas en el output del componente
    metrics.log_metric("rmse", rmse)
    metrics.metadata["rmse_threshold"] = 50000.0

    # Subir el modelo a Vertex AI Model Registry
    aiplatform.init(project=project, location=location)
    uploaded_model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=model_gcs_path,
        serving_container_image_uri=serving_container_image_uri,
        sync=True
    )
    print(f"Modelo registrado en Model Registry: {uploaded_model.resource_name}")
    model_resource_name.write(uploaded_model.resource_name)


@component(
    base_image=KFP_COMPONENTS_IMAGE_URI,
)
def conditional_deploy_model_component_op(
    project: str,
    location: str,
    model_resource_name: str,
    metrics_artifact: Input[Metrics],
    rmse_threshold: float,
    endpoint_display_name: str,
    serving_container_image_uri: str,
    service_account: str,
):
    """
    Componente para desplegar el modelo si el RMSE cumple con el umbral.
    """
    from google.cloud import aiplatform
    import os

    rmse_metric_value = metrics_artifact.get_metric("rmse")

    print(f"Iniciando despliegue condicional del modelo: {model_resource_name}")
    print(f"RMSE del modelo: {rmse_metric_value}, Umbral: {rmse_threshold}")

    if rmse_metric_value <= rmse_threshold:
        print("El RMSE cumple con el umbral. Procediendo al despliegue.")
        aiplatform.init(project=project, location=location)

        model = aiplatform.Model(model_resource_name)

        endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_display_name}"', project=project, location=location)
        if endpoints:
            endpoint = endpoints[0]
            print(f"Endpoint existente encontrado: {endpoint.resource_name}")
        else:
            print(f"Creando nuevo endpoint: {endpoint_display_name}")
            endpoint = aiplatform.Endpoint.create(
                display_name=endpoint_display_name,
                project=project,
                location=location,
                sync=True
            )
            print(f"Endpoint creado: {endpoint.resource_name}")

        print(f"Desplegando modelo {model.display_name} en endpoint {endpoint.display_name}...")
        endpoint.deploy(
            model=model,
            deployed_model_display_name=f"{model.display_name}-deployed",
            machine_type="n1-standard-2",
            min_replica_count=1,
            max_replica_count=1,
            service_account=service_account,
            sync=True
        )
        print("Modelo desplegado exitosamente.")
    else:
        print(f"El RMSE ({rmse_metric_value}) no cumple con el umbral ({rmse_threshold}). No se realizará el despliegue.")


# ==============================================================================
# 3. Definir el Pipeline Principal
# ==============================================================================
@pipeline(
    name="house-price-prediction-training-pipeline",
    description="Pipeline de entrenamiento para el modelo de predicción de precios de casas con evaluación y despliegue condicional.",
    pipeline_root=f"gs://{GCP_VERTEX_BUCKET}/pipelines_root",
)
def house_price_training_pipeline(
    rmse_threshold: float = 50000.0,
    endpoint_display_name: str = "house-price-prediction-endpoint",
    model_display_name: str = "house-price-model-registered",
    serving_container_image_uri: str = "us-central1-docker.pkg.dev/mlops-training-462812/docker-repository/house-price-predictor:latest",
):
    """
    Define el flujo de trabajo completo del pipeline de entrenamiento.
    """
    job_suffix = time.strftime('%Y%m%d-%H%M%S')
    train_job_display_name = f"house-price-train-job-{job_suffix}"
    eval_job_display_name = f"house-price-eval-job-{job_suffix}"
    deploy_job_display_name = f"house-price-deploy-job-{job_suffix}"

    train_task = train_model_component_op(
        project=PROJECT_ID,
        location=REGION,
        training_image_uri=TRAINING_IMAGE_URI,
        gcs_data_path=GCS_DATA_PATH,
        gcs_model_output_dir=GCS_MODEL_OUTPUT_DIR,
        service_account=TRAINING_JOB_SERVICE_ACCOUNT,
        job_display_name=train_job_display_name,
        gcp_vertex_bucket=GCP_VERTEX_BUCKET,
    )

    eval_upload_task = evaluate_and_upload_model_component_op(
        project=PROJECT_ID,
        location=REGION,
        model_gcs_path=GCS_MODEL_OUTPUT_DIR,
        test_data_gcs_path=GCS_TEST_DATA_PATH,
        model_display_name=model_display_name,
        serving_container_image_uri=serving_container_image_uri,
    ).after(train_task)

    conditional_deploy_task = conditional_deploy_model_component_op(
        project=PROJECT_ID,
        location=REGION,
        model_resource_name=eval_upload_task.outputs["model_resource_name"],
        metrics_artifact=eval_upload_task.outputs["metrics"],
        rmse_threshold=rmse_threshold,
        endpoint_display_name=endpoint_display_name,
        serving_container_image_uri=serving_container_image_uri,
        service_account=TRAINING_JOB_SERVICE_ACCOUNT,
    ).after(eval_upload_task)


# ==============================================================================
# 4. Compilar y Ejecutar el Pipeline
# ==============================================================================
if __name__ == "__main__":
    pipeline_json_file = "house_price_training_pipeline.json"

    compiler.Compiler().compile(
        pipeline_func=house_price_training_pipeline,
        package_path=pipeline_json_file,
    )
    print(f"Pipeline compilado a {pipeline_json_file}")

    job = pipeline_jobs.PipelineJob(
        display_name="house-price-training-pipeline-run",
        template_path=pipeline_json_file,
        project=PROJECT_ID,
        location=REGION,
        enable_caching=False,
    )
    print("Lanzando el Pipeline Job: house-price-training-pipeline-run")
    job.run()
    print("Pipeline Job lanzado exitosamente. Revisa la consola de Vertex AI.")
