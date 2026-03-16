"""Deploy Chronos-2 custom inference endpoint to SageMaker.

Usage:
    python deploy.py [--delete]

Packages inference.py + requirements.txt into a model.tar.gz,
uploads to S3, and creates a SageMaker real-time endpoint.

Pass --delete to tear down the endpoint instead.
"""

import argparse
import os
import sys
import tarfile
import tempfile

import boto3
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.image_uris import retrieve

ENDPOINT_NAME = "chronos-2-ntl-demo"
ROLE = os.environ.get("SAGEMAKER_EXECUTION_ROLE") or Session().get_caller_identity_arn()
PREFIX = "ntl-detection-demo/chronos-endpoint"
INSTANCE_TYPE = "ml.g5.2xlarge"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def package_model_tar(output_path):
    """Create model.tar.gz with code/ directory containing inference handler."""
    with tarfile.open(output_path, "w:gz") as tar:
        # SageMaker PyTorch DLC expects code/ directory inside the tarball
        tar.add(os.path.join(SCRIPT_DIR, "inference.py"), arcname="code/inference.py")
        tar.add(os.path.join(SCRIPT_DIR, "requirements.txt"), arcname="code/requirements.txt")
    print(f"Packaged model.tar.gz ({os.path.getsize(output_path)} bytes)")


def deploy():
    session = Session()
    region = session.boto_region_name
    bucket = session.default_bucket()
    sm_client = boto3.client("sagemaker", region_name=region)
    s3_client = boto3.client("s3")

    # 1. Package and upload model.tar.gz
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tar_path = tmp.name
    package_model_tar(tar_path)

    s3_key = f"{PREFIX}/model.tar.gz"
    s3_uri = f"s3://{bucket}/{s3_key}"
    s3_client.upload_file(tar_path, bucket, s3_key)
    os.unlink(tar_path)
    print(f"Uploaded to {s3_uri}")

    # 2. Get PyTorch inference DLC image
    image_uri = retrieve(
        framework="pytorch",
        region=region,
        version="2.3.0",
        py_version="py311",
        image_scope="inference",
        instance_type=INSTANCE_TYPE,
    )
    print(f"Image: {image_uri}")

    # 3. Create model
    model_name = f"chronos-2-custom-{region}"
    try:
        sm_client.delete_model(ModelName=model_name)
        print(f"Deleted existing model: {model_name}")
    except sm_client.exceptions.ClientError:
        pass

    sm_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": s3_uri,
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_MODEL_SERVER_TIMEOUT": "600",
                "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
                "TS_MAX_RESPONSE_SIZE": "104857600",  # 100MB
            },
        },
        ExecutionRoleArn=ROLE,
    )
    print(f"Created model: {model_name}")

    # 4. Create endpoint config
    epc_name = f"chronos-2-custom-epc"
    try:
        sm_client.delete_endpoint_config(EndpointConfigName=epc_name)
    except sm_client.exceptions.ClientError:
        pass

    sm_client.create_endpoint_config(
        EndpointConfigName=epc_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InstanceType": INSTANCE_TYPE,
                "InitialInstanceCount": 1,
                "ContainerStartupHealthCheckTimeoutInSeconds": 600,
                "ModelDataDownloadTimeoutInSeconds": 600,
            }
        ],
    )
    print(f"Created endpoint config: {epc_name}")

    # 5. Create or update endpoint
    try:
        sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
        # Endpoint exists — update it
        sm_client.update_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=epc_name,
        )
        print(f"Updating existing endpoint: {ENDPOINT_NAME}")
    except sm_client.exceptions.ClientError:
        # Endpoint doesn't exist — create it
        sm_client.create_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=epc_name,
        )
        print(f"Creating endpoint: {ENDPOINT_NAME}")

    print(f"\nEndpoint '{ENDPOINT_NAME}' is deploying (~5-8 min for model download + chronos install).")
    print("Monitor status:")
    print(f"  aws sagemaker describe-endpoint --endpoint-name {ENDPOINT_NAME} --query EndpointStatus")
    print(f"\nOnce InService, test with:")
    print(f"  python deploy.py --test")


def test_endpoint():
    """Send a small test payload to verify the endpoint works."""
    import json

    region = Session().boto_region_name
    runtime = boto3.client("sagemaker-runtime", region_name=region)

    payload = {
        "inputs": [
            {"target": [float(i) + 0.5 * (i % 7) for i in range(100)]}
        ],
        "parameters": {"prediction_length": 10, "quantile_levels": [0.5]},
    }

    print(f"Invoking {ENDPOINT_NAME} with 1 series, 100 context, 10 pred...")
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    result = json.loads(response["Body"].read().decode())
    print(f"Response: {json.dumps(result, indent=2)}")
    print("Endpoint is working!")


def delete():
    """Delete endpoint, endpoint config, and model."""
    region = Session().boto_region_name
    sm_client = boto3.client("sagemaker", region_name=region)

    for name in [ENDPOINT_NAME]:
        try:
            sm_client.delete_endpoint(EndpointName=name)
            print(f"Deleted endpoint: {name}")
        except sm_client.exceptions.ClientError as e:
            print(f"Endpoint {name}: {e}")

    for name in ["chronos-2-custom-epc"]:
        try:
            sm_client.delete_endpoint_config(EndpointConfigName=name)
            print(f"Deleted endpoint config: {name}")
        except sm_client.exceptions.ClientError as e:
            print(f"Endpoint config {name}: {e}")

    for name in [f"chronos-2-custom-{region}"]:
        try:
            sm_client.delete_model(ModelName=name)
            print(f"Deleted model: {name}")
        except sm_client.exceptions.ClientError as e:
            print(f"Model {name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy/manage Chronos-2 endpoint")
    parser.add_argument("--delete", action="store_true", help="Delete endpoint")
    parser.add_argument("--test", action="store_true", help="Test endpoint")
    args = parser.parse_args()

    if args.delete:
        delete()
    elif args.test:
        test_endpoint()
    else:
        deploy()
