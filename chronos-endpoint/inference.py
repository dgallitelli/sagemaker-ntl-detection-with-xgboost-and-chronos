"""Custom inference handler for Chronos-2 on SageMaker PyTorch DLC.

Loads amazon/chronos-bolt-base from HuggingFace at startup and serves
forecasts in the same payload format as the JumpStart Chronos-2 endpoint.

Uses BaseChronosPipeline from the chronos-forecasting package.
See: https://huggingface.co/amazon/chronos-bolt-base

Expected input (application/json):
    {
        "inputs": [{"target": [1.0, 2.0, ...]}, ...],
        "parameters": {"prediction_length": 64, "quantile_levels": [0.5]}
    }

Expected output (application/json):
    {
        "predictions": [
            {"0.5": [1.1, 2.2, ...], "mean": [1.1, 2.2, ...]},
            ...
        ]
    }
"""

import json
import logging

import torch

logger = logging.getLogger(__name__)


def model_fn(model_dir):
    """Load Chronos-2 pipeline from HuggingFace Hub."""
    from chronos import BaseChronosPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading Chronos-bolt-base on device: {device}")

    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-base",
        device_map=device,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    logger.info("Chronos pipeline loaded successfully")
    return pipeline


def input_fn(request_body, request_content_type):
    """Parse JSON input into (input_tensors, parameters)."""
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")

    payload = json.loads(request_body)
    inputs = payload.get("inputs", [])
    parameters = payload.get("parameters", {})

    # ChronosBoltPipeline.predict accepts: 1D tensor, list of 1D tensors,
    # or left-padded 2D tensor (batch-first)
    input_tensors = [
        torch.tensor(item["target"], dtype=torch.float32)
        for item in inputs
    ]

    prediction_length = parameters.get("prediction_length", 64)
    quantile_levels = parameters.get("quantile_levels", [0.5])

    return {
        "inputs": input_tensors,
        "prediction_length": prediction_length,
        "quantile_levels": quantile_levels,
    }


# Chronos-Bolt always outputs 9 fixed quantiles: [0.1, 0.2, ..., 0.9]
BOLT_QUANTILES = [round(0.1 * i, 1) for i in range(1, 10)]


def predict_fn(input_data, model):
    """Run Chronos-Bolt forecast.

    ChronosBoltPipeline.predict returns shape:
        (batch_size, 9, prediction_length)
    where the 9 quantiles are fixed at [0.1, 0.2, ..., 0.9].
    """
    inputs = input_data["inputs"]
    prediction_length = input_data["prediction_length"]
    quantile_levels = input_data["quantile_levels"]

    # Returns (n_series, 9, prediction_length)
    forecasts = model.predict(
        inputs=inputs,
        prediction_length=prediction_length,
    )

    # Move to CPU numpy
    forecasts_np = forecasts.cpu().float().numpy()

    # Extract requested quantiles from the fixed 9
    requested_indices = []
    for q in quantile_levels:
        if q in BOLT_QUANTILES:
            requested_indices.append(BOLT_QUANTILES.index(q))
        else:
            # Find nearest quantile
            nearest_idx = min(range(len(BOLT_QUANTILES)),
                              key=lambda i: abs(BOLT_QUANTILES[i] - q))
            requested_indices.append(nearest_idx)

    selected_forecasts = forecasts_np[:, requested_indices, :]

    # Median (0.5) is at index 4 in [0.1, ..., 0.9]
    mean_values = forecasts_np[:, 4, :]

    return {
        "quantile_forecasts": selected_forecasts,
        "mean_values": mean_values,
        "quantile_levels": quantile_levels,
    }


def output_fn(prediction, response_content_type):
    """Format output to match JumpStart Chronos-2 response format."""
    quantile_forecasts = prediction["quantile_forecasts"]
    mean_values = prediction["mean_values"]
    quantile_levels = prediction["quantile_levels"]

    n_series = quantile_forecasts.shape[0]
    predictions = []

    for i in range(n_series):
        pred_entry = {}
        for q_idx, q_level in enumerate(quantile_levels):
            pred_entry[str(q_level)] = quantile_forecasts[i, q_idx, :].tolist()
        pred_entry["mean"] = mean_values[i].tolist()
        predictions.append(pred_entry)

    return json.dumps({"predictions": predictions})
