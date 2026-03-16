"""Shared utilities for NTL detection demo notebooks.

Data loading, preprocessing, feature engineering, and SageMaker helpers.
"""
import json
import os
import tarfile
from pathlib import Path

import boto3
import joblib
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import holidays as holidays_pkg
from sklearn.model_selection import train_test_split

# SageMaker SDK v3 imports
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.image_uris import retrieve
from sagemaker.train.configs import Compute, InputData, OutputDataConfig, SourceCode
from sagemaker.train.model_trainer import ModelTrainer

# --- Constants ---
ROLE = os.environ.get("SAGEMAKER_EXECUTION_ROLE") or Session().get_caller_identity_arn()
PREFIX = "ntl-detection-demo"
RESULTS_DIR = Path(__file__).parent / "results"
TRAINING_DIR = Path(__file__).parent / "training"


def get_sagemaker_session():
    """Initialize SageMaker session and return session, region, bucket."""
    session = Session()
    region = session.boto_region_name
    bucket = session.default_bucket()
    return session, region, bucket


# --- Data Loading ---

def load_sgcc_dataset():
    """Download SGCC dataset from Kaggle and return raw DataFrame."""
    import kagglehub

    dataset_path = kagglehub.dataset_download("chabdullah31222/sgcc-electricity-theft-detection-dataset")
    csv_files = list(Path(dataset_path).rglob("*.csv"))
    raw_df = pd.read_csv(csv_files[0])
    print(f"SGCC dataset: {raw_df.shape[0]} customers, {raw_df.shape[1] - 2} daily readings")
    print(f"Theft rate: {raw_df['FLAG'].mean():.1%}")
    return raw_df


def preprocess(raw_df, test_size=0.2, random_state=42):
    """Clean consumption data and split into train/test.

    Returns:
        customer_ids, labels, consumption_clean, date_columns, train_idx, test_idx
    """
    customer_ids = raw_df["CONS_NO"]
    labels = raw_df["FLAG"]
    consumption = raw_df.drop(columns=["CONS_NO", "FLAG"])
    date_columns = pd.to_datetime(consumption.columns, format="mixed")
    consumption.columns = date_columns

    # Interpolate missing, clip negatives
    consumption_clean = consumption.interpolate(method="linear", axis=1).fillna(0).clip(lower=0)

    # Stratified train/test split on customers
    train_idx, test_idx = train_test_split(
        np.arange(len(labels)), test_size=test_size, random_state=random_state, stratify=labels
    )
    print(f"Train: {len(train_idx)} customers | Test: {len(test_idx)} customers")
    return customer_ids, labels, consumption_clean, date_columns, train_idx, test_idx


# --- External Data Helpers ---

def get_holiday_mask(dates, country="CN"):
    """Return boolean mask for holidays in the given date range."""
    years = sorted(set(dates.year))
    cal = holidays_pkg.country_holidays(country, years=years)
    return np.array([d in cal for d in dates.date])


def get_simulated_weather(dates, base_temp=15.0, amplitude=15.0, noise_std=3.0, seed=42):
    """Simulate daily temperature with seasonal sinusoidal pattern.

    Defaults approximate temperate Chinese climate (SGCC dataset).
    For tropical (Malaysia): base_temp=28, amplitude=3, noise_std=1.5.
    """
    rng = np.random.RandomState(seed)
    doy = dates.dayofyear.values
    seasonal = base_temp + amplitude * np.sin(2 * np.pi * (doy - 80) / 365)
    return seasonal + rng.normal(0, noise_std, len(dates))


def _batch_correlation(matrix, vector):
    """Pearson correlation of each row in matrix with a 1-D vector."""
    m_c = matrix - matrix.mean(axis=1, keepdims=True)
    v_c = vector - vector.mean()
    denom = np.sqrt((m_c ** 2).sum(axis=1) * (v_c ** 2).sum() + 1e-12)
    return (m_c @ v_c) / denom


# --- Feature Engineering ---

def compute_baseline_features(cons_matrix):
    """Approach A: Statistical features on raw consumption."""
    values = cons_matrix.values
    features = pd.DataFrame(index=cons_matrix.index)
    features["mean"] = np.nanmean(values, axis=1)
    features["std"] = np.nanstd(values, axis=1)
    features["max"] = np.nanmax(values, axis=1)
    features["min"] = np.nanmin(values, axis=1)
    features["range"] = features["max"] - features["min"]
    features["median"] = np.nanmedian(values, axis=1)
    features["skew"] = pd.DataFrame(values).apply(lambda x: x.skew(), axis=1).values
    features["kurtosis"] = pd.DataFrame(values).apply(lambda x: x.kurtosis(), axis=1).values
    features["zero_days"] = (values == 0).sum(axis=1)
    features["pct_above_2std"] = (
        (values > (features["mean"].values[:, None] + 2 * features["std"].values[:, None])).sum(axis=1)
    ) / values.shape[1]

    # Coefficient of variation (normalizes variability by consumption level)
    features["cv"] = features["std"] / (features["mean"] + 1e-8)

    # Robust dispersion: IQR and percentiles
    features["p10"] = np.nanpercentile(values, 10, axis=1)
    features["p25"] = np.nanpercentile(values, 25, axis=1)
    features["p75"] = np.nanpercentile(values, 75, axis=1)
    features["p90"] = np.nanpercentile(values, 90, axis=1)
    features["iqr"] = features["p75"] - features["p25"]

    # Fraction below 2 std (theft causes low anomalies)
    features["pct_below_2std"] = (
        (values < (features["mean"].values[:, None] - 2 * features["std"].values[:, None])).sum(axis=1)
    ) / values.shape[1]

    # Longest consecutive zero-day streak
    longest_zero_runs = []
    for i in range(values.shape[0]):
        is_zero = values[i] == 0
        max_run = current_run = 0
        for v in is_zero:
            if v:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        longest_zero_runs.append(max_run)
    features["longest_zero_run"] = longest_zero_runs

    # Lag-1 autocorrelation (theft breaks day-to-day regularity)
    diffs = values[:, 1:] - values[:, :-1]
    features["lag1_autocorr"] = np.array([
        np.corrcoef(values[i, :-1], values[i, 1:])[0, 1]
        if np.std(values[i]) > 1e-8 else 0.0
        for i in range(values.shape[0])
    ])

    # Mean absolute successive difference
    features["mean_abs_diff"] = np.nanmean(np.abs(diffs), axis=1)

    return features.fillna(0)


def compute_enhanced_features(cons_matrix):
    """Approach B: Baseline + temporal patterns, periodicity, trend, external data."""
    features = compute_baseline_features(cons_matrix)
    values = cons_matrix.values
    dates = cons_matrix.columns

    # Day-of-week patterns (ratios only — raw means are redundant with global mean)
    dow = dates.dayofweek.values
    weekday_mask = dow < 5
    weekend_mask = dow >= 5
    weekday_mean = pd.Series(np.nanmean(values[:, weekday_mask], axis=1)).replace(0, np.nan)
    weekend_mean = pd.Series(np.nanmean(values[:, weekend_mask], axis=1)).replace(0, np.nan)
    features["weekend_weekday_ratio"] = (weekend_mean / weekday_mean).fillna(1.0).values

    # Monthly CV
    months = dates.to_period("M")
    monthly_means = []
    for period in months.unique():
        mask = months == period
        monthly_means.append(np.nanmean(values[:, mask], axis=1))
    monthly_arr = np.column_stack(monthly_means)
    features["monthly_cv"] = np.nanstd(monthly_arr, axis=1) / (np.nanmean(monthly_arr, axis=1) + 1e-8)

    # Rolling window volatility
    n_windows = values.shape[1] // 30
    window_means = []
    for i in range(n_windows):
        start = i * 30
        window_means.append(np.nanmean(values[:, start : start + 30], axis=1))
    window_arr = np.column_stack(window_means)
    features["rolling_volatility"] = np.nanstd(window_arr, axis=1)

    # Trend (slope of linear fit)
    x = np.arange(values.shape[1])
    slopes = []
    for i in range(values.shape[0]):
        valid = ~np.isnan(values[i])
        if valid.sum() > 10:
            slope = np.polyfit(x[valid], values[i][valid], 1)[0]
        else:
            slope = 0.0
        slopes.append(slope)
    features["trend_slope"] = slopes

    # Trend acceleration (second-half slope minus first-half slope — captures theft onset)
    half = values.shape[1] // 2
    x_half = np.arange(half)
    trend_accels = []
    for i in range(values.shape[0]):
        v1, v2 = values[i, :half], values[i, half:half * 2]
        valid1, valid2 = ~np.isnan(v1), ~np.isnan(v2)
        if valid1.sum() > 10 and valid2.sum() > 10:
            s1 = np.polyfit(x_half[valid1], v1[valid1], 1)[0]
            s2 = np.polyfit(x_half[valid2], v2[valid2], 1)[0]
            trend_accels.append(s2 - s1)
        else:
            trend_accels.append(0.0)
    features["trend_acceleration"] = trend_accels

    # Periodicity: generic top-5 FFT + targeted weekly (1/7 frequency)
    fft_power_ratios = []
    weekly_power_ratios = []
    for i in range(values.shape[0]):
        series = np.nan_to_num(values[i].copy(), nan=0.0)
        if series.std() < 1e-8:
            fft_power_ratios.append(0.0)
            weekly_power_ratios.append(0.0)
            continue
        fft_vals = np.fft.rfft(series - series.mean())
        power = np.abs(fft_vals) ** 2
        total_power = power[1:].sum() + 1e-8
        top5_power = np.sort(power[1:])[-5:].sum()
        fft_power_ratios.append(top5_power / total_power)
        # Weekly periodicity: power at frequency closest to 1/7
        freqs = np.fft.rfftfreq(len(series), d=1.0)
        weekly_idx = np.argmin(np.abs(freqs[1:] - 1.0 / 7.0)) + 1
        weekly_power_ratios.append(power[weekly_idx] / total_power)
    features["periodicity_strength"] = fft_power_ratios
    features["weekly_periodicity"] = weekly_power_ratios

    # Consumption entropy (fixed bin edges for cross-customer comparability)
    entropies = []
    global_max = np.nanpercentile(values, 99)  # cap at 99th to avoid outlier-driven bins
    bin_edges = np.linspace(0, max(global_max, 1.0), 21)
    for i in range(values.shape[0]):
        series = np.nan_to_num(values[i].copy(), nan=0.0)
        hist, _ = np.histogram(series, bins=bin_edges)
        hist = hist[hist > 0]
        if len(hist) > 1:
            probs = hist / hist.sum()
            entropies.append(sp_stats.entropy(probs))
        else:
            entropies.append(0.0)
    features["consumption_entropy"] = entropies

    # Lag-7 autocorrelation (weekly regularity — complements lag-1 from baseline)
    features["lag7_autocorr"] = np.array([
        np.corrcoef(values[i, :-7], values[i, 7:])[0, 1]
        if np.std(values[i]) > 1e-8 else 0.0
        for i in range(values.shape[0])
    ])

    # Changepoint detection: count of abrupt consumption drops (>50% between 30-day windows)
    n_cp_windows = values.shape[1] // 30
    changepoint_counts = np.zeros(values.shape[0])
    if n_cp_windows > 1:
        cp_means = np.column_stack([
            np.nanmean(values[:, i * 30:(i + 1) * 30], axis=1)
            for i in range(n_cp_windows)
        ])
        for w in range(1, n_cp_windows):
            prev = cp_means[:, w - 1]
            curr = cp_means[:, w]
            # Drop >50% from previous window
            drop = (prev - curr) / (prev + 1e-8)
            changepoint_counts += (drop > 0.5).astype(float)
    features["changepoint_drops"] = changepoint_counts

    # --- External data enrichment: holidays (ratios only) ---
    holiday_mask = get_holiday_mask(dates)
    if holiday_mask.sum() > 0:
        hol_mean = np.nanmean(values[:, holiday_mask], axis=1)
        non_hol_mean = pd.Series(np.nanmean(values[:, ~holiday_mask], axis=1)).replace(0, np.nan)
        features["holiday_ratio"] = (hol_mean / non_hol_mean).fillna(1.0).values
        features["holiday_zero_frac"] = (
            (values[:, holiday_mask] == 0).sum(axis=1) / holiday_mask.sum()
        )

    # --- External data enrichment: simulated weather ---
    temperature = get_simulated_weather(dates)
    cons_clean = np.nan_to_num(values, nan=0.0)
    features["temp_correlation"] = _batch_correlation(cons_clean, temperature)
    features["hdd_correlation"] = _batch_correlation(
        cons_clean, np.maximum(18 - temperature, 0)
    )
    features["cdd_correlation"] = _batch_correlation(
        cons_clean, np.maximum(temperature - 24, 0)
    )

    return features.fillna(0)


def compute_residual_features(residuals_matrix, dates=None, quantile_data=None):
    """Approach C: Features computed on forecast residuals (actual - predicted).

    Args:
        residuals_matrix: np.ndarray (n_customers, n_days) of actual - median_forecast
        dates: DatetimeIndex for holiday-aware features
        quantile_data: dict with keys 'actuals', 'q01', 'q09' — np.ndarray (n_customers, n_days)
            for probabilistic anomaly features from Chronos quantile forecasts
    """
    values = residuals_matrix
    features = pd.DataFrame(index=range(values.shape[0]))

    # --- Basic residual statistics ---
    features["resid_mean"] = np.nanmean(values, axis=1)
    features["resid_std"] = np.nanstd(values, axis=1)
    features["resid_max"] = np.nanmax(values, axis=1)
    features["resid_min"] = np.nanmin(values, axis=1)
    features["resid_median"] = np.nanmedian(values, axis=1)
    features["resid_skew"] = pd.DataFrame(values).apply(lambda x: x.skew(), axis=1).values
    features["resid_kurtosis"] = pd.DataFrame(values).apply(lambda x: x.kurtosis(), axis=1).values
    features["resid_mean_abs"] = np.nanmean(np.abs(values), axis=1)

    resid_std = features["resid_std"].values[:, None]
    resid_mean = features["resid_mean"].values[:, None]
    features["resid_pos_outliers"] = ((values > (resid_mean + 2 * resid_std)).sum(axis=1)) / values.shape[1]
    features["resid_neg_outliers"] = ((values < (resid_mean - 2 * resid_std)).sum(axis=1)) / values.shape[1]

    # Longest run of consecutive negative residuals
    longest_neg_runs = []
    for i in range(values.shape[0]):
        neg = values[i] < 0
        max_run = current_run = 0
        for v in neg:
            if v:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        longest_neg_runs.append(max_run)
    features["resid_longest_neg_run"] = longest_neg_runs

    # Residual trend
    x = np.arange(values.shape[1])
    slopes = []
    for i in range(values.shape[0]):
        valid = ~np.isnan(values[i])
        if valid.sum() > 10:
            slopes.append(np.polyfit(x[valid], values[i][valid], 1)[0])
        else:
            slopes.append(0.0)
    features["resid_trend"] = slopes

    features["resid_pct_large"] = (
        np.abs(values) > np.nanmean(np.abs(values), axis=1, keepdims=True) * 1.5
    ).sum(axis=1) / values.shape[1]

    # --- 6.1: Probabilistic anomaly features from quantile forecasts ---
    if quantile_data is not None:
        actuals = quantile_data["actuals"]
        q01 = quantile_data["q01"]  # 10th percentile forecast
        q09 = quantile_data["q09"]  # 90th percentile forecast
        interval_width = q09 - q01

        # Fraction of days actual falls below q0.1 (strong theft signal)
        features["frac_below_q01"] = np.nanmean(actuals < q01, axis=1)
        # Fraction of days actual falls above q0.9
        features["frac_above_q09"] = np.nanmean(actuals > q09, axis=1)
        # Mean depth of violation when actual < q0.1
        violations = np.where(actuals < q01, q01 - actuals, 0.0)
        features["mean_violation_depth"] = np.nanmean(violations, axis=1)
        # Mean prediction interval width (forecast confidence)
        features["mean_interval_width"] = np.nanmean(interval_width, axis=1)
        # Normalized residual: residual / interval_width (how many "intervals" off)
        safe_width = np.where(interval_width > 1e-6, interval_width, 1e-6)
        features["normalized_resid_mean"] = np.nanmean(values / safe_width, axis=1)
        features["normalized_resid_std"] = np.nanstd(values / safe_width, axis=1)

    # --- 6.2: Temporal structure on residuals (mirrors Approach B) ---
    if dates is not None and len(dates) >= values.shape[1]:
        resid_dates = dates[:values.shape[1]]
        dow = resid_dates.dayofweek.values

        # Weekday/weekend residual patterns
        weekday_mask = dow < 5
        weekend_mask = dow >= 5
        features["resid_weekday_mean"] = np.nanmean(values[:, weekday_mask], axis=1)
        features["resid_weekend_mean"] = np.nanmean(values[:, weekend_mask], axis=1)
        wd = pd.Series(features["resid_weekday_mean"]).replace(0, np.nan)
        we = pd.Series(features["resid_weekend_mean"]).replace(0, np.nan)
        features["resid_weekend_weekday_ratio"] = (we / wd).fillna(1.0).values

        # Monthly CV on residuals
        months = resid_dates.to_period("M")
        monthly_means = []
        for period in months.unique():
            mask = months == period
            monthly_means.append(np.nanmean(values[:, mask], axis=1))
        monthly_arr = np.column_stack(monthly_means)
        features["resid_monthly_cv"] = np.nanstd(monthly_arr, axis=1) / (
            np.nanmean(np.abs(monthly_arr), axis=1) + 1e-8
        )

        # Rolling 30-day volatility on residuals
        n_windows = values.shape[1] // 30
        window_stds = []
        for i in range(n_windows):
            start = i * 30
            window_stds.append(np.nanstd(values[:, start : start + 30], axis=1))
        window_arr = np.column_stack(window_stds)
        features["resid_rolling_volatility"] = np.nanstd(window_arr, axis=1)
        features["resid_max_window_std"] = np.nanmax(window_arr, axis=1)

        # Periodicity on residuals (FFT)
        fft_power_ratios = []
        for i in range(values.shape[0]):
            series = np.nan_to_num(values[i].copy(), nan=0.0)
            if series.std() < 1e-8:
                fft_power_ratios.append(0.0)
                continue
            fft_vals = np.fft.rfft(series - series.mean())
            power = np.abs(fft_vals) ** 2
            top5_power = np.sort(power[1:])[-5:].sum()
            total_power = power[1:].sum() + 1e-8
            fft_power_ratios.append(top5_power / total_power)
        features["resid_periodicity_strength"] = fft_power_ratios

        # Entropy on residuals (fixed bin edges for cross-customer comparability)
        entropies = []
        resid_p01 = np.nanpercentile(values, 1)
        resid_p99 = np.nanpercentile(values, 99)
        resid_bin_edges = np.linspace(min(resid_p01, -1.0), max(resid_p99, 1.0), 21)
        for i in range(values.shape[0]):
            series = np.nan_to_num(values[i].copy(), nan=0.0)
            hist, _ = np.histogram(series, bins=resid_bin_edges)
            hist = hist[hist > 0]
            if len(hist) > 1:
                probs = hist / hist.sum()
                entropies.append(sp_stats.entropy(probs))
            else:
                entropies.append(0.0)
        features["resid_entropy"] = entropies

        # Weather correlation on residuals
        temperature = get_simulated_weather(resid_dates)
        resid_clean = np.nan_to_num(values, nan=0.0)
        features["resid_temp_correlation"] = _batch_correlation(resid_clean, temperature)
        features["resid_hdd_correlation"] = _batch_correlation(
            resid_clean, np.maximum(18 - temperature, 0)
        )
        features["resid_cdd_correlation"] = _batch_correlation(
            resid_clean, np.maximum(temperature - 24, 0)
        )

    # --- Holiday-aware residual features ---
    if dates is not None:
        holiday_mask = get_holiday_mask(dates)
        n = min(values.shape[1], len(holiday_mask))
        hmask = holiday_mask[:n]
        vals = values[:, :n]
        if hmask.sum() > 0:
            features["holiday_resid_mean_abs"] = np.nanmean(np.abs(vals[:, hmask]), axis=1)
            features["non_holiday_resid_mean_abs"] = np.nanmean(np.abs(vals[:, ~hmask]), axis=1)
            non_hol_resid = features["non_holiday_resid_mean_abs"].replace(0, np.nan)
            features["holiday_resid_ratio"] = (
                features["holiday_resid_mean_abs"] / non_hol_resid
            ).fillna(1.0)
            features["holiday_resid_std"] = np.nanstd(vals[:, hmask], axis=1)

    return features.fillna(0)


# --- SageMaker Helpers ---

def upload_to_s3(df, s3_key, bucket, s3_client):
    """Upload a DataFrame as CSV to S3."""
    local_path = f"/tmp/{os.path.basename(s3_key)}"
    df.to_csv(local_path, index=False)
    s3_uri = f"s3://{bucket}/{s3_key}"
    s3_client.upload_file(local_path, bucket, s3_key)
    print(f"Uploaded {df.shape} to {s3_uri}")
    return s3_uri


def train_xgboost_on_sagemaker(
    approach_name, train_s3_uri, test_s3_uri, *,
    region, bucket, role=ROLE, sagemaker_session=None, scale_pos_weight=10.0
):
    """Launch XGBoost training job on SageMaker using SDK v3 ModelTrainer."""
    training_image = retrieve(
        framework="sklearn", region=region, version="1.4-2",
        py_version="py3", image_scope="training",
    )

    source_code_config = SourceCode(
        source_dir=str(TRAINING_DIR.resolve()),
        entry_script="train.py",
        requirements="requirements.txt",
    )

    compute_config = Compute(instance_type="ml.m5.large", instance_count=1, volume_size_in_gb=10)
    output_s3_path = f"s3://{bucket}/{PREFIX}/{approach_name}/output"

    model_trainer = ModelTrainer(
        training_image=training_image,
        source_code=source_code_config,
        compute=compute_config,
        hyperparameters={
            "n-estimators": "200",
            "max-depth": "6",
            "learning-rate": "0.1",
            "scale-pos-weight": str(scale_pos_weight),
            "target-column": "FLAG",
        },
        role=role,
        base_job_name=f"ntl-{approach_name}",
        output_data_config=OutputDataConfig(s3_output_path=output_s3_path),
        sagemaker_session=sagemaker_session,
    )

    train_input = InputData(channel_name="train", data_source=train_s3_uri, content_type="text/csv")
    test_input = InputData(channel_name="test", data_source=test_s3_uri, content_type="text/csv")

    print(f"\nLaunching SageMaker training job: ntl-{approach_name}")
    model_trainer.train(input_data_config=[train_input, test_input], wait=True, logs=True)
    return model_trainer


def download_and_load_model(model_trainer, s3_client):
    """Download model artifacts from S3 and load locally."""
    model_s3_uri = model_trainer._latest_training_job.model_artifacts.s3_model_artifacts
    local_tar = "/tmp/model.tar.gz"
    local_dir = "/tmp/model_output"

    parts = model_s3_uri.replace("s3://", "").split("/", 1)
    s3_client.download_file(parts[0], parts[1], local_tar)

    os.makedirs(local_dir, exist_ok=True)
    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(local_dir)

    model = joblib.load(os.path.join(local_dir, "model.joblib"))

    metrics_path = os.path.join(local_dir, "metrics.json")
    metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

    return model, metrics


# --- Results Persistence ---

def save_results(approach_name, y_test, y_pred, y_prob, feature_names=None):
    """Save predictions and labels for the comparison notebook."""
    RESULTS_DIR.mkdir(exist_ok=True)
    results = {
        "y_test": np.array(y_test).tolist(),
        "y_pred": np.array(y_pred).tolist(),
        "y_prob": np.array(y_prob).tolist(),
    }
    if feature_names is not None:
        results["feature_names"] = feature_names
    path = RESULTS_DIR / f"{approach_name}.json"
    with open(path, "w") as f:
        json.dump(results, f)
    print(f"Results saved to {path}")


def load_results(approach_name):
    """Load saved predictions for comparison."""
    path = RESULTS_DIR / f"{approach_name}.json"
    with open(path) as f:
        data = json.load(f)
    return (
        np.array(data["y_test"]),
        np.array(data["y_pred"]),
        np.array(data["y_prob"]),
    )
