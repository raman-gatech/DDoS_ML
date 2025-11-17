
from dataclasses import dataclass

@dataclass
class CFG:
    # Data
    dataset_path: str = "data/ddos.csv"   # full CSV with features + label column
    label_col: str = "label"             # name of label column in CSV
    id_cols: tuple = ("id", "flow_id", "timestamp")  # columns to drop if present

    test_size: float = 0.2
    random_state: int = 42

    # Preprocessing
    apply_pca: bool = True
    pca_components: int = 50             # will be min(n_features, this)
    scale_before_pca: bool = True

    # Model
    n_estimators: int = 300
    max_depth: int | None = None
    class_weight: str | None = "balanced"

    # Training
    n_jobs: int = -1

    # Paths
    model_path: str = "artifacts/ddos_ensemble.pkl"
    metrics_path: str = "artifacts/metrics.json"

    # API / Runtime
    api_host: str = "0.0.0.0"
    api_port: int = 8000
