import pandas as pd

from concept_tracer.config import Config


def get_config(root: str) -> Config:

    cfg = Config(
        # dataset settings
        root = root, # Root directory
        dataset_name = "my_dataset", # Dataset name
        dataset_specs = {
            "my_dataset": {
                "tasks": [], # Task list, empty list if no tasks (only if train and test labels are explicitly defined in get_data() in helpers.py)
                "concepts": "my_concepts", # Concepts column (e.g., {'A', 'B'; 'B', 'Z'})
                "granularities": {}, # Granularity dict, empty dict if no concept granularities
                "nominal_features": [] # Nominal feature list, empty list if no nominal features
            }
        },

        # experimental settings
        concept_prevalence_threshold = 100, # Prevalence threshold applied to the concepts
        model_name = "TabPFNClassifier", # Model name
        model_specs = {
            "TabPFNClassifier": {
                "layers": list(range(24)) # Model layers
            }
        },
        n_bins = 10, # Number of bins for mutual information estimation
        n_jobs = 20, # Number of jobs for calculations
        n_permutations = 100, # Number of permutations for significance testing
        score_name = "roc_auc", # Score name
        random_seed = 42, # Random seed

        # dashboard settings
        colors = { # Dashboard colors
            "saliency": "#0055B1",
            "selectivity": "#108800",
            "combined": "#700078",
            "knee": "#D00000"
        },
        colorscales = { # Dashboard colorscales
            "saliency": "Blues",
            "selectivity": "Greens",
            "combined": "Purples",
        },
        styles = { # Dashboard styles
            "page": {"padding": "12px", "fontFamily": "system-ui"},
            "row": {"display": "flex", "gap": "12px", "flexWrap": "wrap"},
            "col": {"display": "flex", "flexDirection": "column", "gap": "6px"},
            "tabs": {"display": "flex", "flexDirection": "column", "gap": "6px", "paddingTop": "12px", "minWidth": 0},
            "align_center": {"display": "flex", "gap": "12px", "alignItems": "center"},
            "align_top": {"display": "flex", "alignItems": "flex-start", "gap": "12px"},
            "pane": {"flex": "1 1 520px", "minWidth": "320px"},
            "top_k_label": {"fontSize": "12px", "opacity": 0.9},
            "graph": {"height": "100%", "width": "100%"},
        },

        concept_limit = 500, # Dashboard upper concept limit
        font_size = 14, # Dashboard font size
        heatmap_height = 480, # Dashboard heatmap height
        hist_height = 200, # Dashboard histogram height
        scatter_height = 300, # Dashboard scatter plot height
        template = "plotly_white", # Dashboard Plotly template
        # faster and more readable
        significance_threshold = 0.05 # Dashboard significance threshold
    )

    return cfg


def get_data(
        task: str | None,
        cfg: Config
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:

    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    y_concepts_train = pd.Series()
    y_concepts_test = pd.Series()
    y_labels_train = pd.Series()
    y_labels_test = pd.Series()
    
    return X_train, X_test, y_concepts_train, y_concepts_test, y_labels_train, y_labels_test


def get_results(
        cfg: Config,
        task: str | None = None,
        granularity: str | None = None,
    ) -> pd.DataFrame:

    df = pd.DataFrame()

    return df
