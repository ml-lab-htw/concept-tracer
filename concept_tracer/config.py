import os


class Config:

    def __init__(self, **kwargs):

        # dataset settings
        self.root = os.getcwd() # Root directory
        self.dataset_name = "mimic4ed" # Dataset name
        self.dataset_specs = {
            "mimic4ed": {
                "tasks": [ # Task list, empty list if no tasks (only if train and test labels are explicitly defined in get_data() in helpers.py)
                    "inhospital_mortality",
                    "icu_transfer_12h",
                    "critical",
                    "hospitalization"
                ],
                "concepts": "icd_list", # Concepts column (e.g., {'A', 'B'; 'B', 'Z'})
                "granularities": { # Granularity dict, empty dict if no concept granularities
                    "high_level": lambda cs: {c[0] for c in cs if c[0].isalpha()},
                    "mid_level": lambda cs: {c[:3] for c in cs if c[0].isalpha()},
                    "low_level": lambda cs: {c for c in cs if c[0].isalpha()}
                },
                "nominal_features": [ # Nominal feature list, empty list if no nominal features
                    "gender"
                ]
            }
        }


        # experimental settings
        self.concept_prevalence_threshold = 100 # Prevalence threshold applied to the concepts
        self.model_name = "TabPFNClassifier" # Model name
        self.model_specs = {
            "TabPFNClassifier": {
                "layers": list(range(24)) # Model layers
            }
        }
        self.n_bins = 10 # Number of bins for mutual information estimation
        self.n_jobs = 20 # Number of jobs for calculations
        self.n_permutations = 100 # Number of permutations for significance testing
        self.score_name = "roc_auc" # Score name
        self.random_seed = 42 # Random seed


        # dashboard settings
        self.colors = { # Dashboard colors
            "saliency": "#0055B1",
            "selectivity": "#108800",
            "combined": "#700078",
            "knee": "#D00000"
        }
        self.colorscales = { # Dashboard colorscales
            "saliency": "Blues",
            "selectivity": "Greens",
            "combined": "Purples",
        }
        self.styles = { # Dashboard styles
            "page": {"padding": "12px", "fontFamily": "system-ui"},
            "row": {"display": "flex", "gap": "12px", "flexWrap": "wrap"},
            "col": {"display": "flex", "flexDirection": "column", "gap": "6px"},
            "tabs": {"display": "flex", "flexDirection": "column", "gap": "6px", "paddingTop": "12px", "minWidth": 0},
            "align_center": {"display": "flex", "gap": "12px", "alignItems": "center"},
            "align_top": {"display": "flex", "alignItems": "flex-start", "gap": "12px"},
            "pane": {"flex": "1 1 520px", "minWidth": "320px"},
            "top_k_label": {"fontSize": "12px", "opacity": 0.9},
            "graph": {"height": "100%", "width": "100%"},
        }

        self.concept_limit = 500 # Dashboard upper concept limit
        self.font_size = 14 # Dashboard font size
        self.heatmap_height = 480 # Dashboard heatmap height
        self.hist_height = 200 # Dashboard histogram height
        self.scatter_height = 300 # Dashboard scatter plot height
        self.template = "plotly_white" # Dashboard Plotly template
        # faster and more readable
        self.significance_threshold = 0.05 # Dashboard significance threshold


        # override defaults with keyword arguments
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise KeyError(f"Unknown config key: {key}")
            setattr(self, key, value)
        
        # data path
        self.data_path = os.path.join(self.root, "..", "data_sets", self.dataset_name) # Path for the data
        
        # concepts and embeddings paths
        self.concept_path = os.path.join(self.root, "concepts", "concepts{granularity}{task}.npy") # Path for the concepts
        self.embedding_path = os.path.join(self.root, "embeddings", "embeddings{task}.npy") # Path for the embeddings

        # results paths
        self.baseline_path = os.path.join(self.root, "results", "baseline_scores{granularity}{task}.csv.xz") # Path for the baseline results
        self.interpret_path = os.path.join(self.root, "results", "interpret_scores{granularity}{task}.csv.xz") # Path for the interpretabilty results
        self.test_score_path = os.path.join(self.root, "results", "test_scores{task}.csv") # Path for the test scores

        # create directories if needed
        for path in [
            self.concept_path,
            self.embedding_path,
            self.baseline_path,
            self.interpret_path,
            self.test_score_path
            ]:
            os.makedirs(os.path.dirname(path), exist_ok=True)



    def __str__(self):
        config_items = [
            f"{key}: {value}" for key, value in self.__dict__.items()
        ]
        return "Config:\n" + "\n".join(f"  {item}" for item in config_items)
    