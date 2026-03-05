import ast
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
import time
import warnings

from dash import html
from datetime import datetime
from imblearn.under_sampling import RandomUnderSampler
# from mosek.fusion import Domain, Expr, Matrix, Model, ObjectiveSense, Var
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, KBinsDiscretizer
from tabpfn_extensions import TabPFNClassifier
from tabpfn_extensions.embedding import TabPFNEmbedding

from . import tabpfn_layer_patch
from .config import Config


pd.options.plotting.backend = "plotly"


CONCEPTS = None
EMBEDDINGS = None



def build_l0l2(
        X: np.ndarray,
        y: np.ndarray,
        k: int,
        lambd: float = 1
    ) -> tuple[list[float], float]:
    # https://docs.mosek.com/latest/pythonfusion/case-studies-logistic.html
    # (but with an added intercept and an explicit cardinality constraint k)
    
    n, d = X.shape[0], X.shape[1]

    M = Model()

    # variables
    theta = M.variable(d)
    theta0 = M.variable(1)
    t = M.variable(n)
    reg = M.variable(d, Domain.greaterThan(0.0))
    z = M.variable(d, Domain.binary())

    # constraints
    M.constraint(Expr.sum(z), Domain.lessThan(k))
    for j in range(d):
        M.constraint(
            Var.vstack(z.index(j), reg.index(j), theta.index(j)),
            Domain.inRotatedQCone()
        )

    # objective
    M.objective(
        ObjectiveSense.Minimize,
        Expr.add(Expr.sum(t), Expr.mul(lambd, Expr.sum(reg)))
    )

    signs = [[-1.0 if single_y == 1 else 1.0] for single_y in y]
    softplus(
        M, t, Expr.mulElm(
            Expr.add(
                Expr.mul(Matrix.dense(X.tolist()), theta),
                Expr.mul(Matrix.dense([[1.0]] * n), theta0)
            ),
            signs
        )
    )

    M.solve()

    return theta.level(), theta0.level()[0]



def get_data(
        task: str | None,
        cfg: Config
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    
    # implement your own dataset processing
    if cfg.dataset_name != "mimic4ed":
        raise NotImplementedError("helpers.get_data() only supports MIMIC-IV ED. To use your own dataset, pass get_data_fn=... to calculations.run().")
    
    if cfg.dataset_name == "mimic4ed":

        concepts = cfg.dataset_specs[cfg.dataset_name].get("concepts", "")
        if not concepts:
            raise ValueError("Concepts column must be provided in config.py.")

        if not task:
            raise ValueError("Task(s) must be provided in config.py.")
        
        if not os.path.isdir(cfg.data_path):
            raise FileNotFoundError(f"Data directory not found: {cfg.data_path}. Correct data path must be provided in config.py.")
        
        train_path = os.path.join(cfg.data_path, "train.csv")
        test_path  = os.path.join(cfg.data_path, "test.csv")

        if not os.path.isfile(train_path):
            raise FileNotFoundError(f"Missing train.csv at: {train_path}.")
        if not os.path.isfile(test_path):
            raise FileNotFoundError(f"Missing test.csv at: {test_path}.")
        
        # 353150 training and 88287 test samples
        df_train = pd.read_csv(train_path)
        df_test  = pd.read_csv(test_path)

        # drop rows without ICD codes as concepts -> 21239 training and 5300 test samples
        df_train = df_train.dropna(subset=[concepts])
        df_test = df_test.dropna(subset=[concepts])
        
        # drop training rows in which test set patients are in the training set -> 9031 training samples
        df_train = df_train[~df_train["subject_id"].isin(df_test["subject_id"])]

        features = [
            "age", "gender",
            
            "n_ed_30d", "n_ed_90d", "n_ed_365d",
            "n_hosp_30d", "n_hosp_90d", "n_hosp_365d",
            "n_icu_30d", "n_icu_90d", "n_icu_365d",
            
            "triage_temperature", "triage_heartrate", "triage_resprate",
            "triage_o2sat", "triage_sbp", "triage_dbp", "triage_pain", "triage_acuity",
            
            "chiefcom_chest_pain", "chiefcom_abdominal_pain", "chiefcom_headache",
            "chiefcom_shortness_of_breath", "chiefcom_back_pain", "chiefcom_cough",
            "chiefcom_nausea_vomiting", "chiefcom_fever_chills", "chiefcom_syncope",
            "chiefcom_dizziness",
            
            "cci_MI", "cci_CHF", "cci_PVD", "cci_Stroke", "cci_Dementia",
            "cci_Pulmonary", "cci_Rheumatic", "cci_PUD", "cci_Liver1", "cci_DM1",
            "cci_DM2", "cci_Paralysis", "cci_Renal", "cci_Cancer1", "cci_Liver2",
            "cci_Cancer2", "cci_HIV",
            
            "eci_Arrhythmia", "eci_Valvular", "eci_PHTN", "eci_HTN1", "eci_HTN2",
            "eci_NeuroOther", "eci_Hypothyroid", "eci_Lymphoma", "eci_Coagulopathy",
            "eci_Obesity", "eci_WeightLoss", "eci_FluidsLytes", "eci_BloodLoss",
            "eci_Anemia", "eci_Alcohol", "eci_Drugs","eci_Psychoses", "eci_Depression",

            concepts
        ]

        X_train = df_train[features].copy()
        X_test = df_test[features].copy()
        y_labels_train = df_train[f"outcome_{task}"].copy()
        y_labels_test = df_test[f"outcome_{task}"].copy()
        
        # undersampling
        rus = RandomUnderSampler(
            sampling_strategy="majority", random_state=cfg.random_seed
        )
        X_train, y_labels_train = rus.fit_resample(X_train, y_labels_train)

        # concept handling
        y_concepts_train = X_train[concepts].copy()
        y_concepts_test = X_test[concepts].copy()
        X_train = X_train.drop(columns=[concepts])
        X_test = X_test.drop(columns=[concepts])

        # data type handling
        y_concepts_train = y_concepts_train.apply(lambda x: ast.literal_eval(x))
        y_concepts_test = y_concepts_test.apply(lambda x: ast.literal_eval(x))
    
    return X_train, X_test, y_concepts_train, y_concepts_test, y_labels_train, y_labels_test



def get_embeddings(
        X_train: pd.DataFrame,
        y_labels_train: pd.Series,
        X_test: pd.DataFrame,
        cfg: Config,
        data_source: str = "test",
        layer: int | None = None
    ) -> np.ndarray:

    if cfg.model_name == "TabPFNClassifier":

        tabpfn_layer_patch.set_embedding_layer_idx(layer)
        model = TabPFNClassifier(n_estimators=1, random_state=cfg.random_seed)
        embedding_extractor = TabPFNEmbedding(tabpfn_clf=model)
        embeddings = embedding_extractor.get_embeddings(
            X_train, y_labels_train, X_test, data_source=data_source
        )

    else:

        raise NotImplementedError("The model is not supported yet.")
    
    return embeddings



def get_fitted_model(
        X_train: pd.DataFrame,
        y_labels_train: pd.Series,
        cfg: Config
    ) -> ClassifierMixin:

    if cfg.model_name == "TabPFNClassifier":

        model = TabPFNClassifier(n_estimators=1, random_state=cfg.random_seed)
        model.fit(X_train, y_labels_train)

    else:

        raise NotImplementedError("The model is not supported yet.")
    
    return model



def get_heatmap_dashboard(
        df: pd.DataFrame,
        stats: dict[str, dict[str, float]],
        mode: str,
        cfg: Config
    ) -> go.Figure:

    tmp = df.copy()
    if mode == "combined":
        tmp["combined"] = get_scaled_sum(tmp["saliency"], tmp["selectivity"], stats)

    piv = (
        # max over concepts
        tmp.pivot_table(index="layer", columns="neuron", values=mode, aggfunc="max")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    layers = list(piv.index)

    if layers:
        y0 = min(layers) - 0.5
        y1 = max(layers) + 0.5
    else:
        y0, y1 = -0.5, 0.5

    fig = go.Figure(
        go.Heatmap(
            z=piv,
            x=piv.columns,
            y=layers,
            showscale=False,
            colorscale=cfg.colorscales[mode],
            hovertemplate="layer=%{y}<br>neuron=%{x}<br>value=%{z:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        template=cfg.template,
        title=None,
        font=dict(size=cfg.font_size),
        margin=dict(l=0, r=100, t=0, b=0),
        xaxis=dict(showticklabels=False, fixedrange=True, automargin=False),
        yaxis=dict(showticklabels=False, fixedrange=True, range=[y0, y1], automargin=False),
        dragmode=False
    )

    return fig



def get_hist_plot_dashboard(
        values: pd.Series,
        title: str,
        xlabel: str,
        cfg: Config,
        nbinsx: int = 50
    ) -> go.Figure:

    if xlabel == "Saliency":
        bar_color = cfg.colors["saliency"]
    elif xlabel == "Selectivity":
        bar_color = cfg.colors["selectivity"]

    fig = go.Figure(
        go.Histogram(
            x=values,
            histnorm="percent",
            nbinsx=nbinsx,
            marker=dict(color=bar_color)
        )
    )

    fig.update_layout(
        template=cfg.template,
        height=cfg.hist_height,
        title=title,
        font=dict(size=cfg.font_size),
        margin=dict(l=10, r=10, t=100, b=10),
        xaxis=dict(title=xlabel, fixedrange=True),
        yaxis=dict(title="Percent", fixedrange=True),
        dragmode=False
    )

    return fig



def get_knee_point(
        df: pd.DataFrame,
        stats: dict | None = None
    ) -> object:

    if stats is None:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[["saliency", "selectivity"]])
        summed = scaled[:, 0] + scaled[:, 1]
        knee_point = df.index[np.argmax(summed)]
    else:
        summed = get_scaled_sum(df["saliency"], df["selectivity"], stats)
        knee_point = summed.idxmax()

    return knee_point



def get_l0l2(
        embeddings: np.ndarray,
        concepts: np.ndarray,
        granularity: str | None,
        y_concepts_test_granularity: dict[str | None, list[set]],
        method: str = "exhaustive"
    ) -> list[tuple[str, object, int, int]]:

    d_l, d_s, d_n = embeddings.shape
    embeddings_transposed = embeddings.transpose(0, 2, 1).reshape(d_l * d_n, d_s).T
    concepts_transposed = concepts.T

    # scale embeddings...
    scaler = MinMaxScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_transposed)

    cache = []

    for c, concept in enumerate(sorted(set().union(*y_concepts_test_granularity[granularity]))):

        # ... and build a cardinality-constrained linear probe, one per concept
        if method == "exhaustive":
            # fast exhaustive search for k=1
            selector = SelectKBest(score_func=univariate_log_likelihood, k=1)
            selector.fit(embeddings_scaled, concepts_transposed[:, c])
            best_idx = selector.get_support(indices=True)[0]
        elif method == "mico":
            # slow mixed-integer conic optimization for k=1
            theta, _ = build_l0l2(embeddings_scaled, concepts_transposed[:, c], k=1)
            best_idx = np.argmax(np.abs(theta))
        else:
            raise NotImplementedError("The method is not supported yet.")

        layer, neuron = divmod(best_idx, d_n)
        cache.append(("l0l2", concept, layer, neuron))

    return cache



def get_pareto_front(
        df: pd.DataFrame
    ) -> list[object]:
    # https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    """
    Find the Pareto front
    
    Input: df: pd.DataFrame

    Output: list of Pareto front indices
    """

    scores = df[["selectivity", "saliency"]].values
    is_efficient = np.ones(scores.shape[0], dtype = bool)
    for i, c in enumerate(scores):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(scores[is_efficient] > c, axis=1)
            is_efficient[i] = True
    return df.index[is_efficient].tolist()



def get_results(
        cfg: Config,
        task: str | None = None,
        granularity: str | None = None,
    ) -> pd.DataFrame:

    t0 = time.time()
    print("=" * 80)
    print(f"[{datetime.now()}]")
    print("Start loading results for ConceptTracer Dashboard...")
    print("-" * 80)

    dfs = []
    granularities = cfg.dataset_specs[cfg.dataset_name].get("granularities", {}) or [None]
    granularities_to_load = [granularity] if granularity is not None else granularities

    for g in granularities_to_load:
        path = cfg.interpret_path.format(**resolve_suffixes(cfg, task=task, granularity=g))
        print(f"Loading granularity='{g}' from {path}...", flush=True)

        df = pd.read_csv(
            path,
            usecols=["neuron", "layer", "concept", "saliency", "selectivity", "p_saliency", "p_selectivity", "p_combined"],
        )
        df["granularity"] = g
        dfs.append(df)

    print("-" * 80)
    print("Concatenating dataframes...", flush=True)
    df = pd.concat(dfs, ignore_index=True)

    elapsed = time.time() - t0
    print("-" * 80)
    print(f"Finished loading results in {elapsed:.2f} seconds...")
    print("=" * 80)

    return df



def get_scaled_sum(
        saliency: pd.Series,
        selectivity: pd.Series,
        stats: dict[str, dict[str, float]]
    ) -> pd.Series:

    min_saliency = stats["saliency"]["min"]
    max_saliency = stats["saliency"]["max"]
    if min_saliency == max_saliency:
        print("No variation in saliency, minmax scaling aborted.")
        minmax_saliency = pd.Series(0.0, index=saliency.index)
    else:
        minmax_saliency = (saliency - min_saliency) / (max_saliency - min_saliency)

    min_selectivity = stats["selectivity"]["min"]
    max_selectivity = stats["selectivity"]["max"]
    if min_selectivity == max_selectivity:
        print("No variation in selectivity, minmax scaling aborted.")
        minmax_selectivity = pd.Series(0.0, index=selectivity.index)
    else:
        minmax_selectivity = (selectivity - min_selectivity) / (max_selectivity - min_selectivity)

    return minmax_saliency + minmax_selectivity



def get_scatter_plot_dashboard(
        df: pd.DataFrame,
        stats: dict[str, dict[str, float]],
        title: str,
        cfg: Config
    ) -> go.Figure:

    fig = go.Figure(
        go.Scattergl(
            x=df["selectivity"],
            y=df["saliency"],
            text=df.index,
            mode="markers",
            marker=dict(
                color=cfg.colors["combined"],
                opacity=0.5,
                size=6,
            ),
            hovertemplate="id=%{text}<br>selectivity=%{x:.2f}<br>saliency=%{y:.2f}<extra></extra>",
        )
    )
    
    # Pareto front
    front = df.loc[get_pareto_front(df)].copy().sort_values("selectivity", ascending=True)
    fig.add_trace(
        go.Scattergl(
            x=front["selectivity"],
            y=front["saliency"],
            mode="lines+markers",
            line=dict(color="black", width=2),
            marker=dict(color="black", size=8),
            hoverinfo="skip"
        )
    )

    # Knee point
    if len(front):
        knee_id = get_knee_point(front, stats)
        if knee_id in front.index:
            knee = front.loc[[knee_id]]
            fig.add_trace(
                go.Scattergl(
                    x=knee["selectivity"],
                    y=knee["saliency"],
                    mode="markers",
                    marker=dict(
                        color=cfg.colors["knee"],
                        line=dict(width=1, color="white"),
                        size=10
                    ),
                    hoverinfo="skip"
                )
            )

    fig.update_layout(
        template=cfg.template,
        height=cfg.scatter_height,
        title=title,
        font=dict(size=cfg.font_size),
        margin=dict(l=10, r=10, t=100, b=10),
        showlegend=False,
        xaxis=dict(title="Selectivity", fixedrange=True),
        yaxis=dict(title="Saliency", fixedrange=True),
        dragmode=False
    )

    return fig



def get_scatter_plot_publication(
        df: pd.DataFrame,
        idxs: dict | None = None
    ) -> go.Figure:

    if idxs is None:
        idxs = {}

    fig = df.plot(
        kind="scatter",
        x="selectivity",
        y="saliency",
        hover_data=["concept", "saliency", "selectivity"],
        hover_name=df.index,
        marginal_x="histogram",
        marginal_y="histogram",
        labels={
            "selectivity": "Selectivity",
            "saliency": "Saliency"},
        opacity=0.5,
        range_x=[0, 0.6],
        range_y=[0, 0.3],
        template="plotly_white"
    )

    fig.update_layout(
        font=dict(size=20),
        legend=dict(font=dict(size=20))
    )
    
    if idxs:

        colors = {
            "interpret": "black",
            "shap": "orange",
            "l0l2": "red"
        }

        labels = {
            "interpret": "Global",
            "shap": "SHAP",
            "l0l2": "Optimal"
        }

        for method in ["interpret", "shap", "l0l2"]:

            if f"{method}_front" in idxs:

                present = df.index.intersection(idxs.get(f"{method}_front"))

                if len(present) > 0:
                    rows = df.loc[present].sort_values("selectivity", ascending=True)
                    fig.add_trace(
                        go.Scattergl(
                            x=rows["selectivity"],
                            y=rows["saliency"],
                            mode="lines+markers",
                            name=f"{labels.get(method)}",
                            line=dict(
                                color=colors.get(method),
                                width=2
                            ),
                            marker=dict(
                                color=colors.get(method)
                            )
                        )
                    )

            if f"{method}_knee" in idxs:

                present = df.index.intersection([idxs.get(f"{method}_knee")])

                if len(present) > 0:
                    row = df.loc[present]
                    fig.add_trace(
                        go.Scattergl(
                            x=row["selectivity"],
                            y=row["saliency"],
                            mode="markers",
                            showlegend=False,
                            marker=dict(
                                color=colors.get(method),
                                size=10
                            )
                        )
                    )

    return fig



def get_shap(
        embeddings: np.ndarray,
        concepts: np.ndarray,
        granularity: str | None,
        y_concepts_test_granularity: dict[str | None, list[set]]
    ) -> list[tuple[str, object, int, int]]:

    d_l, d_s, d_n = embeddings.shape
    embeddings_transposed = embeddings.transpose(0, 2, 1).reshape(d_l * d_n, d_s).T
    concepts_transposed = concepts.T

    # scale embeddings...
    scaler = MinMaxScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_transposed)
    
    # ... and build a multilabel linear probe, one per concept
    model = OneVsRestClassifier(LogisticRegression(max_iter=10000))
    model.fit(embeddings_scaled, concepts_transposed)
    
    embeddings_centered = embeddings_scaled - embeddings_scaled.mean(axis=0)

    cache = []

    for c, concept in enumerate(sorted(set().union(*y_concepts_test_granularity[granularity]))):

        # mean absolute interventional SHAP values for a linear model (Lundberg & Lee, 2017)
        shap = model.estimators_[c].coef_[0] * embeddings_centered
        best_idx = np.argmax(np.mean(np.abs(shap), axis=0))
        layer, neuron = divmod(best_idx, d_n)
        cache.append(("shap", concept, layer, neuron))

    return cache



def get_stats(
        j: int,
        cfg: Config,
        estimator_name: str = "normalized_mutual_info_score",
        permutation_seed: int | None = None
    ) -> tuple[int, np.ndarray]:

    stats = []
    concepts = CONCEPTS[j]

    if permutation_seed is not None:
        rng = np.random.default_rng(permutation_seed)
        concepts = concepts[rng.permutation(concepts.shape[0])]

    if estimator_name in ["normalized_mutual_info_score", "mutual_info_score"]:

        # fast, normalized estimation
        if estimator_name == "normalized_mutual_info_score":
            scorer = normalized_mutual_info_score
            
        # fast, unnormalized estimation
        elif estimator_name == "mutual_info_score":
            scorer = mutual_info_score

        for layer in range(EMBEDDINGS.shape[0]):

            disc = KBinsDiscretizer(
                n_bins=cfg.n_bins,
                encode="ordinal",
                strategy="quantile"
            )

            disc_embeddings = disc.fit_transform(EMBEDDINGS[layer])

            for neuron in range(disc_embeddings.shape[1]):
                
                I = scorer(concepts, disc_embeddings[:, neuron])

                stats.append((layer, neuron, I))
    
    # slow, unnormalized estimation
    elif estimator_name == "mutual_info_classif":

        for layer in range(EMBEDDINGS.shape[0]):
            for neuron in range(EMBEDDINGS.shape[2]):

                I = mutual_info_classif(
                    EMBEDDINGS[layer, :, neuron].reshape(-1, 1),
                    concepts,
                    random_state=cfg.random_seed
                )[0]

                stats.append((layer, neuron, I))

    else:
        
        raise NotImplementedError("The estimator is not supported yet.")

    return j, np.asarray(stats).T



def get_test_score(
        X_test: pd.DataFrame,
        y_labels_test: pd.Series,
        model: ClassifierMixin,
        cfg: Config
    ) -> float:

    if cfg.score_name == "roc_auc":

        probabilities = model.predict_proba(X_test)
        score = roc_auc_score(y_labels_test, probabilities[:, 1])

    else:

        raise NotImplementedError("The score is not supported yet.")

    return score



def get_top_k_features(
        task: str | None,
        df: pd.DataFrame,
        idx: int,
        cfg: Config,
        k: int = 3,
        granularity: str | None = None
    ) -> np.ndarray:

    # knee point
    knee_neuron = df.loc[idx, "neuron"]
    knee_layer = df.loc[idx, "layer"]
    knee_concept = df.loc[idx, "concept"]

    _ = resolve_granularity_suffix(cfg, granularity)
    
    # embeddings and concept labels for knee point
    embeddings = np.load(cfg.embedding_path.format(task=resolve_task_suffix(cfg, task)))
    _, X_test, _, y_concepts_test, _, _ = get_data(
        task, cfg
    )
    granularities = cfg.dataset_specs[cfg.dataset_name].get("granularities", {}) or {}
    if granularities:
        y_concepts_test_granularity = {
            g: [granularities[g](cs) for cs in y_concepts_test] for g in granularities
        }
    else:
        y_concepts_test_granularity = {None: y_concepts_test}
    concept = np.fromiter(
        (knee_concept in cs for cs in y_concepts_test_granularity[granularity]),
        bool, count=len(y_concepts_test_granularity[granularity])
    )

    # one-hot encode nominal features
    nominal_features = cfg.dataset_specs[cfg.dataset_name].get("nominal_features", []) or []
    
    transformers = []
    if nominal_features:
        transformers.append(("ohe", OneHotEncoder(drop="if_binary", sparse_output=False), nominal_features))

    # feature selection for input features on knee point activations when concept is present
    pipeline = Pipeline([
        (
            "preprocess", ColumnTransformer(transformers=transformers, remainder="passthrough", verbose_feature_names_out=False)
        ), (
            "top_k", SelectKBest(lambda X, y: mutual_info_regression(X, y, random_state=cfg.random_seed), k=k)
        )
    ])
    _ = pipeline.fit_transform(
        X_test[concept],
        embeddings[knee_layer, concept, knee_neuron]
    )

    # top-3 features that most inform the knee point activations when concept is present
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    top_k_features = feature_names[pipeline.named_steps["top_k"].get_support()]

    return top_k_features



def get_top_k_items(
        df: pd.DataFrame,
        stats: dict[str, dict[str, float]],
        k: int,
        mode: str
    ) -> list[html.Li]:

    if df is None or df.empty:
        return []
    
    if mode in ["saliency", "selectivity"]:
        if not df[mode].notna().any():
            return []
        
        other = "selectivity" if mode == "saliency" else "saliency"
        k = max(1, min(10, k, len(df)))
        top = df.sort_values(by=[mode, other], ascending=[False, False]).head(k)

    elif mode == "combined":
        front = df.loc[get_pareto_front(df)].copy()
        if front.empty:
            return []
        
        front["combined"] = get_scaled_sum(front["saliency"], front["selectivity"], stats)
        k = max(1, min(10, k, len(front)))
        top = (
            front.sort_values(
                by=["combined", "saliency", "selectivity"],
                ascending=[False, False, False]
            )
            .head(k)
            .drop(columns=["combined"])
        )

    else:
        return []
    
    return [
        html.Li(
            f"{idx}: saliency = {row['saliency']:.2f}, "
            f"selectivity = {row['selectivity']:.2f}"
        )
        for idx, row in top.iterrows()
    ]



def resolve_granularity_suffix(
        cfg: Config,
        granularity: str | None = None
    ) -> str:

    granularities = cfg.dataset_specs[cfg.dataset_name].get("granularities", {}) or {}

    if granularities:
        if granularity is None:
            raise ValueError(f"Granularity must be one of {list(granularities.keys())}, got None.")
        if not isinstance(granularity, str):
            raise TypeError(f"Granularity must be a str, got {type(granularity)}.")
        if granularity not in granularities:
            raise ValueError(f"Unknown granularity '{granularity}'. Allowed: {list(granularities.keys())}.")
        granularity_suffix = f"_{granularity}"
    else:
        if granularity is not None:
            raise ValueError("Granularities must be provided in config.py, or set granularity=None.")
        granularity_suffix = ""
    
    return granularity_suffix



def resolve_suffixes(
        cfg: Config,
        task: str | None = None,
        granularity: str | None = None
    ) -> dict[str, str]:
    return {
        "task": resolve_task_suffix(cfg, task),
        "granularity": resolve_granularity_suffix(cfg, granularity),
    }



def resolve_task_suffix(
        cfg: Config,
        task: str | None = None
    ) -> str:

    tasks = cfg.dataset_specs[cfg.dataset_name].get("tasks", []) or []

    if tasks:
        if task is None:
            raise ValueError(f"Task must be one of {tasks}, got None.")
        if not isinstance(task, str):
            raise TypeError(f"Task must be a str, got {type(task)}.")
        if task not in tasks:
            raise ValueError(f"Unknown task '{task}'. Allowed: {tasks}.")
        task_suffix = f"_{task}"
    else:
        if task is not None:
            raise ValueError("Tasks must be provided in config.py, or set task=None.")
        task_suffix = ""

    return task_suffix



def softplus(
        M: "Model",
        t: "Var",
        u: "Expr"
    ) -> None:
    # https://docs.mosek.com/latest/pythonfusion/case-studies-logistic.html

    # variables
    n = t.getShape()[0]
    z1 = M.variable(n)
    z2 = M.variable(n)

    # constraints
    M.constraint(
        Expr.add(z1, z2), Domain.equalsTo(1.0)
    )
    M.constraint(
        Expr.hstack(z1, Expr.constTerm(n, 1.0), Expr.sub(u, t)), Domain.inPExpCone()
    )
    M.constraint(
        Expr.hstack(z2, Expr.constTerm(n, 1.0), Expr.neg(t)), Domain.inPExpCone()
    )



def univariate_log_likelihood(
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:

    d = X.shape[1]
    scores = np.empty(d)

    for j in range(d):

        model = LogisticRegression(max_iter=10000)
        model.fit(X[:, [j]], y)
        scores[j] = -log_loss(y, model.predict_proba(X[:, [j]])[:, 1])

    return scores



def worker_init(
        concepts_path: str,
        embeddings_path: str
    ) -> None:

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"sklearn\.preprocessing\._discretization"
    )

    global CONCEPTS, EMBEDDINGS
    CONCEPTS = np.load(concepts_path, mmap_mode="r")
    EMBEDDINGS = np.load(embeddings_path, mmap_mode="r")
