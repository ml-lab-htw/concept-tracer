import numpy as np
import pandas as pd

from sklearn.utils.parallel import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from . import helpers
from .config import Config


def run_baselines(
        task: str | None,
        granularity: str | None,
        y_concepts_test_granularity: dict[str | None, list[set]],
        cfg: Config,
        method: str = "all"
    ) -> None:
    
    # get embeddings and concepts
    embeddings = np.load(cfg.embedding_path.format(task=helpers.resolve_task_suffix(cfg, task)))
    concepts = np.load(cfg.concept_path.format(**helpers.resolve_suffixes(cfg, task=task, granularity=granularity)))

    cache = []

    # get stats for SHAP values
    if method == "shap" or method == "all":
        cache.extend(helpers.get_shap(
            embeddings, concepts, granularity, y_concepts_test_granularity
        ))

    # get stats for L0-constrained L2-regularized logistic regression
    if method == "l0l2" or method == "all":
        cache.extend(helpers.get_l0l2(
            embeddings, concepts, granularity, y_concepts_test_granularity
        ))

    if not cache:
        raise NotImplementedError("The baseline is not supported yet.")

    # build data frame
    df = pd.DataFrame(
        cache,
        columns=["baseline", "concept", "layer", "neuron"]
    )

    # add meta-data and save
    df.insert(0, "dataset", cfg.dataset_name)
    df.insert(1, "model", cfg.model_name)
    df.insert(2, "concepts", cfg.dataset_specs[cfg.dataset_name].get("concepts", ""))
    df.to_csv(cfg.baseline_path.format(**helpers.resolve_suffixes(cfg, task=task, granularity=granularity)), index=False)



def run_concepts(
        task: str | None,
        granularity: str | None,
        y_concepts_test_granularity: dict[str | None, list[set]],
        cfg: Config
    ) -> None:

    # get concepts
    concepts = []
    for c in sorted(set().union(*y_concepts_test_granularity[granularity])):
        concept = np.fromiter(
            (c in cs for cs in y_concepts_test_granularity[granularity]),
            bool, count=len(y_concepts_test_granularity[granularity])
        )
        concepts.append(concept)

    np.save(cfg.concept_path.format(**helpers.resolve_suffixes(cfg, task=task, granularity=granularity)), np.stack(concepts))



def run_embeddings(
        task: str | None,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_labels_train: pd.Series,
        cfg: Config
    ) -> None:

    # one-hot encode nominal features
    nominal_features = cfg.dataset_specs[cfg.dataset_name].get("nominal_features", []) or []
    
    transformers = []
    if nominal_features:
        transformers.append(("ohe", OneHotEncoder(sparse_output=False), nominal_features))
    
    ct = ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
        verbose_feature_names_out=False
    ).set_output(transform="pandas")
    
    X_train = ct.fit_transform(X_train)
    X_test = ct.transform(X_test)
    
    # get embeddings
    embeddings = []
    for layer in cfg.model_specs[cfg.model_name]["layers"]:
        embedding = helpers.get_embeddings(
            X_train, y_labels_train, X_test,
            cfg, data_source="test", layer=layer
        )
        embeddings.append(embedding)

    np.save(cfg.embedding_path.format(task=helpers.resolve_task_suffix(cfg, task)), np.concatenate(embeddings))



def run_saliency_selectivity(
        task: str | None,
        granularity: str | None,
        y_concepts_test_granularity: dict[str | None, list[set]],
        cfg: Config
    ) -> None:

    concept_list = sorted(set().union(*y_concepts_test_granularity[granularity]))

    # concepts x test samples
    d_c = np.load(cfg.concept_path.format(**helpers.resolve_suffixes(cfg, task=task, granularity=granularity))).shape[0]

    # layers x test samples x embedding dimension
    d_l, _, d_n = np.load(cfg.embedding_path.format(task=helpers.resolve_task_suffix(cfg, task))).shape

    saliency_sum = np.zeros((d_l, d_n))
    cache = []
    
    # get stats
    for j, stats in Parallel(
        n_jobs=cfg.n_jobs,
        pre_dispatch="1*n_jobs",
        initializer=helpers.worker_init,
        initargs=(
            cfg.concept_path.format(**helpers.resolve_suffixes(cfg, task=task, granularity=granularity)),
            cfg.embedding_path.format(task=helpers.resolve_task_suffix(cfg, task))
        ))(
            # pass indices to workers, not arrays
            delayed(helpers.get_stats)(j, cfg)
            for j in tqdm(range(d_c), desc=f"{cfg.dataset_name}, {task}, "
                          f"{granularity + ' ' if granularity is not None else ''}concepts")):
        
        concept = concept_list[j]
        layer, neuron, I = stats
        layer = layer.astype(int)
        neuron = neuron.astype(int)
        saliency = I.astype(float)

        for l, n, s in zip(layer, neuron, saliency):
            saliency_sum[l, n] += s
            cache.append((concept, l, n, s))

    # build data frame
    df = pd.DataFrame(
        cache,
        columns=["concept", "layer", "neuron", "saliency"]
    )
    # handle undefined selectivity
    denominator = saliency_sum[df["layer"], df["neuron"]]
    df["selectivity"] = np.where(
        denominator == 0, np.nan, df["saliency"] / denominator
    )

    # nonparametric permutation testing with max-stat correction
    null_max_saliency = np.empty(cfg.n_permutations)
    null_max_selectivity = np.empty(cfg.n_permutations)

    for k in tqdm(range(cfg.n_permutations), desc=f"{cfg.dataset_name}, {task}, permutations"):

        saliency_sum = np.zeros((d_l, d_n))
        saliency_max = 0.0
        cache = []

        permutation_results = Parallel(
            n_jobs=cfg.n_jobs,
            pre_dispatch="1*n_jobs",
            initializer=helpers.worker_init,
            initargs=(
                cfg.concept_path.format(**helpers.resolve_suffixes(cfg, task=task, granularity=granularity)),
                cfg.embedding_path.format(task=helpers.resolve_task_suffix(cfg, task))
            ))(
                delayed(helpers.get_stats)(j, cfg, permutation_seed=cfg.random_seed + k + 100000)
                for j in range(d_c))
        
        for _, stats in permutation_results:
            layer, neuron, I = stats
            layer = layer.astype(int)
            neuron = neuron.astype(int)
            saliency = I.astype(float)

            # max across neurons
            m = np.nanmax(saliency)
            # max across concepts
            if m > saliency_max:
                saliency_max = m

            for l, n, s in zip(layer, neuron, saliency):
                saliency_sum[l, n] += s
                cache.append((l, n, s))

        null_max_saliency[k] = saliency_max

        cache = np.asarray(cache, dtype=float)
        layer = cache[:, 0].astype(int)
        neuron = cache[:, 1].astype(int)
        saliency = cache[:, 2].astype(float)
        denominator = saliency_sum[layer, neuron]
        selectivity = np.where(
            denominator == 0, np.nan, saliency / denominator
        )

        # max across neurons and concepts
        selectivity_max = np.nanmax(selectivity)
        null_max_selectivity[k] = selectivity_max

    # count(null >= x) = N_PERMUTATIONS - idx_left where idx_left = searchsorted(sorted, x, "left")
    idx_saliency = np.searchsorted(np.sort(null_max_saliency), df["saliency"].to_numpy(), side="left")
    p_saliency = (cfg.n_permutations - idx_saliency + 1) / (cfg.n_permutations + 1)
    df["p_saliency"] = p_saliency

    idx_selectivity = np.searchsorted(np.sort(null_max_selectivity), df["selectivity"].to_numpy(), side="left")
    p_selectivity = np.where(
        np.isnan(df["selectivity"].to_numpy()),
        np.nan,
        (cfg.n_permutations - idx_selectivity + 1) / (cfg.n_permutations + 1)
    )
    df["p_selectivity"] = p_selectivity

    # Bonferroni correction across the two metrics
    p_min = np.where(np.isnan(p_selectivity), p_saliency, np.minimum(p_saliency, p_selectivity))
    df["p_combined"] = np.minimum(1.0, 2.0 * p_min)

    # add meta-data and save
    df.insert(0, "dataset", cfg.dataset_name)
    df.insert(1, "model", cfg.model_name)
    df.insert(2, "concepts", cfg.dataset_specs[cfg.dataset_name].get("concepts", ""))
    df.to_csv(cfg.interpret_path.format(**helpers.resolve_suffixes(cfg, task=task, granularity=granularity)), index=False)



def run_test_scores(
        task: str | None,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_labels_train: pd.Series,
        y_labels_test: pd.Series,
        cfg: Config
    ) -> None:

    model = helpers.get_fitted_model(X_train, y_labels_train, cfg)

    score = helpers.get_test_score(X_test, y_labels_test, model, cfg)

    pd.DataFrame([{
        "dataset": cfg.dataset_name,
        "model": cfg.model_name,
        "concepts": cfg.dataset_specs[cfg.dataset_name].get("concepts", ""),
        "roc_auc": score
    }]).to_csv(cfg.test_score_path.format(task=helpers.resolve_task_suffix(cfg, task)), index=False)
