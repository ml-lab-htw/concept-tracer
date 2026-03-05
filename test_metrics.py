import numpy as np
import pandas as pd
import pytest

from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_classif

from concept_tracer.config import Config



def get_stats(
        concepts: np.ndarray,
        embeddings: np.ndarray,
        j: int,
        cfg: Config,
        estimator_name: str = "normalized_mutual_info_score",
        n_bins: int | None = 10,
        permutation_seed: int | None = None
    ) -> tuple[int, np.ndarray]:

    stats = []
    concepts = concepts[j]

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

        for layer in range(embeddings.shape[0]):

            disc = KBinsDiscretizer(
                n_bins=n_bins,
                encode="ordinal",
                strategy="quantile"
            )

            disc_embeddings = disc.fit_transform(embeddings[layer])

            for neuron in range(disc_embeddings.shape[1]):
                
                I = scorer(concepts, disc_embeddings[:, neuron])

                stats.append((layer, neuron, I))
    
    # slow, unnormalized estimation
    elif estimator_name == "mutual_info_classif":

        for layer in range(embeddings.shape[0]):
            for neuron in range(embeddings.shape[2]):

                I = mutual_info_classif(
                    embeddings[layer, :, neuron].reshape(-1, 1),
                    concepts,
                    random_state=cfg.random_seed
                )[0]

                stats.append((layer, neuron, I))

    else:
        
        raise NotImplementedError("The estimator is not supported yet.")

    return j, np.asarray(stats).T



def run_saliency_selectivity(
        concept_names: list[str],
        concepts: np.ndarray,
        embeddings: np.ndarray,
        cfg: Config,
        estimator_name: str = "normalized_mutual_info_score",
        n_bins: int | None = 10
    ) -> pd.DataFrame:

    # concepts x test samples
    d_c = concepts.shape[0]

    # layers x test samples x embedding dimension
    d_l, _, d_n = embeddings.shape

    saliency_sum = np.zeros((d_l, d_n))
    cache = []
    
    # get stats
    for j in range(d_c):
        _, stats = get_stats(
            concepts=concepts,
            embeddings=embeddings,
            j=j,
            cfg=cfg,
            estimator_name=estimator_name,
            n_bins=n_bins
        )

        concept = concept_names[j]
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

    for k in range(cfg.n_permutations):

        saliency_sum = np.zeros((d_l, d_n))
        saliency_max = 0.0
        cache = []

        for j in range(d_c):
            _, stats = get_stats(
                 concepts=concepts,
                 embeddings=embeddings,
                 j=j,
                 cfg=cfg,
                 estimator_name=estimator_name,
                 n_bins=n_bins,
                 permutation_seed=cfg.random_seed + k + 100000
            )

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

    return df





@pytest.mark.parametrize(
    "estimator_name, n_bins",
    [
        ("normalized_mutual_info_score", 5),
        ("normalized_mutual_info_score", 10),
        ("normalized_mutual_info_score", 15),
        ("mutual_info_score", 5),
        ("mutual_info_score", 10),
        ("mutual_info_score", 15),
        # fails max selectivity and significant selectivity test
        ("mutual_info_classif", None)
    ]
)
def test_case(estimator_name: str, n_bins: int):
    
    cfg = Config()
    rng = np.random.default_rng(cfg.random_seed)

    n_samples = 500
    d_l, d_n, d_c = 2, 10, 5
    signal_layer, signal_neuron, signal_concept = 1, 5, 0

    concept_names = [f"c{i}" for i in range(d_c)]
    concepts = rng.integers(0, 2, size=(d_c, n_samples))
    embeddings = rng.normal(size=(d_l, n_samples, d_n))
    embeddings[signal_layer, :, signal_neuron] = concepts[signal_concept] + 1e-6 * rng.normal(size=n_samples)

    df = run_saliency_selectivity(
        concept_names=concept_names,
        concepts=concepts,
        embeddings=embeddings,
        cfg=cfg,
        estimator_name=estimator_name,
        n_bins=n_bins
    )

    # 1) signal row exists
    hit = df[(df["concept"] == f"c{signal_concept}") & (df["layer"] == signal_layer) & (df["neuron"] == signal_neuron)]
    assert len(hit) == 1
    hit = hit.iloc[0]

    # 2) signal neuron-concept pair has maximal saliency and selectivity
    assert np.isclose(hit["saliency"], df["saliency"].max(), atol=1e-6)
    assert np.isclose(hit["selectivity"], df["selectivity"].max(), atol=1e-6)

    # 3) signal neuron-concept pair is significant
    assert hit["p_saliency"] < cfg.significance_threshold
    assert hit["p_selectivity"] < cfg.significance_threshold
    assert hit["p_combined"] < cfg.significance_threshold

    # 4) no other neuron-concept pairs are significant
    sub = df[~((df["concept"] == f"c{signal_concept}") & (df["layer"] == signal_layer) & (df["neuron"] == signal_neuron) )]
    assert (sub["p_saliency"] > cfg.significance_threshold).all()
    assert (sub["p_selectivity"] > cfg.significance_threshold).all()
    assert (sub["p_combined"] > cfg.significance_threshold).all()

    # 5) selectivities sum to 1.0
    sub = df.dropna(subset=["selectivity"]).groupby(["layer", "neuron"])["selectivity"]
    assert np.allclose(sub.sum(), 1.0, atol=1e-6)

    # 6) p-values are never smaller than 1 / (cfg.n_permutations + 1)
    pmin = 1 / (cfg.n_permutations + 1)
    assert (df["p_saliency"] >= pmin - 1e-6).all()
    assert (df["p_selectivity"].dropna() >= pmin - 1e-6).all()
    assert (df["p_combined"].dropna() >= pmin - 1e-6).all()
    