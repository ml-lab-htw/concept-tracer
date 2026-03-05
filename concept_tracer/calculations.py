import numpy as np
import pandas as pd
import random
import torch

from collections import Counter
from typing import Callable

from . import helpers
from . import processes
from .config import Config


GetDataFn = Callable[
    [str | None, Config],
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series],
]


def run(cfg: Config | None = None, get_data_fn: GetDataFn | None = None):

    if cfg is None:
        cfg = Config()
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    
    if get_data_fn is None:
        get_data_fn = helpers.get_data

    for task in cfg.dataset_specs[cfg.dataset_name].get("tasks", []) or [None]:

        X_train, X_test, _, y_concepts_test, y_labels_train, y_labels_test = get_data_fn(
            task, cfg
        )

        processes.run_test_scores(task, X_train, X_test, y_labels_train, y_labels_test, cfg)

        processes.run_embeddings(task, X_train, X_test, y_labels_train, cfg)

        granularities = cfg.dataset_specs[cfg.dataset_name].get("granularities", {}) or {}
        if granularities:
            y_concepts_test_granularity = {
                granularity: [set(granularities[granularity](cs)) for cs in y_concepts_test]
                for granularity in granularities
            }
        else:
            y_concepts_test_granularity = {None: [set(cs) for cs in y_concepts_test]}
        
        # concept filtering based on concept prevalence
        for granularity, rows in y_concepts_test_granularity.items():
            prevalence = Counter(concept for row in rows for concept in row)
            filtered_concepts = {
                concept for concept, count in prevalence.items()
                if count >= cfg.concept_prevalence_threshold
            }
            y_concepts_test_granularity[granularity] = [row & filtered_concepts for row in rows]

        for granularity, rows in y_concepts_test_granularity.items():

            if not set().union(*rows):
                print(f"[{task}] {granularity}: No concepts survived concept prevalence threshold, skipping downstream steps.")
                continue

            processes.run_concepts(task, granularity, y_concepts_test_granularity, cfg)

            processes.run_baselines(task, granularity, y_concepts_test_granularity, cfg)

            processes.run_saliency_selectivity(task, granularity, y_concepts_test_granularity, cfg)


def main():
    run()


if __name__ == "__main__":
    main()
