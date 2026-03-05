"""
Monkey‑patch for TabPFN to expose the target‑column representation
of any encoder layer as an embedding.

Usage
-----
>>> import tabpfn_layer_patch
>>> from tabpfn_extensions import TabPFNClassifier
>>> from tabpfn_extensions.embedding import TabPFNEmbedding
>>> tabpfn_layer_patch.set_embedding_layer_idx(0)  # first layer
>>> model = TabPFNClassifier(n_estimators=1)
>>> embedding_extractor = TabPFNEmbedding(tabpfn_clf=model)
>>> embeddings = embedding_extractor.get_embeddings(
        X_train, y_labels_train, X_test, data_source="test"
    )

Call ``tabpfn_layer_patch.set_embedding_layer_idx(None)`` (or never call it)
to keep the default behaviour (final encoder layer as an embedding).
"""

import torch

from functools import partial
from torch.utils.checkpoint import checkpoint
from typing import Any, Optional


_LAYER_IDX: Optional[int] = None


def set_embedding_layer_idx(idx: Optional[int]) -> None:
    """Select which encoder layer TabPFN should hand back.

    Parameters
    ----------
    idx : int | None
        * Positive values count from the start (``0`` = first layer).
        * Negative values count from the end (``-1`` = last layer = default behavior).
        * ``None`` resets to the default behaviour (last layer).
    """
    global _LAYER_IDX
    _LAYER_IDX = idx


def _patch_layerstack() -> None:
    """Replace ``LayerStack.forward`` with an index‑aware version."""

    from tabpfn.architectures.base.transformer import LayerStack

    if getattr(LayerStack, "_patched_for_layer_idx", False):
        return
    
    def forward(
        self,
        x: torch.Tensor,
        recompute_layer: bool,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Patched ``forward`` that tracks all hidden states and can return any of them."""

        # Original logic...
        n_layers = torch.randint(
            low=self.min_num_layers_layer_dropout, high=len(self.layers) + 1, size=(1,)
        ).item()

        # ... but with a list for the layers...
        outs: list[torch.Tensor] = []

        for layer in self.layers[:n_layers]:
            if recompute_layer and x.requires_grad:
                x = checkpoint(partial(layer, **kwargs), x, use_reentrant=False)  # type: ignore
            else:
                x = layer(x, **kwargs)
            
            outs.append(x)
        
        # Get specified layer
        idx: Optional[int] = getattr(self, "_layer_idx", _LAYER_IDX)
        if idx is not None:
            x = outs[idx]

        return x

    # Apply monkey‑patch
    LayerStack.forward = forward
    LayerStack._patched_for_layer_idx = True


# Run patch at import
_patch_layerstack()

__all__ = ["set_embedding_layer_idx"]
