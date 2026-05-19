"""Triton Python backend: cuML UMAP -> cuML HDBSCAN for BERTopic.

Drop-in GPU replacement for the CPU BERTopic stage in
``functions/src/functions/cultural_agent/agent.py:1007-1034``. Receives
pre-computed sentence embeddings and emits integer topic labels.

Hyperparameters are read from the ``parameters`` block in ``config.pbtxt``
so prod can tune without redeploying the container. Defaults mirror the
current CPU pipeline (see ``agent.py:1007-1019``).

Mirrors the agent's actual behaviour:
- ``umap_n_neighbors`` is computed dynamically (``min(15, max(5, num_docs*0.05))``)
  rather than fixed, since the CPU code does the same.
- ``hdbscan_min_cluster_size`` and ``min_samples`` track ``min_topic_size``,
  which the agent derives from corpus size (``max(10, min(num_docs*0.015, 120))``).
  The Python backend honours either explicit values from config or the
  agent's heuristic when ``hdbscan_min_cluster_size == 0``.
"""

from __future__ import annotations

import json

import numpy as np
import triton_python_backend_utils as pb_utils

try:
    import cudf  # noqa: F401  # cuML's UMAP/HDBSCAN accept numpy too
    from cuml.cluster import HDBSCAN
    from cuml.manifold import UMAP
except ImportError as exc:  # pragma: no cover — runtime-only path
    raise RuntimeError(
        "cuML is required for the bertopic-gpu backend. Install via "
        "`pip install cuml-cu12` on container boot, or bake a custom "
        "Triton image with RAPIDS preinstalled."
    ) from exc


def _param(model_config: dict, key: str, default: str) -> str:
    return model_config.get("parameters", {}).get(key, {}).get("string_value", default)


class TritonPythonModel:
    """One instance per GPU. Reused across requests."""

    def initialize(self, args):
        cfg = json.loads(args["model_config"])
        self._umap_n_components = int(_param(cfg, "umap_n_components", "5"))
        self._umap_min_dist = float(_param(cfg, "umap_min_dist", "0.0"))
        self._umap_metric = _param(cfg, "umap_metric", "cosine")
        # 0 means "use agent heuristic from num_docs"
        self._hdbscan_min_cluster_size = int(
            _param(cfg, "hdbscan_min_cluster_size", "0")
        )
        self._hdbscan_min_samples = int(_param(cfg, "hdbscan_min_samples", "0"))
        self._hdbscan_csm = _param(cfg, "hdbscan_cluster_selection_method", "eom")

    def _derive_topic_size(self, num_docs: int) -> tuple[int, int]:
        if self._hdbscan_min_cluster_size > 0:
            mcs = self._hdbscan_min_cluster_size
        else:
            # Mirrors agent.py:982
            mcs = max(10, min(int(num_docs * 0.015), 120))
        if self._hdbscan_min_samples > 0:
            ms = self._hdbscan_min_samples
        else:
            # Mirrors agent.py:1016
            ms = max(5, int(mcs * 0.3))
        return mcs, ms

    def _derive_umap_neighbors(self, num_docs: int) -> int:
        # Mirrors agent.py:1008
        return min(15, max(5, int(num_docs * 0.05)))

    def execute(self, requests):
        responses = []
        for request in requests:
            emb_tensor = pb_utils.get_input_tensor_by_name(request, "sentence_embedding")
            embeddings = emb_tensor.as_numpy().astype(np.float32, copy=False)
            num_docs = embeddings.shape[0]

            n_neighbors = self._derive_umap_neighbors(num_docs)
            min_cluster_size, min_samples = self._derive_topic_size(num_docs)

            umap = UMAP(
                n_neighbors=n_neighbors,
                n_components=self._umap_n_components,
                min_dist=self._umap_min_dist,
                metric=self._umap_metric,
                random_state=42,
            )
            reduced = umap.fit_transform(embeddings)

            hdbscan = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
                cluster_selection_method=self._hdbscan_csm,
                prediction_data=True,
            )
            labels = hdbscan.fit_predict(reduced).astype(np.int32, copy=False)

            out = pb_utils.Tensor("topic_labels", labels)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
        return responses

    def finalize(self):
        pass
