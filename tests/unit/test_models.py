import numpy as np
from scipy import sparse

from reco.models.base import minmax
from reco.models.item_knn import fit_item_similarity, prune_top_k
from reco.models.registry import MODEL_REGISTRY


def test_minmax_constant_and_values() -> None:
    np.testing.assert_array_equal(minmax(np.array([3, 3])), np.array([0, 0]))
    np.testing.assert_allclose(minmax(np.array([1, 2, 3])), np.array([0, 0.5, 1]))


def test_top_k_pruning_removes_diagonal() -> None:
    matrix = sparse.csr_matrix(np.array([[9, 3, 2], [4, 8, 1], [7, 6, 5]], dtype=np.float32))
    pruned = prune_top_k(matrix, 1)
    assert max(np.diff(pruned.indptr)) <= 1
    assert np.allclose(pruned.diagonal(), 0)
    assert pruned[0, 1] == 3


def test_item_similarity_is_sparse_and_bounded() -> None:
    history = sparse.csr_matrix(np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.float32))
    similarity = fit_item_similarity(history, neighbors=1)
    assert similarity.shape == (3, 3)
    assert max(np.diff(similarity.indptr)) <= 1


def test_model_registry_has_all_config_names() -> None:
    assert set(MODEL_REGISTRY) == {"popularity", "item_knn", "content", "hybrid"}
