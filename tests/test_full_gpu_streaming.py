"""Correctness and memory regression tests for P0.1's batch transfer path."""

import numpy as np
import pytest
import torch

from rctd import _full


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_stream_gpu_matches_resident_path(monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    # Stub ONLY the IRWLS solver to isolate the transfer/batching behavior.
    def fake_solver(*, P, Y_batch, nUMI_batch, **_):
        k = P.shape[1]
        return (
            Y_batch[:, :k] / nUMI_batch[:, None],
            torch.ones(Y_batch.shape[0], device=Y_batch.device, dtype=torch.bool),
        )

    monkeypatch.setattr(_full, "solve_irwls_batch_shared", fake_solver)
    rng = np.random.default_rng(123)
    n, g, k = 19, 13, 3
    Y = rng.integers(1, 9, size=(n, g)).astype(np.float64)
    u = Y.sum(axis=1)
    P = rng.random((g, k)).astype(np.float64)
    args = dict(
        spatial_counts=Y,
        spatial_numi=u,
        norm_profiles=P,
        cell_type_names=["A", "B", "C"],
        q_mat=np.zeros((3, 4), dtype=np.float64),
        sq_mat=np.zeros((3, 4), dtype=np.float64),
        x_vals=np.arange(4, dtype=np.float64),
        batch_size=7,
        device=device,
    )
    baseline = _full.run_full_mode(**args, stream_gpu=False)
    streamed = _full.run_full_mode(**args, stream_gpu=True)
    np.testing.assert_array_equal(streamed.weights, baseline.weights)
    np.testing.assert_array_equal(streamed.converged, baseline.converged)
    assert streamed.weights.shape == (n, k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_stream_gpu_avoids_full_counts_allocation(monkeypatch):
    def fake_solver(*, P, Y_batch, nUMI_batch, **_):
        k = P.shape[1]
        return (
            torch.zeros((Y_batch.shape[0], k), device=Y_batch.device, dtype=P.dtype),
            torch.ones(Y_batch.shape[0], device=Y_batch.device, dtype=torch.bool),
        )

    monkeypatch.setattr(_full, "solve_irwls_batch_shared", fake_solver)
    n, g, k = 8192, 512, 3
    Y = np.ones((n, g), dtype=np.float64)
    u = np.full(n, g, dtype=np.float64)
    P = np.full((g, k), 1.0 / g, dtype=np.float64)
    args = dict(
        spatial_counts=Y,
        spatial_numi=u,
        norm_profiles=P,
        cell_type_names=["A", "B", "C"],
        q_mat=np.zeros((3, 4)),
        sq_mat=np.zeros((3, 4)),
        x_vals=np.arange(4, dtype=np.float64),
        batch_size=256,
        device="cuda",
    )
    peaks = {}
    for stream in [False, True]:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        result = _full.run_full_mode(**args, stream_gpu=stream)
        torch.cuda.synchronize()
        peaks[stream] = torch.cuda.max_memory_allocated()
        del result
    # 8192 x 512 x 8 bytes = 32 MiB; streaming input is only 1 MiB.
    assert peaks[False] - peaks[True] > 16 * 1024 * 1024, peaks
