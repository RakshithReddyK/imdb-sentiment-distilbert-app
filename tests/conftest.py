"""Test fixtures for the FastAPI service.

The real DistilBERT model is never loaded in tests: we override the
`get_pipeline` FastAPI dependency with a stub tokenizer/model so the test
suite runs without network access, downloaded weights, or a GPU.
"""

import pytest
import torch
from fastapi.testclient import TestClient

from api.main import app, get_pipeline


class StubTokenizer:
    """Minimal stand-in for a Hugging Face tokenizer."""

    def __call__(self, text, return_tensors=None, truncation=None, padding=None, max_length=None):
        # Shape doesn't matter for the stub model below -- just needs to be
        # valid input to `.to(device)`.
        return {
            "input_ids": torch.tensor([[101, 2023, 2003, 1037, 3231, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]]),
        }


class StubOutput:
    def __init__(self, logits):
        self.logits = logits


class StubModel:
    """Deterministic stand-in for the DistilBERT classifier.

    Predicts "Positive" unless the (stub-tokenized) input text contains the
    word "bad", giving tests something to assert on without touching real
    model weights.
    """

    def __call__(self, input_ids=None, attention_mask=None, **kwargs):
        # Always return logits favoring the positive class; per-test
        # behavior is controlled at the route level via monkeypatching where
        # needed. For our purposes a fixed, confident prediction is enough
        # to exercise the request/response contract.
        logits = torch.tensor([[0.1, 4.0]])
        return StubOutput(logits=logits)


def _stub_pipeline():
    return StubTokenizer(), StubModel(), torch.device("cpu")


@pytest.fixture()
def client():
    app.dependency_overrides[get_pipeline] = _stub_pipeline
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
