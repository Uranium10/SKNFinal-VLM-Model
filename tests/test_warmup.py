"""Stub the SDK registration and runtime: these tests never allocate a GPU."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_healthcheck_stays_cheap_and_warmup_loads_without_documents():
    registration = MagicMock()
    runtime = MagicMock(loaded=False, load_seconds=12.3)
    spec = importlib.util.spec_from_file_location('warmup_test_handler', Path(__file__).resolve().parents[1] / 'handler.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'runpod': SimpleNamespace(serverless=SimpleNamespace(start=registration))}), \
         patch('model_runtime.QuotationModelRuntime', return_value=runtime):
        spec.loader.exec_module(module)
        assert module.handler({'input': {'healthcheck': True}})['model_loaded'] is False
        runtime.load.assert_not_called()
        runtime.loaded = True
        with patch.object(module, 'read_documents') as documents:
            result = module.handler({'input': {'warmup': True}})
            assert result['model_loaded'] is True
            runtime.load.assert_called_once()
            runtime.extract.assert_not_called()
            documents.assert_not_called()
