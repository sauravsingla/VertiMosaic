import io
import json
from urllib.error import URLError

from vertimosaic.datasets import registry as registry_module
from vertimosaic.datasets.registry import DatasetRegistry


class _Response(io.BytesIO):
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def test_openml_runtime_license_uses_official_json_metadata(monkeypatch) -> None:
    payload = {"data_set_description": {"licence": "ODbL-1.0"}}

    def fake_urlopen(*_: object, **__: object) -> _Response:
        return _Response(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(registry_module, "urlopen", fake_urlopen)
    registry = DatasetRegistry()
    assert registry.runtime_license("insurance_freq") == "ODbL-1.0"
    assert registry.verify_license_metadata("insurance_freq") is True


def test_openml_license_verification_fails_conservatively(monkeypatch) -> None:
    def fail_urlopen(*_: object, **__: object) -> _Response:
        raise URLError("offline")

    monkeypatch.setattr(registry_module, "urlopen", fail_urlopen)
    registry = DatasetRegistry()
    assert registry.runtime_license("insurance_sev") is None
    assert registry.verify_license_metadata("insurance_sev") is False


def test_uci_registry_license_verification_remains_offline() -> None:
    registry = DatasetRegistry()
    assert registry.verify_license_metadata("bank") is True
    assert registry.verify_license_metadata("telecom") is True
    assert registry.verify_license_metadata("retail") is True
