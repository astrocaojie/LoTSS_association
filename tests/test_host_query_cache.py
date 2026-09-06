import json

import pandas as pd
import pytest

from lotss_association.host_query import HOST_RAW_COLUMNS, HostQueryClient


def test_save_cache_always_writes_jsonl_fallback(tmp_path):
    client = HostQueryClient(tmp_path)
    record = {column: None for column in HOST_RAW_COLUMNS}
    record.update(
        {
            "query_id": "catwise2020_1.0000000_2.0000000_3.00",
            "catalogue": "catwise2020",
            "host_id": "J000000.00+000000.0",
            "host_ra": 1.0,
            "host_dec": 2.0,
        }
    )
    client._cache = pd.DataFrame([record])

    client.save_cache()

    assert client.jsonl_path.exists()
    with client.jsonl_path.open("r", encoding="utf-8") as handle:
        saved = [json.loads(line) for line in handle if line.strip()]
    assert saved[0]["query_id"] == record["query_id"]
    assert saved[0]["host_id"] == record["host_id"]


def test_load_cache_warns_and_skips_malformed_jsonl_rows(tmp_path, caplog):
    path = tmp_path / "host_query_cache.jsonl"
    valid = {column: None for column in HOST_RAW_COLUMNS}
    valid.update({"query_id": "q0", "catalogue": "allwise", "host_id": "h0"})
    path.write_text("not-json\n" + json.dumps(valid) + "\n", encoding="utf-8")

    with caplog.at_level("WARNING"):
        client = HostQueryClient(tmp_path)

    assert len(client._cache) == 1
    assert "malformed host cache row" in caplog.text


def test_save_cache_raises_when_both_persistence_formats_fail(tmp_path, monkeypatch):
    client = HostQueryClient(tmp_path)
    client._cache = pd.DataFrame([{column: None for column in HOST_RAW_COLUMNS}])

    def fail_parquet(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_parquet)

    class FailingPath:
        def __enter__(self):
            raise OSError("permission denied")

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr("lotss_association.host_query.Path.open", lambda *_args, **_kwargs: FailingPath())
    with pytest.raises(RuntimeError, match="Unable to persist host query cache"):
        client.save_cache()
