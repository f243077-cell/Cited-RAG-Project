import pytest

from citerag.eval.metrics import estimate_cost_usd
from citerag.llm.pricing import load_pricing

GOOD = """
retrieved_on: "2026-11-06"
source_url: "https://example.org/model"
reference_model: "vendor/model:free"
models:
  "vendor/model:free": {input_per_mtok: 0.0, output_per_mtok: 0.0}
"""


def _write(tmp_path, text):
    path = tmp_path / "pricing.yaml"
    path.write_text(text)
    return path


def test_reference_entry_is_exposed(tmp_path):
    pricing = load_pricing(_write(tmp_path, GOOD))
    assert pricing["_reference"] == {"input_per_mtok": 0.0, "output_per_mtok": 0.0}
    assert estimate_cost_usd("some/other:free", 1000, 1000, pricing) == 0.0  # falls back to the reference


def test_repo_pricing_file_is_filled_and_free(tmp_path):
    pricing = load_pricing()
    assert pricing["_reference"]["input_per_mtok"] == 0.0 and pricing["_reference"]["output_per_mtok"] == 0.0


@pytest.mark.parametrize(
    "text",
    [
        'reference_model: ""\nmodels: {}\nretrieved_on: "x"\nsource_url: "y"\n',
        'reference_model: "a"\nmodels:\n  "b": {input_per_mtok: 0, output_per_mtok: 0}\nretrieved_on: "x"\nsource_url: "y"\n',
        GOOD.replace('retrieved_on: "2026-11-06"', 'retrieved_on: ""'),
    ],
)
def test_incomplete_file_is_rejected(tmp_path, text):
    with pytest.raises(ValueError):
        load_pricing(_write(tmp_path, text))
