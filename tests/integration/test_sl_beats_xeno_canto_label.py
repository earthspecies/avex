"""Real-data label regression for the SL-BEATs-all classifier.

Loads ``esp_aves2_sl_beats_all`` with its class mapping and runs a real
Xeno-canto recording of a European turtle dove (*Streptopelia turtur*,
XC564654) bundled as a test fixture. The model must keep predicting that
species for that clip — a guard against silent changes to the BEATs forward
path (e.g. attention/frontend) that would shift classifier outputs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

# Real Xeno-canto clip (XC564654), 10s mono 16 kHz PCM, committed as a fixture.
_FIXTURE = Path(__file__).parent / "fixtures" / "xeno_canto" / "XC564654_Streptopelia_turtur.wav"
_MODEL_KEY = "esp_aves2_sl_beats_all"
_EXPECTED_LABEL = "Streptopelia turtur"
_EXPECTED_SAMPLE_RATE = 16_000


def test_sl_beats_all_classifies_xeno_canto_turtle_dove() -> None:
    sf = pytest.importorskip("soundfile")
    from avex import load_model
    from avex.models.utils.load import load_label_mapping
    from avex.models.utils.registry import get_checkpoint_path

    assert _FIXTURE.is_file(), f"Missing fixture audio: {_FIXTURE}"

    # Skip gracefully when the official checkpoint isn't reachable (e.g. no creds).
    if get_checkpoint_path(_MODEL_KEY) is None:
        pytest.skip(f"Checkpoint not available in registry for {_MODEL_KEY!r}")
    try:
        model = load_model(_MODEL_KEY, device="cpu")
    except Exception as exc:  # noqa: BLE001 - network/credential/availability issues
        pytest.skip(f"Unable to load {_MODEL_KEY!r}: {exc}")
    model.eval()

    mapping = load_label_mapping(_MODEL_KEY)
    assert mapping is not None, f"No class mapping for {_MODEL_KEY!r}"
    index_to_label = mapping["index_to_label"]

    waveform, sample_rate = sf.read(str(_FIXTURE), dtype="float32")
    assert sample_rate == _EXPECTED_SAMPLE_RATE, f"Fixture must be {_EXPECTED_SAMPLE_RATE} Hz, got {sample_rate}"
    if waveform.ndim > 1:  # collapse to mono
        waveform = waveform.mean(axis=1)
    audio = torch.from_numpy(waveform).unsqueeze(0)  # [1, T]

    with torch.no_grad():
        output = model(audio)
    logits = output[0] if isinstance(output, (tuple, list)) else output
    assert logits.shape[-1] == len(index_to_label), "Classifier width != label-mapping size"

    top_index = int(logits.argmax(dim=-1)[0])
    # index_to_label may be keyed by int or str depending on how the JSON was loaded.
    predicted_label = index_to_label.get(str(top_index), index_to_label.get(top_index))

    assert predicted_label == _EXPECTED_LABEL, (
        f"Expected {_EXPECTED_LABEL!r} for XC564654, got {predicted_label!r} (index {top_index})"
    )
