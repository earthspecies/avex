"""
AudioProtoPNet species detection example.

Demonstrates two complementary detection modes on a directory of audio files:

  Clip-level (AudioProtoPNet)
    Global max-pool over the prototype activation map → a single ranked list of
    species per audio clip.  Use this for surveys where per-clip presence/absence
    is sufficient.

  Frame-level (AudioProtoPNet SED)
    Frequency max-pool only, sigmoid per time frame → per-second species
    probabilities.  Use this when you need to locate events within a recording.

Both models share the same ConvNeXt-Base backbone and were trained on
BirdSet-XCL (9 736 eBird species, 32 kHz).

Usage
-----
# Demo mode — synthetic noise, no audio files needed:
    python audioprotopnet_species_detection.py --demo

# Single file:
    python audioprotopnet_species_detection.py --audio recording.wav

# Directory of audio files:
    python audioprotopnet_species_detection.py --audio /path/to/recordings/

# Skip frame-level SED (if checkpoint_dir not available):
    python audioprotopnet_species_detection.py --audio /path/to/recordings/ --skip-sed

Requirements
------------
    pip install torchaudio transformers

For the SED model, set --checkpoint-dir to a local directory containing:
    config.pt  backbone_state_dict.pt  head_state_dict.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch

_SAMPLE_RATE = 32_000
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus"}


# ── Audio loading ─────────────────────────────────────────────────────────────


def load_audio(path: Path, target_sr: int = _SAMPLE_RATE) -> torch.Tensor:
    """Load an audio file and resample to *target_sr*.

    Returns
    -------
    torch.Tensor
        Shape ``(num_samples,)`` (mono, float32).
    """
    try:
        import torchaudio
        import torchaudio.transforms as T
    except ImportError:
        print("torchaudio is required.  Install with: pip install torchaudio")
        sys.exit(1)

    waveform, sr = torchaudio.load(path)
    waveform = waveform.mean(dim=0)  # stereo → mono
    if sr != target_sr:
        waveform = T.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
    return waveform


def collect_audio_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in _AUDIO_EXTENSIONS)


# ── Inference helpers ─────────────────────────────────────────────────────────


def top_k_species(
    logits: torch.Tensor,
    label_mapping: dict[int, str],
    k: int = 5,
) -> list[tuple[str, float]]:
    """Return the top-*k* species from a clip-level logit vector.

    Parameters
    ----------
    logits:
        Shape ``(num_classes,)`` — raw logits from ``forward()``.
    label_mapping:
        ``{class_index: common_name}`` dict from ``model.label_mapping``.

    Returns
    -------
    list of (common_name, probability) tuples, highest first.
    """
    probs = torch.sigmoid(logits)
    topk_probs, topk_idx = torch.topk(probs, k=min(k, probs.shape[-1]))
    return [
        (label_mapping.get(idx.item(), f"class_{idx.item()}"), prob.item())
        for prob, idx in zip(topk_probs, topk_idx, strict=False)
    ]


def frame_events(
    frame_probs: torch.Tensor,
    label_mapping: dict[int, str],
    threshold: float = 0.5,
    frames_per_second: float = 7.8,
    top_k: int = 3,
) -> list[dict]:
    """Convert per-frame probabilities to a list of detected events.

    Parameters
    ----------
    frame_probs:
        Shape ``(T, num_classes)`` — sigmoid probabilities from ``forward_frames()``.
    label_mapping:
        ``{class_index: common_name}`` from ``model.label_mapping``.
    threshold:
        Minimum probability to report an activation as a detection.
    frames_per_second:
        Temporal resolution of the frame grid (computed from audio length below).
    top_k:
        Maximum species to report per frame.

    Returns
    -------
    List of dicts: ``{time_s, species, probability}``.
    """
    events = []
    for t, probs in enumerate(frame_probs):
        above = (probs >= threshold).nonzero(as_tuple=True)[0]
        if above.numel() == 0:
            continue
        sorted_idx = above[probs[above].argsort(descending=True)][:top_k]
        time_s = t / frames_per_second
        for idx in sorted_idx:
            events.append(
                {
                    "time_s": round(time_s, 2),
                    "species": label_mapping.get(idx.item(), f"class_{idx.item()}"),
                    "probability": round(probs[idx].item(), 3),
                }
            )
    return events


# ── Clip-level detection ──────────────────────────────────────────────────────


def run_clip_detection(
    audio_files: list[Path],
    device: str,
    model_id: str,
    top_k: int,
) -> None:
    """Run AudioProtoPNet clip-level detection on *audio_files*."""
    from avex.models.audioprotopnet import Model as AudioProtoPNet

    print("\n── Clip-level detection (AudioProtoPNet) ───────────────────────────")
    print(f"Loading {model_id} …")
    model = AudioProtoPNet(pretrained=True, device=device, model_id=model_id)
    model.eval()

    label_mapping = model.label_mapping or {i: model.ebird_codes.get(i, str(i)) for i in range(model.num_classes)}
    print(f"Loaded: {model.num_classes} classes\n")

    for path in audio_files:
        waveform = load_audio(path).unsqueeze(0).to(device)  # [1, T]
        with torch.no_grad():
            logits = model(waveform)[0]  # [C]

        predictions = top_k_species(logits, label_mapping, k=top_k)
        print(f"  {path.name}")
        for rank, (name, prob) in enumerate(predictions, 1):
            bar = "█" * int(prob * 20)
            print(f"    {rank}. {name:<35} {prob:.3f}  {bar}")
        print()


# ── Frame-level SED ───────────────────────────────────────────────────────────


def run_sed_detection(
    audio_files: list[Path],
    device: str,
    checkpoint_dir: Optional[str],
    threshold: float,
    top_k: int,
) -> None:
    """Run AudioProtoPNet SED frame-level detection on *audio_files*."""
    from avex.models.audioprotopnet_sed import Model as AudioProtoPNetSED

    print("\n── Frame-level detection (AudioProtoPNet SED) ──────────────────────")
    print(f"Loading checkpoint from {checkpoint_dir or 'gs://representation-learning/models/sed/audioprotopnet-20/'} …")
    model = AudioProtoPNetSED(pretrained=True, device=device, checkpoint_dir=checkpoint_dir)
    model.eval()

    label_mapping = model.label_mapping or {i: model.ebird_codes.get(i, str(i)) for i in range(model.num_classes)}
    print(f"Loaded: {model.num_classes} classes\n")

    for path in audio_files:
        waveform = load_audio(path)
        duration_s = waveform.shape[-1] / _SAMPLE_RATE
        waveform = waveform.unsqueeze(0).to(device)  # [1, T]

        with torch.no_grad():
            frame_probs = model.forward_frames(waveform)[0]  # [T, C]

        n_frames = frame_probs.shape[0]
        fps = n_frames / duration_s

        events = frame_events(frame_probs, label_mapping, threshold=threshold, frames_per_second=fps, top_k=top_k)

        print(f"  {path.name}  ({duration_s:.1f}s  →  {n_frames} frames, {fps:.1f} fps)")
        if not events:
            print(f"    No detections above threshold {threshold}")
        else:
            for ev in events:
                bar = "█" * int(ev["probability"] * 20)
                print(f"    {ev['time_s']:6.2f}s  {ev['species']:<35} {ev['probability']:.3f}  {bar}")
        print()


# ── Demo mode ─────────────────────────────────────────────────────────────────


def _make_demo_files(n: int = 3, duration_s: int = 5) -> list[Path]:
    """Write *n* silent WAV files to /tmp for demo purposes.

    Returns
    -------
    list[Path]
        Paths to the written WAV files.
    """
    try:
        import torchaudio
    except ImportError:
        print("torchaudio is required.  Install with: pip install torchaudio")
        sys.exit(1)

    out_dir = Path("/tmp/apn_demo_audio")
    out_dir.mkdir(exist_ok=True)
    paths = []
    for i in range(n):
        p = out_dir / f"demo_{i + 1:02d}.wav"
        audio = torch.randn(1, _SAMPLE_RATE * duration_s) * 0.05  # quiet noise
        torchaudio.save(str(p), audio, _SAMPLE_RATE)
        paths.append(p)
    print(f"Demo: wrote {n} synthetic audio files to {out_dir}")
    return paths


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AudioProtoPNet species detection — clip-level and frame-level",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Path to a single audio file or directory of audio files.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate synthetic audio and run both models (no real audio needed).",
    )
    parser.add_argument(
        "--model-id",
        default="DBD-research-group/AudioProtoPNet-20-BirdSet-XCL",
        help="HuggingFace model ID for the clip-level AudioProtoPNet.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Local checkpoint directory for AudioProtoPNet SED "
        "(config.pt / backbone_state_dict.pt / head_state_dict.pt).",
    )
    parser.add_argument(
        "--skip-sed",
        action="store_true",
        help="Skip the frame-level SED model (e.g. if checkpoint-dir is unavailable).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection probability threshold for frame-level events.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top species to display per clip / frame.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device.",
    )
    args = parser.parse_args()

    if not args.demo and args.audio is None:
        parser.error("Provide --audio <path> or --demo.")

    print("AudioProtoPNet species detection")
    print(f"Device: {args.device}")

    audio_files = _make_demo_files() if args.demo else collect_audio_files(args.audio)
    if not audio_files:
        print(f"No audio files found at {args.audio}")
        sys.exit(1)
    print(f"Audio files: {len(audio_files)}")

    # Clip-level
    run_clip_detection(
        audio_files=audio_files,
        device=args.device,
        model_id=args.model_id,
        top_k=args.top_k,
    )

    # Frame-level SED
    if args.skip_sed:
        print("\nSkipping SED (--skip-sed).")
    elif args.checkpoint_dir is None and not args.demo:
        print("\nSkipping SED — no --checkpoint-dir provided.  Set --checkpoint-dir to run frame-level detection.")
    else:
        run_sed_detection(
            audio_files=audio_files,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
            threshold=args.threshold,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
