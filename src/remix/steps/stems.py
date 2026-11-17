"""Split the soundtrack into speech and background (music, effects, ambience) with Demucs."""
from __future__ import annotations

import math

from .. import config
from ..project import Project


def dbfs(rms: float) -> float:
    return round(20 * math.log10(max(rms, 1e-9)), 1)


def run(project: Project, keep_stems: bool = False) -> dict:
    audio_dir = project.root / "audio"
    source = audio_dir / "original.wav"
    if not source.is_file():
        return {"status": "skipped", "reason": "no audio to separate"}

    import numpy as np
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    samples, rate = sf.read(str(source), dtype="float32", always_2d=True)
    model = get_model(config.DEMUCS_MODEL)
    model.eval()
    if rate != model.samplerate:
        raise RuntimeError(f"expected {model.samplerate} Hz audio, got {rate} Hz")

    mix = torch.from_numpy(np.ascontiguousarray(samples.T))
    reference = mix.mean(0)
    mean, std = reference.mean(), reference.std()
    if float(std) < 1e-8:
        return {"status": "skipped", "reason": "audio track is silent"}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # shifts=0 skips Demucs's random time shift, so the same input always gives the same stems.
    with torch.no_grad():
        separated = apply_model(
            model, ((mix - mean) / std)[None], device=device, shifts=0, split=True, overlap=0.25
        )[0]
    separated = (separated * std + mean).cpu()

    names = list(model.sources)
    vocals = names.index("vocals")
    speech = separated[vocals]
    background = sum(separated[i] for i in range(len(names)) if i != vocals)

    # One shared gain keeps the two stems in their original balance if either would clip.
    peak = max(float(speech.abs().max()), float(background.abs().max()))
    gain = 1.0 / peak if peak > 1.0 else 1.0

    def write(path, tensor) -> float:
        data = np.clip((tensor * gain).numpy().T, -1.0, 1.0)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), data, rate, subtype="PCM_16")
        return dbfs(float(np.sqrt(np.mean(np.square(data)))))

    speech_path, background_path = audio_dir / "speech.wav", audio_dir / "background.wav"
    result = {
        "status": "done",
        "model": config.DEMUCS_MODEL,
        "device": device,
        "speech_dbfs": write(speech_path, speech),
        "background_dbfs": write(background_path, background),
        "gain": round(gain, 4),
        "outputs": [project.rel(speech_path), project.rel(background_path)],
    }
    if keep_stems:
        for i, name in enumerate(names):
            path = audio_dir / "stems" / f"{name}.wav"
            write(path, separated[i])
            result["outputs"].append(project.rel(path))
    return result
