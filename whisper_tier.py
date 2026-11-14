#!/usr/bin/env python
# whisper_tier.py
# verbatim.py icin Whisper modulu: OpenAI Whisper large-v3 (openai-whisper
# paketi, PyTorch).
#
# Ses once konusma parcalarina bolunur, her parca Whisper'a ayri verilir;
# sonuc, her bolme parcasinin tek bir aralik oldugu "whisper" tier'idir
# (bolum duzeyi). Bolme kurallari:
#   - Sessizlik: 20 ms'lik pencerelerde (10 ms adimla) enerji, dosyanin kendi
#     gurultu tabanina gore esiklenir (bkz. find_silences).
#   - Uzun bosluk (varsayilan 250 ms) her zaman bolme noktasidir; boylece
#     oncesinde ve sonrasinda uzun bosluk olan konusma, Min'den kisa olsa
#     bile ayri parca olur.
#   - Max'tan (varsayilan 15 s) uzun parca, baslangicindan itibaren
#     [Min, Max] araligindaki en uzun "kisa bosluk"tan (varsayilan >= 50 ms)
#     bolunur. Bu aralikta kisa bosluk yoksa ses 80-1000 Hz band-pass
#     filtreyle suzulur ve suzulmus sesin enerjisinin en dusuk oldugu yerden
#     bolunur. Suzulmus ses yalnizca bolme noktasini bulmak icin kullanilir;
#     Whisper'a ozgun ses verilir.
#   - Bolme noktasi sessizligin ortasidir; parcalar birbirine bitisiktir ve
#     tum sesi kapsar.
#   - Padding secildiyse her parcanin basina ve sonuna 250 ms sessizlik
#     eklenir (yalnizca bellekteki gecici ses; zamanlar etkilenmez).
#
# Model: models/whisper-large-v3/large-v3.pt. Dosya yoksa OpenAI'nin resmi
# adresinden indirilir (openai-whisper, SHA256 dogrulamasiyla).

import os
import sys

import numpy as np

WHISPER_MODEL_NAME = "large-v3"
WHISPER_MODEL_FILE = "large-v3.pt"

FRAME_SEC = 0.020
HOP_SEC = 0.010
PADDING_SEC = 0.250
BANDPASS_HZ = (80.0, 1000.0)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def _import_whisper():
    try:
        import whisper
    except ImportError:
        sys.exit("HATA: openai-whisper kurulu degil. Kurmak icin (verbatim ortaminda):\n"
                 "  pip install openai-whisper")
    return whisper


def ensure_whisper_model(model_dir):
    """models/whisper-large-v3/large-v3.pt yoksa OpenAI'nin resmi adresinden
    indirilir (openai-whisper kendi SHA256 kontrolunu yapar)."""
    path = os.path.join(model_dir, WHISPER_MODEL_FILE)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    whisper = _import_whisper()
    print(f"Whisper modeli bulunamadi: {path}")
    print(f"OpenAI'den indiriliyor: {WHISPER_MODEL_NAME} (yaklasik 3 GB)")
    os.makedirs(model_dir, exist_ok=True)
    try:
        downloaded = whisper._download(whisper._MODELS[WHISPER_MODEL_NAME], model_dir, False)
    except Exception as exc:
        sys.exit(f"HATA: Whisper modeli indirilemedi ({exc}).\n"
                 f"Modeli elle su konuma yerlestirin: {path}")
    if os.path.abspath(downloaded) != os.path.abspath(path):
        os.replace(downloaded, path)
    print("Whisper modeli indirildi.")
    return path


def load_whisper(model_dir, device=None):
    whisper = _import_whisper()
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "mps":
        device = "cpu"  # openai-whisper MPS'te sorunlu (sparse tensorler)
    print(f"Whisper modeli yukleniyor ({WHISPER_MODEL_NAME}, cihaz: {device})...")
    model = whisper.load_model(os.path.join(model_dir, WHISPER_MODEL_FILE), device=device)
    # Ozel yoldan yuklenince hizalama baslari varsayilana duser; resmi
    # large-v3 degerlerini kullan (yalnizca kelime zamanlari icin gerekir).
    model.set_alignment_heads(whisper._ALIGNMENT_HEADS[WHISPER_MODEL_NAME])
    return model


# --------------------------------------------------------------------------
# Sessizlik ve bolme
# --------------------------------------------------------------------------

def frame_energy_db(signal, sr):
    """20 ms pencere, 10 ms adim; her pencerenin RMS enerjisi (dB)."""
    win = int(round(FRAME_SEC * sr))
    hop = int(round(HOP_SEC * sr))
    if len(signal) < win:
        signal = np.pad(signal, (0, win - len(signal)))
    n = 1 + (len(signal) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = signal[idx].astype(np.float64)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    return 20.0 * np.log10(rms + 1e-12)


def silence_threshold_db(energy_db):
    """Dosyaya uyarlanan esik: gurultu tabani (10. yuzdelik) ile konusma
    duzeyi (90. yuzdelik) arasindaki farkin %25'i kadar gurultu tabaninin
    ustu; ama en az 6 dB ustu."""
    floor = np.percentile(energy_db, 10)
    speech = np.percentile(energy_db, 90)
    return floor + max(6.0, 0.25 * (speech - floor))


def find_silences(signal, sr, min_ms):
    """En az min_ms suren sessizlikler: [(baslangic_s, bitis_s)]."""
    energy = frame_energy_db(signal, sr)
    silent = energy < silence_threshold_db(energy)
    out = []
    start = None
    for i, s in enumerate(np.append(silent, False)):
        if s and start is None:
            start = i
        elif not s and start is not None:
            t0 = start * HOP_SEC
            t1 = min((i - 1) * HOP_SEC + FRAME_SEC, len(signal) / sr)
            if (t1 - t0) * 1000.0 >= min_ms:
                out.append((t0, t1))
            start = None
    return out


def bandpass_min_point(signal, sr, t0, t1):
    """[t0, t1] araliginda, 80-1000 Hz band-pass suzulmus sesin (100 ms
    yumusatilmis) enerjisinin en dusuk oldugu an."""
    from scipy.signal import butter, sosfiltfilt
    lo, hi = BANDPASS_HZ
    sos = butter(4, [lo, min(hi, sr / 2 - 1)], btype="bandpass", fs=sr, output="sos")
    a = max(0, int(t0 * sr) - sr)          # filtre kenar etkisi icin 1 s pay
    b = min(len(signal), int(t1 * sr) + sr)
    filtered = sosfiltfilt(sos, signal[a:b].astype(np.float64))
    energy = frame_energy_db(filtered, sr)
    k = max(1, int(round(0.100 / HOP_SEC)))
    smooth = np.convolve(energy, np.ones(k) / k, mode="same")
    times = a / sr + np.arange(len(smooth)) * HOP_SEC + FRAME_SEC / 2
    mask = (times >= t0) & (times <= t1)
    if not mask.any():
        return (t0 + t1) / 2
    return float(times[mask][np.argmin(smooth[mask])])


def split_points(signal, sr, min_s, max_s, long_ms, short_ms):
    """Bolme noktalari: [(saniye, neden)], artan sirada."""
    duration = len(signal) / sr
    silences = find_silences(signal, sr, short_ms)
    hard = [((a + b) / 2, "uzun bosluk") for a, b in silences if (b - a) * 1000 >= long_ms]
    # Dosyanin basindaki/sonundaki sessizlik ayri parca olmasin diye
    # kenarlara cok yakin noktalar atilir.
    hard = [(t, why) for t, why in hard if 0.05 < t < duration - 0.05]

    points = []
    bounds = [0.0] + [t for t, _ in hard] + [duration]
    reasons = [why for _, why in hard]
    for k in range(len(bounds) - 1):
        a, b = bounds[k], bounds[k + 1]
        while b - a > max_s:
            lo, hi = a + min_s, a + max_s
            cands = [(s1 - s0, (s0 + s1) / 2) for s0, s1 in silences
                     if lo <= (s0 + s1) / 2 <= hi]
            if cands:
                _, t = max(cands)
                why = "kisa bosluk"
            else:
                t = bandpass_min_point(signal, sr, lo, hi)
                why = "band-pass enerji minimumu"
            points.append((t, why))
            a = t
        if k < len(reasons):
            points.append((b, reasons[k]))
    return points


# --------------------------------------------------------------------------
# Transkripsiyon
# --------------------------------------------------------------------------

def _all_silent(t0, t1, silences):
    """Parcanin tamami (kenarlardan 20 ms pay ile) sessizlik mi?"""
    return any(s0 <= t0 + 0.02 and s1 >= t1 - 0.02 for s0, s1 in silences)


def _merge_silences(silences, max_gap=0.03):
    """Aralarinda max_gap'ten kisa ses olan sessizlikleri birlestirir (tek
    karelik gurultu sessizligi bolmesin)."""
    merged = []
    for s0, s1 in silences:
        if merged and s0 - merged[-1][1] <= max_gap:
            merged[-1] = (merged[-1][0], s1)
        else:
            merged.append((s0, s1))
    return merged


def _speech_span(t0, t1, silences):
    """[t0, t1] parcasinda konusmanin basladigi ve bittigi an: parcanin
    basindaki ve sonundaki sessizlik disarida birakilir."""
    s0, s1 = t0, t1
    for a, b in silences:
        if a <= t0 + 1e-6 < b:
            s0 = min(b, t1)
        if a < t1 - 1e-6 <= b:
            s1 = max(a, t0)
    return (s0, s1) if s1 > s0 else (t0, t1)


def transcribe_pieces(model, signal, sr, points, language, padding,
                      stage=None, check_cancel=None):
    """Her parcayi Whisper'a verir. [{"xmin", "xmax", "text"}] dondurur.
    Metinli her araligin basinda ve sonunda bir sinir vardir: parcanin
    basindaki ve sonundaki sessizlik ayri bos araliklar olur, metin yalnizca
    konusmanin basladigi ve bittigi anlar arasina yazilir.
    stage(anahtar, **degerler) her parcada asamayi bildirir; check_cancel() kullanici
    durdurduysa istisna firlatir (parcalar arasinda cagrilir)."""
    duration = len(signal) / sr
    edges = [0.0] + [t for t, _ in points] + [duration]
    lang = None if language in ("", "auto") else language
    silences = _merge_silences(find_silences(signal, sr, 0))
    pad = np.zeros(int(round(PADDING_SEC * sr)), dtype=np.float32)
    fp16 = str(getattr(model, "device", "cpu")).startswith("cuda")
    out = []
    for i in range(len(edges) - 1):
        t0, t1 = edges[i], edges[i + 1]
        if check_cancel:
            check_cancel()
        if stage:
            stage("whisper_piece", i=i + 1, n=len(edges) - 1, t0=t0, t1=t1)
        piece = signal[int(round(t0 * sr)):int(round(t1 * sr))].astype(np.float32)
        text = ""
        if len(piece) > 0 and not _all_silent(t0, t1, silences):
            audio = np.concatenate([pad, piece, pad]) if padding else piece
            result = model.transcribe(
                audio, language=lang, task="transcribe", beam_size=5, best_of=5,
                temperature=0.0, condition_on_previous_text=False, fp16=fp16,
                verbose=None)
            text = " ".join(result.get("text", "").split())
        if text:
            s0, s1 = _speech_span(t0, t1, silences)
            pieces = [(t0, s0, ""), (s0, s1, text), (s1, t1, "")]
        else:
            s0, s1 = t0, t1
            pieces = [(t0, t1, "")]
        for a, b, txt in pieces:
            if b - a <= 1e-6:
                continue
            if out and out[-1]["text"] == "" and txt == "":
                out[-1]["xmax"] = b          # ardisik bos araliklari birlestir
            else:
                out.append({"xmin": a, "xmax": b, "text": txt})
        print(f"    whisper {i + 1}/{len(edges) - 1} ({s0:.2f}-{s1:.2f} s): {text}")
    return out


def whisper_intervals(model, signal, sr, opts):
    """verbatim.py'nin cagirdigi giris noktasi."""
    points = split_points(signal, sr, opts["whisper_min_s"], opts["whisper_max_s"],
                          opts["long_pause_ms"], opts["short_pause_ms"])
    for t, why in points:
        print(f"    bolme: {t:.2f} s ({why})")
    return transcribe_pieces(model, signal, sr, points, opts["language"], opts["padding"],
                             opts.get("stage"), opts.get("check_cancel"))

