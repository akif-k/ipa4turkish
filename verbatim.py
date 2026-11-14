#!/usr/bin/env python
# IPA ile verbatim transkripsiyon (verbatim.py)
# facebook/wav2vec2-xlsr-53-espeak-cv-ft modeli ile ses -> IPA fonem hizalama.
# Bagimsiz (standalone) Python uygulamasi; conda ortami "verbatim" icinde
# calistirilir. Model, bu betikle ayni klasordeki models/ altinda yerel
# olarak kullanilir; internet baglantisi gerekmez. Linux, macOS ve Windows'ta
# calisir. Tek disaridan bagimlilik: sistemde kurulu espeak-ng
# (Wav2Vec2PhonemeCTCTokenizer, fonemizer arka ucunu kurarken bunu zorunlu
# kilar; fiili fonemizasyon icin cagrilmaz, sadece tokenizer'in
# yuklenebilmesi icin gereklidir).
#
# Girdi : wav/ klasoru. other_formats/ klasorunde M4A/AAC, MP3, OPUS, AMR,
#         OGG, FLAC, DSS/DS2, WMA, 3GP/3GPP veya CAF dosyasi varsa, ana
#         islemden once convert.py araciligiyla otomatik olarak 16 kHz/
#         16 bit/mono PCM wav'a donusturulup wav/ klasorune yazilir (bu
#         donusum icin ffmpeg gerekir).
# Cikti : txt/ (transkripsiyon), details/ (ayrintili olasilik tablosu,
#         kare basina secilen sayida aday; varsayilan 3),
#         tg/ (TextGrid: secilen tier'lar; hold, prob1-3, segments,
#         syllables, chunks1-N ve istege bagli whisper),
#         whisper/tg ve whisper/txt (whisper tier'i secildiyse: yalnizca
#         whisper tier'li TextGrid ve duz metin)
# Baslangicta bir pencere (gui.py) tier'lari, aday sayisini ve Whisper
# ayarlarini sordurur; --no-gui ile pencere acilmaz, ayarlar komut satiri
# seceneklerinden alinir.
# hold ve chunks1-5 tier sinirlari en yakin sifir gecisine (zero-crossing)
# kaydirilir; prob1-3, segments ve syllables'a dokunulmaz. syllables,
# hold tier'inden rules.json'daki kurallara gore turetilir (bkz.
# syllables.py, build_rules.py, rules.json).
#
# Kullanim:
#   conda activate verbatim
#   python verbatim.py
#
# Klasor konumlarini varsayilanlarindan farkli belirlemek icin:
#   python verbatim.py --wav-dir ... --tg-dir ... --txt-dir ... \
#       --details-dir ... --model-dir ... --config ... --rules ...
#
# Pencere olmadan (orn. sunucuda):
#   python verbatim.py --no-gui --tiers hold,syllables,whisper --top-k 5

import argparse
import glob
import os
import sys

import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

try:
    import librosa
except ImportError:
    librosa = None
    import soundfile as sf

from convert import convert_other_formats, print_progress
from syllables import build_syllable_tier, load_rules
# whisper_tier (ve openai-whisper paketi) yalnizca whisper tier'i secildiginde
# ice aktarilir; secilmediyse Whisper ne yuklenir ne de indirilir.

if sys.platform.startswith("win") and "PHONEMIZER_ESPEAK_LIBRARY" not in os.environ:
    # Windows'ta espeak-ng'nin DLL'i, resmi kurulumdan sonra bile her zaman
    # otomatik bulunamayabiliyor (phonemizer, ctypes ile ararken PATH'e
    # guvenir). Bilinen kurulum konumlarini burada elle deniyoruz.
    for _candidate in (
        r"C:\Program Files\eSpeak NG\libespeak-ng.dll",
        r"C:\Program Files (x86)\eSpeak NG\libespeak-ng.dll",
    ):
        if os.path.exists(_candidate):
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = _candidate
            break

if sys.platform == "darwin" and "PHONEMIZER_ESPEAK_LIBRARY" not in os.environ:
    # macOS'ta Homebrew'un kutuphane klasoru (Apple Silicon'da /opt/homebrew/lib)
    # ctypes'in standart arama yollarinda olmadigi icin espeak-ng bulunamiyor.
    # Homebrew (Apple Silicon / Intel), MacPorts ve conda ortaminin kendi
    # konumlarini sirayla deniyoruz.
    for _candidate in (
        os.path.join(os.environ.get("HOMEBREW_PREFIX", "/opt/homebrew"), "lib", "libespeak-ng.dylib"),
        "/opt/homebrew/lib/libespeak-ng.dylib",
        "/usr/local/lib/libespeak-ng.dylib",
        "/opt/local/lib/libespeak-ng.dylib",
        os.path.join(sys.prefix, "lib", "libespeak-ng.dylib"),
    ):
        if os.path.exists(_candidate):
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = _candidate
            break

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Zero-order hold ve chunks tier'lerine uygulanan sinir kaydirma penceresi.
ZERO_CROSSING_SEARCH_SEC = 0.005

DEFAULT_CHUNK_GAP_THRESHOLDS_MS = [40, 60, 80, 100, 120]

# details/ dosyasinda her kare icin yazilan aday (olasilik) sayisi.
DEFAULT_TOP_K = 3

# Modele tek seferde verilen ses uzunlugu (saniye). Wav2Vec2'nin kendi
# transformer'i self-attention nedeniyle uzun seslerde bellegi karesel
# olarak sisirip cok/uzun kayitlarda cokmeye (OOM) yol aciyordu; ses bu
# sinirin uzerindeyse modele parca parca verilip logit'ler sonradan
# zaman ekseninde birlestiriliyor, geri kalan pipeline degismiyor.
MODEL_CHUNK_DURATION_SEC = 60.0

MODEL_REPO_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
MODEL_FILE_PATTERNS = [
    "config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "vocab.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
]


# Modelin yuklenebilmesi icin model klasorunde bulunmasi ZORUNLU dosyalar.
# Agirlik dosyasi icin iki bicimden (safetensors / bin) biri yeterlidir.
MODEL_REQUIRED_FILES = [
    "config.json",
    "vocab.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
]
MODEL_WEIGHT_FILES = ["model.safetensors", "pytorch_model.bin"]


def missing_model_files(model_dir):
    """Model klasorunde eksik (ya da bos) olan zorunlu dosyalarin listesini
    dondurur; liste bossa model eksiksizdir."""
    def present(name):
        path = os.path.join(model_dir, name)
        return os.path.isfile(path) and os.path.getsize(path) > 0

    missing = [name for name in MODEL_REQUIRED_FILES if not present(name)]
    if not any(present(name) for name in MODEL_WEIGHT_FILES):
        missing.append(" veya ".join(MODEL_WEIGHT_FILES))
    return missing


def ensure_model(model_dir):
    """Model, verilen konumda eksiksiz bulunursa internete baglanilmadan
    kullanilir; klasor yoksa ya da zorunlu dosyalardan biri eksikse (orn.
    yarida kalmis bir indirme) model Hugging Face Hub'dan indirilir."""
    missing = missing_model_files(model_dir)
    if not missing:
        return

    if os.path.isdir(model_dir) and len(missing) < len(MODEL_REQUIRED_FILES) + 1:
        print(f"Model klasoru eksik: {model_dir}")
        print(f"Eksik dosya(lar): {', '.join(missing)}")
    else:
        print(f"Model bulunamadi: {model_dir}")
    print(f"Hugging Face Hub'dan indiriliyor: {MODEL_REPO_ID}")
    os.makedirs(model_dir, exist_ok=True)

    from huggingface_hub import snapshot_download

    try:
        snapshot_download(
            repo_id=MODEL_REPO_ID,
            local_dir=model_dir,
            allow_patterns=MODEL_FILE_PATTERNS,
        )
    except Exception as exc:
        sys.exit(
            f"HATA: Model indirilemedi ({exc}).\n"
            f"Internet baglantinizi kontrol edin veya modeli elle su konuma yerlestirin: {model_dir}"
        )

    missing = missing_model_files(model_dir)
    if missing:
        sys.exit(
            f"HATA: Model indirildi ama beklenen dosyalar eksik: {model_dir}\n"
            f"Eksik dosya(lar): {', '.join(missing)}"
        )

    print("Model indirildi.")


def load_config(config_path):
    if os.path.exists(config_path):
        import json
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "char_mapping": {},
        "exclude_chars": [],
        "min_confidence_percent": 0,
        "chunk_gap_thresholds_ms": list(DEFAULT_CHUNK_GAP_THRESHOLDS_MS),
    }


def process_ipa_token(token_str, prob, pad_token, config):
    if token_str == pad_token or not token_str:
        return ""

    min_confidence = config.get("min_confidence_percent", 0)
    if prob < min_confidence:
        return ""

    char_map = config.get("char_mapping", {})
    if token_str in char_map:
        token_str = char_map[token_str]

    exclude_list = config.get("exclude_chars", [])
    if token_str in exclude_list:
        return ""

    return token_str


def create_chunk_tier(raw_segments, max_gap_sec):
    grouped = []
    for seg in raw_segments:
        if not grouped:
            grouped.append({"xmin": seg["xmin"], "xmax": seg["xmax"], "text": seg["text"]})
        else:
            if grouped[-1]["text"] == seg["text"]:
                grouped[-1]["xmax"] = seg["xmax"]
            else:
                grouped.append({"xmin": seg["xmin"], "xmax": seg["xmax"], "text": seg["text"]})

    intervals = [dict(inv) for inv in grouped]

    while True:
        target_idx = -1
        # Tier'in ilk ve son araligi (genelde bas/son sessizlik) hicbir
        # zaman birlestirme HEDEFI olarak secilmez; boylece komsusu yoksa
        # bile eriyip yutulmaz ve kendi sinirini korur (komsusundan
        # birlesme almaya devam edebilir).
        last_idx = len(intervals) - 1
        for idx, inv in enumerate(intervals):
            if idx == 0 or idx == last_idx:
                continue
            duration = inv["xmax"] - inv["xmin"]
            if duration <= max_gap_sec + 1e-5:
                target_idx = idx
                break

        if target_idx == -1:
            break

        gap = intervals[target_idx]
        g_max = gap["xmax"]
        g_min = gap["xmin"]
        g_text = gap["text"]

        # target_idx hicbir zaman 0 ya da last_idx olamayacagi icin
        # (yukarida elendi), iki komsu da her zaman mevcuttur.
        left_idx = target_idx - 1
        right_idx = target_idx + 1

        if left_idx == 0:
            # Sol komsu tier'in korunan ilk araligiysa, gap'i ona
            # yutturmak onun xmax'ini ileri kaydirip ilk sinirini
            # asindirir; bunun yerine sag komsuya birlestiriyoruz.
            intervals[right_idx]["xmin"] = g_min
            intervals[right_idx]["text"] = g_text + intervals[right_idx]["text"]
        else:
            intervals[left_idx]["xmax"] = g_max
            intervals[left_idx]["text"] += g_text
        intervals.pop(target_idx)

    merged = []
    for inv in intervals:
        if not merged:
            merged.append(inv)
        else:
            if merged[-1]["text"] == inv["text"]:
                merged[-1]["xmax"] = inv["xmax"]
            else:
                merged.append(inv)
    return merged


def nearest_zero_crossing_sample(center_idx, speech_array, sr, search_radius_sec):
    radius = max(1, int(round(search_radius_sec * sr)))
    lo = max(0, center_idx - radius)
    hi = min(len(speech_array) - 1, center_idx + radius)
    if hi <= lo:
        return center_idx

    window = speech_array[lo:hi + 1]
    best_idx = None
    best_dist = None
    for i in range(len(window) - 1):
        a, b = window[i], window[i + 1]
        if a == 0 or (a < 0) != (b < 0):
            idx = lo + i
            dist = abs(idx - center_idx)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = idx

    if best_idx is None:
        best_idx = lo + int(np.argmin(np.abs(window)))

    return best_idx


def split_trailing_hold(hold_intervals, segments_intervals):
    """hold, son gecerli (bos olmayan) fonemi bir sonraki fonem gelene
    kadar tutar; ancak dosyanin sonunda "bir sonraki fonem" hic gelmezse
    bu tutma islemi audio sonuna kadar surer ve segments'teki gercek son
    icerik sinirini (son bos-olmayan karenin bitisini) gizler. Bu sinirin
    kaybolmamasi icin hold'un son araligini o noktadan ikiye bolup, geri
    kalan kismi (segments'teki gercek kuyruk gibi) bos birakiyoruz."""
    if not hold_intervals or hold_intervals[-1]["text"] == "":
        return hold_intervals

    last_content_end = None
    for seg in reversed(segments_intervals):
        if seg["text"] != "":
            last_content_end = seg["xmax"]
            break

    if last_content_end is None:
        return hold_intervals

    last = hold_intervals[-1]
    if last_content_end <= last["xmin"] + 1e-9 or last_content_end >= last["xmax"] - 1e-9:
        return hold_intervals

    result = hold_intervals[:-1]
    result.append({"xmin": last["xmin"], "xmax": last_content_end, "text": last["text"]})
    result.append({"xmin": last_content_end, "xmax": last["xmax"], "text": ""})
    return result


def apply_zero_crossing(intervals, speech_array, sr, audio_duration):
    """Ic sinirlari (0 ve audio_duration haric) en yakin zero-crossing'e
    kaydirir. Ayni sinir birden fazla intervalde paylasildigi icin tek
    seferde hesaplanip her iki tarafa da uygulanir, boylece tier icinde
    bosluk/ust uste binme olusmaz."""
    if len(intervals) <= 1:
        return intervals

    boundary_times = sorted({round(inv["xmin"], 6) for inv in intervals} | {round(inv["xmax"], 6) for inv in intervals})
    snapped = {}
    for t in boundary_times:
        if t <= 0 or t >= audio_duration:
            snapped[t] = t
            continue
        center_idx = int(round(t * sr))
        snapped_idx = nearest_zero_crossing_sample(center_idx, speech_array, sr, ZERO_CROSSING_SEARCH_SEC)
        snapped[t] = snapped_idx / sr

    result = []
    for inv in intervals:
        new_xmin = snapped.get(round(inv["xmin"], 6), inv["xmin"])
        new_xmax = snapped.get(round(inv["xmax"], 6), inv["xmax"])
        if new_xmax <= new_xmin:
            continue
        result.append({"xmin": new_xmin, "xmax": new_xmax, "text": inv["text"]})

    if result:
        result[0]["xmin"] = 0.0
        result[-1]["xmax"] = audio_duration

    return result


def write_textgrid(path, audio_duration, all_tiers):
    tg_lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        '',
        'xmin = 0',
        f'xmax = {audio_duration:.6f}',
        'tiers? <exists>',
        f'size = {len(all_tiers)}',
        'item []:'
    ]

    for tier_idx, (tier_name, intervals) in enumerate(all_tiers, 1):
        tg_lines.append(f'    item [{tier_idx}]:')
        tg_lines.append('        class = "IntervalTier"')
        tg_lines.append(f'        name = "{tier_name}"')
        tg_lines.append('        xmin = 0')
        tg_lines.append(f'        xmax = {audio_duration:.6f}')
        tg_lines.append(f'        intervals: size = {len(intervals)}')

        for inv_idx, inv in enumerate(intervals, 1):
            clean_text = inv["text"].replace('"', '""')
            tg_lines.append(f'        intervals [{inv_idx}]:')
            tg_lines.append(f'            xmin = {inv["xmin"]:.6f}')
            tg_lines.append(f'            xmax = {inv["xmax"]:.6f}')
            tg_lines.append(f'            text = "{clean_text}"')

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(tg_lines))


def compute_logits(speech_array, sr, processor, model, device):
    chunk_samples = int(MODEL_CHUNK_DURATION_SEC * sr)
    if len(speech_array) <= chunk_samples:
        input_values = processor(speech_array, sampling_rate=sr, return_tensors="pt").input_values
        input_values = input_values.to(device)
        with torch.no_grad():
            return model(input_values).logits

    logits_chunks = []
    for start in range(0, len(speech_array), chunk_samples):
        chunk = speech_array[start:start + chunk_samples]
        input_values = processor(chunk, sampling_rate=sr, return_tensors="pt").input_values
        input_values = input_values.to(device)
        with torch.no_grad():
            logits_chunks.append(model(input_values).logits)
    return torch.cat(logits_chunks, dim=1)


def load_audio(audio_path):
    if librosa is not None:
        speech_array, sr = librosa.load(audio_path, sr=16000)
        return speech_array, sr
    # librosa yoksa soundfile + basit yeniden orneklemeye dus (fallback)
    speech_array, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if speech_array.ndim > 1:
        speech_array = speech_array.mean(axis=1)
    if sr != 16000:
        raise RuntimeError(
            f"librosa kurulu degil ve dosya 16000 Hz degil ({sr} Hz). "
            "librosa'yi kurun (pip install librosa) ya da dosyayi 16 kHz'e cevirin."
        )
    return speech_array, sr


def main():
    parser = argparse.ArgumentParser(description="IPA ile verbatim transkripsiyon")
    parser.add_argument("--wav-dir", default=os.path.join(BASE_DIR, "wav"))
    parser.add_argument("--other-formats-dir", default=os.path.join(BASE_DIR, "other_formats"))
    parser.add_argument("--tg-dir", default=os.path.join(BASE_DIR, "tg"))
    parser.add_argument("--txt-dir", default=os.path.join(BASE_DIR, "txt"))
    parser.add_argument("--details-dir", default=os.path.join(BASE_DIR, "details"))
    parser.add_argument("--model-dir", default=os.path.join(BASE_DIR, "models", "wav2vec2-xlsr-53-espeak-cv-ft"))
    parser.add_argument("--config", default=os.path.join(BASE_DIR, "config.json"))
    parser.add_argument("--rules", default=os.path.join(BASE_DIR, "rules.json"))
    parser.add_argument("--device", default=None, help="cpu / cuda / cuda:0 ... (varsayilan: otomatik algila)")
    parser.add_argument("--whisper-dir", default=os.path.join(BASE_DIR, "whisper"))
    parser.add_argument("--whisper-model-dir",
                        default=os.path.join(BASE_DIR, "models", "whisper-large-v3"))
    parser.add_argument("--no-gui", action="store_true",
                        help="Ayar penceresini acma; asagidaki secenekleri kullan")
    parser.add_argument("--tiers", default=None,
                        help="Virgulle ayrilmis tier listesi "
                             "(varsayilan: hold,prob1,prob2,prob3,syllables,whisper)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help="details/ dosyasinda kare basina aday sayisi (varsayilan 3)")
    parser.add_argument("--whisper-min", type=float, default=10.0, help="Whisper parcasi min (s)")
    parser.add_argument("--whisper-max", type=float, default=15.0, help="Whisper parcasi max (s)")
    parser.add_argument("--long-pause-ms", type=int, default=250,
                        help="Her zaman bolunen sessizlik (ms)")
    parser.add_argument("--short-pause-ms", type=int, default=50,
                        help="Uzun parcalari bolmek icin en kisa sessizlik (ms)")
    parser.add_argument("--no-padding", action="store_true",
                        help="Whisper parcalarina 250 ms sessizlik ekleme")
    parser.add_argument("--language", default="tr", help="Whisper dili (varsayilan tr; 'auto')")
    args = parser.parse_args()

    config = load_config(args.config)
    chunk_gap_thresholds_ms = config.get("chunk_gap_thresholds_ms", DEFAULT_CHUNK_GAP_THRESHOLDS_MS)
    # Varsayilan secim: segments ve chunks disindakiler (whisper dahil).
    default_tiers = ["hold", "prob1", "prob2", "prob3", "syllables", "whisper"]

    opts = {
        "tiers": [t.strip() for t in args.tiers.split(",")] if args.tiers else list(default_tiers),
        "top_k": args.top_k,
        "whisper_min_s": args.whisper_min,
        "whisper_max_s": args.whisper_max,
        "long_pause_ms": args.long_pause_ms,
        "short_pause_ms": args.short_pause_ms,
        "padding": not args.no_padding,
        "language": args.language,
    }
    if args.no_gui:
        sys.exit(run_console(args, opts, config, chunk_gap_thresholds_ms))
    try:
        from gui import run_app
    except ImportError as exc:  # tkinter yok (orn. sunucu)
        print(f"UYARI: pencere acilamadi ({exc}); komut satiri ayarlari kullaniliyor.")
        sys.exit(run_console(args, opts, config, chunk_gap_thresholds_ms))
    try:
        code = run_app(chunk_gap_thresholds_ms, opts,
                       lambda chosen, ui: run_pipeline(args, chosen, config,
                                                       chunk_gap_thresholds_ms, ui))
    except Exception as exc:  # ekran yok vb. (tkinter.TclError)
        print(f"UYARI: pencere acilamadi ({exc}); komut satiri ayarlari kullaniliyor.")
        code = run_console(args, opts, config, chunk_gap_thresholds_ms)
    if code == 3:
        print("Islem iptal edildi.")
    sys.exit(code)


def run_console(args, opts, config, chunk_gap_thresholds_ms):
    """Pencere olmadan calistirir; cikis kodunu dondurur."""
    from gui import ConsoleUI
    run_pipeline(args, opts, config, chunk_gap_thresholds_ms, ConsoleUI())
    return 0


def run_pipeline(args, opts, config, chunk_gap_thresholds_ms, ui):
    """Asil islem. ui: stage(anahtar, **degerler; metinler gui.STAGES), file(sira, toplam, ad), check_cancel()
    (kullanici durdurduysa PipelineCancelled firlatir). Pencerede ayri bir
    is parcaciginda calisir."""
    chunk_tier_names = [f"chunks{i}" for i in range(1, len(chunk_gap_thresholds_ms) + 1)]
    base_tiers = ["hold", "prob1", "prob2", "prob3", "segments", "syllables"] + chunk_tier_names
    unknown = [t for t in opts["tiers"] if t not in base_tiers + ["whisper"]]
    if unknown:
        sys.exit(f"HATA: bilinmeyen tier(lar): {', '.join(unknown)}\n"
                 f"Gecerli tier'lar: {', '.join(base_tiers + ['whisper'])}")
    if opts["top_k"] < 1:
        sys.exit("HATA: aday sayisi (--top-k) en az 1 olmali.")
    use_whisper = "whisper" in opts["tiers"]
    opts = dict(opts, check_cancel=ui.check_cancel)
    print(f"Tier'lar: {', '.join(opts['tiers'])} | details aday sayisi: {opts['top_k']}")
    if use_whisper:
        print(f"Whisper: {opts['whisper_min_s']:g}-{opts['whisper_max_s']:g} s, "
              f"uzun/kisa bosluk {opts['long_pause_ms']}/{opts['short_pause_ms']} ms, "
              f"padding {'var' if opts['padding'] else 'yok'}, dil {opts['language']}")

    wav_dir = args.wav_dir
    other_formats_dir = args.other_formats_dir
    tg_dir = args.tg_dir
    txt_dir = args.txt_dir
    details_dir = args.details_dir
    model_dir = args.model_dir

    # whisper/tg: yalnizca whisper tier'li TextGrid; whisper/txt: duz metin
    whisper_tg_dir = os.path.join(args.whisper_dir, "tg")
    whisper_txt_dir = os.path.join(args.whisper_dir, "txt")
    folders = [wav_dir, other_formats_dir, txt_dir, tg_dir, details_dir]
    if use_whisper:
        folders += [whisper_tg_dir, whisper_txt_dir]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    ui.stage("convert")
    convert_other_formats(other_formats_dir, wav_dir)

    ui.stage("check_w2v")
    ensure_model(model_dir)
    if use_whisper:
        from whisper_tier import ensure_whisper_model, load_whisper, whisper_intervals
        ui.stage("check_whisper")
        ensure_whisper_model(args.whisper_model_dir)
    ui.check_cancel()

    wav_files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    if not wav_files:
        sys.exit(f"HATA: wav klasorunde islenecek .wav dosyasi bulunamadi: {wav_dir}")

    rules = load_rules(args.rules)

    chunk_thresholds_sec = [ms / 1000.0 for ms in chunk_gap_thresholds_ms]

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon
    else:
        device = "cpu"
    print(f"IPA ile verbatim transkripsiyon baslatildi. ({len(wav_files)} dosya, cihaz: {device})")

    ui.stage("load_w2v")
    print("Model yukleniyor...")
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    pad_token = processor.tokenizer.pad_token
    whisper_model = None
    if use_whisper:
        ui.stage("load_whisper")
        whisper_model = load_whisper(args.whisper_model_dir, args.device)
    ui.check_cancel()

    total = len(wav_files)

    for idx, audio_path in enumerate(wav_files, 1):
        ui.check_cancel()
        file_name = os.path.basename(audio_path)
        base_name = os.path.splitext(file_name)[0]
        ui.file(idx, total, file_name)
        print_progress(idx - 1, total, file_name)

        output_txt_path = os.path.join(txt_dir, f"{base_name}.txt")
        output_textgrid_path = os.path.join(tg_dir, f"{base_name}.TextGrid")
        output_details_path = os.path.join(details_dir, f"{base_name}_details.txt")

        ui.stage("read", file=file_name)
        speech_array, sr = load_audio(audio_path)
        audio_duration = len(speech_array) / sr

        ui.stage("w2v", file=file_name, dur=audio_duration)
        logits = compute_logits(speech_array, sr, processor, model, device)

        ui.stage("tiers", file=file_name)
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(transcription)

        logits_frame = logits[0]
        probabilities = torch.softmax(logits_frame, dim=-1)
        top_k = min(opts["top_k"], probabilities.shape[-1])
        top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=-1)

        num_frames = logits_frame.shape[0]
        time_per_frame = audio_duration / num_frames

        header = "Baslangic_s\tBitis_s\t" + "\t".join(
            f"Aday_{r}\tOlasilik_{r}(%)" for r in range(1, top_k + 1))
        details_lines = [header]

        hold_intervals = []
        prob_intervals = [[] for _ in range(top_k)]
        segments_intervals = []

        last_non_empty_char = ""

        for frame_idx in range(num_frames):
            start_time = frame_idx * time_per_frame
            end_time = (frame_idx + 1) * time_per_frame
            if frame_idx == num_frames - 1:
                end_time = audio_duration

            row = [f"{start_time:.3f}", f"{end_time:.3f}"]

            for rank in range(top_k):
                token_id = top_indices[frame_idx][rank].item()
                prob = top_probs[frame_idx][rank].item() * 100

                raw_token = processor.tokenizer.convert_ids_to_tokens(token_id)
                tg_text = process_ipa_token(raw_token, prob, pad_token, config)

                detail_token = "<PAD>" if tg_text == "" else tg_text
                row.append(detail_token)
                row.append(f"{prob:.1f}")

                if rank == 0:
                    if tg_text != "":
                        last_non_empty_char = tg_text
                    segments_intervals.append({"xmin": start_time, "xmax": end_time, "text": tg_text})

                target_list = prob_intervals[rank]
                if not target_list:
                    target_list.append({"xmin": start_time, "xmax": end_time, "text": tg_text})
                else:
                    if target_list[-1]["text"] == tg_text:
                        target_list[-1]["xmax"] = end_time
                    else:
                        target_list.append({"xmin": start_time, "xmax": end_time, "text": tg_text})

            details_lines.append("\t".join(row))

            hold_text = last_non_empty_char
            if not hold_intervals:
                hold_intervals.append({"xmin": start_time, "xmax": end_time, "text": hold_text})
            else:
                if hold_intervals[-1]["text"] == hold_text:
                    hold_intervals[-1]["xmax"] = end_time
                else:
                    hold_intervals.append({"xmin": start_time, "xmax": end_time, "text": hold_text})

        with open(output_details_path, "w", encoding="utf-8") as f:
            f.write("\n".join(details_lines))

        chunk_tiers = []
        for threshold, name in zip(chunk_thresholds_sec, chunk_tier_names):
            chunk_tier = create_chunk_tier(segments_intervals, threshold)
            chunk_tier = apply_zero_crossing(chunk_tier, speech_array, sr, audio_duration)
            chunk_tiers.append((name, chunk_tier))

        hold_intervals = split_trailing_hold(hold_intervals, segments_intervals)
        hold_intervals = apply_zero_crossing(hold_intervals, speech_array, sr, audio_duration)

        syllable_intervals = build_syllable_tier(hold_intervals, rules)

        all_tiers = [("hold", hold_intervals)]
        all_tiers += [(f"prob{r + 1}", prob_intervals[r]) for r in range(min(3, top_k))]
        all_tiers += [
            ("segments", segments_intervals),
            ("syllables", syllable_intervals),
        ] + chunk_tiers

        if whisper_model is not None:
            print()  # ilerleme cubugunun satirindan ayir
            ui.check_cancel()
            ui.stage("whisper_split", file=file_name)
            whisper_tier = whisper_intervals(
                whisper_model, speech_array, sr,
                dict(opts, stage=lambda key, f=file_name, **kw: ui.stage(key, file=f, **kw)))
            whisper_tier = apply_zero_crossing(whisper_tier, speech_array, sr, audio_duration)
            write_textgrid(os.path.join(whisper_tg_dir, f"{base_name}.TextGrid"),
                           audio_duration, [("whisper", whisper_tier)])
            # duz metin: her bolme parcasi bir satir (bos parcalar atlanir)
            with open(os.path.join(whisper_txt_dir, f"{base_name}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(inv["text"] for inv in whisper_tier if inv["text"]) + "\n")
            all_tiers.append(("whisper", whisper_tier))

        ui.stage("write_tg", file=file_name)
        selected = set(opts["tiers"])
        all_tiers = [(name, inv) for name, inv in all_tiers if name in selected]
        write_textgrid(output_textgrid_path, audio_duration, all_tiers)

        print_progress(idx, total, file_name)

    print("\nIslem tamamlandi: TextGrid dosyalari 'tg', fonem tablolari 'txt', ayrintili olasiliklar 'details' klasorunde."
          + (" Whisper TextGrid'leri 'whisper/tg', duz metinleri 'whisper/txt' klasorunde."
             if use_whisper else ""))


if __name__ == "__main__":
    main()
