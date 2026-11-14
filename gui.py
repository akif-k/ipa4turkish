#!/usr/bin/env python
# gui.py
# verbatim.py icin pencere (tkinter, Python standart kutuphanesi).
#
# Sol tarafta ayarlar: TextGrid'e yazilacak tier'lar, details/ dosyasindaki
# aday (olasilik) sayisi ve whisper tier'i secildiyse Whisper bolme/analiz
# ayarlari. Sag tarafta islem durumu: o anki asama, uzerinde calisilan
# dosya, dosya ilerlemesi ve islem gunlugu. Sag ustteki TR/EN secicisi
# pencerenin dilini degistirir (secim gui_settings.json'da hatirlanir).
#
# "Devam et" ayarlari kilitler ve islemi arka planda (ayri bir is
# parcaciginda) baslatir; pencere acik kalir ve ilerlemeyi gosterir. Islem
# bitince "Kapat" dugmesi cikar. "Iptal et" islem baslamadan pencereyi
# kapatir; islem sirasinda ise islemi o anki adim bitince durdurur.
# Ayrintili aciklamalar README'de.

import json
import os
import queue
import sys
import threading
import traceback
import webbrowser
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox, ttk

WHISPER_LANGUAGES = ["tr", "en", "de", "fr", "ar", "fa", "ru", "az", "ku", "auto"]
FONT_SIZE = 12
LOG_FONT_SIZE = 11
TOOLTIP_FONT_SIZE = 11
TOOLTIP_DELAY_MS = 500
# Pencerenin altindaki kunye satiri (CITATION.cff ile ayni)
AUTHOR = "Mehmet Akif Kılıç"
DOI = "10.5281/zenodo.22081832"
SETTINGS_PATH =os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui_settings.json")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CANCELLED = 3

# --------------------------------------------------------------------------
# Metinler (TR / EN)
# --------------------------------------------------------------------------

STRINGS = {
    "tr": {
        "title": "ipa4turkish: IPA ile verbatim transkripsiyon",
        "tiers": "Tier'lar",
        "hold": "Tutulan fonem",
        "prob1": "Birinci ihtimal",
        "prob2": "İkinci ihtimal",
        "prob3": "Üçüncü ihtimal",
        "segments": "20 ms'lik parçalar",
        "syllables": "Heceler",
        "chunks": "Gruplandırma (≤ {ms} ms parçalar birleştirilir)",
        "whisper": "Whisper large-v3 metni",
        "details": "details",
        "top_k": "Olasılık sayısı:",
        "whisper_box": "Whisper",
        "min": "Min (s):",
        "max": "Max (s):",
        "long": "Uzun boşluk (ms):",
        "short": "Kısa boşluk (ms):",
        "lang": "Dil:",
        "padding": "Padding ekle (250 ms)",
        "status": "İşlem durumu",
        "stage": "Aşama:",
        "file": "Dosya:",
        "ready": "Ayarları seçip 'Devam et'e basın.",
        "starting": "Başlatılıyor...",
        "stopping": "Durduruluyor (o anki adım bitince)...",
        "done": "Tamamlandı.",
        "stopped": "Durduruldu.",
        "failed": "Hata ile sonlandı (ayrıntılar günlükte).",
        "cancel": "İptal et",
        "continue": "Devam et",
        "stop": "Durdur",
        "close": "Kapat",
        "err_title": "Hatalı değer",
        "err_number": "Sayısal alanlara geçerli bir sayı girin.",
        "err_no_tier_title": "Tier seçilmedi",
        "err_no_tier": "En az bir tier seçin.",
        "err_top_k": "Olasılık sayısı 1-20 arasında olmalı.",
        "err_whisper": "0 < Min < Max ve 0 < kısa boşluk ≤ uzun boşluk olmalı.",
        "stop_title": "Durdur",
        "stop_ask": "İşlem o anki adım bitince durdurulsun mu?",
        # Imlec uzerine gelince acilan ayrintili aciklamalar
        "tip_hold": "Tutulan fonem (zero-order hold): 1. adayın son geçerli fonemi, bir "
                    "sonraki fonem gelene kadar sürdürülür; aradaki boş kareler doldurulur. "
                    "Sınırlar en yakın sıfır geçişine kaydırılır.",
        "tip_prob1": "Her 20 ms'lik karede en olası aday. Ardışık aynı adaylar tek aralıkta "
                     "birleştirilir.",
        "tip_prob2": "Her 20 ms'lik karede ikinci en olası aday.",
        "tip_prob3": "Her 20 ms'lik karede üçüncü en olası aday.",
        "tip_segments": "Her 20 ms'lik karenin 1. adayı, birleştirilmeden ayrı ayrı.",
        "tip_syllables": "hold tier'inden, rules.json'daki Türkçe hece kurallarına göre "
                         "türetilen heceler.",
        "tip_chunks": "segments'te ardışık aynı fonemler birleştirilir; ardından {ms} ms ve "
                      "daha kısa parçalar (fonem ya da boşluk) soldaki komşusuna katılır "
                      "(sol komşu tier'in ilk aralığıysa sağdakine). Tier'in ilk ve son "
                      "aralığı korunur; sınırlar sıfır geçişine kaydırılır.",
        "tip_whisper": "OpenAI Whisper large-v3 ile her bölme parçasının metni (bölüm "
                       "düzeyi). Seçilmezse Whisper ne yüklenir ne de indirilir. Sonuç "
                       "ayrıca whisper/tg (TextGrid) ve whisper/txt (düz metin) klasörlerine "
                       "yazılır.",
        "tip_top_k": "details/ dosyasında her 20 ms'lik kare için yazılan en olası aday "
                     "sayısı (1-20). prob tier'ları bu sayıdan fazla olamaz.",
        "tip_min": "İdeal aralığın alt sınırı. Öncesinde ve sonrasında en az 'uzun boşluk' "
                   "kadar sessizlik olan konuşma, bundan kısa olsa bile ayrı parça olur.",
        "tip_max": "İdeal aralığın üst sınırı. Daha uzun konuşma, Min-Max aralığındaki en "
                   "uzun 'kısa boşluk'tan bölünür. O aralıkta kısa boşluk yoksa ses "
                   "80-1000 Hz band-pass filtreyle süzülür ve enerjinin en düşük olduğu "
                   "yerden bölünür; süzülmüş ses yalnızca bölme noktasını bulmak için "
                   "kullanılır.",
        "tip_long": "Bu süre ve üzerindeki sessizlikler her zaman bölme noktasıdır.",
        "tip_short": "Max'tan uzun parçaları bölmek için aranan en kısa sessizlik.",
        "tip_lang": "Whisper'ın dili (ISO kodu, varsayılan tr). 'auto' dili Whisper'a "
                    "buldurur.",
        "tip_padding": "Her parçanın başına ve sonuna 250 ms sessizlik eklenir. Bu yalnızca "
                       "Whisper'a verilen geçici (bellekteki) sestir; zamanlar etkilenmez.",
    },
    "en": {
        "title": "ipa4turkish: Verbatim transcription with IPA for Turkish",
        "tiers": "Tiers",
        "hold": "Zero-order hold",
        "prob1": "First candidate",
        "prob2": "Second candidate",
        "prob3": "Third candidate",
        "segments": "20 ms frames",
        "syllables": "Syllables",
        "chunks": "Grouping (pieces ≤ {ms} ms are merged)",
        "whisper": "Whisper large-v3 text",
        "details": "details",
        "top_k": "Candidate count:",
        "whisper_box": "Whisper",
        "min": "Min (s):",
        "max": "Max (s):",
        "long": "Long pause (ms):",
        "short": "Short pause (ms):",
        "lang": "Language:",
        "padding": "Add padding (250 ms)",
        "status": "Status",
        "stage": "Stage:",
        "file": "File:",
        "ready": "Choose the settings and press 'Continue'.",
        "starting": "Starting...",
        "stopping": "Stopping (after the current step)...",
        "done": "Done.",
        "stopped": "Stopped.",
        "failed": "Finished with an error (see the log).",
        "cancel": "Cancel",
        "continue": "Continue",
        "stop": "Stop",
        "close": "Close",
        "err_title": "Invalid value",
        "err_number": "Enter a valid number in the numeric fields.",
        "err_no_tier_title": "No tier selected",
        "err_no_tier": "Select at least one tier.",
        "err_top_k": "Candidate count must be between 1 and 20.",
        "err_whisper": "Required: 0 < Min < Max and 0 < short pause ≤ long pause.",
        "stop_title": "Stop",
        "stop_ask": "Stop processing after the current step?",
        "tip_hold": "Zero-order hold: the last valid phoneme of the 1st candidate is held "
                    "until the next phoneme arrives; empty frames in between are filled. "
                    "Boundaries are shifted to the nearest zero crossing.",
        "tip_prob1": "The most likely candidate in each 20 ms frame. Consecutive identical "
                     "candidates are merged into one interval.",
        "tip_prob2": "The second most likely candidate in each 20 ms frame.",
        "tip_prob3": "The third most likely candidate in each 20 ms frame.",
        "tip_segments": "The 1st candidate of every 20 ms frame, one by one, unmerged.",
        "tip_syllables": "Syllables derived from the hold tier according to the Turkish "
                         "syllable rules in rules.json.",
        "tip_chunks": "Consecutive identical phonemes in segments are merged; then every "
                      "piece (phoneme or gap) of {ms} ms or less is merged into its left "
                      "neighbor (into the right one if the left is the tier's first "
                      "interval). The tier's first and last intervals are preserved; "
                      "boundaries are shifted to the nearest zero crossing.",
        "tip_whisper": "The text of each split piece from OpenAI Whisper large-v3 (piece "
                       "level). If not selected, Whisper is neither loaded nor downloaded. "
                       "The result is also written to whisper/tg (TextGrid) and whisper/txt "
                       "(plain text).",
        "tip_top_k": "How many top candidates are written to the details/ file for each "
                     "20 ms frame (1-20). There can be at most this many prob tiers.",
        "tip_min": "Lower bound of the ideal range. Speech with at least a 'long pause' of "
                   "silence before and after it becomes its own piece even if it is "
                   "shorter than this.",
        "tip_max": "Upper bound of the ideal range. Longer speech is split at the longest "
                   "'short pause' within Min-Max. If there is none, the audio is filtered "
                   "with an 80-1000 Hz band-pass filter and split where the energy is "
                   "lowest; the filtered audio is used only to find the split point.",
        "tip_long": "Silences of this length or longer are always split points.",
        "tip_short": "The shortest silence searched for when splitting pieces longer than "
                     "Max.",
        "tip_lang": "Whisper's language (ISO code, default tr). 'auto' lets Whisper detect "
                    "it.",
        "tip_padding": "250 ms of silence is added before and after each piece. This is only "
                       "the temporary (in-memory) audio passed to Whisper; times are not "
                       "affected.",
    },
}

# verbatim.py / whisper_tier.py'nin bildirdigi asamalar (anahtar -> metin).
STAGES = {
    "tr": {
        "convert": "Diğer formatlardaki ses dosyaları kontrol ediliyor (other_formats)",
        "check_w2v": "wav2vec2 modeli kontrol ediliyor",
        "check_whisper": "Whisper modeli kontrol ediliyor",
        "load_w2v": "wav2vec2 modeli yükleniyor",
        "load_whisper": "Whisper modeli yükleniyor",
        "read": "{file}: ses okunuyor",
        "w2v": "{file}: wav2vec2 ile fonem olasılıkları hesaplanıyor ({dur:.1f} s ses)",
        "tiers": "{file}: txt ve details yazılıyor, tier'lar oluşturuluyor",
        "whisper_split": "{file}: Whisper için bölme noktaları aranıyor",
        "whisper_piece": "{file}: Whisper parça {i}/{n} ({t0:.1f}-{t1:.1f} s)",
        "write_tg": "{file}: TextGrid yazılıyor",
    },
    "en": {
        "convert": "Checking audio files in other formats (other_formats)",
        "check_w2v": "Checking the wav2vec2 model",
        "check_whisper": "Checking the Whisper model",
        "load_w2v": "Loading the wav2vec2 model",
        "load_whisper": "Loading the Whisper model",
        "read": "{file}: reading audio",
        "w2v": "{file}: computing phoneme probabilities with wav2vec2 ({dur:.1f} s of audio)",
        "tiers": "{file}: writing txt and details, building tiers",
        "whisper_split": "{file}: finding split points for Whisper",
        "whisper_piece": "{file}: Whisper piece {i}/{n} ({t0:.1f}-{t1:.1f} s)",
        "write_tg": "{file}: writing TextGrid",
    },
}

_ASCII = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")


def stage_text(lang, key, **kw):
    return STAGES[lang][key].format(**kw)


def _load_ui_lang():
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            lang = json.load(f).get("ui_language")
        return lang if lang in STRINGS else "tr"
    except (OSError, ValueError):
        return "tr"


def _save_ui_lang(lang):
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump({"ui_language": lang}, f)
    except OSError:
        pass


class Tooltip:
    """Imlec bir widget'in uzerinde TOOLTIP_DELAY_MS kadar durunca acilan
    kucuk aciklama penceresi. Metin her gosterimde get_text()'ten alinir;
    boylece pencere dili degisince ipucu da degisir."""

    def __init__(self, widget, get_text):
        self.widget = widget
        self.get_text = get_text
        self.tip = None
        self.after_id = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None):
        self._cancel()
        self.after_id = self.widget.after(TOOLTIP_DELAY_MS, self._show)

    def _cancel(self):
        if self.after_id is not None:
            self.widget.after_cancel(self.after_id)
            self.after_id = None

    def _show(self):
        self.after_id = None
        if self.tip is not None:
            return
        x = self.widget.winfo_pointerx() + 16
        y = self.widget.winfo_pointery() + 14
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.get_text(), justify="left", wraplength=440,
                 background="#ffffe8", foreground="#222", relief="solid", borderwidth=1,
                 padx=8, pady=6, font=("TkDefaultFont", TOOLTIP_FONT_SIZE)).pack()

    def _hide(self, _event=None):
        self._cancel()
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


class PipelineCancelled(Exception):
    """Kullanici islemi durdurdu."""


class ConsoleUI:
    """Pencere olmadan calisirken ayni arayuz: asamalar terminale (Turkce,
    projenin terminal ciktilari gibi ASCII) yazilir."""

    def stage(self, key, **kw):
        print(f"-- {stage_text('tr', key, **kw).translate(_ASCII)}")

    def file(self, index, total, name):
        pass

    def check_cancel(self):
        pass


class _QueueStream:
    """print/tqdm ciktisini hem asil konsola hem pencerenin gunlugune yollar."""

    def __init__(self, q, original):
        self.q = q
        self.original = original

    def write(self, s):
        if s:
            self.q.put(("log", s))
            if self.original is not None:
                try:
                    self.original.write(s)
                except Exception:
                    pass
        return len(s)

    def flush(self):
        if self.original is not None:
            try:
                self.original.flush()
            except Exception:
                pass

    def isatty(self):
        return False


class _WindowUI:
    """Is parcacigindan pencereye asama/dosya bilgisi yollar."""

    def __init__(self, q, cancel_event):
        self.q = q
        self.cancel_event = cancel_event

    def stage(self, key, **kw):
        self.q.put(("stage", (key, kw)))

    def file(self, index, total, name):
        self.q.put(("file", (index, total, name)))

    def check_cancel(self):
        if self.cancel_event.is_set():
            raise PipelineCancelled()


# --------------------------------------------------------------------------
# Pencere
# --------------------------------------------------------------------------

def run_app(chunk_thresholds_ms, defaults, worker):
    """Pencereyi gosterir; "Devam et"te worker(opts, ui)'yi arka planda
    calistirir. Cikis kodunu dondurur (0 basarili, 1 hata, 3 iptal).

    defaults: {"tiers": [...], "top_k": int, "whisper_min_s": float,
               "whisper_max_s": float, "long_pause_ms": int,
               "short_pause_ms": int, "padding": bool, "language": str}"""
    root = tk.Tk()
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
        tkfont.nametofont(name).configure(size=FONT_SIZE)
    root.option_add("*TCombobox*Listbox.font", tkfont.nametofont("TkTextFont"))
    bold = tkfont.nametofont("TkDefaultFont").copy()
    bold.configure(weight="bold")

    ui_lang = tk.StringVar(value=_load_ui_lang())
    # Dil degisince yeniden yazilacak metinler: (widget, anahtar, bicim)
    texts = []
    # o anki asama: ("str", anahtar) ya da ("stage", (anahtar, kw))
    current = {"stage": ("str", "ready")}
    state = {"exit": EXIT_CANCELLED, "running": False, "finished": False,
             "close_when_done": False}
    q = queue.Queue()
    cancel_event = threading.Event()

    def t(key):
        return STRINGS[ui_lang.get()][key]

    def tr_widget(widget, key, **fmt):
        texts.append((widget, key, fmt))
        widget.configure(text=t(key).format(**fmt))
        return widget

    def tip(key, *widgets, **fmt):
        """widget'larin uzerine gelince STRINGS'teki 'tip_<key>' aciklamasi."""
        for w in widgets:
            Tooltip(w, lambda: t(f"tip_{key}").format(**fmt))

    outer = ttk.Frame(root, padding=14)
    outer.grid(sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    outer.columnconfigure(1, weight=1)
    outer.rowconfigure(1, weight=1)

    # --- Dil secici (sag ust) --------------------------------------------
    lang_bar = ttk.Frame(outer)
    lang_bar.grid(row=0, column=0, columnspan=2, sticky="e", pady=(0, 6))
    for col, code in enumerate(("tr", "en")):
        ttk.Radiobutton(lang_bar, text=code.upper(), value=code, variable=ui_lang).grid(
            row=0, column=col, padx=4)

    left = ttk.Frame(outer)
    left.grid(row=1, column=0, sticky="nw")
    settings_widgets = []

    # --- Tier secimi -----------------------------------------------------
    tiers_box = tr_widget(ttk.LabelFrame(left, padding=10), "tiers")
    tiers_box.grid(row=0, column=0, sticky="ew")
    tier_rows = [(n, n, {}) for n in ("hold", "prob1", "prob2", "prob3", "segments", "syllables")]
    tier_rows += [(f"chunks{i}", "chunks", {"ms": f"{ms:g}"})
                  for i, ms in enumerate(chunk_thresholds_ms, 1)]
    tier_rows.append(("whisper", "whisper", {}))
    tier_vars = {}
    tier_checks = {}
    for row, (name, key, fmt) in enumerate(tier_rows):
        var = tk.BooleanVar(value=name in defaults["tiers"])
        chk = ttk.Checkbutton(tiers_box, text=name, variable=var)
        chk.grid(row=row, column=0, sticky="w", padx=(0, 16), pady=2)
        desc = tr_widget(ttk.Label(tiers_box, foreground="#555"), key, **fmt)
        desc.grid(row=row, column=1, sticky="w")
        tip(key, chk, desc, **fmt)
        tier_vars[name] = var
        tier_checks[name] = chk
        settings_widgets.append(chk)

    # --- details aday sayisi --------------------------------------------
    det_box = tr_widget(ttk.LabelFrame(left, padding=10), "details")
    det_box.grid(row=1, column=0, sticky="ew", pady=(10, 0))
    top_k_label = tr_widget(ttk.Label(det_box), "top_k")
    top_k_label.grid(row=0, column=0, sticky="w")
    top_k = tk.IntVar(value=defaults["top_k"])
    top_k_spin = ttk.Spinbox(det_box, from_=1, to=20, width=4, textvariable=top_k)
    top_k_spin.grid(row=0, column=1, sticky="w", padx=8)
    tip("top_k", top_k_label, top_k_spin)
    settings_widgets.append(top_k_spin)

    # --- Whisper ayarlari -------------------------------------------------
    wh_box = tr_widget(ttk.LabelFrame(left, padding=10), "whisper_box")
    wh_box.grid(row=2, column=0, sticky="ew", pady=(10, 0))
    wmin = tk.DoubleVar(value=defaults["whisper_min_s"])
    wmax = tk.DoubleVar(value=defaults["whisper_max_s"])
    long_pause = tk.IntVar(value=defaults["long_pause_ms"])
    short_pause = tk.IntVar(value=defaults["short_pause_ms"])
    padding = tk.BooleanVar(value=defaults["padding"])
    language = tk.StringVar(value=defaults["language"])

    wh_widgets = []
    for r, (key, var) in enumerate([("min", wmin), ("max", wmax),
                                    ("long", long_pause), ("short", short_pause)]):
        label = tr_widget(ttk.Label(wh_box), key)
        label.grid(row=r, column=0, sticky="w", pady=2)
        ent = ttk.Entry(wh_box, textvariable=var, width=7)
        ent.grid(row=r, column=1, sticky="w", padx=8, pady=2)
        wh_widgets.append(ent)
        tip(key, label, ent)
    lang_label = tr_widget(ttk.Label(wh_box), "lang")
    lang_label.grid(row=4, column=0, sticky="w", pady=2)
    lang_box = ttk.Combobox(wh_box, textvariable=language, values=WHISPER_LANGUAGES, width=6)
    lang_box.grid(row=4, column=1, sticky="w", padx=8)
    wh_widgets.append(lang_box)
    tip("lang", lang_label, lang_box)
    pad_chk = tr_widget(ttk.Checkbutton(wh_box, variable=padding), "padding")
    pad_chk.grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))
    wh_widgets.append(pad_chk)
    tip("padding", pad_chk)

    # --- Islem durumu (sag) ----------------------------------------------
    right = tr_widget(ttk.LabelFrame(outer, padding=10), "status")
    right.grid(row=1, column=1, sticky="nsew", padx=(14, 0))
    right.columnconfigure(1, weight=1)
    right.rowconfigure(3, weight=1)
    tr_widget(ttk.Label(right), "stage").grid(row=0, column=0, sticky="nw")
    stage_label = ttk.Label(right, font=bold, wraplength=780)
    stage_label.grid(row=0, column=1, sticky="w", padx=6)
    tr_widget(ttk.Label(right), "file").grid(row=1, column=0, sticky="w", pady=(4, 0))
    file_var = tk.StringVar(value="-")
    ttk.Label(right, textvariable=file_var).grid(row=1, column=1, sticky="w", padx=6, pady=(4, 0))
    progress = ttk.Progressbar(right, mode="determinate", maximum=1, value=0)
    progress.grid(row=2, column=0, columnspan=2, sticky="ew", pady=8)
    log_frame = ttk.Frame(right)
    log_frame.grid(row=3, column=0, columnspan=2, sticky="nsew")
    log_frame.columnconfigure(0, weight=1)
    log_frame.rowconfigure(0, weight=1)
    log = tk.Text(log_frame, width=105, height=24, wrap="word", state="disabled",
                  font=("Consolas", LOG_FONT_SIZE))
    log.grid(row=0, column=0, sticky="nsew")
    scroll = ttk.Scrollbar(log_frame, orient="vertical", command=log.yview)
    scroll.grid(row=0, column=1, sticky="ns")
    log.configure(yscrollcommand=scroll.set)

    def show_stage():
        kind, value = current["stage"]
        if kind == "str":
            stage_label.configure(text=t(value))
        else:
            key, kw = value
            stage_label.configure(text=stage_text(ui_lang.get(), key, **kw))

    def set_stage_str(key):
        current["stage"] = ("str", key)
        show_stage()

    def append_log(s):
        log.configure(state="normal")
        parts = s.split("\n")
        for i, part in enumerate(parts):
            if "\r" in part:  # ilerleme cubuklari: ayni satiri yeniden yaz
                part = part.rsplit("\r", 1)[1]
                log.delete("end-1c linestart", "end-1c")
            log.insert("end", part)
            if i < len(parts) - 1:
                log.insert("end", "\n")
        log.see("end")
        log.configure(state="disabled")

    # --- Dugmeler ---------------------------------------------------------
    # --- Alt satir: kunye (solda) ve dugmeler (sagda) -----------------------
    footer = ttk.Frame(outer)
    footer.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(14, 0))
    footer.columnconfigure(0, weight=1)
    credit = ttk.Frame(footer)
    credit.grid(row=0, column=0, sticky="w")
    small = tkfont.nametofont("TkDefaultFont").copy()
    small.configure(size=FONT_SIZE - 2)
    link_font = small.copy()
    link_font.configure(underline=True)
    ttk.Label(credit, text=f"{AUTHOR} · DOI: ", font=small, foreground="#666").grid(
        row=0, column=0)
    doi_link = ttk.Label(credit, text=DOI, font=link_font, foreground="#1a5fb4", cursor="hand2")
    doi_link.grid(row=0, column=1)
    doi_link.bind("<Button-1>", lambda _e: webbrowser.open(f"https://doi.org/{DOI}"))
    buttons = ttk.Frame(footer)
    buttons.grid(row=0, column=1, sticky="e")
    cancel_btn = ttk.Button(buttons)
    cancel_btn.grid(row=0, column=0, padx=4)
    cont_btn = ttk.Button(buttons)
    cont_btn.grid(row=0, column=1, padx=4)

    def apply_language(*_):
        lang = ui_lang.get()
        root.title(t("title"))
        for widget, key, fmt in texts:
            widget.configure(text=STRINGS[lang][key].format(**fmt))
        cancel_btn.configure(text=t("stop") if state["running"] else t("cancel"))
        cont_btn.configure(text=t("close") if state["finished"] else t("continue"))
        show_stage()
        _save_ui_lang(lang)

    def refresh(*_):
        if state["running"] or state["finished"]:
            return
        wstate = "normal" if tier_vars["whisper"].get() else "disabled"
        for w in wh_widgets:
            w.configure(state=wstate)
        try:
            k = int(top_k.get())
        except (tk.TclError, ValueError):
            k = 3
        for i in (1, 2, 3):
            name = f"prob{i}"
            if i > k:
                tier_vars[name].set(False)
                tier_checks[name].configure(state="disabled")
            else:
                tier_checks[name].configure(state="normal")

    tier_vars["whisper"].trace_add("write", refresh)
    top_k.trace_add("write", refresh)
    ui_lang.trace_add("write", apply_language)
    refresh()
    apply_language()

    def read_values():
        try:
            values = {
                "tiers": [n for n, v in tier_vars.items() if v.get()],
                "top_k": int(top_k.get()),
                "whisper_min_s": float(wmin.get()),
                "whisper_max_s": float(wmax.get()),
                "long_pause_ms": int(long_pause.get()),
                "short_pause_ms": int(short_pause.get()),
                "padding": bool(padding.get()),
                "language": language.get().strip() or "tr",
            }
        except (tk.TclError, ValueError):
            messagebox.showerror(t("err_title"), t("err_number"), parent=root)
            return None
        if not values["tiers"]:
            messagebox.showerror(t("err_no_tier_title"), t("err_no_tier"), parent=root)
            return None
        if not 1 <= values["top_k"] <= 20:
            messagebox.showerror(t("err_title"), t("err_top_k"), parent=root)
            return None
        if "whisper" in values["tiers"] and not (
                0 < values["whisper_min_s"] < values["whisper_max_s"]
                and 0 < values["short_pause_ms"] <= values["long_pause_ms"]):
            messagebox.showerror(t("err_title"), t("err_whisper"), parent=root)
            return None
        return values

    def run_worker(values):
        ui = _WindowUI(q, cancel_event)
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = _QueueStream(q, old_out)
        sys.stderr = _QueueStream(q, old_err)
        code = EXIT_OK
        try:
            worker(values, ui)
        except PipelineCancelled:
            code = EXIT_CANCELLED
        except SystemExit as exc:  # sys.exit("HATA: ...")
            if exc.code not in (None, 0):
                code = EXIT_ERROR
                if isinstance(exc.code, str):
                    q.put(("log", exc.code + "\n"))
        except Exception:
            code = EXIT_ERROR
            q.put(("log", traceback.format_exc()))
        finally:
            sys.stdout, sys.stderr = old_out, old_err
        q.put(("done", code))

    def on_continue():
        if state["running"]:
            return
        if state["finished"]:
            root.destroy()
            return
        values = read_values()
        if values is None:
            return
        state["running"] = True
        for w in settings_widgets + wh_widgets:
            w.configure(state="disabled")
        cont_btn.configure(state="disabled")
        cancel_btn.configure(text=t("stop"))
        set_stage_str("starting")
        threading.Thread(target=run_worker, args=(values,), daemon=True).start()

    def on_cancel():
        if not state["running"]:
            root.destroy()
            return
        if cancel_event.is_set():
            return
        if messagebox.askyesno(t("stop_title"), t("stop_ask"), parent=root):
            cancel_event.set()
            cancel_btn.configure(state="disabled")
            set_stage_str("stopping")

    def on_close():
        if state["running"]:
            on_cancel()
            if cancel_event.is_set():
                state["close_when_done"] = True
        else:
            root.destroy()

    def poll():
        try:
            while True:
                kind, payload = q.get_nowait()
                if kind == "log":
                    append_log(payload)
                elif kind == "stage":
                    current["stage"] = ("stage", payload)
                    show_stage()
                    key, kw = payload
                    append_log(f"-- {stage_text(ui_lang.get(), key, **kw)}\n")
                elif kind == "file":
                    index, total, name = payload
                    file_var.set(f"{index}/{total}  {name}")
                    progress.configure(maximum=max(1, total), value=index - 1)
                elif kind == "done":
                    state["running"] = False
                    state["finished"] = True
                    state["exit"] = payload
                    if payload == EXIT_OK:
                        set_stage_str("done")
                        progress.configure(value=progress.cget("maximum"))
                    elif payload == EXIT_CANCELLED:
                        set_stage_str("stopped")
                    else:
                        set_stage_str("failed")
                    cancel_btn.grid_remove()
                    cont_btn.configure(text=t("close"), state="normal")
                    cont_btn.focus_set()
                    if state["close_when_done"]:
                        root.destroy()
                        return
        except queue.Empty:
            pass
        root.after(100, poll)

    cont_btn.configure(command=on_continue)
    cancel_btn.configure(command=on_cancel)
    cont_btn.focus_set()
    root.bind("<Return>", lambda _e: on_continue())
    root.bind("<Escape>", lambda _e: on_cancel())
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(100, poll)
    root.mainloop()
    return state["exit"]
