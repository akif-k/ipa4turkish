# ipa_verbatim — Multi-Tier Verbatim Transcription for Turkish

[Turkce](README.md) | English

A Python command-line application that aligns audio files to IPA
(International Phonetic Alphabet) symbols using the
`facebook/wav2vec2-xlsr-53-espeak-cv-ft` model. The model is downloaded
to the `models/` folder on first use, and can be copied to another
computer if needed (dependencies such as espeak-ng, ffmpeg, and the
Python packages also need to be installed separately during setup, see
[Setup](#setup)). Runs on Linux, macOS, and Windows (see
[Platform-specific notes](#platform-specific-notes)).

Takes `.wav` files in the `wav/` folder as input; for each file it
produces a text consisting of IPA symbols (`txt/`), a multi-tier Praat
TextGrid (`tg/`), and a frame-level probability table (`details/`) (see
[Output format](#output-format)).

Audio files in other formats — M4A/AAC, MP3, OPUS, AMR, OGG, FLAC,
DSS/DS2, WMA, 3GP/3GPP, or CAF — can also be placed in the
`other_formats/` folder; before the main process starts, these are
automatically converted to 16 kHz, 16-bit, mono PCM wav by `convert.py`
and written to `wav/`, then processing continues normally (see
[Supported audio formats](#supported-audio-formats)).

## Contents

```
ipa_verbatim/
├── verbatim.py                        the main application
├── syllables.py                       module that derives the syllables tier from the hold tier, per rules.json
├── convert.py                         other_formats/ -> wav/ format conversion (via ffmpeg)
├── config.json                        phoneme mapping / filtering and TextGrid settings
├── rules.json                         C/V syllable values and syllable-type rules for the syllables tier
├── environment.yml                    to create the "verbatim" conda environment
├── run.sh                             shortcut that activates the conda environment and runs verbatim.py (Linux/macOS)
├── run.bat                            the same shortcut for Windows
├── LISANS.txt / LISANS.en.txt         license and third-party components (Turkish / English)
├── README.md / README.en.md           this document (Turkish / English)
├── models/                            folder for model files (input, empty; automatically filled on
│                                       first run with the wav2vec2-xlsr-53-espeak-cv-ft/ subfolder,
│                                       see Setup > Model files)
├── wav/                               folder for .wav files to be processed (input)
│   └── sample.wav                     example audio recording
├── other_formats/                     folder for audio files in other formats (input, optional, empty)
├── tg/                                generated TextGrid files (output, empty)
├── txt/                               generated phoneme transcripts (output, empty)
└── details/                           frame-level probability tables (output, empty)
```

## Setup

### 1) Requirements

- [Miniconda / Anaconda](https://docs.conda.io/en/latest/miniconda.html)
- **espeak-ng** (OS package). Required for the model tokenizer
  (`Wav2Vec2PhonemeCTCTokenizer`) to load successfully (it is not called
  for the actual phonemization step, its mere presence is enough for
  the library to load):

  ```bash
  # Ubuntu / Debian
  sudo apt install espeak-ng

  # macOS (Homebrew)
  brew install espeak-ng
  ```

  For Windows, see [Platform-specific notes](#platform-specific-notes).

- **ffmpeg** — only needed if you use the `other_formats/` folder (see
  [Supported audio formats](#supported-audio-formats)). It's declared
  as a conda-forge dependency in `environment.yml` and is installed
  automatically with the environment in the step below; no separate
  system package is required.

- (Optional) GPU acceleration — used automatically if available,
  otherwise falls back to CPU: NVIDIA GPU + up-to-date driver (CUDA) on
  Linux/Windows, Apple Silicon (MPS) on macOS.

### 2) Python package dependencies

`environment.yml` also automatically installs the following Python
packages from conda-forge/pip when creating the "verbatim" environment;
no manual installation is needed:

| Package | Purpose |
|---|---|
| `torch` | PyTorch, runs model inference |
| `transformers` | Wav2Vec2 model and tokenizer (`Wav2Vec2Processor`, `Wav2Vec2ForCTC`) |
| `librosa` | reads `.wav` files, resamples to 16 kHz mono |
| `safetensors` | loading the model weights (`model.safetensors`) |
| `phonemizer` | the espeak-ng binding required for the tokenizer to load |
| `numpy` | array/numeric operations |

### 3) Create the "verbatim" conda environment

```bash
cd ~/ipa_verbatim
conda env create -f environment.yml
```

`environment.yml` installs torch with the CUDA 12.8 wheel by default.
If you don't have a GPU or need a different CUDA version, remove the
`torch` line from the file and install the version matching your
platform separately, using the command from
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

To update an existing environment:

```bash
conda env update -n verbatim -f environment.yml --prune
```

### 4) Model files

The `models/` folder ships empty with this package. The first time
`verbatim.py` is run, it automatically downloads `config.json`,
`model.safetensors` (~1.2 GB), `vocab.json`, `tokenizer_config.json`,
`preprocessor_config.json`, `special_tokens_map.json` from the Hugging
Face Hub (`facebook/wav2vec2-xlsr-53-espeak-cv-ft`) into
`models/wav2vec2-xlsr-53-espeak-cv-ft/` — this requires an internet
connection. On subsequent runs this folder is already populated, so it
is not downloaded again.

If you copy the downloaded model folder
(`models/wav2vec2-xlsr-53-espeak-cv-ft/`) to another computer and place
it in the same location there, that computer won't need to download the
model separately. (The rest of the setup — the conda environment,
espeak-ng, ffmpeg, and other dependencies — still needs to be installed
separately on that computer, with an internet connection, see
[Setup](#setup).)

## Usage

1. Place the `.wav` files to be processed in the `wav/` folder. The
   sample rate/bit depth/channel count of files in `wav/` does not
   matter — `verbatim.py` automatically converts them to 16 kHz mono
   while reading them with `librosa`. If you have recordings in a
   different container/codec (M4A/AAC, MP3, OPUS, AMR, OGG, FLAC,
   DSS/DS2, WMA, 3GP/3GPP, CAF) — i.e. files that are not `.wav` — you
   can place them in the `other_formats/` folder instead of `wav/`;
   `verbatim.py` automatically converts them to wav and writes them to
   `wav/` when run, then processing continues normally (see [Supported
   audio formats](#supported-audio-formats)).
2. Run it:

   ```bash
   # Linux / macOS (make it executable the first time)
   chmod +x run.sh
   ./run.sh
   ```

   ```bat
   :: Windows
   run.bat
   ```

   or manually:

   ```bash
   conda activate verbatim
   python verbatim.py
   ```

3. Results are written to the `tg/`, `txt/`, and `details/` folders;
   progress is shown in the terminal with a progress bar.
4. When run via `run.sh` / `run.bat`, once processing completes
   without errors the `ipa_verbatim` folder automatically opens in the
   file manager (this does not happen if you run `python verbatim.py`
   directly).

### Desktop shortcut

**Linux:** A `.desktop` shortcut that runs `run.sh` can be added to the
desktop manually (this is specific to freedesktop/GNOME, does not work
on macOS or Windows — see below). If the file manager shows an "Allow
Launching" warning on the first double-click, right-click the shortcut
and select **Allow Launching**.

**macOS:** `.desktop` files are not supported. To make `run.sh`
double-clickable from Finder, it's enough to create a symbolic link
with a `.command` extension (a link, not a copy, so the shortcut stays
automatically up to date if `run.sh` is updated):

```bash
ln -s ~/ipa_verbatim/run.sh ~/Desktop/ipa_verbatim.command
chmod +x ~/ipa_verbatim/run.sh
```

**Windows:** Right-click `run.bat` and select **Send to > Desktop
(create shortcut)** to create a standard Windows shortcut; no
additional script is needed.

### Changing folder locations

By default, all folders are looked up in the same directory as this
script. To specify different locations:

```bash
python verbatim.py \
  --wav-dir /other/path/wav \
  --other-formats-dir /other/path/other_formats \
  --tg-dir /other/path/tg \
  --txt-dir /other/path/txt \
  --details-dir /other/path/details \
  --model-dir /other/path/models/wav2vec2-xlsr-53-espeak-cv-ft \
  --config /other/path/config.json \
  --rules /other/path/rules.json \
  --device cpu   # or cuda, cuda:0, mps ... (default: auto-detect)
```

## Supported audio formats

The following formats placed in `other_formats/` are converted to 16
kHz, 16-bit, mono PCM wav by `convert.py` using `ffmpeg`:

| Format | Extension |
|---|---|
| M4A / AAC | `.m4a`, `.aac` |
| MP3 | `.mp3` |
| Opus | `.opus` |
| AMR | `.amr` |
| Ogg (Vorbis/Opus) | `.ogg`, `.oga` |
| FLAC | `.flac` |
| Olympus/Philips Digital Speech Standard | `.dss`, `.ds2` |
| Windows Media Audio | `.wma` |
| 3GP / 3GPP | `.3gp`, `.3gpp` |
| Apple Core Audio Format | `.caf` |

If a converted file with the same base name (`<name>.wav`) already
exists in `wav/`, conversion is skipped; otherwise it's created. If a
file with the same name exists in multiple formats inside
`other_formats/`, which one gets processed depends on glob order — it's
recommended to keep only a single source file per name.

To run it standalone (e.g. if you only want to convert and run
verbatim.py afterwards):

```bash
conda activate verbatim
python convert.py
```

If you need other formats, it's enough to add the extension to the
`OTHER_FORMAT_EXTENSIONS` set in `convert.py` — no other change is
needed as long as ffmpeg already supports it.

## Output format

**`txt/<file>.txt`** — the phoneme sequence produced by the model, a
single line.

**`tg/<file>.TextGrid`** — a Praat TextGrid; contains 6 fixed tiers plus
as many `chunks` tiers as the length of the `chunk_gap_thresholds_ms`
list in `config.json` (with this repo's current `config.json`, 3
thresholds → 9 tiers total; the number of tiers can be adjusted by
changing the number of thresholds):

| Tier | Description |
|---|---|
| `hold` | intervals where the last valid (non-empty) phoneme is held until the next phoneme |
| `prob1` / `prob2` / `prob3` | separate tiers for the top 3 candidate phonemes per frame |
| `segments` | raw frame-level most-likely phoneme sequence (unmerged) |
| `syllables` | the syllable tier derived from `hold`, according to the rules in `rules.json` (see [rules.json and syllables](#rulesjson-and-syllables)) |
| `chunks1`..`chunksN` | N progressively coarser-grained versions of `segments`, produced by merging small pieces with their neighbor according to the `chunk_gap_thresholds_ms` thresholds in `config.json` (40/80/120 ms by default in this repo) |

Boundaries in the `hold` and `chunks1-N` tiers are shifted to the
nearest zero-crossing; `prob1-3`, `segments`, and `syllables` are left
untouched.

**`details/<file>_details.txt`** — for each frame, the start/end time
and the top 3 candidate phonemes + probability percentage (tab-separated
table).

## config.json

- `char_mapping`: to map some of the phoneme symbols produced by the
  model to another symbol (e.g. `"ɡ": "g"`).
- `exclude_chars`: list of symbols/special tokens that will not be
  shown (will be turned into a blank) in the TextGrid and details
  output (stress marks, the `<pad>`/`<s>`/`</s>`/`<unk>` special
  tokens, phonemes belonging to unused model languages, etc.).
- `min_confidence_percent`: the minimum probability percentage
  required for a candidate to be shown; candidates below it are
  treated as empty.
- `chunk_gap_thresholds_ms`: the merge thresholds (in milliseconds)
  used for the `chunks1`..`chunksN` tiers. As many `chunks` tiers are
  produced as the length of the list.

## rules.json and syllables

The `syllables` tier is created by classifying every interval in the
`hold` tier as V (vowel, value `vowel_value`, default 8), C (consonant,
value 1..`vowel_value`-1), or excluded (value 0 or not in the table)
according to the `hece_degerleri` table in `rules.json`, and then
splitting them into syllables according to the rules in `rules.json`.
**All the rules are kept as data in `rules.json`; `syllables.py` is a
generic engine that interprets them.** So for the changes below you
normally don't need to touch `syllables.py`, editing `rules.json` is
enough:

| What you want to change | rules.json field |
|---|---|
| A symbol's syllable value (C/V group) | `hece_degerleri` |
| Which value counts as the "vowel/nucleus" | `vowel_value` |
| Allowed syllable types (CV, CVC, VC, CVCC, V, or a new pattern) | `syllable_types_priority` |
| The ordering rule for 2+ consonants in the coda/onset | `coda_ordering` / `onset_ordering` |

Rules applied:

- The syllable nucleus's (V) syllable value is always `vowel_value`
  (default 8).
- Three consonants cannot be adjacent within a syllable — this follows
  naturally from the fact that no pattern in the `syllable_types_priority`
  list has a 3-consonant cluster (onset at most 1, coda at most 2
  consonants); if a new pattern containing a 3-consonant onset/coda
  (e.g. `"CVCCC"`) is added to the list, this limit updates
  automatically.
- Of the consonants between two vowels, at most `max_onset` (derived
  from the patterns, currently 1) can become the next syllable's onset,
  and at most `max_coda` (currently 2) can become the previous
  syllable's coda (maximal onset principle).
- If an (onset length, coda length) pair is not in the
  `syllable_types_priority` list, OR the `coda_ordering`/`onset_ordering`
  rule is not satisfied (default: for coda, **"descending"** — the
  syllable value must DECREASE moving away from the nucleus; based on
  the sonority scale, e.g. the "nd" coda: n=6 > d=2, consistent with
  Turkish examples like "kalp", "kirk", "genc"): the consonant(s)
  FARTHEST from the nucleus are dropped from the cluster one by one and
  left unassigned/empty, and the rule is checked again with the
  remaining cluster. The coda/onset must always stay ADJACENT to the
  nucleus — that's why what's dropped is always the farthest one(s),
  never the one(s) adjacent to the nucleus. (The rule that in CVCC the
  3rd segment's syllable value must be GREATER than the 4th is a
  specific consequence of this general mechanism with
  `coda_ordering: "descending"`; if the order isn't satisfied, the 4th
  segment — the one farther from the nucleus — is dropped, falling back
  to CVC.)
- Excluded segments (value 0, symbols not in the table, or ones that
  cannot be assigned due to the rules above) do not participate in
  syllable structure; they remain as a separate, empty interval at
  their own `hold` boundary in the `syllables` tier.

**The syllable boundary is also determined by sonority.** For the
onset, the default is `onset_ordering: "ascending"` (the syllable
value must INCREASE approaching the nucleus — an onset rising toward
the nucleus on the sonority scale, the mirror of
`coda_ordering: "descending"`). Since the `syllable_types_priority`
list currently has no 2+-consonant onset pattern (e.g. `"CCV"`)
(`max_onset = 1`), this rule is not yet ACTUALLY in effect — with a
single-consonant onset there is no "order" to check. However, to
support rising-sonority onset clusters seen in loanwords in Turkish
like "tren", "grup", "plan", if `"CCV"` (and `"CCVC"` if needed) is
added to `syllable_types_priority` in the future, this rule kicks in
automatically: a 2-consonant onset cluster whose sonority does NOT
increase toward the nucleus is considered invalid, and the consonant
FARTHEST from the nucleus is dropped (by the same "stay adjacent to
the nucleus" principle), falling back to a single-consonant onset.

Changing the **order** of `syllable_types_priority` (without
deleting/adding patterns) does not change the result — priority is
already structurally guaranteed by the maximal onset principle above.
What affects the result is **adding/removing** a pattern from the
list.

Since `char_mapping` in `config.json` changes some raw symbols (e.g.
`ɡ` -> `g`, `ɚ` -> `ər`) before they enter the `hold` tier, the
syllable values for these mapped symbols are also separately defined
in `rules.json` (with the same value as the source symbol).

### Notation for rule changes

When you make a rule change (a new syllable type, a different ordering
rule, a changed syllable value...), you can express the expected
syllabification using the notation below.

**Notation** (one cell = one test): symbols are separated by spaces.

- The first symbol automatically belongs to syllable 1.
- A `|` with no preceding space is added before the **first** symbol of
  a new syllable. E.g.: `k a |p s a` -> `ka` + `psa` (2 syllables).
- A symbol expected to be excluded/unassigned (either excluded, or
  outside a capacity/ordering rule) is put in parentheses: `(n)`.
  E.g.: `s a |n a n (d) |r i` -> `sa` + `nan` (trailing d excluded) +
  `ri`.

### Reporting / requesting a fix for a wrong syllabification

When you notice a wrong syllabification in the `syllables` tier of a
TextGrid in `tg/` that you want fixed:

1. Note the file name, the problem location (time or phoneme sequence),
   and **what it should be**. You can read the faulty IPA symbol
   sequence in Praat by looking at the `syllables` tier (or the
   `hold`/`segments` tiers for reference).
2. You can convey this in natural language, e.g.: *"in sample.wav,
   around 0.7-1.0s, 'g ʌ n' stays as separate 'gʌ' + '' + 'n', but it
   should be a single syllable 'gʌn'."* or, if ready, directly with the
   notation above: *"'g ʌ |n' was expected, 'g ʌ (n)' was produced."*
3. With this information (you, or whoever is working on the code) first
   determine the ROOT CAUSE: does the problem come from a syllable
   value (`hece_degerleri` in `rules.json`), the allowed syllable types
   (`syllable_types_priority`), or an ordering rule
   (`coda_ordering`/`onset_ordering`)?
4. Fix the relevant `rules.json` field and try again end-to-end with a
   real audio file (`python verbatim.py`) to confirm you see the
   expected result in the TextGrid.

## Platform-specific notes

`verbatim.py` itself is platform-independent (pure Python + torch +
transformers + librosa); the following are platform differences that
only apply to the setup and run shortcuts (`run.sh`/`run.bat`, desktop
shortcut):

| Topic | Linux | macOS | Windows |
|---|---|---|---|
| Run shortcut | `./run.sh` | `./run.sh` | `run.bat` |
| espeak-ng installation | `sudo apt install espeak-ng` (or your distro's package manager) | `brew install espeak-ng` | See below |
| Opening the folder afterward | `xdg-open` (automatic inside run.sh) | `open` (automatic inside run.sh) | `explorer` (automatic inside run.bat) |
| GPU acceleration | NVIDIA + CUDA (`cuda`) | Apple Silicon (`mps`) | NVIDIA + CUDA (`cuda`) |
| Desktop shortcut | `.desktop` file (see above) | `.command` symbolic link (see above) | "Create shortcut" from `run.bat` (see above) |

`run.sh` tries the `xdg-open` and `open` commands in order, so you
don't need to change anything by hand between Linux/macOS — the script
automatically detects which platform it's running on. `run.bat`
likewise uses the Windows-specific `explorer` command.

### Windows-specific setup

- **espeak-ng:** not available as a conda-forge package; download and
  install the official installer (`.msi`) from
  [espeak-ng GitHub Releases](https://github.com/espeak-ng/espeak-ng/releases).
  After installation, `libespeak-ng.dll` is usually under
  `C:\Program Files\eSpeak NG\`; `verbatim.py` automatically looks for
  this location (and the `Program Files (x86)` equivalent) when run and
  sets the `PHONEMIZER_ESPEAK_LIBRARY` environment variable itself. If
  you installed it to a different location, set it manually before
  running:

  ```bat
  set PHONEMIZER_ESPEAK_LIBRARY=C:\path\to\libespeak-ng.dll
  ```

- **conda environment:** `environment.yml` is set up with the same
  command as on Linux, inside Anaconda Prompt / PowerShell:
  `conda env create -f environment.yml`. The CUDA 12.8 wheel used for
  `torch` is also provided for Windows; if you don't have a GPU, again
  remove the `torch` line and use the Windows command from
  [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).
- If `conda` is not on PATH, `run.bat` automatically tries common
  install locations (`%USERPROFILE%\miniconda3`,
  `%USERPROFILE%\anaconda3`, etc.); if it can't find it, it shows a
  clear error message.

> Note: `run.bat` could not be tested on an actual Windows machine
> during this transition (only Linux is available in this environment);
> the logic is identical to `run.sh`, but please report it if you run
> into an issue on first run.

## Contact

You can report any issues you run into, examples of faulty output, or
other feedback to the following email address:

makilic@yahoo.com

## License

See [LISANS.en.txt](LISANS.en.txt) (Turkish: [LISANS.txt](LISANS.txt)).
The model is licensed under Apache 2.0;
this package's own code (`verbatim.py`, `syllables.py`, `config.json`,
`rules.json`, `run.sh`, `run.bat`) was written specifically for this
project. espeak-ng (GPLv3) is not included in the package, users are
expected to install it separately from their own operating system.

## Alternative setup: Praat plugin package

Setting up this repo requires installing dependencies separately —
espeak-ng, ffmpeg, and the Python packages (see [Setup](#setup)). If
you'd rather not install these one by one, you can download a Praat
plugin bundling all of these dependencies, under the "Transkripsiyon
Paketi" folder, from the address below (downloading the package
requires an internet connection, but once it's installed you won't
need to separately install espeak-ng/ffmpeg/Python packages):

https://drive.google.com/drive/folders/0B4QKykn2bzqQa0JQRTZtU08yZWs?resourcekey=0-kYz6XS59uRJCKj4YuhaqhQ
