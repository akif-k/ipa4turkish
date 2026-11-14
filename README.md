# ipa_verbatim — Türkçe İçin Çok Katmanlı Verbatim Transkripsiyon

[English](README.en.md) | Türkçe

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22081832.svg)](https://doi.org/10.5281/zenodo.22081832)

`facebook/wav2vec2-xlsr-53-espeak-cv-ft` modeli ile ses dosyalarını IPA
(Uluslararası Fonetik Alfabe) sembolleriyle hizalayan bir Python komut
satırı uygulaması. Model ilk kullanımda `models/` klasörüne indirilir,
istenirse kopyalanıp başka bir bilgisayara taşınabilir (kurulum
sırasında espeak-ng, ffmpeg ve Python paketleri gibi bağımlılıkların da
ayrıca kurulması gerekir, bkz. [Kurulum](#kurulum)). Linux, macOS ve
Windows üzerinde çalışır (bkz. [Platforma özel
notlar](#platforma-özel-notlar)).

Girdi olarak `wav/` klasöründeki `.wav` dosyalarını alır; her dosya için
UFA kodlarından oluşan metin (`txt/`), çok tier'lı bir Praat TextGrid'i
(`tg/`) ve kare bazlı olasılık tablosu (`details/`) üretir (bkz. [Çıktı
formatı](#çıktı-formatı)).

`other_formats/` klasörüne M4A/AAC, MP3, OPUS, AMR, OGG, FLAC, DSS/DS2,
WMA, 3GP/3GPP veya CAF gibi farklı formatlarda ses dosyaları da
konulabilir; ana işlem
başlamadan önce bunlar `convert.py` tarafından otomatik olarak 16 kHz,
16 bit, mono PCM wav'a dönüştürülüp `wav/` klasörüne yazılır, sonra
işlem normal şekilde devam eder (bkz. [Desteklenen ses
formatları](#desteklenen-ses-formatları)).

## İçerik

```
ipa_verbatim/
├── verbatim.py                        asıl uygulama
├── gui.py                             pencere: ayarlar (tier, aday sayısı, Whisper) ve işlem durumu
├── whisper_tier.py                    isteğe bağlı whisper tier'ı (OpenAI Whisper large-v3); yalnızca
│                                       whisper tier'ı seçildiğinde yüklenir
├── syllables.py                       hold tier'ından rules.json'a göre syllables tier'ı üreten modül
├── convert.py                         other_formats/ -> wav/ format dönüşümü (ffmpeg ile)
├── config.json                        fonem eşleştirme / filtreleme ve TextGrid ayarları
├── rules.json                         syllables tier'ı için C/V hece değerleri ve hece tipi kuralları
├── environment.yml                    "verbatim" conda ortamını oluşturmak için
├── run.sh                             conda ortamını etkinleştirip verbatim.py'yi çalıştıran kısayol (Linux/macOS)
├── run.bat                            aynı kısayolun Windows karşılığı
├── LICENSE                            bu paketin kendi kodunun lisansı (GPLv3)
├── THIRD-PARTY-NOTICES.txt            üçüncü taraf bileşenlerin lisans bilgisi (İngilizce)
├── README.md / README.en.md           bu belge (Türkçe / İngilizce)
├── CITATION.cff                       atıf bilgisi (Citation File Format)
├── icons/                             uygulama simgeleri (masaüstü kısayolları için):
│                                       ipa4turkish.ico Windows, ipa4turkish.png Linux/macOS
├── models/                            model dosyaları için klasör (girdi, boş; ilk çalıştırmada
│                                       wav2vec2-xlsr-53-espeak-cv-ft/ alt klasörüyle otomatik doldurulur,
│                                       bkz. Kurulum > Model dosyaları)
├── wav/                               işlenecek .wav dosyalarının konulduğu klasör (girdi)
│   └── sample.wav                     örnek ses kaydı
├── other_formats/                     farklı formatlardaki ses dosyalarının konulduğu klasör (girdi, isteğe bağlı, boş)
├── tg/                                üretilen TextGrid dosyaları (çıktı, boş)
├── txt/                               üretilen fonem metinleri (çıktı, boş)
├── details/                           kare bazlı olasılık tabloları (çıktı, boş)
└── whisper/                           Whisper çıktıları (whisper tier'ı seçildiğinde oluşturulur)
    ├── tg/                            yalnızca whisper tier'ını içeren TextGrid'ler
    └── txt/                           Whisper metni, düz metin
```

## Kurulum

### 1) Gereksinimler

- [Miniconda / Anaconda](https://docs.conda.io/en/latest/miniconda.html)
- **espeak-ng** (işletim sistemi paketi). Model tokenizer'ı
  (`Wav2Vec2PhonemeCTCTokenizer`) yüklenirken bunun sistemde bulunmasını
  zorunlu kılar (fiilî fonemizasyon için çağrılmaz, sadece kütüphanenin
  var olması yeterlidir):

  ```bash
  # Ubuntu / Debian
  sudo apt install espeak-ng

  # macOS (Homebrew)
  brew install espeak-ng
  ```

  macOS'ta `verbatim.py`, `libespeak-ng.dylib`'i Homebrew
  (`/opt/homebrew/lib`, `/usr/local/lib`), MacPorts (`/opt/local/lib`) ve
  conda ortamının `lib` klasöründe kendisi arar ve
  `PHONEMIZER_ESPEAK_LIBRARY` ortam değişkenini ayarlar. Başka bir yere
  kurduysanız bu değişkeni elle tanımlayın.

  Windows için bkz. [Platforma özel notlar](#platforma-özel-notlar).

- **ffmpeg** — yalnızca `other_formats/` klasörünü kullanacaksanız
  gerekir (bkz. [Desteklenen ses formatları](#desteklenen-ses-formatları)).
  `environment.yml` içinde conda-forge bağımlılığı olarak tanımlıdır,
  aşağıdaki adımda ortamla birlikte otomatik kurulur; ayrıca sistem
  paketi kurmanıza gerek yoktur.

- (İsteğe bağlı) GPU hızlandırması — varsa otomatik kullanılır, yoksa
  CPU'ya düşer: Linux/Windows'ta NVIDIA GPU + güncel sürücü (CUDA),
  macOS'ta Apple Silicon (MPS).

### 2) Python paket bağımlılıkları

`environment.yml`, "verbatim" ortamını oluştururken aşağıdaki Python
paketlerini de conda-forge/pip'ten otomatik kurar; elle kurmanıza gerek
yoktur:

| Paket | Amaç |
|---|---|
| `torch` | model çıkarımını çalıştıran PyTorch |
| `transformers` | Wav2Vec2 model ve tokenizer (`Wav2Vec2Processor`, `Wav2Vec2ForCTC`) |
| `librosa` | `.wav` dosyalarını okuma, 16 kHz mono'ya yeniden örnekleme |
| `safetensors` | model ağırlıklarının (`model.safetensors`) yüklenmesi |
| `phonemizer` | tokenizer'ın yüklenmesi için gereken espeak-ng bağlayıcısı |
| `numpy` | dizi/sayısal işlemler |
| `openai-whisper` | isteğe bağlı `whisper` tier'ı (OpenAI Whisper large-v3); yalnızca bu tier seçildiğinde kullanılır |
| `scipy` | `whisper` tier'ı için bölme noktası aramasında 80-1000 Hz band-pass filtre |

`tkinter` (ayar penceresi) Python ile birlikte gelir. Bulunamazsa (örn.
ekransız bir sunucuda) program pencereyi atlayıp komut satırı
ayarlarıyla devam eder.

Linux'ta conda-forge bazen Tk'nin `noxft` derlemesini kurar; bu derleme
eski X fontlarını kullandığı için pencerede Türkçe karakterler (ş, ğ, ı,
İ ...) bozuk görünür. `run.sh` bu durumu algılayıp Xft destekli derlemeyi
otomatik kurar. Elle düzeltmek için:

```bash
conda install -n verbatim -c conda-forge "tk=*=xft*"
```

### 3) "verbatim" conda ortamını oluştur

```bash
cd ~/ipa_verbatim
conda env create -f environment.yml
```

`environment.yml`, torch'u varsayılan olarak CUDA 12.8 wheel'i ile kurar.
GPU'nuz yoksa veya farklı bir CUDA sürümü gerekiyorsa, dosyadaki `torch`
satırını kaldırıp [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
adresindeki komutla platformunuza uygun sürümü ayrıca kurun.

Ortam zaten varsa güncellemek için:

```bash
conda env update -n verbatim -f environment.yml --prune
```

### 4) Model dosyaları

`models/` klasörü bu paketle birlikte boş gelir. `verbatim.py` ilk
çalıştırıldığında, `config.json`, `model.safetensors` (~1.2 GB),
`vocab.json`, `tokenizer_config.json`, `preprocessor_config.json`,
`special_tokens_map.json` dosyalarını Hugging Face Hub'dan
(`facebook/wav2vec2-xlsr-53-espeak-cv-ft`) otomatik olarak
`models/wav2vec2-xlsr-53-espeak-cv-ft/` altına indirir — bunun için
internet bağlantısı gerekir. Sonraki çalıştırmalarda zorunlu dosyaların
hepsi (ağırlık dosyası dahil) klasörde bulunduğundan tekrar indirme
yapılmaz ve internete bağlanılmaz. Dosyalardan biri eksikse (örn. yarıda
kalmış bir indirme), program eksik dosyaları listeler ve modeli yeniden
indirir.

İndirilen model klasörünü (`models/wav2vec2-xlsr-53-espeak-cv-ft/`)
başka bir bilgisayara kopyalayıp oradaki aynı konuma yerleştirirseniz, o
bilgisayarda modelin ayrıca indirilmesi gerekmez. (Kurulumun geri
kalanı — conda ortamı, espeak-ng, ffmpeg gibi bağımlılıklar — yine de o
bilgisayarda ayrıca ve internet bağlantısıyla kurulmalıdır, bkz.
[Kurulum](#kurulum).)

**Whisper modeli (isteğe bağlı).** `whisper` tier'ı seçilmediği sürece
Whisper modeli ne yüklenir ne de indirilir. Seçildiğinde
`models/whisper-large-v3/large-v3.pt` (~3 GB) aranır; yoksa OpenAI'nin
resmî adresinden indirilir (`openai-whisper` paketi dosyayı SHA256 ile
doğrular). İndirilen `models/whisper-large-v3/` klasörü de başka bir
bilgisayara kopyalanabilir.

## Kullanım

1. İşlenecek `.wav` dosyalarını `wav/` klasörüne koyun. `wav/`
   klasöründeki dosyaların örnekleme hızı/bit derinliği/kanal sayısı
   önemli değildir — `verbatim.py`, `librosa` ile okurken bunları
   otomatik olarak 16 kHz mono'ya çevirir. Farklı bir konteyner/codec'te
   (M4A/AAC, MP3, OPUS, AMR, OGG, FLAC, DSS/DS2, WMA, 3GP/3GPP, CAF)
   kayıtlarınız varsa (yani `.wav` uzantılı olmayan dosyalar), bunları
   `wav/` yerine `other_formats/` klasörüne koyabilirsiniz;
   `verbatim.py` çalışırken bunları otomatik olarak wav'a çevirip
   `wav/` klasörüne yazar, sonra normal işlem devam eder (bkz.
   [Desteklenen ses formatları](#desteklenen-ses-formatları)).
2. Çalıştırın:

   ```bash
   # Linux / macOS (ilk seferde çalıştırma izni gerekir)
   chmod +x run.sh
   ./run.sh
   ```

   ```bat
   :: Windows
   run.bat
   ```

   veya elle:

   ```bash
   conda activate verbatim
   python verbatim.py
   ```

3. Bir pencere açılır (bkz. [Ayar penceresi](#ayar-penceresi)). **Devam
   et** seçilen ayarlarla işlemi başlatır; pencere açık kalır ve sağdaki
   **İşlem durumu** panelinde ilerlemeyi gösterir. **İptal et** işlem
   başlamadan pencereyi kapatır.
4. Sonuçlar `tg/`, `txt/` ve `details/` klasörlerine (whisper tier'ı
   seçildiyse ayrıca `whisper/tg/` ve `whisper/txt/` klasörlerine) yazılır; ilerleme terminalde
   bir ilerleme çubuğu ile gösterilir.
5. `run.sh` / `run.bat` ile çalıştırıldığında, işlem hatasız tamamlanınca
   `ipa_verbatim` klasörü dosya yöneticisinde otomatik olarak açılır
   (`python verbatim.py` ile doğrudan çalıştırırsanız bu olmaz).

### Ayar penceresi

- **Tier'lar:** TextGrid'e yazılacak tier'lar işaretlenir (bkz. [Çıktı
  formatı](#çıktı-formatı)). Varsayılan olarak `hold`, `prob1-3`,
  `syllables` ve `whisper` seçilidir; `segments` ve `chunks1-N` seçili
  değildir. `whisper` seçili olduğu için ilk çalıştırmada Whisper modeli
  (~3 GB) iner; istemiyorsanız işaretini kaldırın.
- **details/ olasılık (aday) sayısı:** `details/` dosyasında her 20 ms'lik
  kare için yazılan en olası aday sayısı (varsayılan 3, en çok 20).
  `prob` tier'ları en fazla bu sayı kadar olabilir (sayı 2 ise `prob3`
  seçilemez).
- **Whisper ayarları:** yalnızca `whisper` tier'ı işaretliyse etkinleşir
  (bkz. [Whisper tier'ı](#whisper-tierı)).
- **Açıklamalar:** imleci bir tier'ın, alanın ya da kutunun üzerinde
  yarım saniye tutunca ayrıntılı açıklaması küçük bir pencerede açılır.
- **TR / EN (sağ üstte):** pencerenin dili. Seçim `gui_settings.json`
  dosyasında hatırlanır. İşlem günlüğündeki program çıktıları Türkçe
  kalır; aşama satırları seçilen dilde yazılır.
- **İşlem durumu (sağda):** o anki aşama (örn. model yükleniyor, fonem
  olasılıkları hesaplanıyor, Whisper parça 2/5), üzerinde çalışılan dosya
  ve sırası, dosya ilerleme çubuğu ve işlem günlüğü.
- İşlem sırasında ayarlar kilitlenir; **İptal et** düğmesi **Durdur**
  olur ve işlemi o anki adım bitince durdurur. İşlem bitince **Kapat**
  düğmesi çıkar.

Pencereyi açmadan çalıştırmak için (örn. ekransız bir sunucuda):

```bash
python verbatim.py --no-gui --tiers hold,syllables,whisper --top-k 5 \
  --whisper-min 10 --whisper-max 15 --long-pause-ms 250 --short-pause-ms 50 \
  --language tr          # --no-padding ile padding kapatılır
```

`--tiers` verilmezse pencerenin varsayılan seçimi kullanılır (`hold`,
`prob1-3`, `syllables`, `whisper`).

### Masaüstü kısayolu

**Linux:** `run.sh`'ı çalıştıran bir `.desktop` kısayolu masaüstüne elle
eklenebilir (freedesktop/GNOME'a özgüdür, macOS ve Windows'ta çalışmaz —
aşağıya bakın). İlk çift tıklamada dosya yöneticisi "Başlatmaya izin
ver" uyarısı gösterirse, kısayola sağ tıkla > **Başlatmaya izin ver**
seçilmelidir.

**macOS:** `.desktop` dosyaları desteklenmez. `run.sh`'ı Finder'dan çift
tıklanabilir yapmak için `.command` uzantılı bir sembolik bağlantı
oluşturmak yeterli (kopya değil bağlantı, böylece `run.sh` güncellenirse
kısayol da otomatik güncel kalır):

```bash
ln -s ~/ipa_verbatim/run.sh ~/Desktop/ipa_verbatim.command
chmod +x ~/ipa_verbatim/run.sh
```

**Windows:** `run.bat` dosyasına sağ tıklayıp **Gönder > Masaüstü
(kısayol oluştur)** seçilerek standart bir Windows kısayolu
oluşturulabilir; ayrıca bir betik gerekmez.

### Klasör konumlarını değiştirme

Varsayılan olarak tüm klasörler bu betikle aynı dizinde aranır. Farklı
konumlar belirtmek için:

```bash
python verbatim.py \
  --wav-dir /baska/yol/wav \
  --other-formats-dir /baska/yol/other_formats \
  --tg-dir /baska/yol/tg \
  --txt-dir /baska/yol/txt \
  --details-dir /baska/yol/details \
  --whisper-dir /baska/yol/whisper \
  --model-dir /baska/yol/models/wav2vec2-xlsr-53-espeak-cv-ft \
  --whisper-model-dir /baska/yol/models/whisper-large-v3 \
  --config /baska/yol/config.json \
  --rules /baska/yol/rules.json \
  --device cpu   # veya cuda, cuda:0, mps ... (varsayılan: otomatik algıla)
```

## Desteklenen ses formatları

`other_formats/` klasörüne konulan aşağıdaki formatlar, `convert.py`
tarafından `ffmpeg` ile 16 kHz, 16 bit, mono PCM wav'a dönüştürülür:

| Format | Uzantı |
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

Dönüştürülen dosya, `wav/` klasöründe aynı taban adla (`<ad>.wav`)
zaten varsa atlanır; yoksa oluşturulur. Aynı adı taşıyan bir
dosya birden fazla formatta `other_formats/` içinde bulunursa,
hangisinin işleneceği glob sırasına bağlıdır — aynı ad için tek bir
kaynak dosya bırakmanız önerilir.

Tek başına çalıştırmak için (örn. sadece dönüşüm yapıp verbatim.py'yi
sonra çalıştırmak istiyorsanız):

```bash
conda activate verbatim
python convert.py
```

Başka formatlar gerekiyorsa, `convert.py` içindeki
`OTHER_FORMAT_EXTENSIONS` kümesine uzantıyı eklemek yeterlidir — ffmpeg
zaten destekliyorsa başka bir değişiklik gerekmez.

## Çıktı formatı

**`txt/<dosya>.txt`** — modelin ürettiği fonem dizisi, tek satır.

**`tg/<dosya>.TextGrid`** — Praat TextGrid'i; ayar penceresinde seçilen
tier'ları aşağıdaki sırayla içerir. Hepsi seçildiğinde: 6 sabit tier +
`config.json` içindeki `chunk_gap_thresholds_ms` listesinin uzunluğu
kadar `chunks` tier'ı (bu depodaki mevcut `config.json` ile 3 eşik → 9
tier; eşik sayısı değiştirilerek tier sayısı ayarlanabilir) + en sonda
isteğe bağlı `whisper`:

| Tier | Açıklama |
|---|---|
| `hold` | son geçerli (boş olmayan) fonemin bir sonraki foneme kadar tutulduğu aralıklar |
| `prob1` / `prob2` / `prob3` | her karede en yüksek 3 aday fonem için ayrı tier'lar |
| `segments` | ham kare bazlı en olası fonem dizisi (birleştirilmemiş) |
| `syllables` | `hold`'dan, `rules.json` içindeki kurallara göre türetilen hece tier'ı (bkz. [rules.json ve syllables](#rulesjson-ve-syllables)) |
| `chunks1`..`chunksN` | `segments`'in, `config.json` içindeki `chunk_gap_thresholds_ms` eşiklerine (bu depoda varsayılan 40/80/120 ms) göre küçük parçaların komşu parçayla birleştirilmesiyle oluşturulan, gitgide daha kaba taneli N versiyonu. Önce ardışık aynı fonemler birleştirilir; sonra eşikten kısa ya da eşiğe eşit her parça (fonem ya da boşluk) soldaki komşusuna katılır (soldaki tier'ın ilk aralığıysa sağdakine); tier'ın ilk ve son aralığı korunur |
| `whisper` | isteğe bağlı; OpenAI Whisper large-v3 ile her bölme parçasının metni (bölüm düzeyi, bkz. [Whisper tier'ı](#whisper-tierı)) |

`hold`, `chunks1-N` ve `whisper` tier'larındaki sınırlar en yakın sıfır
geçişine (zero-crossing) kaydırılır; `prob1-3`, `segments` ve
`syllables`'a dokunulmaz.

**`details/<dosya>_details.txt`** — her kare için başlangıç/bitiş zamanı
ve ilk K aday fonem + olasılık yüzdesi (tab ile ayrılmış tablo; K ayar
penceresindeki aday sayısı, varsayılan 3).

**`whisper/tg/<dosya>.TextGrid`** — `whisper` tier'ı seçildiyse, yalnızca
bu tier'ı içeren TextGrid.

**`whisper/txt/<dosya>.txt`** — `whisper` tier'ı seçildiyse, Whisper
metni düz metin olarak; her bölme parçası bir satır (boş parçalar
yazılmaz).

## Whisper tier'ı

`whisper` tier'ı seçildiğinde ses önce parçalara bölünür, her parça
OpenAI Whisper large-v3'e ayrı verilir ve her parçanın metni o parçanın
aralığına yazılır.

- **Sessizlik:** 20 ms'lik pencerelerde (10 ms adımla) ölçülen enerji,
  dosyanın kendi gürültü tabanına göre eşiklenir (gürültü tabanı ile
  konuşma düzeyi arasındaki farkın %25'i kadar üstü, en az 6 dB).
- **Uzun boşluk (varsayılan 250 ms):** her zaman bölme noktasıdır. Böylece
  öncesinde ve sonrasında uzun boşluk olan konuşma, Min'den kısa olsa bile
  ayrı parça olur.
- **Min / Max (varsayılan 10 / 15 s):** Max'tan uzun bir parça,
  başlangıcından itibaren Min-Max aralığındaki en uzun **kısa boşluktan**
  (varsayılan 50 ms ve üzeri) bölünür. O aralıkta kısa boşluk yoksa ses
  80-1000 Hz band-pass filtreyle süzülür ve süzülmüş sesin enerjisinin en
  düşük olduğu yerden bölünür. Süzülmüş ses yalnızca bölme noktasını
  bulmak için kullanılır; Whisper'a özgün ses verilir.
- Bölme noktası sessizliğin ortasıdır; parçalar birbirine bitişiktir ve
  tüm sesi kapsar. Tamamen sessiz parçalar Whisper'a verilmez (boş aralık).
- Tier'da her metnin başında ve sonunda bir sınır vardır: parçanın
  başındaki ve sonundaki sessizlik ayrı boş aralıklar olur, metin yalnızca
  konuşmanın başladığı ve bittiği anlar arasına yazılır (aynı sessizlik
  eşiğiyle bulunur).
- **Padding ekle:** her parçanın başına ve sonuna 250 ms sessizlik eklenir.
  Bu yalnızca Whisper'a verilen geçici (bellekteki) sestir; zamanlar
  etkilenmez.
- **Dil:** varsayılan `tr`; `auto` dili Whisper'a buldurur.

**`whisper` tier'ı için NVIDIA GPU (CUDA) önerilir.** Whisper, NVIDIA GPU
varsa onu kullanır; yoksa CPU'da çalışır ve large-v3 için oldukça
yavaştır (uzun kayıtlarda dakikalar sürebilir). macOS'ta Whisper her
zaman CPU'da çalışır (openai-whisper Apple Silicon GPU'sunda (MPS)
sorunlu olduğu için); wav2vec2 tier'ları ise MPS'i kullanmaya devam
eder. CUDA yoksa ve hız önemliyse pencerede `whisper` tier'ını
kapatabilir ya da `--tiers` ile dışarıda bırakabilirsiniz.

## config.json

- `char_mapping`: modelin ürettiği bazı fonem sembollerini başka bir
  sembole eşleştirmek için (örn. `"ɡ": "g"`).
- `exclude_chars`: TextGrid ve details çıktısında gösterilmeyecek
  (boşluğa çevrilecek) sembol/özel token listesi (vurgu işaretleri,
  `<pad>`/`<s>`/`</s>`/`<unk>` özel token'ları, model diline ait
  kullanılmayan fonemler vb.).
- `min_confidence_percent`: bir adayın gösterilmesi için gereken minimum
  olasılık yüzdesi; altındaki adaylar boş sayılır.
- `chunk_gap_thresholds_ms`: `chunks1`..`chunksN` tier'ları için
  kullanılan birleştirme eşikleri (milisaniye). Liste uzunluğu kadar
  `chunks` tier'ı üretilir.

## rules.json ve syllables

`syllables` tier'ı, `hold` tier'ındaki her aralığı `rules.json`'daki
`hece_degerleri` tablosuna göre V (ünlü, değer `vowel_value`, varsayılan
8), C (ünsüz, değer 1..`vowel_value`-1) ya da hariç (değer 0 ya da
tabloda yok) olarak sınıflandırıp, `rules.json`'daki kurallara göre
hecelere ayırarak oluşturulur. **Kuralların tamamı `rules.json`'da veri
olarak tutulur; `syllables.py` bunları yorumlayan genel bir motordur.**
Yani aşağıdaki değişiklikler için normalde `syllables.py`'ye
dokunmanıza gerek yoktur, sadece `rules.json`'u düzenlemeniz yeterlidir:

| Değiştirmek istediğiniz | rules.json alanı |
|---|---|
| Bir sembolün hece değeri (C/V grubu) | `hece_degerleri` |
| Hangi değerin "ünlü/çekirdek" sayıldığı | `vowel_value` |
| İzin verilen hece tipleri (CV, CVC, VC, CVCC, V, ya da yeni bir kalıp) | `syllable_types_priority` |
| Koda/onset'teki 2+ ünsüzün hece değeri sırası kuralı | `coda_ordering` / `onset_ordering` |

Uygulanan kurallar:

- Hece çekirdeğinin (V) hece değeri her zaman `vowel_value`'dur (varsayılan 8).
- Bir hecede 3 ünsüz yan yana olamaz — bu, `syllable_types_priority`
  listesindeki hiçbir kalıpta 3 ünsüzlük bir küme olmamasından (onset
  en fazla 1, koda en fazla 2 ünsüz) doğal olarak sağlanır; listeye 3
  ünsüzlük bir onset/koda içeren yeni bir kalıp (örn. `"CVCCC"`)
  eklenirse bu sınır da otomatik güncellenir.
- İki ünlü arasındaki ünsüzlerin en fazla `max_onset` (kalıplardan
  türetilir, şu an 1) tanesi bir sonraki hecenin onset'i, en fazla
  `max_coda` (şu an 2) tanesi bir önceki hecenin kodası olabilir
  (maksimal onset ilkesi).
- Bir (onset uzunluğu, koda uzunluğu) ikilisi `syllable_types_priority`
  listesinde yoksa, YA DA `coda_ordering`/`onset_ordering` kuralı
  (varsayılan: koda için **"descending"** — çekirdekten uzaklaştıkça
  hece değeri AZALMALI; sonorluk skalasına göre, örn. "nd" kodası: n=6
  > d=2, Türkçe'deki "kalp", "kırk", "genç"
  gibi örnekleriyle tutarlı) sağlanmıyorsa: kümeden çekirdekten EN UZAK
  unsur(lar) sırayla hece dışı/boş bırakılır ve kalan kümeyle kural
  tekrar kontrol edilir. Koda/onset her zaman çekirdeğe BİTİŞİK
  kalmalıdır — bu yüzden düşen her zaman en uzaktaki(ler)dir, çekirdeğe
  bitişik olan(lar) değil. (CVCC'de 3. segmentin hece değeri 4.'ten
  BÜYÜK olmalı kuralı, bu genel mekanizmanın `coda_ordering:
  "descending"` ile özel bir sonucudur; sıra sağlanmazsa 4. segment
  -çekirdekten daha uzaktaki- düşürülür, CVC'ye gerilenir.)
- Hariç tutulan segmentler (değer 0, tabloda olmayan semboller, ya da
  yukarıdaki kurallar gereği atanamayanlar) hece yapısına katılmaz;
  `syllables` tier'ında kendi `hold` sınırında ayrı ve boş bir aralık
  olarak kalır.

**Hece sınırı da sonorluğa göre belirlenir.** Onset için varsayılan
`onset_ordering: "ascending"` (çekirdeğe yaklaştıkça hece değeri
ARTMALI — sonorluk skalasında çekirdeğe doğru yükselen bir onset,
`coda_ordering: "descending"`'in aynası). Şu an `syllable_types_priority`
listesinde 2+ ünsüzlük bir onset kalıbı (örn. `"CCV"`) bulunmadığından
(`max_onset = 1`), bu kural henüz FİİLEN devrede değildir -- tek
unsurluk bir onsette sıralama kontrol edilecek bir "sıra" yoktur. Ancak
Türkçe'deki yabancı kökenli kelimelerde görülen "tren", "grup", "plan"
gibi yükselen sonorluklu onset kümelerini desteklemek için ileride
`syllable_types_priority`'ye `"CCV"` (ve gerekirse `"CCVC"`) eklenirse,
bu kural otomatik olarak devreye girer: sonorluğu çekirdeğe doğru
ARTMAYAN bir 2'li onset kümesi geçersiz sayılır ve çekirdekten EN UZAK
unsur (aynı "çekirdeğe bitişik kal" ilkesiyle) düşürülerek tek unsurluk
onsete gerilenir.

`syllable_types_priority`'nin **sırasını** değiştirmek (kalıpları
SİLMEDEN/EKLEMEDEN) sonucu değiştirmez — öncelik, yukarıdaki maksimal
onset ilkesiyle yapısal olarak zaten sağlanır. Sonuç üzerinde etkili
olan, listeye yeni bir kalıp **eklemek/çıkarmak**tır.

`config.json`'daki `char_mapping` bazı ham sembolleri (örn. `ɡ` -> `g`,
`ɚ` -> `ər`) `hold` tier'ına girmeden önce değiştirdiğinden, `rules.json`
içinde bu eşlenmiş sembollerin hece değerleri de (kaynak sembolle aynı
değer) ayrıca tanımlıdır.

### Kural değişikliklerinde notasyon

Bir kural değişikliği (yeni hece tipi, farklı bir sıra kuralı, değişen
bir hece değeri...) yaptığınızda, beklenen hecelemeyi aşağıdaki
notasyonla ifade edebilirsiniz.

**Notasyon** (bir hücre = tek bir test): semboller boşluk ile ayrılır.

- İlk sembol otomatik olarak 1. hecenin parçasıdır.
- Yeni bir hecenin **ilk** sembolünün başına boşluksuz `|` eklenir.
  Örn: `k a |p s a` -> `ka` + `psa` (2 hece).
- Hece dışı/atanmamış (hariç ya da kapasite/sıra kuralı dışı) olması
  beklenen bir sembol parantez içine alınır: `(n)`.
  Örn: `s a |n a n (d) |r i` -> `sa` + `nan` (sondaki d hariç) + `ri`.

### Yanlış bir hecelemeyi bildirme / düzeltme isteme

`tg/` klasöründeki bir TextGrid'in `syllables` tier'ında yanlış
gördüğünüz bir hecelemeyi düzeltmek istediğinizde:

1. Dosya adını, sorunlu yeri (zaman ya da fonem dizisi) ve **olması
   gerekeni** not edin. Praat'ta `syllables` tier'ına (ya da referans
   için `hold`/`segments` tier'larına) bakarak hatalı IPA sembol
   dizisini okuyabilirsiniz.
2. Bunu doğal dille (Türkçe) şöyle aktarabilirsiniz, örn.:
   *"turku.wav'da 0.7-1.0s civarında 'g ʌ n' 'gʌ' + '' + 'n' olarak
   ayrı ayrı kalıyor, oysa 'gʌn' olarak tek hece olmalı."*
   ya da hazırsanız doğrudan yukarıdaki notasyonla:
   *"'g ʌ |n' bekleniyordu, 'g ʌ (n)' çıktı alındı."*
3. Bu bilgiyle (siz ya da kod üzerinde çalışan biri) önce KÖKENİ
   belirler: sorun bir hece değerinden mi (`rules.json`'daki
   `hece_degerleri`), izin verilen hece tiplerinden mi
   (`syllable_types_priority`), yoksa sıralama kuralından mı
   (`coda_ordering`/`onset_ordering`) kaynaklanıyor?
4. İlgili `rules.json` alanını düzeltip gerçek bir ses dosyasıyla
   (`python verbatim.py`) uçtan uca tekrar deneyip TextGrid'de beklenen
   sonucu gördüğünüzü teyit edin.

## Platforma özel notlar

`verbatim.py`'nin kendisi platformdan bağımsızdır (saf Python + torch +
transformers + librosa); aşağıdakiler yalnızca kurulum ve çalıştırma
kısayollarında (`run.sh`/`run.bat`, masaüstü kısayolu) geçerli platform
farklarıdır:

| Konu | Linux | macOS | Windows |
|---|---|---|---|
| Çalıştırma kısayolu | `./run.sh` | `./run.sh` | `run.bat` |
| espeak-ng kurulumu | `sudo apt install espeak-ng` (veya dağıtımınızın paket yöneticisi) | `brew install espeak-ng` | Aşağıya bakın |
| İşlem sonunda klasör açma | `xdg-open` (run.sh içinde otomatik) | `open` (run.sh içinde otomatik) | `explorer` (run.bat içinde otomatik) |
| GPU hızlandırma | NVIDIA + CUDA (`cuda`) | Apple Silicon (`mps`) | NVIDIA + CUDA (`cuda`) |
| Masaüstü kısayolu | `.desktop` dosyası (bkz. yukarıda) | `.command` sembolik bağlantısı (bkz. yukarıda) | `run.bat`'tan "Kısayol oluştur" (bkz. yukarıda) |

`run.sh`, `xdg-open` ve `open` komutlarını sırayla dener, bu yüzden
Linux/macOS arasında elle bir şey değiştirmeniz gerekmez — betik hangi
platformda çalıştığını otomatik anlar. `run.bat` de aynı şekilde
Windows'a özgü `explorer` komutunu kullanır.

### Windows'a özgü kurulum

- **espeak-ng:** conda-forge'da paket olarak bulunmuyor; resmî kurulum
  dosyasını [espeak-ng GitHub Releases](https://github.com/espeak-ng/espeak-ng/releases)
  sayfasından (`.msi`) indirip kurun. Kurulumdan sonra `libespeak-ng.dll`
  çoğunlukla `C:\Program Files\eSpeak NG\` altında olur; `verbatim.py`
  bu konumu (ve `Program Files (x86)` karşılığını) çalışırken otomatik
  arar ve `PHONEMIZER_ESPEAK_LIBRARY` ortam değişkenini kendisi ayarlar.
  Farklı bir konuma kurduysanız, çalıştırmadan önce elle ayarlayın:

  ```bat
  set PHONEMIZER_ESPEAK_LIBRARY=C:\yol\libespeak-ng.dll
  ```

- **conda ortamı:** `environment.yml`, Anaconda Prompt / PowerShell
  içinde Linux'takiyle aynı komutla kurulur: `conda env create -f
  environment.yml`. `torch` için kullanılan CUDA 12.8 wheel'i Windows
  için de sağlanır; GPU'nuz yoksa yine `torch` satırını kaldırıp
  [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
  adresindeki Windows komutunu kullanın.
- `run.bat`, sistemde `conda` PATH'te değilse yaygın kurulum
  konumlarını (`%USERPROFILE%\miniconda3`, `%USERPROFILE%\anaconda3`,
  vb.) otomatik dener; bulamazsa açık bir hata mesajı gösterir.

> Not: `run.bat` bu geçiş sırasında gerçek bir Windows makinesinde test
> edilememiştir (bu ortamda sadece Linux mevcuttur); mantık `run.sh` ile
> birebir aynıdır ama ilk çalıştırmada bir sorunla karşılaşırsanız
> bildirin.

## İletişim

Karşılaştığınız sorunları, hatalı çıktı örneklerini ya da geri
bildirimlerinizi aşağıdaki e-posta adresine bildirebilirsiniz:

makilic@yahoo.com

## Lisans

Bu paketin kendi kodu (`verbatim.py`, `gui.py`, `whisper_tier.py`,
`syllables.py`, `convert.py`,
`config.json`, `rules.json`, `run.sh`, `run.bat`) [GNU General Public
License v3.0 (GPLv3)](LICENSE) ile lisanslanmıştır.

Üçüncü taraf bileşenler için bkz.
[THIRD-PARTY-NOTICES.txt](THIRD-PARTY-NOTICES.txt) (İngilizce): Model
Apache 2.0 lisanslıdır; espeak-ng (GPLv3) pakete dahil değildir,
kullanıcının kendi işletim sisteminden ayrıca kurması beklenir.

## Alternatif kurulum: Praat plugin paketi

Bu depodaki kurulum, espeak-ng, ffmpeg ve Python paketleri gibi
bağımlılıkların ayrıca kurulmasını gerektirir (bkz. [Kurulum](#kurulum)).
Bunları tek tek kurmak istemiyorsanız, tüm bu bağımlılıkları içeren bir
Praat plugin'ini, "Transkripsiyon Paketi" klasörü altında aşağıdaki
adresten indirebilirsiniz (paketin indirilmesi için internet bağlantısı
gerekir, ancak kurulduktan sonra ayrıca espeak-ng/ffmpeg/Python paketi
kurmanız gerekmez):

https://drive.google.com/drive/folders/0B4QKykn2bzqQa0JQRTZtU08yZWs?resourcekey=0-kYz6XS59uRJCKj4YuhaqhQ

## Geliştirme Notu

Bu projenin kod ve dokümantasyonunun yazılmasında Anthropic'in Claude Code
aracından yardım alınmıştır.
