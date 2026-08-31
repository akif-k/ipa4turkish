# ipa_verbatim — Turkce Icin Cok Katmanli Verbatim Transkripsiyon

[English](README.en.md) | Turkce

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22081832.svg)](https://doi.org/10.5281/zenodo.22081832)

`facebook/wav2vec2-xlsr-53-espeak-cv-ft` modeli ile ses dosyalarini IPA
(Uluslararasi Fonetik Alfabe) sembolleriyle hizalayan bir Python komut
satiri uygulamasi. Model ilk kullanimda `models/` klasorune indirilir,
istenirse kopyalanip baska bir bilgisayara tasinabilir (kurulum
sirasinda espeak-ng, ffmpeg ve Python paketleri gibi bagimliliklarin da
ayrica kurulmasi gerekir, bkz. [Kurulum](#kurulum)). Linux, macOS ve
Windows uzerinde calisir (bkz. [Platforma ozel
notlar](#platforma-ozel-notlar)).

Girdi olarak `wav/` klasorundeki `.wav` dosyalarini alir; her dosya icin
UFA kodlarindan olusan metin (`txt/`), cok tier'li bir Praat TextGrid'i
(`tg/`) ve kare-bazli olasilik tablosu (`details/`) uretir (bkz. [Cikti
formati](#cikti-formati)).

`other_formats/` klasorune M4A/AAC, MP3, OPUS, AMR, OGG, FLAC, DSS/DS2,
WMA, 3GP/3GPP veya CAF gibi farkli formatlarda ses dosyalari da
konulabilir; ana islem
baslamadan once bunlar `convert.py` tarafindan otomatik olarak 16 kHz,
16 bit, mono PCM wav'a donusturulup `wav/` klasorune yazilir, sonra
islem normal sekilde devam eder (bkz. [Desteklenen ses
formatlari](#desteklenen-ses-formatlari)).

## Icerik

```
ipa_verbatim/
├── verbatim.py                        asil uygulama
├── syllables.py                       hold tier'inden rules.json'a gore syllables tier'i ureten modul
├── convert.py                         other_formats/ -> wav/ format donusumu (ffmpeg ile)
├── config.json                        fonem eslestirme / filtreleme ve TextGrid ayarlari
├── rules.json                         syllables tier'i icin C/V hece degerleri ve hece tipi kurallari
├── environment.yml                    "verbatim" conda ortamini olusturmak icin
├── run.sh                             conda ortamini etkinlestirip verbatim.py'yi calistiran kisayol (Linux/macOS)
├── run.bat                            ayni kisayolun Windows karsiligi
├── LICENSE                            bu paketin kendi kodunun lisansi (GPLv3)
├── THIRD-PARTY-NOTICES.txt            ucuncu taraf bilesenlerin lisans bilgisi (Ingilizce)
├── README.md / README.en.md           bu belge (Turkce / Ingilizce)
├── models/                            model dosyalari icin klasor (girdi, bos; ilk calistirmada
│                                       wav2vec2-xlsr-53-espeak-cv-ft/ alt klasoruyle otomatik doldurulur,
│                                       bkz. Kurulum > Model dosyalari)
├── wav/                               islenecek .wav dosyalarinin konuldugu klasor (girdi)
│   └── sample.wav                     ornek ses kaydi
├── other_formats/                     farkli formatlardaki ses dosyalarinin konuldugu klasor (girdi, opsiyonel, bos)
├── tg/                                uretilen TextGrid dosyalari (cikti, bos)
├── txt/                               uretilen fonem metinleri (cikti, bos)
└── details/                           kare-bazli olasilik tablolari (cikti, bos)
```

## Kurulum

### 1) Gereksinimler

- [Miniconda / Anaconda](https://docs.conda.io/en/latest/miniconda.html)
- **espeak-ng** (isletim sistemi paketi). Model tokenizer'i
  (`Wav2Vec2PhonemeCTCTokenizer`) yuklenirken bunun sistemde bulunmasini
  zorunlu kilar (fiili fonemizasyon icin cagrilmaz, sadece kutuphanenin
  var olmasi yeterlidir):

  ```bash
  # Ubuntu / Debian
  sudo apt install espeak-ng

  # macOS (Homebrew)
  brew install espeak-ng
  ```

  Windows icin bkz. [Platforma ozel notlar](#platforma-ozel-notlar).

- **ffmpeg** — yalnizca `other_formats/` klasorunu kullanacaksaniz
  gerekir (bkz. [Desteklenen ses formatlari](#desteklenen-ses-formatlari)).
  `environment.yml` icinde conda-forge bagimliligi olarak tanimlidir,
  asagidaki adimda ortamla birlikte otomatik kurulur; ayrica sistem
  paketi kurmaniza gerek yoktur.

- (Opsiyonel) GPU hizlandirmasi — varsa otomatik kullanilir, yoksa
  CPU'ya duser: Linux/Windows'ta NVIDIA GPU + guncel surucu (CUDA),
  macOS'ta Apple Silicon (MPS).

### 2) Python paket bagimliliklari

`environment.yml`, "verbatim" ortamini olustururken asagidaki Python
paketlerini de conda-forge/pip'ten otomatik kurar; elle kurmaniza gerek
yoktur:

| Paket | Amac |
|---|---|
| `torch` | model cikarimini calistiran PyTorch |
| `transformers` | Wav2Vec2 model ve tokenizer (`Wav2Vec2Processor`, `Wav2Vec2ForCTC`) |
| `librosa` | `.wav` dosyalarini okuma, 16 kHz mono'ya yeniden orneklem |
| `safetensors` | model agirliklarinin (`model.safetensors`) yuklenmesi |
| `phonemizer` | tokenizer'in yuklenmesi icin gereken espeak-ng baglayicisi |
| `numpy` | dizi/sayisal islemler |

### 3) "verbatim" conda ortamini olustur

```bash
cd ~/ipa_verbatim
conda env create -f environment.yml
```

`environment.yml`, torch'u varsayilan olarak CUDA 12.8 wheel'i ile kurar.
GPU'nuz yoksa veya farkli bir CUDA surumu gerekiyorsa, dosyadaki `torch`
satirini kaldirip [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
adresindeki komutla platformunuza uygun surumu ayrica kurun.

Ortam zaten varsa guncellemek icin:

```bash
conda env update -n verbatim -f environment.yml --prune
```

### 4) Model dosyalari

`models/` klasoru bu paketle birlikte bos gelir. `verbatim.py` ilk
calistirildiginda, `config.json`, `model.safetensors` (~1.2 GB),
`vocab.json`, `tokenizer_config.json`, `preprocessor_config.json`,
`special_tokens_map.json` dosyalarini Hugging Face Hub'dan
(`facebook/wav2vec2-xlsr-53-espeak-cv-ft`) otomatik olarak
`models/wav2vec2-xlsr-53-espeak-cv-ft/` altina indirir — bunun icin
internet baglantisi gerekir. Sonraki calistirmalarda bu klasor dolu
bulundugundan tekrar indirme yapilmaz.

Indirilen model klasorunu (`models/wav2vec2-xlsr-53-espeak-cv-ft/`)
baska bir bilgisayara kopyalayip oradaki ayni konuma yerlestirirseniz, o
bilgisayarda modelin ayrica indirilmesi gerekmez. (Kurulumun geri
kalani — conda ortami, espeak-ng, ffmpeg gibi bagimliliklar — yine de o
bilgisayarda ayrica ve internet baglantisiyla kurulmalidir, bkz.
[Kurulum](#kurulum).)

## Kullanim

1. Islenecek `.wav` dosyalarini `wav/` klasorune koyun. `wav/`
   klasorundeki dosyalarin ornekleme hizi/bit derinligi/kanal sayisi
   onemli degildir — `verbatim.py`, `librosa` ile okurken bunlari
   otomatik olarak 16 kHz mono'ya cevirir. Farkli bir konteyner/codec'te
   (M4A/AAC, MP3, OPUS, AMR, OGG, FLAC, DSS/DS2, WMA, 3GP/3GPP, CAF)
   kayitlariniz varsa (yani `.wav` uzantili olmayan dosyalar), bunlari
   `wav/` yerine `other_formats/` klasorune koyabilirsiniz;
   `verbatim.py` calisirken bunlari otomatik olarak wav'a cevirip
   `wav/` klasorune yazar, sonra normal islem devam eder (bkz.
   [Desteklenen ses formatlari](#desteklenen-ses-formatlari)).
2. Calistirin:

   ```bash
   # Linux / macOS (ilk seferde calistirma izni gerekir)
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

3. Sonuclar `tg/`, `txt/` ve `details/` klasorlerine yazilir; ilerleme
   terminalde bir ilerleme cubugu ile gosterilir.
4. `run.sh` / `run.bat` ile calistirildiginda, islem hatasiz tamamlaninca
   `ipa_verbatim` klasoru dosya yoneticisinde otomatik olarak acilir
   (`python verbatim.py` ile dogrudan calistirirsaniz bu olmaz).

### Masaustu kisayolu

**Linux:** `run.sh`'i calistiran bir `.desktop` kisayolu masaustune elle
eklenebilir (freedesktop/GNOME'a ozgudur, macOS ve Windows'ta calismaz —
asagiya bakin). Ilk cift tiklamada dosya yoneticisi "Baslatmaya izin
ver" uyarisi gosterirse, kisayola sag tikla > **Baslatmaya izin ver**
secilmelidir.

**macOS:** `.desktop` dosyalari desteklenmez. `run.sh`'i Finder'dan cift
tiklanabilir yapmak icin `.command` uzantili bir sembolik baglanti
olusturmak yeterli (kopya degil baglanti, boylece `run.sh` guncellenirse
kisayol da otomatik guncel kalir):

```bash
ln -s ~/ipa_verbatim/run.sh ~/Desktop/ipa_verbatim.command
chmod +x ~/ipa_verbatim/run.sh
```

**Windows:** `run.bat` dosyasina sag tiklayip **Gonder > Masaustu
(kisayol olustur)** secilerek standart bir Windows kisayolu
olusturulabilir; ayrica bir betik gerekmez.

### Klasor konumlarini degistirme

Varsayilan olarak tum klasorler bu betikle ayni dizinde aranir. Farkli
konumlar belirtmek icin:

```bash
python verbatim.py \
  --wav-dir /baska/yol/wav \
  --other-formats-dir /baska/yol/other_formats \
  --tg-dir /baska/yol/tg \
  --txt-dir /baska/yol/txt \
  --details-dir /baska/yol/details \
  --model-dir /baska/yol/models/wav2vec2-xlsr-53-espeak-cv-ft \
  --config /baska/yol/config.json \
  --rules /baska/yol/rules.json \
  --device cpu   # veya cuda, cuda:0, mps ... (varsayilan: otomatik algila)
```

## Desteklenen ses formatlari

`other_formats/` klasorune konulan asagidaki formatlar, `convert.py`
tarafindan `ffmpeg` ile 16 kHz, 16 bit, mono PCM wav'a donusturulur:

| Format | Uzanti |
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

Donusturulen dosya, `wav/` klasorunde ayni taban adla (`<ad>.wav`)
zaten varsa atlanir; yoksa olusturulur. Bu betikle ayni ad tasiyan bir
dosya birden fazla formatta `other_formats/` icinde bulunursa,
hangisinin islenecegi glob sirasina baglidir — ayni ad icin tek bir
kaynak dosya birakmaniz onerilir.

Tek basina calistirmak icin (orn. sadece donusum yapip verbatim.py'yi
sonra calistirmak istiyorsaniz):

```bash
conda activate verbatim
python convert.py
```

Baska formatlar gerekiyorsa, `convert.py` icindeki
`OTHER_FORMAT_EXTENSIONS` kumesine uzantiyi eklemek yeterlidir — ffmpeg
zaten destekliyorsa baska bir degisiklik gerekmez.

## Cikti formati

**`txt/<dosya>.txt`** — modelin ürettigi fonem dizisi, tek satir.

**`tg/<dosya>.TextGrid`** — Praat TextGrid'i; 6 sabit tier + `config.json`
icindeki `chunk_gap_thresholds_ms` listesinin uzunlugu kadar `chunks`
tier'i icerir (bu depodaki mevcut `config.json` ile 3 esik → toplam 9
tier; esik sayisi degistirilerek tier sayisi ayarlanabilir):

| Tier | Aciklama |
|---|---|
| `hold` | son gecerli (bos olmayan) fonemin bir sonraki foneme kadar tutuldugu araliklar |
| `prob1` / `prob2` / `prob3` | her karede en yuksek 3 aday fonem icin ayri tier'lar |
| `segments` | ham kare-bazli en olasi fonem dizisi (birlestirilmemis) |
| `syllables` | `hold`'dan, `rules.json` icindeki kurallara gore turetilen hece tier'i (bkz. [rules.json ve syllables](#rulesjson-ve-syllables)) |
| `chunks1`..`chunksN` | `segments`'in, `config.json` icindeki `chunk_gap_thresholds_ms` esiklerine (bu depoda varsayilan 40/80/120 ms) gore kucuk parcalarin komsu parcayla birlestirilmesiyle olusturulan, gitgide daha kaba taneli N versiyonu |

`hold` ve `chunks1-N` tier'larindaki sinirlar en yakin sifir gecisine
(zero-crossing) kaydirilir; `prob1-3`, `segments` ve `syllables`'a
dokunulmaz.

**`details/<dosya>_details.txt`** — her kare icin baslangic/bitis zamani
ve ilk 3 aday fonem + olasilik yuzdesi (tab ile ayrilmis tablo).

## config.json

- `char_mapping`: modelin urettigi bazi fonem sembollerini baska bir
  sembole eslestirmek icin (orn. `"ɡ": "g"`).
- `exclude_chars`: TextGrid ve details ciktisinda gosterilmeyecek
  (bosluga cevrilecek) sembol/ozel token listesi (vurgu isaretleri,
  `<pad>`/`<s>`/`</s>`/`<unk>` ozel tokenlari, model diline ait
  kullanilmayan fonemler vb.).
- `min_confidence_percent`: bir adayin gosterilmesi icin gereken minimum
  olasilik yuzdesi; altindaki adaylar bos sayilir.
- `chunk_gap_thresholds_ms`: `chunks1`..`chunksN` tier'lari icin
  kullanilan birlestirme esikleri (milisaniye). Liste uzunlugu kadar
  `chunks` tier'i uretilir.

## rules.json ve syllables

`syllables` tier'i, `hold` tier'indeki her araligi `rules.json`'daki
`hece_degerleri` tablosuna gore V (unlu, deger `vowel_value`, varsayilan
8), C (unsuz, deger 1..`vowel_value`-1) ya da haric (deger 0 ya da
tabloda yok) olarak siniflandirip, `rules.json`'daki kurallara gore
hecelere ayirarak olusturulur. **Kurallarin tamami `rules.json`'da veri
olarak tutulur; `syllables.py` bunlari yorumlayan genel bir motordur.**
Yani asagidaki degisiklikler icin normalde `syllables.py`'ye
dokunmaniza gerek yoktur, sadece `rules.json`'u duzenlemeniz yeterlidir:

| Degistirmek istediginiz | rules.json alani |
|---|---|
| Bir sembolun hece degeri (C/V grubu) | `hece_degerleri` |
| Hangi degerin "unlu/cekirdek" sayildigi | `vowel_value` |
| Izin verilen hece tipleri (CV, CVC, VC, CVCC, V, ya da yeni bir kalip) | `syllable_types_priority` |
| Koda/onset'teki 2+ unsuzun hece degeri sirasi kurali | `coda_ordering` / `onset_ordering` |

Uygulanan kurallar:

- Hece cekirdeginin (V) hece degeri her zaman `vowel_value`'dur (varsayilan 8).
- Bir hecede 3 unsuz yan yana olamaz — bu, `syllable_types_priority`
  listesindeki hic bir kalipta 3 unsuzluk bir kume olmamasindan (onset
  en fazla 1, koda en fazla 2 unsuz) dogal olarak saglanir; listeye 3
  unsuzluk bir onset/koda iceren yeni bir kalip (orn. `"CVCCC"`)
  eklenirse bu sinir da otomatik guncellenir.
- Iki unlu arasindaki unsuzlarin en fazla `max_onset` (kaliplardan
  turetilir, su an 1) tanesi bir sonraki hecenin onset'i, en fazla
  `max_coda` (su an 2) tanesi bir onceki hecenin kodasi olabilir
  (maksimal onset ilkesi).
- Bir (onset uzunlugu, koda uzunlugu) ikilisi `syllable_types_priority`
  listesinde yoksa, YA DA `coda_ordering`/`onset_ordering` kurali
  (varsayilan: koda icin **"descending"** — cekirdekten uzaklastikca
  hece degeri AZALMALI; sonorluk skalasina gore, orn. "nd" kodasi: n=6
  > d=2, Turkce'deki "kalp", "kirk", "genc"
  gibi ornekleriyle tutarli) saglanmiyorsa: kumeden cekirdekten EN UZAK
  unsur(lar) sirayla hece disi/bos birakilir ve kalan kumeyle kural
  tekrar kontrol edilir. Koda/onset her zaman cekirdege BITISIK
  kalmalidir — bu yuzden dusen her zaman en uzaktaki(ler)dir, cekirdege
  bitisik olan(lar) degil. (CVCC'de 3. segmentin hece degeri 4.'ten
  BUYUK olmali kurali, bu genel mekanizmanin `coda_ordering:
  "descending"` ile ozel bir sonucudur; sira saglanmazsa 4. segment
  -cekirdekten daha uzaktaki- dusurulur, CVC'ye gerilenir.)
- Haric tutulan segmentler (deger 0, tabloda olmayan semboller, ya da
  yukaridaki kurallar geregi atanamayanlar) hece yapisina katilmaz;
  `syllables` tier'inde kendi `hold` sinirinda ayri ve bos bir aralik
  olarak kalir.

**Hece siniri da sonorluga gore belirlenir.** Onset icin varsayilan
`onset_ordering: "ascending"` (cekirdege yaklastikca hece degeri
ARTMALI — sonorluk skalasinda cekirdege dogru yukselen bir onset,
`coda_ordering: "descending"`'in aynasi). Su an `syllable_types_priority`
listesinde 2+ unsuzluk bir onset kalibi (orn. `"CCV"`) bulunmadigindan
(`max_onset = 1`), bu kural henuz FIILEN devrede degildir -- tek
unsurluk bir onsette siralama kontrol edilecek bir "sira" yoktur. Ancak
Turkce'deki yabanci kokenli kelimelerde gorulen "tren", "grup", "plan"
gibi yukselen-sonorluklu onset kumelerini desteklemek icin ileride
`syllable_types_priority`'ye `"CCV"` (ve gerekirse `"CCVC"`) eklenirse,
bu kural otomatik olarak devreye girer: sonorlugu cekirdege dogru
ARTMAYAN bir 2'li onset kumesi gecersiz sayilir ve cekirdekten EN UZAK
unsur (ayni "cekirdege bitisik kal" ilkesiyle) dusurulerek tek unsurluk
onsete gerilenir.

`syllable_types_priority`'nin **sirasini** degistirmek (kaliplari
SILMEDEN/EKLEMEDEN) sonucu degistirmez — oncelik, yukaridaki maksimal
onset ilkesiyle yapisal olarak zaten saglanir. Sonuc uzerinde etkili
olan, listeye yeni bir kalip **eklemek/cikarmak**tir.

`config.json`'daki `char_mapping` bazi ham sembolleri (orn. `ɡ` -> `g`,
`ɚ` -> `ər`) `hold` tier'ine girmeden once degistirdiginden, `rules.json`
icinde bu eslenmis sembollerin hece degerleri de (kaynak sembolle ayni
deger) ayrica tanimlidir.

### Kural degisikliklerinde notasyon

Bir kural degisikligi (yeni hece tipi, farkli bir sira kurali, degisen
bir hece degeri...) yaptiginizda, beklenen hecelemeyi asagidaki
notasyonla ifade edebilirsiniz.

**Notasyon** (bir hucre = tek bir test): semboller bosluk ile ayrilir.

- Ilk sembol otomatik olarak 1. hecenin parcasidir.
- Yeni bir hecenin **ilk** sembolunun basina bosluksuz `|` eklenir.
  Orn: `k a |p s a` -> `ka` + `psa` (2 hece).
- Hece disi/atanmamis (haric ya da kapasite/sira kurali disi) olmasi
  beklenen bir sembol parantez icine alinir: `(n)`.
  Orn: `s a |n a n (d) |r i` -> `sa` + `nan` (sondaki d haric) + `ri`.

### Yanlis bir hecelemeyi bildirme / duzeltme isteme

`tg/` klasorundeki bir TextGrid'in `syllables` tier'inde yanlis
gordugunuz bir hecelemeyi duzeltmek istediginizde:

1. Dosya adini, sorunlu yeri (zaman ya da fonem dizisi) ve **olmasi
   gerekeni** not edin. Praat'ta `syllables` tier'ine (ya da referans
   icin `hold`/`segments` tier'lerine) bakarak hatali IPA sembol
   dizisini okuyabilirsiniz.
2. Bunu dogal dille (Turkce) soyle aktarabilirsiniz, orn.:
   *"turku.wav'da 0.7-1.0s civarinda 'g ʌ n' 'gʌ' + '' + 'n' olarak
   ayri ayri kaliyor, oysa 'gʌn' olarak tek hece olmali."*
   ya da hazirsaniz dogrudan yukaridaki notasyonla:
   *"'g ʌ |n' bekleniyordu, 'g ʌ (n)' cikti alindi."*
3. Bu bilgiyle (siz ya da kod uzerinde calisan biri) once KOKENI
   belirler: sorun bir hece degerinden mi (`rules.json`'daki
   `hece_degerleri`), izin verilen hece tiplerinden mi
   (`syllable_types_priority`), yoksa siralama kuralindan mi
   (`coda_ordering`/`onset_ordering`) kaynaklaniyor?
4. Ilgili `rules.json` alanini duzeltip gercek bir ses dosyasiyla
   (`python verbatim.py`) uctan uca tekrar deneyip TextGrid'de beklenen
   sonucu gordugunuzu teyit edin.

## Platforma ozel notlar

`verbatim.py`'nin kendisi platformdan bagimsizdir (saf Python + torch +
transformers + librosa); asagidakiler yalnizca kurulum ve calistirma
kisayollarinda (`run.sh`/`run.bat`, masaustu kisayolu) gecerli platform
farklari:

| Konu | Linux | macOS | Windows |
|---|---|---|---|
| Calistirma kisayolu | `./run.sh` | `./run.sh` | `run.bat` |
| espeak-ng kurulumu | `sudo apt install espeak-ng` (veya dagitiminizin paket yoneticisi) | `brew install espeak-ng` | Asagiya bakin |
| Isletim sonu klasor acma | `xdg-open` (run.sh icinde otomatik) | `open` (run.sh icinde otomatik) | `explorer` (run.bat icinde otomatik) |
| GPU hizlandirma | NVIDIA + CUDA (`cuda`) | Apple Silicon (`mps`) | NVIDIA + CUDA (`cuda`) |
| Masaustu kisayolu | `.desktop` dosyasi (bkz. yukarida) | `.command` sembolik baglantisi (bkz. yukarida) | `run.bat`'tan "Kisayol olustur" (bkz. yukarida) |

`run.sh`, `xdg-open` ve `open` komutlarini sirayla dener, bu yuzden
Linux/macOS arasinda elle bir sey degistirmeniz gerekmez — script hangi
platformda calistigini otomatik anlar. `run.bat` de ayni sekilde
Windows'a ozgu `explorer` komutunu kullanir.

### Windows'a ozgu kurulum

- **espeak-ng:** conda-forge'da paket olarak bulunmuyor; resmi kurulum
  dosyasini [espeak-ng GitHub Releases](https://github.com/espeak-ng/espeak-ng/releases)
  sayfasindan (`.msi`) indirip kurun. Kurulumdan sonra `libespeak-ng.dll`
  cogunlukla `C:\Program Files\eSpeak NG\` altinda olur; `verbatim.py`
  bu konumu (ve `Program Files (x86)` karsiligini) calisirken otomatik
  arar ve `PHONEMIZER_ESPEAK_LIBRARY` ortam degiskenini kendisi ayarlar.
  Farkli bir konuma kurduysaniz, calistirmadan once elle ayarlayin:

  ```bat
  set PHONEMIZER_ESPEAK_LIBRARY=C:\yol\libespeak-ng.dll
  ```

- **conda ortami:** `environment.yml`, Anaconda Prompt / PowerShell
  icinde Linux'takiyle ayni komutla kurulur: `conda env create -f
  environment.yml`. `torch` icin kullanilan CUDA 12.8 wheel'i Windows
  icin de saglanir; GPU'nuz yoksa yine `torch` satirini kaldirip
  [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
  adresindeki Windows komutunu kullanin.
- `run.bat`, sistemde `conda` PATH'te degilse yaygin kurulum
  konumlarini (`%USERPROFILE%\miniconda3`, `%USERPROFILE%\anaconda3`,
  vb.) otomatik dener; bulamazsa acik bir hata mesaji gosterir.

> Not: `run.bat` bu gecis sirasinda gercek bir Windows makinesinde test
> edilememistir (bu ortamda sadece Linux mevcuttur); mantik `run.sh` ile
> birebir aynidir ama ilk calistirmada bir sorunla karsilasirsaniz
> bildirin.

## Iletisim

Karsilastiginiz sorunlari, hatali cikti orneklerini ya da geri
bildirimlerinizi asagidaki e-posta adresine bildirebilirsiniz:

makilic@yahoo.com

## Lisans

Bu paketin kendi kodu (`verbatim.py`, `syllables.py`, `convert.py`,
`config.json`, `rules.json`, `run.sh`, `run.bat`) [GNU General Public
License v3.0 (GPLv3)](LICENSE) ile lisanslanmıştır.

Üçüncü taraf bileşenler için bkz.
[THIRD-PARTY-NOTICES.txt](THIRD-PARTY-NOTICES.txt) (İngilizce): Model
Apache 2.0 lisanslıdır; espeak-ng (GPLv3) pakete dahil değildir,
kullanıcının kendi işletim sisteminden ayrıca kurması beklenir.

## Alternatif kurulum: Praat plugin paketi

Bu depodaki kurulum, espeak-ng, ffmpeg ve Python paketleri gibi
bagimliliklarin ayrica kurulmasini gerektirir (bkz. [Kurulum](#kurulum)).
Bunlari tek tek kurmak istemiyorsaniz, tum bu bagimliliklari iceren bir
Praat plugin'ini, "Transkripsiyon Paketi" klasoru altinda asagidaki
adresten indirebilirsiniz (paketin indirilmesi icin internet baglantisi
gerekir, ancak kurulduktan sonra ayrica espeak-ng/ffmpeg/Python paketi
kurmaniz gerekmez):

https://drive.google.com/drive/folders/0B4QKykn2bzqQa0JQRTZtU08yZWs?resourcekey=0-kYz6XS59uRJCKj4YuhaqhQ

## Gelistirme Notu

Bu projenin gelistirilmesinde, kod yazimi ve dokumantasyon calismalarina
destek olarak Anthropic'in Claude Code araci kullanilmistir.
