#!/usr/bin/env python
# hold tier'inden kural tabanli "syllables" (hece) tier'i uretir.
#
# TUM hece kurallari rules.json'da tanimlidir; bu dosya sadece o
# kurallari YORUMLAYAN genel bir motordur. Yeni bir kural degisikligi
# yapmak icin (yeni bir hece tipi eklemek/kaldirmak, oncelik sirasini
# degistirmek, siralama kuralini degistirmek, C/V hece degerlerini
# guncellemek) NORMALDE bu dosyayi degil, sadece rules.json'u (veya
# hece_degerleri.xlsx + build_rules.py) duzenlemeniz yeterlidir:
#
#   - "hece_degerleri": sembol -> hece degeri (0 = haric, vowel_value =
#     unlu/cekirdek, digerleri = unsuz siniflari). hece_degerleri.xlsx'i
#     duzenleyip `python build_rules.py` calistirarak yeniden uretilir.
#   - "vowel_value": hangi degerin "unlu/cekirdek" sayildigi (varsayilan 8).
#   - "syllable_types_priority": izin verilen hece kaliplari, oncelik
#     sirasina gore (orn. ["CV","CVC","VC","CVCC","V"]). Yeni bir kalip
#     eklemek/kaldirmak, izin verilen onset/koda uzunluklarini ve
#     eslesmelerini degistirir (asagidaki parse_syllable_types()
#     tarafindan bu listeden otomatik turetilir). NOT: Listenin sirasini
#     degistirmek (mevcut kaliplari SILMEDEN/EKLEMEDEN) algoritmanin
#     sonucunu degistirmez -- oncelik, "maksimal onset" ilkesiyle zaten
#     yapisal olarak saglanir (bkz. asagidaki algoritma ozeti). Oncelik
#     sirasi ancak birden fazla kalip AYNI ONSET/KODA uzunlugu ile
#     CAKISTIGINDA anlam kazanir (bu durumda rules.json'a yeni bir
#     alan eklemek gerekir; bkz. PRIORITY_RANK kullanimini genisletmek
#     icin asagidaki not).
#   - "coda_ordering" / "onset_ordering": bir koda/onset kumesi 2+
#     unsuzdan olustugunda, bu unsuzlarin hece degerleri arasinda
#     aranacak sira kurali. Gecerli degerler: "descending" (cekirdekten
#     uzaklastikca deger azalmali -- hece_degerleri.xlsx'teki sonorluk
#     skalasina gore, orn. "nd" kodasi: n=6 > d=2, Turkce'deki "kalp",
#     "kirk", "genc" gibi CVCC ornekleriyle tutarli), "ascending"
#     (tersi) ya da "none" (kural yok). Varsayilan: koda icin
#     "descending". Yeni bir siralama kurali eklemek icin
#     ORDERING_CHECKS sozlugune yeni bir fonksiyon eklemeniz yeterlidir.
#
# Algoritma ozeti:
#   1. Sembol dizisi rules.json'daki "hece_degerleri" degerine gore V
#      (deger == vowel_value), C (0 < deger < vowel_value) veya HARIC
#      (deger == 0 ya da sembol tabloda yok) olarak siniflandirilir.
#   2. HARIC semboller hece olusumuna katilmaz.
#   3. Her unlu (V) bir hece cekirdegi olusturur. Iki unlu arasindaki
#      unsuz(lar), "maksimal onset" ilkesiyle paylastirilir: cekirdege
#      en yakin olanlardan en fazla max_onset (kaliplardan turetilir,
#      su an 1) tanesi SONRAKI hecenin onset'i olur; kalanlardan en
#      fazla max_coda (su an 2) tanesi ONCEKI hecenin kodasi olur. Bu,
#      CV'nin (dusuk onset/koda) CVC/CVCC'den (yuksek onset/koda) once
#      "denenmis" gibi davranmasini saglar.
#   4. Bir (onset_uzunlugu, koda_uzunlugu) ikilisi "syllable_types_priority"
#      listesinde yoksa YA DA koda/onset siralama kurali saglanmiyorsa,
#      kumeden EN UZAK unsur(lar) (cekirdege bitisik OLMAYANLAR) hece
#      disi/bos birakilir ve kalan kumeyle kural tekrar kontrol edilir
#      (kapasiteyi asan durumlar da ayni mekanizmayla cozulur). Koda/
#      onset, cekirdege daima BITISIK kalmalidir; bu yuzden dusen
#      unsur(lar) her zaman cekirdekten en uzaktakilerdir, en yakinlar
#      degil.
#   5. Cekirdege atanamayan/haric semboller "syllables" tier'inde kendi
#      hold sinirinda ayri ve bos ("") kalir; ayni heceye ait ardisik
#      semboller tek bir aralikta birlesir (bkz. build_syllable_tier).
#
# Yeni kural kombinasyonlarini kod degistirmeden dogrulamak icin
# test_cases.xlsx + run_tests.py kullanilabilir (bkz. README.md).

import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RULES_PATH = os.path.join(BASE_DIR, "rules.json")


def load_rules(rules_path=DEFAULT_RULES_PATH):
    with open(rules_path, "r", encoding="utf-8") as f:
        return json.load(f)


def classify(text, rules):
    """Bir sembolu 'V', 'C' ya da None (haric) olarak siniflandirir;
    hece degerini de dondurur (haric ise 0)."""
    vowel_value = rules["vowel_value"]
    hece_degerleri = rules["hece_degerleri"]

    value = hece_degerleri.get(text)
    if value is None or value == 0:
        return None, 0
    if value == vowel_value:
        return "V", value
    return "C", value


def parse_syllable_types(rules):
    """rules['syllable_types_priority'] listesini (orn. ["CV","CVC",...])
    ayristirir. Her kalip tam olarak bir 'V' icermelidir; 'V'den onceki
    'C' sayisi onset uzunlugu, sonraki 'C' sayisi koda uzunlugudur.
    Donen deger: {"types": [(onset_len, coda_len, pattern), ...],
    "allowed_pairs": {(onset_len, coda_len), ...}, "priority_rank": {...},
    "max_onset": int, "max_coda": int}."""
    patterns = rules["syllable_types_priority"]
    types = []
    for pattern in patterns:
        if pattern.count("V") != 1 or any(ch not in "CV" for ch in pattern):
            raise ValueError(
                f"Gecersiz hece tipi deseni: {pattern!r} "
                f"(yalnizca 'C'/'V' iceren, tam olarak bir 'V' iceren bir dize olmali)"
            )
        v_pos = pattern.index("V")
        onset_len = v_pos
        coda_len = len(pattern) - v_pos - 1
        types.append((onset_len, coda_len, pattern))

    allowed_pairs = {(o, c) for o, c, _p in types}
    if (0, 0) not in allowed_pairs:
        raise ValueError(
            "syllable_types_priority listesinde tek basina 'V' (cekirdek) "
            "tipi bulunmali; algoritma kapasite/sira kurali saglanamadiginda "
            "en son bu sekle geriler."
        )

    priority_rank = {(o, c): idx for idx, (o, c, _p) in enumerate(types)}
    max_onset = max(o for o, _c, _p in types)
    max_coda = max(c for _o, c, _p in types)

    return {
        "types": types,
        "allowed_pairs": allowed_pairs,
        "priority_rank": priority_rank,
        "max_onset": max_onset,
        "max_coda": max_coda,
    }


def _ascending_ok(values):
    return all(values[i] < values[i + 1] for i in range(len(values) - 1))


def _descending_ok(values):
    return all(values[i] > values[i + 1] for i in range(len(values) - 1))


def _no_ordering_check(_values):
    return True


ORDERING_CHECKS = {
    "ascending": _ascending_ok,
    "descending": _descending_ok,
    "none": _no_ordering_check,
}


def get_ordering_check(rules, key):
    name = rules.get(key, "none")
    if name not in ORDERING_CHECKS:
        raise ValueError(
            f"{key!r} icin bilinmeyen deger: {name!r}. Gecerli degerler: "
            f"{sorted(ORDERING_CHECKS)} (yeni bir siralama kurali eklemek "
            f"icin syllables.py'deki ORDERING_CHECKS sozlugune fonksiyon ekleyin)."
        )
    return ORDERING_CHECKS[name]


def assign_syllable_ids(symbols, rules):
    """symbols: [str, ...] (IPA sembol dizisi, zamanlamasiz).
    Donen deger: her sembol icin ait oldugu hece numarasi (int, 0'dan
    baslar, sirali) ya da None (haric/atanamamis), symbols ile ayni
    uzunlukta liste. Sadece mantiksal hecelemeyi hesaplar; zamanlama/
    tier birlestirme icin bkz. build_syllable_tier."""
    syl_types = parse_syllable_types(rules)
    coda_ordering_check = get_ordering_check(rules, "coda_ordering")
    onset_ordering_check = get_ordering_check(rules, "onset_ordering")

    allowed_pairs = syl_types["allowed_pairs"]
    max_onset = syl_types["max_onset"]
    max_coda = syl_types["max_coda"]

    n = len(symbols)
    kinds = [None] * n
    values = [0] * n
    for i, sym in enumerate(symbols):
        kinds[i], values[i] = classify(sym, rules)

    syllable_id = [None] * n
    vowel_indices = [i for i in range(n) if kinds[i] == "V"]

    next_syl_no = 0
    onset_len_of = {}
    prev_vowel_i = None

    def resolve_coda(coda_candidates, onset_len_for_owner):
        """coda_candidates: cekirdekten uzaklasan sirada (indeks 0 = cekirdege
        en yakin) unsur indeksleri. Koda, cekirdege BITISIK olmak zorundadir;
        bu yuzden kapasiteyi asan ya da siralama kuralini saglamayan
        durumda, cekirdekten EN UZAK unsur(lar) (listenin sonu) sirayla
        dusurulur -- cekirdege bitisik olan(lar) daima korunur."""
        coda = list(coda_candidates)
        while coda and not (
            len(coda) <= max_coda
            and (onset_len_for_owner, len(coda)) in allowed_pairs
            and coda_ordering_check([values[i] for i in coda])
        ):
            coda.pop()
        return coda

    for vi in vowel_indices:
        syl_no = next_syl_no
        next_syl_no += 1
        syllable_id[vi] = syl_no

        c_block_start = (prev_vowel_i + 1) if prev_vowel_i is not None else 0
        c_block = [i for i in range(c_block_start, vi) if kinds[i] == "C"]

        onset_take = c_block[-max_onset:] if max_onset > 0 else []
        # Onset da cekirdege bitisik olmali; siralama kurali basarisiz
        # olursa cekirdekten EN UZAK unsur (listenin basi) dusurulur.
        while len(onset_take) > 1 and not onset_ordering_check([values[i] for i in onset_take]):
            onset_take = onset_take[1:]

        for i in onset_take:
            syllable_id[i] = syl_no
        onset_len_of[syl_no] = len(onset_take)

        onset_set = set(onset_take)
        coda_candidates = [i for i in c_block if i not in onset_set]

        if prev_vowel_i is not None:
            prev_syl_no = syllable_id[prev_vowel_i]
            coda = resolve_coda(coda_candidates, onset_len_of[prev_syl_no])
            for i in coda:
                syllable_id[i] = prev_syl_no
            # coda_candidates - coda: hece disi/bos kalir (syllable_id=None).

        prev_vowel_i = vi

    if vowel_indices:
        last_vi = vowel_indices[-1]
        last_syl_no = syllable_id[last_vi]
        tail_block = [i for i in range(last_vi + 1, n) if kinds[i] == "C"]
        coda = resolve_coda(tail_block, onset_len_of[last_syl_no])
        for i in coda:
            syllable_id[i] = last_syl_no

    return syllable_id


def build_syllable_tier(hold_intervals, rules):
    """hold_intervals: [{"xmin", "xmax", "text"}, ...] (hold tier'i, ayni
    sinirlarla). Donen deger: syllables tier'i icin interval listesi.
    Ayni heceye ait ardisik hold araliklari tek bir aralikta birlesir
    (text = hecedeki fonemlerin birlesimi, orn. "pat"); haric tutulan ya
    da hecelenmemis (kapasite/sira kurali disi) araliklar kendi hold
    sinirinda ayri ve bos ("") kalir."""

    symbols = [inv["text"] for inv in hold_intervals]
    syllable_id = assign_syllable_ids(symbols, rules)

    result = []
    current = None  # {"xmin", "xmax", "text", "syl_no"}

    def flush():
        if current is not None:
            result.append({"xmin": current["xmin"], "xmax": current["xmax"], "text": current["text"]})

    for i, inv in enumerate(hold_intervals):
        syl_no = syllable_id[i]
        if syl_no is None:
            flush()
            current = None
            result.append({"xmin": inv["xmin"], "xmax": inv["xmax"], "text": ""})
            continue

        if current is not None and current["syl_no"] == syl_no:
            current["xmax"] = inv["xmax"]
            current["text"] += inv["text"]
        else:
            flush()
            current = {"xmin": inv["xmin"], "xmax": inv["xmax"], "text": inv["text"], "syl_no": syl_no}

    flush()

    return result
