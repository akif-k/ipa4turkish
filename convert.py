#!/usr/bin/env python
# Farkli ses formatlarindaki (M4A/AAC, MP3, OPUS, AMR, OGG, FLAC, DSS/DS2,
# WMA, 3GP/3GPP, CAF) kayitlari 16 kHz, 16 bit, mono PCM wav'a donusturur
# (ffmpeg ile).
#
# verbatim.py tarafindan otomatik cagrilir (other_formats/ -> wav/, ana
# islemden once); ayrica tek basina da calistirilabilir:
#   python convert.py [--other-formats-dir ...] [--wav-dir ...]

import argparse
import glob
import os
import shutil
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OTHER_FORMAT_EXTENSIONS = {
    ".m4a", ".aac", ".mp3", ".opus", ".amr", ".ogg", ".oga", ".flac", ".dss", ".ds2",
    ".wma", ".3gp", ".3gpp", ".caf",
}


def print_progress(idx, total, filename):
    bar_width = 30
    filled = min(bar_width, round(bar_width * idx / max(total, 1)))
    bar = "#" * filled + "-" * (bar_width - filled)
    sys.stdout.write(f"\r[{bar}] {idx}/{total}  {filename:<40}")
    sys.stdout.flush()


def convert_other_formats(other_formats_dir, wav_dir):
    """other_formats_dir icindeki desteklenen formatlari 16 kHz / 16 bit /
    mono PCM wav'a donusturup wav_dir'e yazar. wav_dir'de ayni isimde
    (ayni taban ad + .wav) bir dosya zaten varsa, o dosya atlanir."""
    if not os.path.isdir(other_formats_dir):
        return

    source_files = sorted(
        p for p in glob.glob(os.path.join(other_formats_dir, "*"))
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in OTHER_FORMAT_EXTENSIONS
    )
    if not source_files:
        return

    if shutil.which("ffmpeg") is None:
        sys.exit(
            "HATA: other_formats/ klasorunde donusturulecek dosya var ama ffmpeg bulunamadi.\n"
            "'conda env update -n verbatim -f environment.yml --prune' ile kurun, ya da "
            "sisteminize ayrica kurun (orn. 'sudo apt install ffmpeg' / 'brew install ffmpeg' / "
            "https://ffmpeg.org/download.html)."
        )

    os.makedirs(wav_dir, exist_ok=True)

    print(f"other_formats/ icinde {len(source_files)} dosya bulundu, wav'a donusturuluyor...")
    total = len(source_files)
    converted = 0
    skipped = 0
    failed = []

    for idx, src_path in enumerate(source_files, 1):
        file_name = os.path.basename(src_path)
        base_name = os.path.splitext(file_name)[0]
        dst_path = os.path.join(wav_dir, f"{base_name}.wav")
        print_progress(idx - 1, total, file_name)

        if os.path.exists(dst_path):
            skipped += 1
            print_progress(idx, total, file_name)
            continue

        result = subprocess.run(
            [
                "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
                "-i", src_path,
                "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                dst_path,
            ],
            capture_output=True, text=True,
        )

        if result.returncode != 0 or not os.path.exists(dst_path):
            err_lines = result.stderr.strip().splitlines() if result.stderr else []
            failed.append((file_name, err_lines[-1] if err_lines else "bilinmeyen hata"))
        else:
            converted += 1

        print_progress(idx, total, file_name)

    print()
    if converted:
        print(f"{converted} dosya wav/ klasorune donusturuldu.")
    if skipped:
        print(f"{skipped} dosya atlandi (wav/ icinde ayni isimde dosya zaten var).")
    if failed:
        print(f"UYARI: {len(failed)} dosya donusturulemedi:")
        for name, err in failed:
            print(f"  - {name}: {err}")


def main():
    parser = argparse.ArgumentParser(
        description="other_formats/ klasorundeki ses dosyalarini 16 kHz/16 bit/mono PCM wav'a donustur"
    )
    parser.add_argument("--other-formats-dir", default=os.path.join(BASE_DIR, "other_formats"))
    parser.add_argument("--wav-dir", default=os.path.join(BASE_DIR, "wav"))
    args = parser.parse_args()

    os.makedirs(args.other_formats_dir, exist_ok=True)
    convert_other_formats(args.other_formats_dir, args.wav_dir)


if __name__ == "__main__":
    main()
