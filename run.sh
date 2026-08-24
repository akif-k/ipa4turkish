#!/usr/bin/env bash
# IPA ile verbatim transkripsiyonu calistirir.
# "verbatim" adli conda ortamini kullanir; wav/ klasorundeki dosyalari
# isleyip tg/, txt/, details/ klasorlerine yazar.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# conda'yi bul ve "verbatim" ortamini etkinlestir.
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$HOME/miniconda3"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$HOME/anaconda3"
else
    echo "HATA: conda bulunamadi. Once 'verbatim' ortamini olusturun:" >&2
    echo "  conda env create -f environment.yml" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate verbatim

set +e
python "$SCRIPT_DIR/verbatim.py" "$@"
status=$?
set -e

if [ "$status" -eq 0 ]; then
    # Islem basariyla bittiginde ipa_verbatim klasorunu dosya yoneticisinde ac.
    # Arka plana atilmiyor: terminal penceresi script bitince kapandigi
    # icin arka plandaki cagri D-Bus istegini tamamlamadan kesilebiliyor.
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$SCRIPT_DIR" >/dev/null 2>&1
    elif command -v open >/dev/null 2>&1; then
        open "$SCRIPT_DIR" >/dev/null 2>&1
    fi
    sleep 1
fi

exit "$status"
