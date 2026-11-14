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

# Linux'ta conda-forge'un "noxft" Tk derlemesi eski X fontlarini kullanir;
# bu fontlarda s, g, i (Turkce) gibi harfler bozuk gorunur. Bu derleme
# kuruluysa pencere fontlari icin Xft destekli derlemeye gecilir (bir kez).
if [ "$(uname -s)" = "Linux" ] && conda list "^tk$" 2>/dev/null | grep -q noxft; then
    echo "-- Tk Xft destekli derlemeyle degistiriliyor (Turkce karakterler icin, bir kez)"
    conda install -y -n verbatim -c conda-forge "tk=*=xft*" >/dev/null \
        || echo "UYARI: Xft destekli Tk kurulamadi; pencerede Turkce karakterler bozuk gorunebilir." >&2
fi

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

# 3: kullanici pencerede "Iptal et" / "Durdur" dedi; hata degil.
if [ "$status" -eq 3 ]; then
    exit 0
fi
exit "$status"
