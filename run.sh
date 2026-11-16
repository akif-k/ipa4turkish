#!/usr/bin/env bash
# IPA ile verbatim transkripsiyonu calistirir.
# "verbatim" adli conda ortamini kullanir; wav/ klasorundeki dosyalari
# isleyip tg/, txt/, details/ klasorlerine yazar.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Terminal penceresi olmadan calisirken (masaustu kisayolu, Terminal=false)
# kisayol IPA_VERBATIM_HIDDEN=1 verir: hata mesajlari o zaman ayrica bir
# mesaj kutusunda (zenity varsa) ya da bildirimde gosterilir.
show_error() {
    echo "$1" >&2
    if [ "${IPA_VERBATIM_HIDDEN:-}" = "1" ]; then
        if [ "$(uname -s)" = "Darwin" ]; then
            # Mesaj osascript'e argv ile verilir (tirnak sorunu olmasin)
            osascript -e 'on run argv' \
                -e 'display dialog (item 1 of argv) with title "ipa4turkish" buttons {"Tamam"} with icon stop' \
                -e 'end run' "$1" >/dev/null 2>&1 || true
        elif command -v zenity >/dev/null 2>&1; then
            zenity --error --title="ipa4turkish" --text="$1" >/dev/null 2>&1 || true
        elif command -v notify-send >/dev/null 2>&1; then
            notify-send "ipa4turkish" "$1" || true
        fi
    fi
}

# conda'yi bul ve "verbatim" ortamini etkinlestir.
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$HOME/miniconda3"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$HOME/anaconda3"
# Masaustu kisayolundan baslatilinca PATH'te conda olmayabilir (ozellikle
# macOS .app): yaygin diger kurulum konumlari
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$HOME/miniforge3"
elif [ -f "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="/opt/homebrew/Caskroom/miniconda/base"
elif [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="/opt/miniconda3"
else
    show_error "HATA: conda bulunamadi. Once 'verbatim' ortamini olusturun:
  conda env create -f environment.yml"
    exit 1
fi

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
if ! conda activate verbatim; then
    show_error "HATA: 'verbatim' conda ortami etkinlestirilemedi. Once olusturun:
  conda env create -f environment.yml"
    exit 1
fi

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
