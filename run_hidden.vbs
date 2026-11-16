' run_hidden.vbs
' ipa4turkish'i konsol (cmd) penceresi olmadan baslatir: run.bat'i gizli
' pencerede calistirir. Windows masaustu kisayolu bu dosyayi gostermeli
' (bkz. README > Masaustu kisayolu). Programin tum ciktisi pencerenin
' islem gunlugunde gorunur; pencere acilmadan olusan hatalar ayri bir
' hata penceresinde gosterilir.
Option Explicit

Dim shell, fso, folder, q
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
folder = fso.GetParentFolderName(WScript.ScriptFullName)
q = Chr(34)

' run.bat ve verbatim.py konsolun gorunmedigini bu degiskenden anlar
shell.Environment("PROCESS")("IPA_VERBATIM_HIDDEN") = "1"
shell.CurrentDirectory = folder

' 0: gizli pencere; False: bitmesini bekleme
shell.Run "cmd /c " & q & q & folder & "\run.bat" & q & q, 0, False
