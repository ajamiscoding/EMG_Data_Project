# EMG_Data_Project


## 1) Depoyu Klonla
```bash
git clone <[REPO_URL](https://github.com/ajamiscoding/EMG_Data_Project)>
cd <EMG_Data_Project>
```

## 2) Sanal Ortamı Oluştur ve Aktifleştir
**macOS / Linux**
```bash
python3 -m venv emg_venv
source emg_venv/bin/activate
```
**Windows (PowerShell)**
```powershell
python -m venv emg_venv
emg_venv\Scripts\Activate.ps1
```

## 3) Gereksinimleri Yükle
```bash
pip install -r requirements.txt
```

## 4) Projeyi Çalıştır

 tek tek yöntemleri çalıştır:
```bash
python Upsampling_linear.py
python Upsampling_cubic.py
python Upsampling_polyphase.py
```
Karıştırma Kodu:
```bash
python compare_emg_upsampling.py
```

## 5) Örnek Görsel
`saved_plots/` klasörüne bir görsel koy (dosya adını kendine göre değiştir) ve aşağıdaki gibi göster:
```markdown
![Örnek Çıktı](saved_plots/BasParmakEkstansiyon__emg2__overlay_native.png)
```

