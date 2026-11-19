# Deteksi Dini Cacat Injection Molding 

Kasus: 01 - Pengendalian Mutu Injection Molding · Aplikasi: https://mkb-ml-kasus01-qc-injection-molding-duxjjjld6uwamktmsdmnmc.streamlit.app/ · Video: [Tautan Video Demo]

## Anggota

| Anggota | NIM | Bagian yang dikerjakan |
| :--- | :--- | :--- |
| Nafira Zahra Anantha | 2405832 | Audit data mentah, pembersihan anomali sensor (`src/cleaning.py`), dan EDA (`notebooks/01`) | Inferensi & validasi input ekstrem (`src/predict.py`), API (`app/api.py`), aplikasi (`app/streamlit_app.py`), dan pengujian (`tests/`) |
| Nailah Khoirun Widad | 2401678 | Rekayasa fitur fisika proses (`src/features.py`), pemodelan, dan optimasi threshold (`src/train.py`) | Penyusunan laporan (`LAPORAN.md`), deployment aplikasi daring, dan video demo |

## Masalah dan pengguna

Pengguna sistem ini adalah **operator mesin injection molding dan tim Quality Control (QC)** di pabrik komponen plastik Cikarang (`IM-01` s.d. `IM-04`). Model ini membantu operator mengambil keputusan secara real-time di setiap siklus cetak apakah komponen perlu **langsung disisihkan untuk inspeksi** atau **lolos QC**, sekaligus memberikan rekomendasi penyesuaian parameter mesin sebelum ratusan komponen cacat terlanjur diproduksi.

## Data dan pembersihan

| Masalah yang ditemukan | Jumlah baris | Cara menangani |
| :--- | :---: | :--- |
| Baris duplikat eksak pada `data/raw.csv` (dari 25.350 baris) | 350 | Dihapus dengan `drop_duplicates()` sehingga tersisa 25.000 baris unik |
| Inkonsistensi penulisan kategori `bahan` (`P.P`, `pp`, `Abs`) & `shift` | Seluruh variasi | Dinormalisasi menjadi `PP`, `ABS`, `PC` dan `pagi`, `siang`, `malam` |
| Kesalahan satuan suhu (`suhu_leleh_C > 350`, tercatat dalam °F) | 579 | Dikonversi ke °C menggunakan rumus $(T - 32) \times 5/9$ |
| Tekanan injeksi bernilai <= 0 bar (sensor dropout) | 125 | Diubah menjadi `NaN`, lalu diimputasi median per jenis bahan (fit hanya pada train) |
| Waktu siklus bernilai <= 0 detik (kode `-1`) | 75 | Diubah menjadi `NaN`, lalu direkonstruksi dari `waktu_pendinginan_s + median_non_cooling` |
| Suhu cetakan tidak wajar (< 12 °C) | 1 | Diubah menjadi `NaN`, lalu diimputasi median per jenis bahan |
| Nilai hilang (`NaN`) pada `kelembaban_bahan_pct` & `waktu_pendinginan_s` | Seluruh `NaN` | Diimputasi median per jenis bahan di dalam pipeline disertai flag biner missing |

## Hasil

| Model | PR-AUC validation | PR-AUC test |
| :--- | :---: | :---: |
| Baseline: `LogisticRegression` (fitur asli bersih) | 0.5182 | 0.4938 |
| Final: `HistGradientBoostingClassifier` + Fitur Fisika | **0.6798** | **0.6727** |

Threshold yang dipakai: **0.065**, karena biaya komponen cacat lolos ke pelanggan ($FN = \text{Rp500.000}$) 20 kali lebih besar dibandingkan biaya komponen bagus ikut disisihkan ($FP = \text{Rp25.000}$); pemilihan $\tau = 0,065$ pada validation set meminimalkan total kerugian bisnis pada test set dari **Rp48.675.000** (baseline) menjadi **Rp36.550.000** (menghemat **Rp12.125.000** per 3.750 siklus).

## Cara menjalankan

```bash
pip install -r requirements.txt
uvicorn app.api:app --reload            # API
streamlit run app/streamlit_app.py      # aplikasi
