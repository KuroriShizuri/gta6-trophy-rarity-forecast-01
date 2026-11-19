# Keputusan model — gta6-trophy-rarity-forecast-01

**Status: FINAL (2026-11-19).** Keputusan user (18:47 WIB): *"1,2,3,4,5 kamu harus
jawab berdasarkan data gta 5. Karena itu satu satunya data prior yang ada."*
Semua lima pertanyaan dijawab dari data GTA V. Terimplementasi di `main.py`:
`MODEL = "poisson-hazard"` + prior GTA V per tipe (`TROPHY_PRIOR`,
`PRIOR_SATURATION`, `LAMBDA_PRIOR`, `E_INF_CAP_MULT`, `LAMBDA_PENALTY_W`,
`P_SLOPE_CAP`). Mengganti model = mengubah konstanta + parameter itu — tidak
ada restruktur kode.

## Keputusan final (ringkasan)

**Kombinasi: median poisson-hazard + prior GTA V + spread bersama.**
Median dari model poisson-hazard (satu-satunya keluarga yang memodelkan
pembilang dan penyebut secara terpisah — mekanisme refresh), diregularisasi
oleh prior GTA V: (a) cap keras E_inf per tipe dari saturasi GTA V, (b) penalti
log-λ lemah menuju pusat prior per tipe. Spread dipertahankan dari provisional
(STALE_SIGMA=1,5 / STALE_TAU=8 jam). Median BOLEH turun mengantisipasi
refresh, tapi lewat mekanisme eksplisit P(t) di model poisson — bukan offset
ad-hoc — dan laju pertumbuhan P(t) dibatasi cap 0,01/jam.

## Sumber data GTA V (diambil 2026-11-19)

**Sumber: Exophase GTA V PS3** — `https://www.exophase.com/game/grand-theft-auto-v-ps3/trophies/`
(daftar launch-era, 50 trofi dasar). Halaman PSN/PS5 juga diambil sebagai
pembanding (78 trofi; populasi PS5 condong ke Online, jadi PS3 yang dipakai).
**Yang data ini ADALAH:** rarity kumulatif HARI INI (ekuilibrium 13 tahun),
per trofi. **Yang BUKAN:** kurva minggu pertama — tidak ada snapshot
launch-week yang bisa diambil: Wayback Machine PUNYA snapshot halaman
Exophase PS3 per 2013-10-29 (6 minggu pasca-rilis), tapi fetch-nya ditolak
(403) dan tidak diulang. Jadi tidak ada λ / kurva minggu pertama dari GTA V;
yang dipakai dari GTA V adalah **level saturasi per tipe + pengelompokan
tipe**, dan level dipakai sebagai SHAPE prior (bukan level — level tetap dari
rate GTA VI sendiri).

Tabel saturasi GTA V (median rarity kumulatif PS3 per kelompok analog):

| kelompok prior GTA V | trofi GTA V (n) | median | min–max |
|---|---|---|---|
| story | 11 misi cerita | **56,00%** | 47,81–96,42 |
| robbery | 5 misi heist cerita | **57,86%** | 51,31–70,91 |
| side | 15 aktivitas/ekonomi/eksplorasi/stranger | **20,38%** | 11,21–65,22 |
| collectible | 3 set collectible | **10,72%** | 9,96–10,91 |
| grind | 70 gold medal 5,70%; 100% completion 7,03%; Close Shave 6,64% | **6,64%** | 5,70–7,03 |
| platinum | 1 | **3,90%** | — |

Pemetaan 55 trofi GTA VI → kelompok prior (dari `kind` di data guide):
story ids 2–13 (12); robbery 34, 39, 40, 41, 43 (5); collectible 30, 31 (2);
grind 14 (100% completion), 15 + 17 (aktivitas gold-medal — analog 70 gold
medal GTA V); platinum 1; side = sisanya 32 (activity, stranger, misc,
exploration, economy, combat — tidak ada analog tunggal yang bersih per kind,
jadi dipakai median gabungan 20,38%).

Headroom dari rate GTA VI saat ini (modal 1,1%) ke saturasi tipe, skala logit:
story +4,74; robbery +4,82; side +3,14; collectible +2,38; grind +1,86;
platinum +1,29. Sisi bawah (refresh yang teramati 638→3.868, faktor 0,165):
−1,81 logit pada rate 1,1%; −2,41 pada rate 50%.

## Fakta data yang membatasi semua opsi

1. **History practice datar total.** Semua read di p00/p01 identik: 638 pemain
   terlacak, semua 55 rate sama, dari 2026-11-18T23:50Z sampai checkpoint.
   Artinya: slope/trend/hazard **tidak teridentifikasi** dari history practice.
   Semua model runtuh ke carry-forward median di practice; yang membedakan
   mereka di practice hanyalah lebar spread (dan, untuk poisson, sedikit
   kenaikan median ~1% — lihat backtest).
2. **Refresh Exophase itu lompatan, bukan drift halus.** Di read 04:30Z pemain
   terlacak melonjak 638 → 3.868 dan semua rate (kecuali 100%) TURUN. Rate =
   earners / tracked: penyebut melonjak → rate turun meski earners naik.
   Model yang hanya mengekstrapolasi rate tanpa memikirkan penyebut akan
   salah arah tepat setelah refresh.
3. **Final beda rezim dengan practice.** Final (f00) punya ~48 jam history
   dengan read tiap 10 menit — di sana ada gerakan nyata untuk di-fit, dan
   horizon 4,5 hari (6.510 menit), bukan 90 menit. Model yang "menang" di
   practice belum tentu menang di final; practice hanya menguji format output
   dan apakah spread sanggup menampung satu refresh yang belum terlihat.
4. **Jawaban practice tidak tersedia lokal.** Kit hanya berisi input p00/p01,
   bukan `reference/answer.json`. Backtest di bawah memakai **proxy jawaban**:
   "frozen earners" — rate jawaban = rate terakhir × 638/3.868 (dilusi murni
   kalau jumlah earners tidak berubah). Itu **batas bawah** jawaban sebenarnya
   (pemain baru juga mendapat trofi, jadi rate sebenarnya lebih tinggi), bukan
   jawaban itu sendiri. Angka backtest di bawah adalah angka-vs-proxy, dan
   dilaporkan apa adanya sebagai proxy.

## Jawaban lima pertanyaan (dari data GTA V)

### 1. Keluarga model final: KOMBINASI — median poisson-hazard + prior GTA V
Median dari poisson-hazard (E(t) = E_inf·(1−exp(−λt)) pada jumlah earners +
forecast P(t) log-linear), diregularisasi prior GTA V (poin 2), dengan spread
bersama dari logit-linear (random walk + staleness + ketidakpastian tren).
Alasan dari data GTA V: (a) saturasi per tipe GTA V adalah objek yang
persis dimodelkan E_inf — prior dan model bicara tentang besaran yang sama;
(b) poisson satu-satunya keluarga yang memisahkan earners dari tracked
players, jadi arah efek refresh (poin 5) dimodelkan, bukan ditampung spread;
(c) bentuknya jenuh (asymptote E_inf) sehingga tidak overshoot seperti garis
lurus logit-linear pada horizon 4,5 hari. Logit-linear TIDAK dipakai untuk
median (ekstrapolasi garis lurus 4,5 hari overshoot trofi yang jenuh cepat);
prior-shape TIDAK dipakai sebagai keluarga (faktor bentuk butuh kurva minggu
pertama yang tidak ada — memakai rarity kumulatif sebagai level adalah
kesalahan objek, lihat Opsi 3 lama).

### 2. Bentuk hazard + regularisasi λ per tier: eksponensial tunggal + cap E_inf + penalti log-λ
Bentuk hazard: **eksponensial tunggal** E(t) = E_inf·(1−exp(−λt)), seperti
yang sudah terimplementasi (grid λ 10^−4–10^2/jam). Dua-fase/power-law tidak
dipakai: tidak ada data launch-week untuk membedakan bentuk, dan satu
parameter λ per trofi sudah cukup dengan regularisasi.
Regularisasi per kelompok (dua lapis):
- **Cap keras E_inf** (lapis utama, menghilangkan sudut degenerat):
  E_inf ≤ max(E_last, 1,5 × S_kelompok × P_target). S dari tabel GTA V di
  atas. Cap ini membunuh mode kegagalan "λ kecil + E_inf raksasa → forecast
  naik ~25% palsu" — tanpa menyentuh level yang sudah teramati (E_last selalu
  lolos). Sebagai cap rate: story 84%, robbery 87%, side 31%, collectible
  16%, grind 10%, platinum 5,9%.
- **Penalti log-λ lemah** (lapis kedua, pemutus seri): skor pemilihan =
  SSE/(n·mean_e²) + 0,001·(log10 λ − log10 λ0)², dengan λ0 per kelompok:
  story 0,0417; robbery 0,0208; side 0,0139; collectible 0,0060; grind
  0,0030; platinum 0,0014 per jam (τ = 1/λ0 = 24 jam / 48 jam / 72 jam /
  7 hari / 14 hari / 30 hari — waktu menuju 63% saturasi).
  **λ0 adalah ASUMSI yang dinyatakan eksplisit, bukan data GTA V** (tidak ada
  kurva launch-week untuk mengestimasinya; lihat flag di bawah). Bobotnya
  sengaja lemah: hanya memutus seri pada history datar (di sana semua λ
  jenuh memberi SSE=0) dan dikalahkan data begitu ada gerakan nyata.
  Verifikasi: pada history sintetis yang naik (earners 2x tiap 12 jam),
  median story 33,76% (cap 84%, tidak mengikat) dan median grind 9,14%
  (cap 10%, MENGIKAT — persis fungsi prior).

### 3. Prior-shape: sumber + pemakaian — Exophase PS3, SHAPE ONLY
Sumber: **Exophase GTA V PS3** (tabel di atas), dipakai sebagai **shape only**:
level saturasi per tipe mengatur cap + penalti λ di dalam model poisson,
sementara LEVEL forecast tetap dari rate GTA VI sendiri. TIDAK dipakai sebagai
level (tidak ada faktor pengali rate dari rarity kumulatif — itu menarik
forecast ke ekuilibrium 13 tahun, objek yang salah untuk horizon 7 hari).
Tidak ada kurva minggu pertama yang di-bake (Wayback 2013-10-29 tidak bisa
diambil). Model `prior-shape` di kode tetap carry-forward (faktor 1,0) dan
tidak dipakai sebagai keluarga final.

### 4. Kalibrasi spread: DIPERTAHANKAN — STALE_SIGMA=1,5 / STALE_TAU=8 jam
Dari data GTA V + refresh teramati: sisi bawah refresh −1,81 logit (rate
1,1%) sampai −2,41 (rate 50%) muat di dalam kuantil 5% pada σ≈1,0–1,5; sisi
atas headroom ke saturasi +1,29 (platinum) sampai +4,82 (robbery) logit
menuntut band yang lebar ke atas — σ di bawah ~1,0 membiarkan upside trofi
yang tumbuh tidak tertutup. Sensitivitas vs proxy: σ 1,02 → 1,0742; σ 1,49 →
0,9710; σ 2,02 → 0,9367 — loss vs proxy terus turun, TAPI itu artefak proxy
(proxy = batas bawah ekstrem; jawaban sebenarnya lebih tinggi, jadi σ
optimal sebenarnya lebih kecil dari σ-optimal-vs-proxy). σ≈1,02 (provisional)
berada di zona yang benar; melebarkan ke σ≈2 berarti over-wide di final.
Karena itu 1,5/8 jam dipertahankan. (Catatan flag: suku staleness menyempit
untuk trofi yang lama tidak berubah — lihat flag di bawah.)

### 5. Median BOLEH turun mengantisipasi refresh — lewat P(t), bukan offset
**Ya, median digeser turun — tapi lewat mekanisme eksplisit model poisson:**
rate forecast = 100·E(t_target)/P(t_target), dengan P(t) di-fit log-linear
pada players_tracked. Ketika penyebut tumbuh, median turun sebesar itu
dengan sendirinya; tidak ada offset ad-hoc. Bukti arah dari data GTA VI
sendiri (bukan GTA V — GTA V tidak punya event refresh yang teramati):
refresh 638→3.868 menurunkan semua rate dengan faktor 0,165. Besaran
pergeseran sepenuhnya data-driven dari history 48 jam final: kalau P(t)
datar di final, tidak ada pergeseran (benar — tidak ada refresh, tidak ada
dilusi). **Batas pengaman (temuan diagnostik):** fit log-linear P(t) pada
lompatan refresh awal memberi slope ~0,04/jam yang diekstrapolasi jadi
~100x pemain di hari ke-7 dan meruntuhkan semua median ke ~0 (terukur di
diagnostik sintetis: median 0,086% vs last rate 2,93%). Karena itu slope
P(t) di-cap **P_SLOPE_CAP = 0,01/jam** (mengizinkan 3,3x pertumbuhan
penyebut selama horizon 5 hari — lonjakan launch-week yang masuk akal —
sambil membatasi dilusi). Setelah cap: median sintetis 2,29% vs last rate
2,93% — dilusi termodelkan, tidak runtuh. **Cap ini judgment call** (tidak
ada kurva P(t) GTA V untuk mengkalibrasinya) — di-flag di bawah.

## Hasil backtest FINAL (vs PROXY jawaban "frozen earners", p00+p01 digabung)

Mean pinball loss (skala logit, 550 item: 2 case × 55 trofi × 5 kuantil):

| model | loss vs proxy | keterangan |
|---|---|---|
| **poisson-hazard + prior GTA V (FINAL)** | **1,0840** (p00 1,0633; p01 1,1046) | median ≈ carry-forward ×1,01 di practice; typical miss 0,736 pp; coverage 90% vs proxy = 0 |
| logit-linear provisional | 1,0727 (p00 1,0510; p01 1,0944) | median = carry-forward |
| poisson-hazard tanpa prior (lama) | 1,0744 | median ≈ carry-forward |
| prior-shape (faktor 1.0) | 1,0727 | median = carry-forward |
| house baseline (starter, sigma 0,125) | 1,6877 | coverage 90% = 0 |
| carry-forward titik | 1,7767 | = mean \|logit error\| vs proxy |

**Jujur dibaca:** model FINAL 0,011 LEBIH BURUK dari provisional vs proxy.
Itu bukan regresi nyata: di history datar semua median ≈ carry-forward, dan
selisih 0,011 berasal dari median poisson yang duduk ~1% DI ATAS
carry-forward (kurva hazard masih sedikit naik pada horizon 90 menit +
satu read t=0 yang tak ter-fit). Terhadap jawaban SEBENARNYA — yang berada
DI ATAS proxy (pemain baru juga mendapat trofi) — median yang sedikit lebih
tinggi itu netral sampai membantu, bukan merugikan. Practice TIDAK BISA
membedakan ketiga keluarga (median identik); nilai prior baru terlihat di
final, saat history bergerak. Jangan overclaim dari angka ini.

## Verifikasi (2026-11-19, setelah kunci model)

- `python3 eval-forecast.py`: format OK (55 trofi, id 1..55, kuantil terurut,
  tanpa NaN), deterministik byte-identik, **0,10 dtk/case** (<1 dtk).
  Stress test history bergerak: poisson merespons (id4 median 0,83 vs last
  0,79; id6 median 2,56 vs last 2,11) — prior tidak membekukan model.
- Diagnostik `../diag-prior.py`: cap E_inf MENGIKAT pada grind yang naik
  (median 9,14% vs cap 10%) dan tidak mengikat pada story (median 33,76%
  vs cap 84%); dilusi P(t) termodelkan dan terbatas oleh cap slope
  (median 2,29% vs last rate 2,93% pada skenario P×6, dari 0,086% sebelum
  cap — runtuh — diperbaiki).
- Rerun backtest setelah cap slope P(t): angka tidak berubah (P datar di
  practice, cap tidak mengikat).

## Yang di-flag untuk parent (tidak diputuskan diam-diam)

1. **λ0 per kelompok adalah asumsi** (τ 24 jam–30 hari dari sifat konten),
   bukan estimasi dari data GTA V — tidak ada kurva launch-week yang bisa
   diambil (Wayback 2013-10-29 ada tapi fetch ditolak 403). Bobot penalti
   sengaja lemah (0,001) sehingga asumsi ini hanya memutus seri; kalau
   history final menunjukkan λ yang jauh berbeda, data menang.
2. **P_SLOPE_CAP = 0,01/jam adalah judgment call** (bukan dari data GTA V).
   Kalau pertumbuhan tracked players GTA VI hari ke-2→7 ternyata >3,3x,
   median akan kurang turun (under-dilusi) — tapi spread menampungnya, dan
   arahnya konservatif (lebih baik daripada runtuh ke nol).
3. **Suku staleness menyempit untuk trofi yang lama tidak berubah**
   (stale = 1,5·exp(−gap/8 jam)): trofi langka yang dormant 48 jam di final
   mendapat σ≈1,06 (hanya random walk), sementara upside headroom-nya ke
   saturasi tipe bisa +3 logit — kuantil 95% mungkin tidak mencapai
   saturasi. Tidak diubah (di luar pertanyaan 4, dan practice tidak bisa
   mengukurnya); kalau parent mau, perbaikan yang benar adalah suku spread
   berbasis headroom-ke-saturasi, bukan mengubah STALE_SIGMA.
4. **Kelompok "side" adalah gabungan** (15 trofi GTA V, median 20,38%,
   rentang lebar 11–65%): activity/stranger/misc/exploration/economy/combat
   tidak punya analog tunggal yang bersih. Kalau satu kind ternyata
   sistematis berbeda (mis. economy jauh lebih lambat), prior-nya terlalu
   kasar — tapi cap 1,5x memberi ruang, dan spread menampung sisanya.
5. **Backtest practice tidak memvalidasi prior** (semua median runtuh ke
   carry-forward). Keyakinan pada prior ini berasal dari mekanisme + data
   GTA V, bukan dari angka backtest. Jangan laporkan 1,0840 sebagai "skor".

## Opsi yang tidak dipilih (alasan singkat)

- **Logit-linear sebagai median:** ekstrapolasi garis lurus 4,5 hari
  overshoot trofi yang jenuh cepat; tidak memodelkan penyebut — refresh
  hanya ditampung spread, bukan median.
- **Prior-shape sebagai keluarga (faktor pengali rate):** butuh kurva minggu
  pertama; memakai rarity kumulatif 13 tahun sebagai level adalah kesalahan
  objek. Dipakai sebagai shape di dalam poisson saja.
- **Poisson tanpa prior:** mode kegagalan λ kecil + E_inf raksasa sudah
  terbukti (loss 1,3032 sebelum grid diperbaiki); cap + penalti adalah
  pengaman yang sama dengan harga murah.
- **Spread lebih lebar (σ≈2):** menang vs proxy hanya karena proxy adalah
  batas bawah ekstrem; jawaban sebenarnya lebih tinggi → over-wide di final.
