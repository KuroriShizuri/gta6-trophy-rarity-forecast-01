# Keputusan model — gta6-trophy-rarity-forecast-01

**Status: FINAL v2 (2026-11-19).** Revisi dari FINAL v1 mengikuti review
Claude Sonnet 5.5 pasca-submit (19:26 WIB, sesi Kiro 03864765, 0,49 kredit;
Sonnet menilai dari dokumen ini saja — main.py tidak bisa dilampirkan).
Keputusan user (19:29 WIB): *"Bisa direvisi bisa bertahan, ikutin panduan
sonnet pokok nya."* Revisi ini menggantikan submission 01a1f3bd (claim
AF-BJGHA) — submission lama tetap jadi fallback sampai revisi ini terverifikasi.

**Yang berubah di v2 (per poin Sonnet):**
1. **Data launch-curve dicoba lagi, gagal, didokumentasikan.** Wayback CDX API
   403 (jangan di-retry — runtime melarang), PSNProfiles/TrueTrophies tidak
   punya time series. λ0 tetap asumsi (lihat §1).
2. **P(t) jadi scenario mixture** (bukan log-linear + P_SLOPE_CAP).
   P_SLOPE_CAP DIHAPUS. Penyebut dimodelkan sebagai proses EVENT: probabilitas
   refresh + ukuran refresh (aditif, 3.230 pemain — data dari lompatan
   638→3.868). Kuantil bawah dari massa refresh, kuantil atas dari
   pertumbuhan earners. Median turun kalau refresh mungkin, tidak turun kalau
   tidak ada refresh yang diharapkan.
3. **Arah staleness diperbaiki.** Suku lama (1,5·exp(−gap/8 jam)) yang
   MENYEMPITKAN band untuk trofi dormant DIHAPUS. Diganti suku overdue yang
   MELEBAR saat refresh makin overdue + suku headroom-ke-saturasi. Spread
   jadi campuran BIMODAL (refresh downside + earner upside), bukan band
   simetris.
4. **Dipertahankan:** median poisson-hazard untuk earners (dekomposisi
   rate = earners/tracked benar), cap E_inf + penalti log-λ lemah, prior
   bentuk GTA V untuk cap.
5. **Simulator refresh dibangun** (`../sim-refresh.py`) dan dijalankan
   SEBELUM finalisasi — hasilnya di bawah.

Implementasi di `main.py`: `MODEL = "poisson-hazard"` + prior GTA V
(`TROPHY_PRIOR`, `PRIOR_SATURATION`, `LAMBDA_PRIOR`, `E_INF_CAP_MULT`,
`LAMBDA_PENALTY_W`) + mixture penyebut (`REFRESH_JUMP_RATIO`,
`PRIOR_REFRESH_RATE/ALPHA/BETA_H`, `PRIOR_REFRESH_SIZE`, `P_REFRESH_MAX`,
`SIZE_SIGMA`, `OVERDUE_SIGMA/TAU_HOURS`, `HEADROOM_FRAC`, `SIGMA_RW/CAP/FLOOR`).
Konstanta v1 yang DIHAPUS: `P_SLOPE_CAP`, `STALE_SIGMA`, `STALE_TAU`.

## Keputusan final v2 (ringkasan)

**Kombinasi: median poisson-hazard + prior GTA V + scenario mixture untuk P(t)
+ spread bimodal.** Median dari model poisson-hazard (satu-satunya keluarga
yang memodelkan pembilang dan penyebut secara terpisah), diregularisasi prior
GTA V (cap E_inf + penalti log-λ). Penyebut TIDAK lagi di-fit log-linear:
ia adalah campuran dua skenario — (A) tidak ada refresh (P datar, earners
tumbuh) dan (B) refresh (P + N×3.230 pemain, earners tumbuh) — dengan bobot
p = probabilitas ≥1 refresh sebelum target. Kuantil dihitung dari CDF
campuran (bisection), jadi kuantil bawah ditarik massa refresh dan kuantil
atas ditarik pertumbuhan earners. Spread adalah campuran bimodal, bukan band
simetris yang dilebarkan dua arah.

## 1. Data launch-curve: dicoba lagi, tidak ada yang bisa diambil

Sonnet meminta data yang bisa membedakan model SEBELUM memilih model — sumber
satu-satunya kurva launch adalah snapshot Wayback 2013-10-29 halaman Exophase
GTA V PS3. Yang dicoba (bounded effort, 2026-11-19):
- **Wayback CDX API** (`web.archive.org/cdx/search/cdx?...`): HTTP **403
  (access denied)**. Runtime menginstruksikan TIDAK me-retry (termasuk dengan
  argumen berbeda atau lewat tool/endpoint lain) dan tidak melewati penolakan
  — jadi bentuk URL `id_` suffix TIDAK dicoba (penolakan yang sama).
- **PSNProfiles / TrueTrophies** (pencarian web): keduanya hanya menerbitkan
  rarity SAAT INI, bukan time series. PocketPSN menampilkan persentase saat
  ini + tanggal first-earned, bukan kurva populasi.
- **Game terbaru dengan history launch-week:** tidak ada sumber yang
  menerbitkan kurva rarity minggu pertama yang bisa diambil.

**Kesimpulan: TIDAK ADA data launch-curve yang bisa diambil.** λ0 tetap
ASUMSI (τ dari sifat konten, penalti lemah 0,001 hanya pemutus seri) — tidak
teridentifikasi dari data GTA V saja (rarity kumulatif hari ini adalah satu
titik = E_inf; kurva launch butuh ≥2 titik). Ini dinyatakan eksplisit di
flag di bawah, bukan disembunyikan.

## Sumber data GTA V (diambil 2026-11-19) — tidak berubah dari v1

**Sumber: Exophase GTA V PS3** — `https://www.exophase.com/game/grand-theft-auto-v-ps3/trophies/`
(daftar launch-era, 50 trofi dasar). Halaman PSN/PS5 juga diambil sebagai
pembanding (78 trofi; populasi PS5 condong ke Online, jadi PS3 yang dipakai).
**Yang data ini ADALAH:** rarity kumulatif HARI INI (ekuilibrium 13 tahun),
per trofi. **Yang BUKAN:** kurva minggu pertama (lihat §1 — tidak bisa
diambil). Yang dipakai dari GTA V adalah **level saturasi per tipe +
pengelompokan tipe**, dan level dipakai sebagai SHAPE prior (bukan level —
level tetap dari rate GTA VI sendiri).

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

## Fakta data yang membatasi semua opsi — tidak berubah dari v1

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

## 2. P(t) sebagai scenario mixture (perubahan utama v2)

**Log-linear P(t) + P_SLOPE_CAP DIHAPUS.** Penyebut adalah proses EVENT:

- **Deteksi refresh (adaptif).** `detect_refreshes()` menemukan lompatan P:
  sebuah langkah dihitung refresh hanya jika rasionya ≥1,5x DAN log-increment-
  nya ≥5× median |log-step| history. Syarat kedua ini penting: pertumbuhan
  halus dari basis kecil (mis. 638→1.038, ×1,627) BUKAN refresh — detektor
  lama (rasio saja) salah menandainya refresh dan over-dilusi 95%.
  Lompatan 6,06x yang teramati pada history datar lolos kedua syarat.
- **Probabilitas refresh p.** Refresh adalah proses Poisson dengan rate λ.
  Rate adalah **posterior mean Gamma** (shape=ALPHA, rate=BETA_H):
  λ = (ALPHA + n_lompatan) / (BETA_H + span_jam). Prior: ALPHA=1, BETA_H=24
  jam → mean prior 1/hari (STATED ASSUMPTION — satu refresh per hari adalah
  kadensi launch-night; tidak ada kadensi launch-week yang bisa diestimasi).
  History datar MENURUNKAN rate (bukti melawan refresh): 0 lompatan dalam
  48 jam → λ = 1/72 per jam (0,33/hari), bukan 1/24. p = 1−exp(−λ·horizon),
  di-cap 0,95.
- **Ukuran refresh: ADITIF, bukan rasio.** Refresh MENAMBAH pemain yang baru
  terlacak. Ukuran = median penambahan yang teramati (per lompatan), kalau
  tidak ada lompatan = **3.230 pemain** (3.868−638 — DATA dari lompatan
  launch-night, bukan asumsi). Bentuk aditif dipilih karena rasio
  meng-compound secara absurd (refresh kedua ×6 akan membawa P ke ~23 ribu);
  jumlah pemain yang baru terlacak per refresh kira-kira seorde tiap kali.
  Rasio skenario R = (P_last + N·ukuran) / P_last — menyusut saat P tumbuh,
  jadi refresh kedua mendilusi jauh lebih sedikit dari catch-up launch-night.
- **Jumlah refresh yang diharapkan.** Skenario refresh TIDAK hanya satu
  refresh: ia memakai E[N | N≥1] = μ/(1−exp(−μ)), μ = λ·horizon — jumlah
  refresh yang diharapkan SEBELUM target kalau setidaknya satu terjadi.
  Ini ditemukan simulator: mixture satu-refresh UNDER-dilusi dan kalah dari
  log-linear lama yang di-cap, bahkan saat proses refresh cocok dengan prior.
  Untuk horizon 4,5 hari dengan λ≈1/hari, E[N|N≥1]≈4,5.
- **Kuantil dari CDF campuran (bimodal).** Kuantil q adalah solusi bisection
  dari (1−p)·Φ((q−μ_A)/σ_A) + p·Φ((q−μ_B)/σ_B) = level, dengan μ_A =
  logit rate tanpa refresh, μ_B = logit rate dengan refresh. Kuantil bawah
  ditarik massa refresh, kuantil atas ditarik pertumbuhan earners.
  **Angka practice:** p00/p01: lompatan=0, p=0,055, ukuran=3.230,
  E[N|N≥1]=1,03, rasio=6,21, gap=3,15/3,65 jam. **Angka final-like**
  (history 48 jam dengan lompatan 6x): lompatan=1, p=0,95, ukuran=3.230,
  E[N|N≥1]=3,17, rasio=3,65, gap=43,5 jam.
- **Median bergeser turun kalau refresh mungkin, tidak kalau tidak.**
  Pada practice (p≈0,05) median ≈ carry-forward (id4 median 1,03% vs last
  1,1%). Pada final-like (p=0,95) median terdilusi ~3,4x (id4 story median
  0,053% vs last 0,18% pada earners datar — angka ini naik kalau earners
  tumbuh, seperti di final sungguhan).

## 3. Spread: arah staleness diperbaiki, bimodal (perubahan v2)

**Suku staleness lama DIHAPUS** (STALE_SIGMA/STALE_TAU). Suku itu menyempitkan
band saat trofi dormant — arah yang salah untuk risiko refresh (rentang datar
yang panjang berarti refresh PALING overdue). Diganti dua suku:

- **Suku overdue (tumbuh saat dormant):**
  σ_overdue = OVERDUE_SIGMA·(1−exp(−gap/OVERDUE_TAU_HOURS)), OVERDUE_SIGMA=1,0,
  OVERDUE_TAU_HOURS=24 jam. gap = jam sejak refresh terakhir (atau sejak
  read pertama kalau belum ada refresh). Dormant = spread LEBIH besar untuk
  risiko refresh, bukan lebih kecil.
- **Suku headroom-ke-saturasi (skenario A):** σ_A menampung random walk
  (SIGMA_RW=0,5) + sebagian headroom logit ke cap rate E_inf
  (HEADROOM_FRAC=0,25 × headroom_logit). Trofi yang jauh dari saturasi
  mendapat band atas yang lebih lebar; trofi yang dekat saturasi tidak.
- **Skenario B (refresh):** σ_B = sqrt(σ_A² + SIZE_SIGMA² + σ_overdue²),
  SIZE_SIGMA=0,5 (ketidakpastian ukuran refresh — satu ukuran yang teramati
  adalah satu draw). σ di-floor 0,05, di-cap 2,5 (SIGMA_FLOOR/CAP).
- **Hasilnya bimodal, bukan simetris:** kuantil bawah mencapai drop refresh
  (−1,8 sampai −2,4 logit pada 5%), kuantil atas mencapai headroom earners
  (+1,3 sampai +4,8 logit). Band simetris yang dilebarkan dua arah akan
  membuang massa di tengah yang tidak dibutuhkan.

## 4. Yang dipertahankan (Sonnet: ini baik-baik saja)

- **Median poisson-hazard untuk earners.** E(t) = E_inf·(1−exp(−λt)) pada
  jumlah earners, grid λ 10^−4–10^2/jam. Dekomposisi rate = earners/tracked
  benar; hazard satu-satunya keluarga yang memisahkan pembilang dari penyebut.
- **Cap E_inf + penalti log-λ lemah.** Cap keras E_inf ≤
  max(E_last, 1,5 × S_kelompok × P_last) — v2 memakai P_last (bentuk yang
  mengikat/konservatif), bukan P_target. Sebagai cap rate: story 84%,
  robbery 87%, side 31%, collectible 16%, grind 10%, platinum 5,9%.
  Penalti log-λ: SSE/(n·mean_e²) + 0,001·(log10 λ − log10 λ0)², λ0 per
  kelompok: story 0,0417; robbery 0,0208; side 0,0139; collectible 0,0060;
  grind 0,0030; platinum 0,0014 per jam (τ 24 jam / 48 jam / 72 jam /
  7 hari / 14 hari / 30 hari). **λ0 adalah ASUMSI yang dinyatakan eksplisit**
  (lihat §1 — tidak ada kurva launch-week untuk mengestimasinya).
- **Prior bentuk GTA V untuk cap.** Exophase PS3, shape only (lihat tabel
  di atas). Model `prior-shape` di kode tetap carry-forward (faktor 1,0)
  dan tidak dipakai sebagai keluarga final.

## 5. Simulator / held-out check untuk proses penyebut (baru di v2)

`../sim-refresh.py` (bukan bagian entry): mensimulasikan proses refresh yang
DIKETAHUI (seeded RNG) — earners mengikuti hazard dengan λ_true yang
dipilih supaya rate di t=4 jam cocok dengan last rate p00 sungguhan, dan
P(t) mengikuti proses Poisson refresh (tiap refresh MENAMBAH pemain,
A ~ LogNormal(median=A_med, σ=0,4)). History 48 jam (read tiap 30 menit),
target +108 jam (horizon final). Dibandingkan model BARU (v2) vs model LAMA
(v1, dari `git show f22a527:main.py`) pada kasus sintetis yang sama — mean
pinball loss skala logit vs rate target SEBENARNYA + coverage band 90% +
|logit median error|. 4 skenario × 6 seed. Ukuran refresh simulator BERBEDA
dari asumsi model di S2/S3 (anti-sirkularitas: model tidak diuji pada
asumsinya sendiri).

| skenario (proses refresh sebenarnya) | pinball baru (v2) | pinball lama (v1) | coverage90 baru/lama | \|logit med err\| baru/lama |
|---|---|---|---|---|
| S1 cocok: 1/hari ×3.230 | **0,4398** | 0,6557 | 1,00 / 0,95 | 0,411 / 0,760 |
| S2 lambat/kecil: 0,5/hari ×1.500 | **0,6259** | 0,7114 | 0,98 / 0,98 | 0,886 / 1,096 |
| S3 cepat/besar: 2/hari ×6.000 | 0,4584 | **0,4297** | 1,00 / 1,00 | 0,375 / 0,338 |
| S4 tidak ada (tanpa refresh) | 1,6162 | **0,5239** | 0,85 / 0,98 | 2,915 / 0,703 |

**Dibaca jujur:** v2 menang jelas di S1 dan S2 (skenario refresh yang
realistis — refresh memang terjadi di launch night), hampir seri di S3
(kalah tipis pinball 0,029 dan median log-error 0,037), dan **kalah telak di
S4**. S4 adalah dunia tanpa refresh sama sekali — yang BERTENTANGAN dengan
refresh launch-night yang teramati. Kelemahan v2: prior refresh (1/hari)
cukup kuat sehingga setelah 48 jam datar pun p atas horizon 4,5 hari masih
~0,78 dan median over-dilusi kalau tracker berhenti refresh total. Mitigasi:
(a) rate adalah posterior Gamma — history datar yang lebih panjang akan
menurunkan p lebih jauh; (b) history final sungguhan MENGANDUNG refresh
(lompatan 6x), jadi rate final diestimasi dari data, bukan hanya prior.
Kalau tracker berhenti total, v2 akan over-dilusi — itu risiko yang
dinyatakan, bukan disembunyikan. Dua perbaikan dari simulator yang sudah
masuk ke v2: detektor langkah adaptif (poin 2) dan E[N|N≥1] (poin 2) —
keduanya ditemukan karena v2 awal kalah dari v1 di S1/S2 pada run pertama
simulator. Simulator deterministik (seed integer tetap per skenario —
run pertama memakai `hash()` string yang diacak per proses Python, jadi
angka tidak reprodusibel; diperbaiki ke seed integer 1000×s_idx+seed dan
diverifikasi dua run identik).

## Hasil backtest FINAL v2 (vs PROXY jawaban "frozen earners", p00+p01)

Mean pinball loss (skala logit, 550 item: 2 case × 55 trofi × 5 kuantil):

| model | loss vs proxy | keterangan |
|---|---|---|
| **poisson-hazard + prior GTA V + mixture (FINAL v2)** | **0,9643** (p00 0,9642; p01 0,9643) | typical miss 0,679 pp; coverage 90% vs proxy = 0,818 |
| poisson-hazard v1 (FINAL v1) | 1,0840 | coverage 90% vs proxy = 0 |
| logit-linear / prior-shape (tidak dikirim) | 1,6471 | spread-nya kini sempit (suku staleness yang salah arah dihapus) — band-nya tidak lagi menampung refresh secara kebetulan |
| house baseline (starter, sigma 0,125) | 1,6877 | coverage 90% = 0 |
| carry-forward titik | 1,7767 | = mean \|logit error\| vs proxy |

**Jujur dibaca:** v2 lebih baik 0,12 dari v1 vs proxy — TAPI itu bukan bukti
modelnya lebih baik. Proxy adalah batas bawah jawaban sebenarnya, dan
backtest practice tidak bisa memvalidasi penanganan refresh (semua median ≈
carry-forward di history datar; yang berubah hanyalah kuantil bawah kini
mencapai rate yang terdilusi — coverage 0,818 vs 0). Nilai mixture baru
terlihat di final dan di simulator (§5). **Jangan laporkan 0,9643 sebagai
"skor". Jangan tune parameter apa pun ke proxy** — HEADROOM_FRAC=0,25
ditetapkan SEBELUM run eval, bukan di-tune ke angka ini.

## Verifikasi v2 (2026-11-19)

- `python3 eval-forecast.py`: format OK (55 trofi, id 1..55, kuantil terurut,
  tanpa NaN), deterministik byte-identik, **0,07–0,10 dtk/case** (<1 dtk).
  Mixture: p00/p01 lompatan=0, p≈0,055, ukuran=3.230, E[N|N≥1]=1,03,
  rasio=6,21. Stress test history bergerak: poisson merespons (id4 median
  0,83 vs last 0,79; id6 median 2,55 vs last 2,11) — prior tidak membekukan
  model; id4 yang trennya turun di sintetis tetap dekat last rate (bentuk
  hazard tidak bisa turun — batasan yang diketahui, lihat flag 6).
- Diagnostik `../diag-prior.py`: cap E_inf tidak mengikat pada story yang
  naik (median 5,28% vs cap 84%) maupun grind (median 1,38% vs cap 10%)
  pada run v2 — median lebih rendah dari v1 (33,76% / 9,14%) karena
  skenario refresh kini mendilusi ~3,2x yang diharapkan atas horizon 4,5
  hari. Skenario P×6: median 3,56% vs last rate 2,93% (DI ATAS — pertumbuhan
  earners yang berlipat mengalahkan dilusi pada sintetis itu; band lebar
  0,14–45,68% menampung kedua arah).
- `python3 sim-refresh.py`: lihat §5 (v2 menang S1/S2, seri S3, kalah S4).

## Parameter baru v2 (lengkap, dengan status data/asumsi)

| parameter | nilai | status |
|---|---|---|
| PRIOR_REFRESH_RATE (mean prior) | 1/24 per jam (1/hari) | ASUMSI (kadensi launch-night; tidak ada data kadensi) |
| PRIOR_REFRESH_ALPHA / BETA_H | 1,0 / 24 jam | ASUMSI (prior Gamma; mean 1/hari) |
| PRIOR_REFRESH_SIZE (aditif) | 3.230 pemain | DATA (3.868−638, lompatan teramati) |
| P_REFRESH_MAX | 0,95 | judgment (jangan pernah yakin penuh) |
| SIZE_SIGMA | 0,5 | judgment (satu ukuran teramati = satu draw) |
| OVERDUE_SIGMA / OVERDUE_TAU_HOURS | 1,0 / 24 jam | judgment (arah benar; besaran tidak terkalibrasi) |
| HEADROOM_FRAC | 0,25 | judgment (ditetapkan sebelum eval; tidak di-tune ke proxy) |
| SIGMA_RW / SIGMA_CAP / SIGMA_FLOOR | 0,5 / 2,5 / 0,05 | dipertahankan dari v1 |
| REFRESH_JUMP_RATIO | 1,5x + 5× median \|log-step\| | judgment (ambang deteksi; 6,06x teramati lolos jauh) |
| λ0 per kelompok + penalti 0,001 | lihat §4 | ASUMSI (tidak ada kurva launch-week) |
| E_INF_CAP_MULT | 1,5x saturasi GTA V | DATA (saturasi GTA V) + judgment (faktor 1,5x) |

## Yang di-flag untuk parent (tidak diputuskan diam-diam)

1. **λ0 per kelompok adalah asumsi** (τ 24 jam–30 hari dari sifat konten),
   bukan estimasi dari data GTA V — tidak ada kurva launch-week yang bisa
   diambil (Wayback CDX 403, jangan di-retry; PSNProfiles/TrueTrophies tidak
   punya time series). Bobot penalti sengaja lemah (0,001) sehingga asumsi
   ini hanya memutus seri; kalau history final menunjukkan λ yang jauh
   berbeda, data menang.
2. **Bobot scenario mixture adalah asumsi kalau tidak tervalidasi data.**
   Rate refresh prior (1/hari), ukuran refresh (3.230 pemain — data, tapi
   satu draw), HEADROOM_FRAC=0,25, SIZE_SIGMA=0,5, OVERDUE_SIGMA=1,0:
   semuanya judgment call yang tidak terkalibrasi dari data (tidak ada
   kadensi refresh launch-week yang bisa diambil). Simulator (§5) memvalidasi
   ARAH-nya (mixture menang saat refresh terjadi), bukan BESARAN-nya.
3. **S4: kalau tracker berhenti refresh total, v2 over-dilusi.** Median
   akan duduk ~3x terlalu rendah (pinball simulator 1,62 vs 0,52 v1).
   Ini harga dari prior refresh yang kuat. Diterima karena refresh
   launch-night membuktikan tracker refresh; history final mengandung
   refresh, jadi rate final dari data.
4. **Suku staleness lama dihapus — model non-shipped makin buruk vs proxy.**
   Logit-linear/prior-shape kini 1,6471 vs proxy (dari 1,0727 di v1) karena
   band-nya tidak lagi menampung refresh secara kebetulan. Itu perilaku yang
   BENAR (mereka bukan model yang dikirim), bukan regresi model final.
5. **Kelompok "side" adalah gabungan** (15 trofi GTA V, median 20,38%,
   rentang lebar 11–65%): activity/stranger/misc/exploration/economy/combat
   tidak punya analog tunggal yang bersih. Kalau satu kind ternyata
   sistematis berbeda (mis. economy jauh lebih lambat), prior-nya terlalu
   kasar — tapi cap 1,5x memberi ruang, dan spread menampung sisanya.
6. **Bentuk hazard tidak bisa memodelkan earners yang TURUN.** Pada history
   sintetis yang trennya turun, median duduk dekat last rate (bukan di
   bawahnya) karena E(t) monoton naik. Di final sungguhan earners tidak
   turun (pemain hanya bertambah), jadi batasan ini tidak mengikat — tapi
   dicatat.
7. **Backtest practice tidak memvalidasi prior maupun mixture** (semua median
   runtuh ke carry-forward). Keyakinan berasal dari mekanisme + data GTA V +
   simulator (§5), bukan dari angka backtest. Jangan laporkan 0,9643 sebagai
   "skor".

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
- **P(t) log-linear + P_SLOPE_CAP (v1):** DIHAPUS di v2. Cap 0,01/jam adalah
  konstanta tak terkalibrasi yang memindahkan median ~1 logit lewat satu
  angka; keluarga yang salah untuk proses lompatan (Sonnet poin 1).
  Simulator mengonfirmasi mixture lebih baik saat refresh terjadi (S1/S2).
- **Ukuran refresh sebagai rasio (×6,06):** ditolak — rasio meng-compound
  secara absurd pada refresh kedua. Bentuk aditif (3.230 pemain) dipakai.
- **Mixture satu-refresh (v2 awal):** ditolak oleh simulator — under-dilusi
  dan kalah dari v1 di S1/S2. Diganti E[N|N≥1] (jumlah refresh yang
  diharapkan kalau setidaknya satu terjadi).
