# Keputusan model — gta6-trophy-rarity-forecast-01

**Status: BELUM FINAL. Dokumen ini diajukan ke user untuk diputuskan.**
Model provisional yang terimplementasi di `main.py` adalah `logit-linear`
(dengan spread yang peka terhadap staleness halaman). Itu placeholder yang
masuk akal, bukan keputusan final. Mengganti model final = mengubah satu
konstanta `MODEL` di `main.py` (plus parameternya) — tidak perlu restruktur
kode.

## Fakta data yang membatasi semua opsi

1. **History practice datar total.** Semua read di p00/p01 identik: 638 pemain
   terlacak, semua 55 rate sama, dari 2026-11-18T23:50Z sampai checkpoint.
   Artinya: slope/trend/hazard **tidak teridentifikasi** dari history practice.
   Semua model runtuh ke carry-forward median di practice; yang membedakan
   mereka di practice hanyalah lebar spread.
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

## Opsi 1 — Logit-linear (PROVISIONAL, yang terimplementasi sebagai default)

OLS logit(rate) terhadap jam, di-fit pada 24 jam terakhir, slope di-shrink
(`n/(n+4)`, cap |slope| ≤ 0,10 logit/jam, cap pergeseran median ≤ 3 logit),
ditambah spread gabungan (random walk 0,5·√hari + staleness + ketidakpastian
slope). Ini kerabat dekat house curve baseline (beda: jendela fit, shrinkage,
dan suku staleness).

- **Pro:** sederhana, robust, transparan; di final ia benar-benar memakai
  gerakan 48 jam history; tidak butuh data luar (jalan offline); slope yang
  di-shrink tidak meledak pada ekstrapolasi 4,5 hari.
- **Kontra:** logit-linear adalah aproksimasi lokal — kurva adopsi nyata
  berbentuk S / meluruh, bukan garis lurus di logit; pada horizon 4,5 hari
  garis lurus bisa overshoot (terutama trofi cerita awal yang jenuh cepat);
  tidak memodelkan penyebut (tracked players) secara eksplisit — refresh
  yang menurunkan semua rate sekaligus hanya ditampung lewat spread, bukan
  lewat median.
- **Yang dibutuhkan untuk memutuskan:** apakah user nyaman dengan ekstrapolasi
  garis lurus 4,5 hari, atau minta bentuk kurva yang melengkung (mis. slope
  yang meluruh seiring waktu).

## Opsi 2 — Poisson / decaying-hazard (terimplementasi, bisa dipilih)

Per trofi, jumlah earners E(t) dimodelkan dari hazard yang meluruh:
E(t) = E_inf · (1 − exp(−λ·t)), t jam sejak rilis; λ dipilih lewat grid
least-squares per trofi, E_inf skala linier; pemain terlacak P(t) di-fit
log-linear; rate forecast = 100·E/P. Ini satu-satunya opsi yang memodelkan
pembilang dan penyebut secara terpisah — persis mekanisme refresh (P melonjak
→ rate turun).

- **Pro:** intuisi paling benar soal mekanisme (hazard tertinggi di awal lalu
  meluruh; trofi cerita vs collectible/grind punya λ berbeda); memisahkan
  earners dari tracked players, jadi arah efek refresh dimodelkan, bukan
  cuma ditampung spread; bentuknya jenuh (asymptote E_inf), tidak overshoot
  seperti garis lurus.
- **Kontra:** λ **tidak teridentifikasi** dari data launch-night yang datar
  (fit mengembalikan boundary datar); E_inf dan λ hanya bisa diestimasi
  setelah history final menunjukkan gerakan — dan dengan 48 jam data pun,
  ekstrapolasi ke hari ke-7 tetap sensitif terhadap λ (hazard tail yang
  panjang vs pendek mengubah E_inf jauh); asumsi satu λ per trofi +
  satu bentuk (1 − exp(−λt)) mungkin terlalu kaku untuk trofi grind
  (hazard naik dulu saat pemain mencapai kontennya, baru meluruh); butuh
  keputusan soal prior/regularisasi λ per tier (tanpa itu, trofi yang
  datar di history bisa dapat λ liar).
- **Yang dibutuhkan untuk memutuskan:** bentuk hazard (eksponensial murni
  vs power-law vs dua-fase), dan bagaimana me-regularisasi λ/E_inf saat
  data masih sedikit — mis. prior per tier (cerita vs collectible vs
  grind vs platinum) atau pooling antar trofi se-tier.

## Opsi 3 — Prior bentuk GTA IV/V (terimplementasi sebagai kerangka, faktor 1.0)

Median = rate terakhir × faktor bentuk per tier, dari kurva minggu pertama
GTA IV/V. **Saat ini semua faktor = 1.0 (carry-forward)**, karena tidak ada
kurva minggu pertama yang kredibel untuk di-bake: yang tersedia di tracker
(Exophase/PSNProfiles/TrueTrophies) adalah rarity KUMULATIF hari ini
(hasil bertahun-tahun) — objek yang salah untuk forecast 4,5 hari ke depan
(akan menarik forecast ke rate ekuilibrium jangka panjang). GTA V lebih
dekat sebagai analog daripada GTA IV (50 trofi dasar, open-world, launch
besar), tapi masalah datanya sama.

- **Pro:** kalau ada sumber kurva minggu pertama yang kredibel, ini satu-
  satunya opsi yang membawa informasi dari luar history 48 jam GTA VI
  sendiri; bentuk kurva launch game Rockstar sebelumnya mungkin mirip
  (ledakan story trophy di hari 1–2, grind/collectible merangkak).
- **Kontra:** tanpa sumber kurva minggu pertama, opsi ini kosong (faktor
  1.0); memakai rarity kumulatif jangka panjang sebagai prior adalah
  kesalahan objek; perbedaan antar game (durasi cerita, desain trofi,
  ukuran install base, pola sync Exophase) bisa membuat prior menyesatkan;
  prior yang salah + spread sempit = pinball loss besar di semua trofi.
- **Yang dibutuhkan untuk memutuskan:** sumber kurva minggu pertama yang
  kredibel dan bisa di-bake ke repo (offline!), atau keputusan eksplisit
  untuk TIDAK memakai prior historis; kalau dipakai, apakah sebagai bentuk
  (shape prior — level tetap dari data GTA VI) atau sebagai level.

## Rekomendasi awal (provisional, bukan final)

**Default provisional: logit-linear + spread staleness.** Alasan: di antara
ketiga opsi, ini satu-satunya yang (a) benar-benar bekerja dengan data yang
ada di tangan (history GTA VI sendiri, tanpa data luar yang tidak ada),
(b) perilakunya terdefinisi baik di kedua rezim (datar → carry-forward;
bergerak → trend yang di-shrink), dan (c) paling mudah diaudit. Poisson-
hazard adalah kandidat final yang paling menarik secara mekanisme, tapi
finalisasinya butuh keputusan regularisasi λ yang belum dibuat. Prior
GTA IV/V tidak bisa direkomendasikan sebagai final sampai ada sumber kurva
minggu pertama yang kredibel.

## Yang harus diputuskan user (sebelum kunci model final)

1. Keluarga model final: logit-linear / poisson-hazard / prior-shape
   (atau kombinasi — mis. median poisson-hazard + spread logit-linear)?
2. Kalau poisson-hazard: bentuk hazard + regularisasi λ per tier?
3. Kalau prior-shape: sumber kurva minggu pertama apa, dan dipakai sebagai
   bentuk saja atau sebagai level?
4. Kalibrasi spread: parameter staleness provisional (STALE_SIGMA=1,5,
   STALE_TAU=8 jam) — lihat sensitivitas di bawah. Apakah spread boleh
   selebar itu, atau user minta lebih sempit/lebih lebar?
5. Apakah median boleh digeser turun mengantisipasi refresh (arah dilusi
   diketahui dari rules: rate turun saat penyebut melonjak), atau median
   harus tetap netral (carry-forward/trend) dan refresh hanya ditampung
   spread? Ini keputusan arah — bukan netralitas — jadi perlu persetujuan
   user.

## Hasil backtest (vs PROXY jawaban "frozen earners", p00+p01 digabung)

Proxy: rate jawaban = rate terakhir × 638/3.868 ≈ ×0,165 (semua trofi
kecuali 100%). Batas bawah jawaban sebenarnya. Mean pinball loss (skala
logit, 550 item: 2 case × 55 trofi × 5 kuantil):

| model | loss vs proxy | keterangan |
|---|---|---|
| house baseline (starter, sigma 0,125) | 1,6877 | dihitung ulang vs proxy (rules mengutip ~1,51 vs jawaban SEBENARNYA — angka proxy tidak bisa dibandingkan langsung dengan itu); coverage 90% = 0 |
| logit-linear provisional (sigma ~1,02) | **1,0727** (p00 1,0510; p01 1,0944) | median = carry-forward (history datar); typical miss 0,726 pp; coverage 90% vs proxy = 0 |
| poisson-hazard (grid diperbaiki, lihat catatan) | 1,0744 (p00 1,0529; p01 1,0958) | median ≈ carry-forward setelah grid λ diperluas ke rezim jenuh; sebelum perbaikan grid, loss 1,3032 karena grid terpotong memaksa forecast naik ~25% (artefak, bukan efek model) |
| prior-shape (faktor 1.0) | 1,0727 | median = carry-forward |
| carry-forward titik (semua kuantil = rate terakhir) | 1,7767 | = mean \|logit error\| vs proxy |

(Catatan: ketiga model median-nya identik di practice — yang diukur di sini
hanyalah spread. Perbedaan poisson-hazard vs logit-linear (0,0017) murni
pembulatan/artefak grid, bukan sinyal.)

Pelajaran yang benar-benar bisa diambil dari practice: **bukan soal median**
(semua model median-nya sama — history datar), **melainkan soal spread**:
spread yang menampung lompatan refresh (~1,8 logit ke bawah untuk trofi
jarang) mengalahkan baseline sempit dengan selisih besar vs proxy. Sensitivitas
spread (sigma total) vs proxy: sigma 0,12 → 1,6877; 0,51 → 1,4216;
0,69 → 1,2991; 1,02 (provisional) → 1,0742; 1,49 → 0,9710; 2,02 → 0,9367.
Loss vs proxy TERUS turun sampai sigma ~2 — TAPI itu artefak proxy: proxy
adalah batas bawah ekstrem (dilusi murni, −1,8 logit), jawaban sebenarnya
lebih tinggi, jadi sigma optimal sebenarnya lebih kecil. Mengejar
sigma-optimal-vs-proxy berarti over-wide di final. Karena itu STALE_SIGMA
provisional dipertahankan 1,5 (sigma ~1,02), dan kalibrasi spread final
diajukan ke user (poin 4 di atas) — bukan diputuskan diam-diam di sini.
