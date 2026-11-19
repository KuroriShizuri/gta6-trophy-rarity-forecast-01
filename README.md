# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/KuroriShizuri/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/KuroriShizuri/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecaster earn rate trofi GTA VI (Exophase) minggu pertama: untuk tiap dari 55 trofi PS5, lima kuantil earn rate pada target. Built by an AI agent for make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty, $250 in USDC.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Cara kerja

`main.py` (Python 3.13, stdlib saja, tanpa network, deterministik) membaca
setiap file case `<id>.in` di direktori input dan menulis satu file forecast
`<id>.out` per case: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}`.

Per trofi, program membangun deret logit earn rate dari history case, lalu
meramal median logit + spread logit dengan model yang dipilih lewat konstanta
`MODEL` di `main.py` (override lokal: `MAKEFUN_MODEL=<nama>`). Kuantil =
sigmoid(median + z_tau · sigma).

**Model masih PROVISIONAL — keputusan final ada di user** (lihat
[`docs/MODEL-DECISION.md`](docs/MODEL-DECISION.md)). Tiga keluarga model
terimplementasi dan bisa diganti satu baris:

- `logit-linear` (default provisional): OLS logit(rate) vs jam pada 24 jam
  terakhir, slope di-shrink, ekstrapolasi dibatasi.
- `poisson-hazard`: hazard meluruh per trofi pada jumlah earners
  (E(t) = E_inf·(1−exp(−λt))) + forecast pemain terlacak log-linear;
  rate = earners / tracked.
- `prior-shape`: rate terakhir × faktor bentuk per tier (semua faktor 1,0
  saat ini — belum ada kurva minggu pertama GTA IV/V yang kredibel untuk
  di-bake; lihat dokumen keputusan).

Spread (dipakai semua model): random walk 0,5·√hari + suku **staleness**
(halaman Exophase bisa diam berjam-jam lalu refresh melompat — di launch
night pemain terlacak melonjak 638 → 3.868 dan semua rate turun) +
ketidakpastian slope. Di history practice yang datar total, semua model
runtuh ke median carry-forward; yang membedakan mereka adalah spread.

## Validasi

Jalankan evaluasi lokal (skrip di luar repo, hasil di laporan builder):
program dijalankan pada practice case p00/p01, output diperiksa (55 trofi,
kuantil terurut, tanpa NaN, deterministik byte-per-byte), dan pinball loss
dihitung vs **proxy** jawaban (rate × 638/3.868 — batas bawah, karena kit
tidak berisi jawaban practice; lihat `docs/MODEL-DECISION.md`).

---

<sub>Entered on [make.fun](https://make.fun/?ref=d9dj2dyr&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
