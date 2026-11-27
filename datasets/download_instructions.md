# Download instructions

All three datasets are hosted on Mendeley Data. No login is required; the DOIs below resolve to each dataset's landing page with a "Download all" button that produces a ZIP.

## Target layout

```
data_raw/
├── SLD/
│   ├── Sett_rot/*.jpg
│   ├── Healthy/*.jpg
│   └── ... (11 class folders)
├── DK/
│   ├── Healthy/*.jpg
│   └── ... (5 class folders)
└── BD/
    ├── Red_rot/*.jpg
    └── ... (5 class folders)
```

All scripts in this repository expect this layout and read it via `preprocessing/dataset.py`.

## 1. SLD — Sugarcane Leaf Dataset

```
https://doi.org/10.17632/355y629ynj.1
```

Unzip into `data_raw/SLD/`. The release ships one folder per class. Keep the class-folder names exactly as the release provides them.

## 2. D&K — Daphal & Koli

```
https://doi.org/10.17632/9424skmnrk.1
```

Unzip into `data_raw/DK/`.

## 3. BD — SugarcaneLD-BD

```
https://doi.org/10.17632/n8mpzb7p4k.1
```

Unzip into `data_raw/BD/`.

## Verification

After download, the first script to run is:

```bash
python scripts/build_manifest.py --root ./data_raw --out manifests/main_manifest.csv
```

Its first output line reports the number of images found per dataset. If any of the three does not match the paper's "Released images" row of Table 5 (6,748 / 2,521 / 638), stop and re-download the mismatching one before continuing: an incomplete download will silently reduce components and change the split.

## Licenses

All three datasets are released under CC BY 4.0. This repository does not redistribute the images; it operates on files fetched locally by the user.
