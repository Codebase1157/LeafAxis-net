# Datasets

Three publicly released sugarcane-leaf collections are used. All three are downloaded, deduplicated and split by the scripts in this repository; no modified copies of the source images are redistributed here.

## 1. Sugarcane Leaf Dataset (SLD)

- **Reference**: Thite, Suryawanshi, Patil, Chumchu (2024), *Data in Brief* 53, 110268.
- **Mendeley DOI**: 10.17632/355y629ynj.1
- **License**: CC BY 4.0
- **Released images**: 6,748
- **Classes (11)**: Sett rot, Healthy, Dried leaves, Yellow leaf, Grassy shoot, Mosaic, Brown spot, Brown rust, Banded chlorosis, Pokkah boeng, Smut.
- **Audit reference**: Sabrigiriraj & Manoharan (2026), *Data in Brief* 67, 113069. The deduplication audit in this paper's Table 5 reproduces the SLD column of that letter.

## 2. Daphal & Koli (D&K)

- **Reference**: Daphal & Koli (2024), *Heliyon* 10(8), e29438.
- **Mendeley DOI**: 10.17632/9424skmnrk.1
- **License**: CC BY 4.0
- **Released images**: 2,521
- **Classes (5)**: Healthy, Rust, Yellow, Mosaic, Red rot.

## 3. SugarcaneLD-BD (BD)

- **Reference**: Arman, Abdullah, Saiem, Asha, Sakib (2025).
- **Mendeley DOI**: 10.17632/n8mpzb7p4k.1
- **License**: CC BY 4.0
- **Released images**: 638
- **Classes (5)**: Red rot, Eye spot, Healthy, Ring spot, Red leaf spot.

## Shared / provisional categories across releases

- **Confirmed shared** (used in zero-shot transfer primary analysis): Healthy, Mosaic, Yellow leaf.
- **Provisional**: Rust (named differently across releases; no pathogen identified in SugarcaneLD-BD's "red rot" vs D&K's "red rot"; see Section 5.6 of the paper).
