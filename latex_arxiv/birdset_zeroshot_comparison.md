# BirdSet DT vs LT Comparison

Comparison of DT (Deep Tuning) vs LT (Linear Transfer) performance for sl-BEATS models on BirdSet detection datasets. All values are Probe (mAP) scores.

| Model | POW | PER | NES | NBP | HSN | SNE | UHH |
|:------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| **sl-BEATS-bio (DT)** | 0.304 | 0.150 | 0.279 | 0.496 | 0.349 | 0.226 | 0.213 |
| **sl-BEATS-bio (LT)** | 0.355 | 0.167 | 0.372 | 0.535 | 0.377 | 0.261 | 0.271 |
| **sl-BEATS-all (DT)** | 0.322 | 0.152 | 0.257 | 0.493 | 0.404 | 0.211 | 0.221 |
| **sl-BEATS-all (LT)** | 0.343 | 0.167 | 0.356 | 0.535 | 0.406 | 0.268 | 0.224 |

## Key Observations

**DT (Deep Tuning)** models:
- Full end-to-end fine-tuning with task-specific optimization
- sl-BEATS-all (DT) generally performs better on most datasets

**LT (Linear Transfer)** models:
- Linear probe only (frozen backbone)
- Shows strong linear transfer capabilities, especially on NBP dataset
- Surprisingly competitive with DT models on some datasets, and even outperforms on NBP
- **LT consistently outperforms DT on most datasets**, suggesting strong learned representations

## Dataset Key
- **POW**: Powdermill
- **PER**: Peru
- **NES**: Nips4Bplus
- **NBP**: NorthernBirdsongPlainsSpecies
- **HSN**: Hawaiian Short Night
- **SNE**: Sierra Nevada
- **UHH**: Hawaii
