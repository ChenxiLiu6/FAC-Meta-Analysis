# FAC-Meta-Analysis

Code and data for the meta-analysis section of:

> **Foreign Accent Conversion: Methods, Models, and Research Challenges**  
> Chenxi Liu and Israel Cohen, Fellow, IEEE  
> *IEEE Signal Processing Magazine*, 2025 (Draft)

---

## Overview

This repository contains the Python code and evaluation data used to conduct a systematic meta-analysis of Foreign Accent Conversion (FAC) systems published between 2021–2025. The meta-analysis quantifies accent reduction effectiveness and naturalness preservation across 13 FAC systems using standardized effect sizes (Hedges' *g*).

Three perceptual scales are analyzed:
- **9-Point Accentedness** (1=native, 9=heavy accent) — 5 studies, Seq2Seq & TTS-guided methods
- **5-Point Accentedness** (1=heavy, 5=native) — 3 studies, Token-based methods
- **5-Point Naturalness** (1=unnatural, 5=natural) — 5 studies

---

## Repository Structure

```
FAC-Meta-Analysis/
│
├── Meta-Analysis-code/
│   ├── cross-scale-analysis.py       # Cross-scale accent reduction & accent-naturalness trade-off
│   ├── Naturalness-5-pt.py           # 5-point naturalness meta-analysis
│   └── sensitivity-wer-analysis.py  # Sensitivity analysis (CV imputation) + MOS vs WER correlation
│
├── Analysis-Results/
│   ├── cross_analysis/               # Cross-scale forest plot outputs
│   ├── naturalness_5pt/              # Naturalness forest plot + results
│   └── computational_json/           # Token-based FAC study TokAN's RTF json file (evaluated by both CPU and GPU)
│
├── table3_evaluation_data.csv        # Full evaluation data (Table III from paper)
├── requirements.txt
└── README.md
```

---

## Methods Summary

### 1. Missing SD Imputation (CV Method)
Two studies (ZS-NoRef 2022, TTS-Guided-NoPar 2023) reported only mean values without standard deviations. Missing SDs were estimated using the **Coefficient of Variation (CV) method**:

$$\hat{\sigma}_{missing} = \mu_{missing} \times \text{median}(CV_{known})$$

### 2. Effect Size: Hedges' *g*
Raw MOS differences are standardized using Hedges' *g* to enable comparison across studies with varying baselines and sample sizes:

$$g = d \times J(df), \quad J(df) = 1 - \frac{3}{4df - 1}$$

### 3. Random-Effects Meta-Analysis
The **DerSimonian-Laird** random-effects model is used to pool effect sizes, weighting studies by their precision:

$$\bar{g} = \frac{\sum w_i \times g_i}{\sum w_i}, \quad w_i = \frac{1}{\sigma_i^2 + \tau^2}$$

### 4. Heterogeneity Analysis
- **Cochran's Q**: tests whether variability across studies exceeds sampling error
- **I²**: percentage of variance due to genuine between-study differences
- **τ²**: between-study variance

---

## Data

`evaluation_data.md` contains the full evaluation data for all 13 FAC systems reviewed in the paper, including:
- Accentedness and Naturalness MOS scores (mean ± SD) for converted and L2 speech
- Word Error Rate (WER) and Speaker Encoder Cosine Similarity (SECS)
- Rater count, sample size, and rater nationality
- System category (S2S = Seq2Seq, TTS = TTS-Guided, Token = Token-based)

> `*` SD imputed via CV method  
> `†` Inverted 9-pt scale (1=heavy, 9=native)  
> `‡` Sample size imputed (N=380)

This data is provided to support reproducibility and to serve as a reference for future FAC meta-analyses, since Table III is not included in the published paper due to space constraints.

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, numpy, pandas, matplotlib, scipy

---

## Usage

Run each analysis independently:

```bash
# Cross-scale accent reduction + accent-naturalness trade-off
python Meta-Analysis-code/cross-scale-analysis.py

# 5-point naturalness meta-analysis
python Meta-Analysis-code/Naturalness-5-pt.py

# Sensitivity analysis (CV imputation) + MOS vs WER correlation
python Meta-Analysis-code/sensitivity-wer-analysis.py
```

Results (forest plots as `.pdf`/`.png`, effect size tables as `.csv`) are saved to the `Analysis-Results/` directory.

---

## Key Results

| Analysis | k | Pooled Hedges' *g* | 95% CI | I² |
|---|---|---|---|---|
| Cross-scale Accentedness | 8 | 21.30 | [13.82, 28.78] | 99.9% |
| 9-pt Accentedness | 5 | 18.93 | — | 99.9% |
| 5-pt Accentedness | 3 | 25.30 | — | 99.9% |
| 5-pt Naturalness | 5 | −5.37 | [−8.27, −2.48] | 99.9% |

A **moderate negative correlation** (Pearson *r* = −0.64, Spearman *ρ* = −0.60) was observed between accent gap closure and naturalness preservation across systems reporting both metrics, indicating a quality–accent trade-off.

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{liu2025fac,
  title={Foreign Accent Conversion: Methods, Models, and Research Challenges},
  author={Liu, Chenxi and Cohen, Israel},
  journal={IEEE Signal Processing Magazine},
  year={2025}
}
```

---

## Contact

Chenxi Liu — chenxi.liu@campus.technion.ac.il  
Andrew and Erna Viterbi Faculty of Electrical and Computer Engineering  
Technion – Israel Institute of Technology, Haifa, Israel
