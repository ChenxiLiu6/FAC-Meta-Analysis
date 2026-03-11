[evaluation_data.md](https://github.com/user-attachments/files/25897909/evaluation_data.md)
# Comprehensive Evaluation of Foreign Accent Conversion Systems (2021–2025)

> Data supporting the meta-analysis in: **Foreign Accent Conversion: Methods, Models, and Research Challenges**, Chenxi Liu and Israel Cohen, *IEEE Signal Processing Magazine*, 2025.

---

## Table III: Evaluation Scores and System Metadata

| Ref | System | Cat | Year | WER↓ | SECS↑ | Scale | Accent Conv | Accent L2 | Natural. Conv | Natural. L2 | R | N | C |
|-----|--------|-----|------|------|-------|-------|-------------|-----------|---------------|-------------|---|---|---|
| [3] | FAC-NoRef | S2S | 2021 | – | – | 9pt | 5.33 ± 0.28 | 6.77 ± 0.20 | 3.22 ± 0.10 | 3.70 ± 0.06 | 20 | 800 | US |
| [4] | Accentron | S2S | 2022 | – | – | 9pt | 3.30 ± 0.26 | 7.11 ± 0.21 | 3.43 ± 0.12 | 3.67 ± 0.28 | 18 | 360 | US |
| [5] | ZS-NoRef | S2S | 2022 | – | – | 9pt | 2.23 \* | 6.78 \* | 3.03 \* | 4.44 \* | 20 | 800 | US |
| [6] | AC-Articulatory | S2S | 2024 | 17.8 | – | 9pt † | 7.32 ± 0.51 | 2.07 ± 0.37 | 2.45 ± 0.33 | 3.36 ± 0.43 | 20 | 300 | US |
| [2] | TTS-Guided-NoPar | TTS | 2023 | 15.8 | – | 9pt | 2.87 \* | 7.98 \* | – | – | 25 | 250 | US |
| [1] | TTS-Transfer | TTS | 2024 | 17.7 | – | – | – | – | 3.71 ± 0.08 | 4.43 ± 0.08 | 20 | 400 | – |
| [11] | Diffusion-TTS | TTS | 2024 | – | – | BW | 0.16 | – | 50.0 ᴹ | 51.2 ᴹ | 13 | 390 | – |
| [8] | Streaming-NAR | TTS | 2025 | 14.1 | 0.85 | 5pt ‡ | 3.78 ± 0.18 | 1.12 ± 0.15 | – | – | 10 | – | – |
| [9] | KD-AC | TTS | 2025 | 12.4 | 0.83 | 5pt ‡ | 3.87 ± 0.08 | 1.67 ± 0.05 | – | – | 10 | – | – |
| [7] | AC-DiscreteUnits | Token | 2024 | – | – | 5pt ‡ | 4.42 ± 0.11 | 2.12 ± 0.05 | – | – | 20 | 1000 | US |
| [12] | SpeechAccentLLM | Token | 2025 | 9.1 | 0.63 | 10pt | 1.86 | – | 4.07 ± 0.10 | – | 20 | 2000 | US |
| [10] | TokAN | Token | 2025 | 16.2 | 0.87 | MUSHRA | 26.1 ± 1.8 | 46.4 ± 2.5 | 60.4 ᴹ | 60.5 ᴹ | 21 | 420 | – |
| [13] | FAC-FACodec | Token | 2025 | 11.0 | 0.86 | – | – | – | – | – | – | – | – |

**Column definitions:** Cat = Category; WER = Word Error Rate (%); SECS = Speaker Encoder Cosine Similarity; Scale = Perceptual rating scale; Accent / Natural. = Accentedness / Naturalness MOS score for converted (Conv) and original L2 speech; R = Number of raters; N = Sample size (utterances); C = Raters' origin country/citizenship.

**Notes:**
- Cat: S2S = Seq2Seq, TTS = TTS-Guided, Token = Token-based
- Scale: 9pt (1=native, 9=heavy accent ↓); 5pt ‡ (1=heavy accent, 5=native ↑); 10pt (1=native, 10=heavy ↓); MUSHRA (0–100); BW = Best-Worst Scaling
- \* SD imputed via CV method (study reported mean only)
- † Inverted 9-pt scale (1=heavy accent, 9=native)
- ᴹ MUSHRA score (0–100 scale)
- ‡ Sample size imputed (N = 380)

**Meta-analysis inclusion:**
- 9-pt accentedness: [3], [4], [5], [2], [6] (k=5)
- 5-pt nativeness: [7], [8], [9] (k=3, analyzed separately — cannot pool with 9-pt)
- 5-pt naturalness: [3], [4], [5], [1], [6] (k=5); [12] excluded (no L2 baseline)
- MUSHRA / Best-Worst excluded (different evaluation protocols)

---

## References

| Ref | Authors | Title | Venue | Year |
|-----|---------|-------|-------|------|
| [1] | X. Chen, J. Pei, L. Xue, M. Zhang | Transfer the Linguistic Representations from TTS to Accent Conversion with Non-Parallel Data | ICASSP | 2024 |
| [2] | Y. Zhou, Z. Wu, M. Zhang, X. Tian, H. Li | TTS-Guided Training for Accent Conversion without Parallel Data | IEEE Signal Processing Letters | 2023 |
| [3] | G. Zhao, S. Ding, R. Gutierrez-Osuna | Converting Foreign Accent Speech without a Reference | IEEE/ACM Trans. Audio, Speech, Language Process. | 2021 |
| [4] | S. Ding, G. Zhao, R. Gutierrez-Osuna | Accentron: Foreign Accent Conversion to Arbitrary Non-Native Speakers Using Zero-Shot Learning | Computer Speech & Language | 2022 |
| [5] | W. Quamer, A. Das, J. Levis, E. Chukharev-Hudilainen, R. Gutierrez-Osuna | Zero-Shot Foreign Accent Conversion without a Native Reference | Interspeech | 2022 |
| [6] | Y. M. Siriwardena et al. | Accent Conversion with Articulatory Representations | Interspeech | 2024 |
| [7] | T. N. Nguyen, N. Q. Pham, A. Waibel | Accent Conversion Using Discrete Units with Parallel Data Synthesized from Controllable Accented TTS | arXiv:2410.03734 | 2024 |
| [8] | T.-N. Nguyen, N.-Q. Pham, S. Akti, A. Waibel | Streaming Non-Autoregressive Model for Accent Conversion and Pronunciation Improvement | arXiv:2506.16580 | 2025 |
| [9] | T. N. Nguyen, S. Akti, N. Q. Pham, A. Waibel | Improving Pronunciation and Accent Conversion through Knowledge Distillation and Synthetic Ground-Truth from Native TTS | ICASSP | 2025 |
| [10] | Q. Bai, S. Inoue, S. Wang, Z. Jiang, Y. Wang, H. Li | Accent Normalization Using Self-Supervised Discrete Tokens with Non-Parallel Data | Interspeech | 2025 |
| [11] | Q. Bai et al. | Diffusion-Based Method with TTS Guidance for Foreign Accent Conversion | IEEE ISCSLP | 2024 |
| [12] | Z. Cheng et al. | SpeechAccentLLM: A Unified Framework for Foreign Accent Conversion and Text to Speech | arXiv:2507.01348 | 2025 |
| [13] | Y. Halychanskyi, C. Churchwell, Y. Wen, V. Kindratenko | FAC-FACodec: Controllable Zero-Shot Foreign Accent Conversion with Factorized Speech Codec | arXiv:2510.10785 | 2025 |
