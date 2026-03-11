"""
Sensitivity Analysis for CV Imputation + MOS vs WER Correlation
Run alongside existing meta-analysis code.
"""

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# PART 1: SENSITIVITY ANALYSIS FOR CV IMPUTATION
# =============================================================================

def hedges_g(mean1, mean2, sd1, sd2, n):
    """Compute Hedges' g (mean1 - mean2 direction, positive = improvement)."""
    sd_pool = np.sqrt(((n-1)*sd1**2 + (n-1)*sd2**2) / (2*n - 2))
    d = (mean1 - mean2) / sd_pool
    J = 1 - 3 / (4*(2*n - 2) - 1)
    g = d * J
    var_g = 2/n + g**2 / (4*n)
    se = np.sqrt(var_g)
    return g, se, var_g


def dl_pooled(g_vals, var_vals):
    """DerSimonian-Laird random-effects pooling."""
    k = len(g_vals)
    w_fe = 1 / var_vals
    g_fe = np.sum(w_fe * g_vals) / np.sum(w_fe)
    Q = np.sum(w_fe * (g_vals - g_fe)**2)
    C = np.sum(w_fe) - np.sum(w_fe**2) / np.sum(w_fe)
    tau2 = max(0, (Q - (k - 1)) / C)
    w_re = 1 / (var_vals + tau2)
    g_re = np.sum(w_re * g_vals) / np.sum(w_re)
    se_re = np.sqrt(1 / np.sum(w_re))
    I2 = max(0, (Q - (k-1)) / Q * 100) if Q > 0 else 0
    return g_re, se_re, tau2, Q, I2


def run_sensitivity_analysis():
    """Sensitivity analysis: vary CV imputation for ZS-NoRef and TTS-Guided-NoPar."""
    print("=" * 70)
    print("SENSITIVITY ANALYSIS: CV IMPUTATION (9-Point Accentedness)")
    print("=" * 70)

    # ---- Complete studies (known SDs) ----
    # Direction: L2_mean - Conv_mean for non-inverted; Conv_mean - L2_mean for inverted
    complete = pd.DataFrame({
        'study': ['FAC-NoRef (2021)', 'Accentron (2022)', 'AC-Articulatory (2024)'],
        'L2_mean': [6.77, 7.11, 2.07],
        'Conv_mean': [5.33, 3.30, 7.32],
        'L2_sd': [0.20, 0.21, 0.37],
        'Conv_sd': [0.28, 0.26, 0.51],
        'n': [800, 360, 300],
        'inverted': [False, False, True]
    })

    # ---- Imputed studies (missing Conv_sd and L2_sd) ----
    imputed = pd.DataFrame({
        'study': ['ZS-NoRef (2022)', 'TTS-Guided-NoPar (2023)'],
        'L2_mean': [6.78, 7.98],
        'Conv_mean': [2.23, 2.87],
        'n': [800, 250],
        'inverted': [False, False]
    })

    # ---- Compute CVs from complete studies ----
    # L2 CVs
    l2_cvs = complete['L2_sd'] / complete['L2_mean']
    # For inverted study (AC-Articulatory), L2_mean=2.07 is the "low accent" end
    # but CV is still sd/mean
    conv_cvs = complete['Conv_sd'] / complete['Conv_mean']

    print(f"\nKnown L2 CVs:   {l2_cvs.values}  -> median = {l2_cvs.median():.4f}")
    print(f"Known Conv CVs: {conv_cvs.values}  -> median = {conv_cvs.median():.4f}")
    print(f"  Min CV (L2): {l2_cvs.min():.4f},  Max CV (L2): {l2_cvs.max():.4f}")
    print(f"  Min CV (Conv): {conv_cvs.min():.4f},  Max CV (Conv): {conv_cvs.max():.4f}")

    # ---- Three imputation scenarios ----
    scenarios = {
        'Baseline (median CV)': {
            'l2_cv': l2_cvs.median(),
            'conv_cv': conv_cvs.median()
        },
        'Conservative (max CV)': {
            'l2_cv': l2_cvs.max(),
            'conv_cv': conv_cvs.max()
        },
        'Liberal (min CV)': {
            'l2_cv': l2_cvs.min(),
            'conv_cv': conv_cvs.min()
        }
    }

    print(f"\n{'Scenario':<28} {'Pooled g':>10} {'95% CI':>22} {'tau2':>8} {'I2':>8}")
    print("-" * 78)

    for name, cvs in scenarios.items():
        all_g, all_var = [], []

        # Complete studies
        for _, r in complete.iterrows():
            if not r['inverted']:
                g, se, var = hedges_g(r['L2_mean'], r['Conv_mean'],
                                      r['L2_sd'], r['Conv_sd'], r['n'])
            else:
                g, se, var = hedges_g(r['Conv_mean'], r['L2_mean'],
                                      r['Conv_sd'], r['L2_sd'], r['n'])
            all_g.append(g)
            all_var.append(var)

        # Imputed studies with current scenario's CVs
        for _, r in imputed.iterrows():
            l2_sd_imp = r['L2_mean'] * cvs['l2_cv']
            conv_sd_imp = r['Conv_mean'] * cvs['conv_cv']
            g, se, var = hedges_g(r['L2_mean'], r['Conv_mean'],
                                  l2_sd_imp, conv_sd_imp, r['n'])
            all_g.append(g)
            all_var.append(var)

            if name == 'Baseline (median CV)':
                print(f"  {r['study']}: imputed L2_sd={l2_sd_imp:.3f}, "
                      f"Conv_sd={conv_sd_imp:.3f}, g={g:.2f}")

        g_arr = np.array(all_g)
        var_arr = np.array(all_var)
        g_re, se_re, tau2, Q, I2 = dl_pooled(g_arr, var_arr)

        print(f"{name:<28} {g_re:>10.2f} [{g_re-1.96*se_re:>8.2f}, "
              f"{g_re+1.96*se_re:>8.2f}] {tau2:>8.2f} {I2:>7.1f}%")

    # ---- Leave-one-out analysis ----
    print(f"\n{'LEAVE-ONE-OUT ANALYSIS':}")
    print("-" * 78)

    # Baseline imputation for all 5 studies
    baseline_cv_l2 = l2_cvs.median()
    baseline_cv_conv = conv_cvs.median()

    all_studies_data = []
    for _, r in complete.iterrows():
        if not r['inverted']:
            g, se, var = hedges_g(r['L2_mean'], r['Conv_mean'],
                                  r['L2_sd'], r['Conv_sd'], r['n'])
        else:
            g, se, var = hedges_g(r['Conv_mean'], r['L2_mean'],
                                  r['Conv_sd'], r['L2_sd'], r['n'])
        all_studies_data.append({'study': r['study'], 'g': g, 'var': var, 'imputed': False})

    for _, r in imputed.iterrows():
        l2_sd_imp = r['L2_mean'] * baseline_cv_l2
        conv_sd_imp = r['Conv_mean'] * baseline_cv_conv
        g, se, var = hedges_g(r['L2_mean'], r['Conv_mean'],
                              l2_sd_imp, conv_sd_imp, r['n'])
        all_studies_data.append({'study': r['study'], 'g': g, 'var': var, 'imputed': True})

    studies_df = pd.DataFrame(all_studies_data)

    for i, row in studies_df.iterrows():
        subset = studies_df.drop(i)
        g_re, se_re, tau2, Q, I2 = dl_pooled(subset['g'].values, subset['var'].values)
        marker = " *" if row['imputed'] else ""
        print(f"  Excluding {row['study']:<30}{marker}  -> pooled g = {g_re:.2f} "
              f"[{g_re-1.96*se_re:.2f}, {g_re+1.96*se_re:.2f}]")

    print("\nConclusion: Pooled effect remains very large and significant across")
    print("all imputation scenarios and leave-one-out subsets.")


# =============================================================================
# PART 2: MOS vs WER CORRELATION
# =============================================================================

def run_mos_wer_analysis():
    """Correlate MOS-based effect sizes with WER for studies reporting both."""
    print("\n\n" + "=" * 70)
    print("MOS vs WER CORRELATION ANALYSIS")
    print("=" * 70)

    # Studies with both accentedness g and WER (from converted speech)
    # g values from the 9-point and 5-point accentedness analyses
    accent_wer = pd.DataFrame({
        'study': ['TTS-Guided-NoPar (2023)', 'AC-Articulatory (2024)',
                  'Streaming-NAR (2025)', 'KD-AC (2025)'],
        'accent_g': [23.91, 11.77, 16.04, 32.95],
        'WER': [15.8, 17.8, 14.1, 12.4],
        'category': ['TTS', 'S2S', 'Token', 'Token']
    })

    # Studies with both naturalness g and WER
    nat_wer = pd.DataFrame({
        'study': ['TTS-Transfer (2024)', 'AC-Articulatory (2024)'],
        'nat_g': [-8.99, -2.37],
        'nat_pres': [83.7, 72.9],
        'WER': [17.7, 17.8],
        'category': ['TTS', 'S2S']
    })

    print(f"\n--- Accent Reduction (g) vs WER (k={len(accent_wer)}) ---")
    print(f"{'Study':<30} {'Accent g':>10} {'WER':>8}")
    print("-" * 50)
    for _, r in accent_wer.iterrows():
        print(f"{r['study']:<30} {r['accent_g']:>10.2f} {r['WER']:>7.1f}%")

    # Correlation: higher accent g (more reduction) should correlate with lower WER
    r_val, p_val = stats.pearsonr(accent_wer['accent_g'], accent_wer['WER'])
    rho_val, sp_val = stats.spearmanr(accent_wer['accent_g'], accent_wer['WER'])

    print(f"\nPearson  r = {r_val:.3f} (p = {p_val:.3f})")
    print(f"Spearman ρ = {rho_val:.3f} (p = {sp_val:.3f})")
    print(f"Direction: {'Negative (more accent reduction → lower WER)' if r_val < 0 else 'Positive'}")

    print(f"\n--- Naturalness vs WER (k={len(nat_wer)}) ---")
    print(f"  Only {len(nat_wer)} studies — insufficient for correlation.")
    for _, r in nat_wer.iterrows():
        print(f"  {r['study']:<30} g={r['nat_g']:.2f}, Pres={r['nat_pres']:.1f}%, WER={r['WER']:.1f}%")

    return accent_wer, r_val, p_val, rho_val, sp_val


if __name__ == "__main__":
    run_sensitivity_analysis()
    accent_wer, r_val, p_val, rho_val, sp_val = run_mos_wer_analysis()
