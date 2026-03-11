"""
Cross-Analysis: Cross-Scale Accent Reduction & Accent–Naturalness Trade-off
Combines results from:
  - 9-Point Accentedness (↓) Meta-Analysis (k=5, Seq2Seq + TTS)
  - 5-Point Naturalness (↑) Meta-Analysis (k=5)
  - 5-Point Nativeness (↑) Meta-Analysis (k=3, Token-based)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os

class CrossAnalysis:
    def __init__(self, output_dir="results/cross_analysis"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.colors = {
            'S2S': '#3498db',       # Blue for Seq2Seq
            'TTS': '#f39c12',       # Orange for TTS-Guided  
            'Token': '#2ecc71',     # Green for Token-based
            'pooled': '#8e44ad',    # Purple for pooled
        }
    
    # =========================================================================
    # DATA LOADING
    # =========================================================================
    
    def load_all_data(self):
        """Load and merge data from all three analyses."""
        print("\n" + "="*70)
        print("LOADING DATA FROM ALL ANALYSES")
        print("="*70)
        
        # --- 9-Point Accentedness (↓): L2 is high, L1 is low ---
        self.accent_9pt = pd.DataFrame({
            'study': ['FAC-NoRef (2021)', 'Accentron (2022)', 'ZS-NoRef (2022) *',
                      'TTS-Guided-NoPar (2023) *', 'AC-Articulatory (2024) \u2020'],
            'category': ['S2S', 'S2S', 'S2S', 'TTS', 'S2S'],
            'scale': ['9pt-accent'] * 5,
            'L1_mean': [1.04, 1.06, 1.10, 2.03, 8.56],
            'L2_mean': [6.77, 7.11, 6.78, 7.98, 2.07],
            'Conv_mean': [5.33, 3.30, 2.23, 2.87, 7.32],
            'L2_sd': [0.20, 0.21, 0.195, 0.230, 0.37],
            'Conv_sd': [0.28, 0.26, 0.064, 0.083, 0.51],
            'n': [800, 360, 800, 250, 300],
            'inverted': [False, False, False, False, True]
        })
        
        # --- 5-Point Naturalness (higher = more natural) ---
        self.nat_5pt = pd.DataFrame({
            'study': ['FAC-NoRef (2021)', 'Accentron (2022)', 'ZS-NoRef (2022)',
                      'TTS-Transfer (2024)', 'AC-Articulatory (2024)'],
            'category': ['S2S', 'S2S', 'S2S', 'TTS', 'S2S'],
            'L1_mean': [4.63, 4.90, 4.63, np.nan, 3.46],
            'L2_mean': [3.70, 3.67, 4.44, 4.43, 3.36],
            'Conv_mean': [3.22, 3.43, 3.03, 3.71, 2.45],
            'L2_sd': [0.06, 0.28, 0.074, 0.08, 0.43],
            'Conv_sd': [0.10, 0.12, 0.050, 0.08, 0.33],
            'n': [800, 360, 800, 400, 300]
        })
        
        # --- 5-Point Nativeness (higher = more native) ---
        self.native_5pt = pd.DataFrame({
            'study': ['AC-DiscreteUnits (2024)', 'Streaming-NAR (2025) \u2021', 'KD-AC (2025) \u2021'],
            'category': ['Token', 'TTS', 'TTS'],
            'scale': ['5pt-native'] * 3,
            'L2_mean': [2.12, 1.12, 1.67],
            'L2_sd': [0.05, 0.15, 0.05],
            'Conv_mean': [4.42, 3.78, 3.87],
            'Conv_sd': [0.11, 0.18, 0.08],
            'n': [1000, 380, 380]
        })
        
        self._compute_all_effect_sizes()
        return self
    
    def _hedges_g(self, mean1, mean2, sd1, sd2, n, direction='positive'):
        """Compute Hedges' g. direction='positive' means mean1-mean2 is good."""
        sd_pool = np.sqrt(((n-1)*sd1**2 + (n-1)*sd2**2) / (2*n-2))
        d = (mean1 - mean2) / sd_pool if direction == 'positive' else (mean2 - mean1) / sd_pool
        J = 1 - 3/(4*(2*n-2) - 1)
        g = d * J
        var_g = 2/n + g**2/(4*n)
        se = np.sqrt(var_g)
        return g, se, var_g, g - 1.96*se, g + 1.96*se
    
    def _compute_all_effect_sizes(self):
        """Compute effect sizes for all studies."""
        
        # 9-Point Accentedness: positive g = accent reduction
        results_accent = []
        for _, r in self.accent_9pt.iterrows():
            n = r['n']
            if not r['inverted']:
                g, se, var_g, ci_lo, ci_hi = self._hedges_g(
                    r['L2_mean'], r['Conv_mean'], r['L2_sd'], r['Conv_sd'], n, 'positive')
                gap = (r['L2_mean'] - r['Conv_mean']) / (r['L2_mean'] - r['L1_mean']) * 100
            else:
                g, se, var_g, ci_lo, ci_hi = self._hedges_g(
                    r['Conv_mean'], r['L2_mean'], r['Conv_sd'], r['L2_sd'], n, 'positive')
                gap = (r['Conv_mean'] - r['L2_mean']) / (r['L1_mean'] - r['L2_mean']) * 100
            results_accent.append({
                'study': r['study'], 'category': r['category'], 'scale': '9pt-accent',
                'g': g, 'se': se, 'var': var_g, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
                'gap_pct': gap, 'n': n
            })
        self.accent_es = pd.DataFrame(results_accent)
        
        # 5-Point Naturalness: negative g = degradation (Conv - L2)
        results_nat = []
        for _, r in self.nat_5pt.iterrows():
            g, se, var_g, ci_lo, ci_hi = self._hedges_g(
                r['Conv_mean'], r['L2_mean'], r['Conv_sd'], r['L2_sd'], r['n'], 'positive')
            pres = r['Conv_mean'] / r['L2_mean'] * 100
            results_nat.append({
                'study': r['study'], 'category': r['category'],
                'g_nat': g, 'se_nat': se, 'pres_pct': pres, 'n': r['n']
            })
        self.nat_es = pd.DataFrame(results_nat)
        
        # 5-Point Nativeness: positive g = improvement (Conv - L2)
        results_native = []
        for _, r in self.native_5pt.iterrows():
            g, se, var_g, ci_lo, ci_hi = self._hedges_g(
                r['Conv_mean'], r['L2_mean'], r['Conv_sd'], r['L2_sd'], r['n'], 'positive')
            results_native.append({
                'study': r['study'], 'category': r['category'], 'scale': '5pt-native',
                'g': g, 'se': se, 'var': var_g, 'ci_lo': ci_lo, 'ci_hi': ci_hi, 'n': r['n']
            })
        self.native_es = pd.DataFrame(results_native)
        
        print(f"  9-Point Accentedness: {len(self.accent_es)} studies")
        print(f"  5-Point Naturalness:  {len(self.nat_es)} studies")
        print(f"  5-Point Nativeness:   {len(self.native_es)} studies")
    
    # =========================================================================
    # ANALYSIS 1: ACCENT-NATURALNESS TRADE-OFF (text + table, no figure)
    # =========================================================================
    
    def tradeoff_analysis(self):
        """Correlate accent reduction (Gap%) with naturalness preservation (L2 Pres%)."""
        print("\n" + "="*70)
        print("ANALYSIS 1: ACCENT-NATURALNESS TRADE-OFF")
        print("="*70)
        
        # Merge overlapping studies (strip footnote markers for matching)
        accent_clean = self.accent_es.copy()
        accent_clean['study_base'] = accent_clean['study'].str.replace(r'\s*[\*\u2020\u2021]', '', regex=True).str.strip()
        nat_clean = self.nat_es.copy()
        nat_clean['study_base'] = nat_clean['study'].str.replace(r'\s*[\*\u2020\u2021]', '', regex=True).str.strip()
        
        merged = accent_clean[['study', 'study_base', 'category', 'gap_pct', 'g']].merge(
            nat_clean[['study_base', 'pres_pct', 'g_nat']], on='study_base', how='inner')
        merged.columns = ['study', 'study_base', 'category', 'accent_gap_pct', 'accent_g', 'nat_pres_pct', 'nat_g']
        
        self.tradeoff_df = merged
        
        print(f"\n  Overlapping studies (k={len(merged)}):")
        print(f"  {'Study':<30} {'Gap%':>8} {'L2 Pres%':>10} {'Accent g':>10} {'Nat g':>10}")
        print(f"  {'-'*68}")
        for _, r in merged.iterrows():
            print(f"  {r['study']:<30} {r['accent_gap_pct']:>7.1f}% {r['nat_pres_pct']:>9.1f}% "
                  f"{r['accent_g']:>9.2f} {r['nat_g']:>9.2f}")
        
        # Correlations
        if len(merged) >= 3:
            r_pct, p_pct = stats.pearsonr(merged['accent_gap_pct'], merged['nat_pres_pct'])
            r_g, p_g = stats.pearsonr(merged['accent_g'], merged['nat_g'])
            rho_pct, sp_pct = stats.spearmanr(merged['accent_gap_pct'], merged['nat_pres_pct'])
            rho_g, sp_g = stats.spearmanr(merged['accent_g'], merged['nat_g'])
            
            print(f"\n  CORRELATION RESULTS:")
            print(f"  Gap% vs L2 Pres%:  Pearson r = {r_pct:.3f} (p = {p_pct:.3f}), "
                  f"Spearman \u03c1 = {rho_pct:.3f} (p = {sp_pct:.3f})")
            print(f"  Accent g vs Nat g: Pearson r = {r_g:.3f} (p = {p_g:.3f}), "
                  f"Spearman \u03c1 = {rho_g:.3f} (p = {sp_g:.3f})")
            
            self.tradeoff_corr = {
                'r_pct': r_pct, 'p_pct': p_pct, 'rho_pct': rho_pct, 'sp_pct': sp_pct,
                'r_g': r_g, 'p_g': p_g, 'rho_g': rho_g, 'sp_g': sp_g, 'k': len(merged)
            }
        else:
            self.tradeoff_corr = None
        
        return merged
    
    # =========================================================================
    # ANALYSIS 2: CROSS-SCALE COMPARISON
    # =========================================================================
    
    def cross_scale_comparison(self):
        """Compare Hedges' g across 9-point and 5-point accentedness scales."""
        print("\n" + "="*70)
        print("ANALYSIS 2: CROSS-SCALE COMPARISON (Hedges' g)")
        print("="*70)
        
        accent_all = pd.concat([
            self.accent_es[['study', 'category', 'scale', 'g', 'se', 'var', 'ci_lo', 'ci_hi', 'n', 'gap_pct']],
            self.native_es[['study', 'category', 'scale', 'g', 'se', 'var', 'ci_lo', 'ci_hi', 'n']]
        ], ignore_index=True)
        self.accent_all = accent_all
        
        print(f"\n  {'Study':<35} {'Cat':<6} {'Scale':<10} {'g':>7} {'95% CI':>20} {'N':>6}")
        print(f"  {'-'*84}")
        for _, r in accent_all.sort_values('g').iterrows():
            print(f"  {r['study']:<35} {r['category']:<6} {r['scale']:<10} "
                  f"{r['g']:>6.2f} [{r['ci_lo']:>6.2f}, {r['ci_hi']:>6.2f}] {int(r['n']):>6}")
        
        # Category summaries
        print(f"\n  CATEGORY SUMMARY:")
        for cat in ['S2S', 'TTS', 'Token']:
            subset = accent_all[accent_all['category'] == cat]
            if len(subset) > 0:
                w = 1/subset['var'].values
                g_pooled = np.sum(w * subset['g'].values) / np.sum(w)
                se_pooled = np.sqrt(1/np.sum(w))
                print(f"  {cat:<8}: k={len(subset)}, pooled g={g_pooled:.2f} "
                      f"[{g_pooled-1.96*se_pooled:.2f}, {g_pooled+1.96*se_pooled:.2f}], "
                      f"range={subset['g'].min():.2f}\u2013{subset['g'].max():.2f}")
        
        # Statistical comparison
        g9 = accent_all[accent_all['scale'] == '9pt-accent']
        g5 = accent_all[accent_all['scale'] == '5pt-native']
        if len(g9) > 1 and len(g5) > 1:
            t_stat, t_p = stats.ttest_ind(g9['g'], g5['g'], equal_var=False)
            u_stat, u_p = stats.mannwhitneyu(g9['g'], g5['g'], alternative='two-sided')
            print(f"\n  SCALE COMPARISON (9pt vs 5pt):")
            print(f"  9pt mean g = {g9['g'].mean():.2f} (k={len(g9)})")
            print(f"  5pt mean g = {g5['g'].mean():.2f} (k={len(g5)})")
            print(f"  Welch's t = {t_stat:.2f} (p = {t_p:.3f})")
            print(f"  Mann-Whitney U = {u_stat:.1f} (p = {u_p:.3f})")
        
        # DerSimonian-Laird pooled effect across all k=8
        g_vals = accent_all['g'].values
        var_vals = accent_all['var'].values
        k = len(g_vals)
        w_fe = 1/var_vals
        g_fe = np.sum(w_fe * g_vals) / np.sum(w_fe)
        Q = np.sum(w_fe * (g_vals - g_fe)**2)
        C = np.sum(w_fe) - np.sum(w_fe**2)/np.sum(w_fe)
        tau2 = max(0, (Q - (k-1)) / C)
        w_re = 1/(var_vals + tau2)
        g_re = np.sum(w_re * g_vals) / np.sum(w_re)
        se_re = np.sqrt(1/np.sum(w_re))
        I2 = max(0, (Q - (k-1))/Q * 100) if Q > 0 else 0
        
        self.pooled_cross = {
            'g': g_re, 'se': se_re, 
            'ci_lo': g_re - 1.96*se_re, 'ci_hi': g_re + 1.96*se_re,
            'tau2': tau2, 'Q': Q, 'I2': I2, 'k': k
        }
        print(f"\n  POOLED (Random-Effects, k={k}):")
        print(f"  g = {g_re:.2f} [{g_re-1.96*se_re:.2f}, {g_re+1.96*se_re:.2f}]")
        print(f"  \u03c4\u00b2 = {tau2:.2f}, Q = {Q:.1f}, I\u00b2 = {I2:.1f}%")
        
        return accent_all
    
    # =========================================================================
    # FOREST PLOT - matches style of accentedness & naturalness plots
    # =========================================================================
    
    def create_forest_plot(self):
        """Publication-quality cross-scale forest plot.
        Style matched to accentedness_forest_plot and naturalness_forest_plot:
        figsize=(15,8), big_fs=18, small_fs=16, footnotes, same layout.
        """
        fig, ax = plt.subplots(figsize=(17, 8))
        big_fs = 20
        small_fs = 18
        
        df = self.accent_all.copy()
        scale_map = {'9pt-accent': '9-point', '5pt-native': '5-point'}
        
        # Build sorted study list -- no gaps between groups
        categories_order = ['S2S', 'TTS', 'Token']
        sorted_studies = []
        y = 0
        group_y_ranges = {}
        for cat in categories_order:
            cat_df = df[df['category'] == cat].sort_values('g')
            if len(cat_df) == 0:
                continue
            group_start = y + 1
            for _, r in cat_df.iterrows():
                y += 1
                sorted_studies.append((r, y))
            group_y_ranges[cat] = (group_start, y)
        
        n_studies = y
        
        # Column x-positions
        x_study = -2
        x_ci = 40
        x_gap = 65
        x_method = 75
        
        
        # Plot each study
        for r, y_pos in sorted_studies:
            color = self.colors[r['category']]
            ax.errorbar(r['g'], y_pos, xerr=[[r['g']-r['ci_lo']], [r['ci_hi']-r['g']]],
                       fmt='s', markersize=10, color=color, capsize=5, capthick=2, elinewidth=2)
            
            ax.text(x_study, y_pos, r['study'], ha='right', va='center', fontsize=small_fs)
            ax.text(x_ci, y_pos, f"{r['g']:.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]",
                   ha='left', va='center', fontsize=small_fs, family='monospace')
            
            
            if 'gap_pct' in r and pd.notna(r['gap_pct']):
                gap_val = f"{r['gap_pct']:.1f}%"
            else:
                gap_val = "-"
            ax.text(x_gap, y_pos, gap_val, ha='center', va='center', fontsize=small_fs, family='monospace')
            ax.text(x_method, y_pos, r['category'], ha='center', va='center', fontsize=small_fs,
                   color=self.colors[r['category']], fontweight='bold')
        
        # Thin separator lines between groups
        for cat, (y_lo, y_hi) in group_y_ranges.items():
            if y_lo > 1:
                ax.axhline(y=y_lo - 0.5, color='gray', linewidth=0.8, alpha=0.4)
        
        # Pooled effect (matching accentedness/naturalness style)
        p = self.pooled_cross
        ax.errorbar(p['g'], 0, xerr=[[p['g']-p['ci_lo']], [p['ci_hi']-p['g']]],
                   fmt='o', markersize=12, color=self.colors['pooled'], capsize=6, capthick=2.5,
                   elinewidth=2.5, markeredgecolor='white', markeredgewidth=2)
        
        ax.text(x_study, 0, "Pooled Effect (Random)", ha='right', va='center', 
               fontsize=big_fs, fontweight='bold')
        ax.text(x_ci, 0, f"{p['g']:.2f} [{p['ci_lo']:.2f}, {p['ci_hi']:.2f}]",
               ha='left', va='center', fontsize=small_fs, family='monospace', fontweight='bold')
        
        # Reference line at g = 0
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Headers
        header_y = n_studies + 1
        ax.text(x_study, header_y, "Study", ha='right', fontsize=big_fs, fontweight='bold')
        ax.text(x_ci, header_y, "Hedges' g [95% CI]", ha='left', fontsize=big_fs, fontweight='bold')
        ax.text(x_gap, header_y, "Gap Closure", ha='center', fontsize=big_fs, fontweight='bold')
        ax.text(x_method, header_y, "Method", ha='center', fontsize=big_fs, fontweight='bold')
        
        # Formatting
        ax.set_xlim(-5, 78)
        ax.set_ylim(-1.5, n_studies + 1.8)
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Title
        ax.set_title("Cross-Scale Forest Plot: Accent Reduction Effect Sizes (k=8)\n"
                     "Positive effect = accent improvement toward native speech",
                     fontsize=big_fs, fontweight='bold', pad=15)
        
        # X-axis
        ax.set_xlabel("Hedges' g (Standardized Effect Size)", fontsize=small_fs)
        ax.tick_params(axis='x', labelsize=small_fs)
        
        # Direction labels
        ax.text(-0.2, -1.15, "\u2190 No effect ", ha='right', fontsize=small_fs)
        ax.text(-0.2, -1.15, "| Accent improvement \u2192", ha='left', fontsize=small_fs)
        
        # Footnote (matching other plots)
        het_text = (f"Heterogeneity: I\u00b2 = {p['I2']:.1f}%, "
                    f"\u03c4\u00b2 = {p['tau2']:.2f}, Q = {p['Q']:.1f}")
        ax.text(70, -1.15, het_text, ha='right', fontsize=small_fs, style='italic')
        
        #fig.text(0.5, 0.01, 
        #        "* SD imputed via CV method    \u2020 Inverted scale (1=heavy, 9=native)    \u2021 Sample size imputed (N=380)", 
        #        ha='center', fontsize=13, style='italic')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.10)
        
        for fmt in ['pdf', 'png']:
            plt.savefig(f"{self.output_dir}/cross_scale_forest.{fmt}",
                       dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n\u2713 Saved cross-scale forest plot to {self.output_dir}/")
        plt.show()
    
    # =========================================================================
    # PRINT INSIGHTS
    # =========================================================================
    
    def print_insights(self):
        """Print key findings for paper."""
        print("\n" + "="*70)
        print("KEY FINDINGS")
        print("="*70)
        
        tc = self.tradeoff_corr
        s2s = self.accent_all[self.accent_all['category'] == 'S2S']
        tts = self.accent_all[self.accent_all['category'] == 'TTS']
        tok = self.accent_all[self.accent_all['category'] == 'Token']
        all_9pt = self.accent_all[self.accent_all['scale'] == '9pt-accent']
        all_5pt = self.accent_all[self.accent_all['scale'] == '5pt-native']
        
        print(f"""
  TRADE-OFF (k={tc['k']} overlapping studies):
    Gap% vs L2 Pres%:  r = {tc['r_pct']:.2f}, \u03c1 = {tc['rho_pct']:.2f}
    Accent g vs Nat g: r = {tc['r_g']:.2f}, \u03c1 = {tc['rho_g']:.2f}

  CROSS-SCALE:
    9pt mean g = {all_9pt['g'].mean():.2f} (k={len(all_9pt)})
    5pt mean g = {all_5pt['g'].mean():.2f} (k={len(all_5pt)})

  BY CATEGORY:
    S2S:   g = {s2s['g'].min():.1f}\u2013{s2s['g'].max():.1f} (k={len(s2s)})
    TTS:   g = {tts['g'].values[0]:.1f} (k={len(tts)})
    Token: g = {tok['g'].min():.1f}\u2013{tok['g'].max():.1f} (k={len(tok)})
""")
    
    # =========================================================================
    # SAVE & RUN
    # =========================================================================
    
    def save_all_results(self):
        """Save all analysis results to CSV."""
        self.tradeoff_df.to_csv(f"{self.output_dir}/tradeoff_data.csv", index=False)
        self.accent_all.to_csv(f"{self.output_dir}/all_accent_effects.csv", index=False)
        self.nat_es.to_csv(f"{self.output_dir}/naturalness_effects.csv", index=False)
        if self.tradeoff_corr:
            pd.DataFrame([self.tradeoff_corr]).to_csv(
                f"{self.output_dir}/tradeoff_correlation.csv", index=False)
        print(f"\n\u2713 All results saved to {self.output_dir}/")
    
    def run(self):
        """Execute complete cross-analysis."""
        print("\n" + "\u2554"+"\u2550"*68+"\u2557")
        print(" \u2551  CROSS-ANALYSIS: Accent x Naturalness x Scale Comparison  \u2551")
        print(" \u255a"+"\u2550"*68+"\u255d")
        
        self.load_all_data()
        self.tradeoff_analysis()
        self.cross_scale_comparison()
        self.create_forest_plot()
        self.print_insights()
        self.save_all_results()


if __name__ == "__main__":
    analysis = CrossAnalysis()
    analysis.run()
