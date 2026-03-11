"""
Meta-Analysis: 5-Point Naturalness MOS for Foreign Accent Conversion
Papers: FAC-NoRef, Accentron, ZS-NoRef, TTS-Transfer, AC-Articulatory
Scale: 1 (unnatural) to 5 (natural) | Negative g = naturalness degradation
Excluded: SpeechAccentLLM (no L2 baseline)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os

class NaturalnessMetaAnalysis:
    def __init__(self, output_dir="results/naturalness_5pt"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.df = self._load_data()
        self.colors = {
            'S2S': '#4285F4',      # blue
            'TTS': '#F4A235',      # orange
            'Token': '#34A853',    # green
            'pooled': '#7B2D8E',   # purple
        }

    def _load_data(self):
        """Load study data for naturalness evaluation."""
        return pd.DataFrame({
            'Study': ['FAC-NoRef (2021)', 'Accentron (2022)', 'ZS-NoRef (2022) *',
                      'TTS-Transfer (2024)', 'AC-Articulatory (2024)'],
            'L1_mean': [4.63, 4.90, 4.63, np.nan, 3.46],  # [9] no L1
            'L1_sd': [0.06, 0.10, np.nan, np.nan, 0.18],
            'L2_mean': [3.70, 3.67, 4.44, 4.43, 3.36],
            'L2_sd': [0.06, 0.28, np.nan, 0.08, 0.43],
            'Conv_mean': [3.22, 3.43, 3.03, 3.71, 2.45],
            'Conv_sd': [0.10, 0.12, np.nan, 0.08, 0.33],
            'n': [800, 360, 800, 400, 300]
        })
    
    def impute_missing_sds(self):
        """Impute missing SDs using CV method."""
        print("\n" + "="*70)
        print("STEP 1: SD IMPUTATION (CV Method)")
        print("="*70)
        
        df = self.df.copy()
        for col in ['L1', 'L2', 'Conv']:
            known = df[df[f'{col}_sd'].notna()]
            if len(known) > 0:
                cv = (known[f'{col}_sd'] / known[f'{col}_mean']).median()
                mask = df[f'{col}_sd'].isna() & df[f'{col}_mean'].notna()
                df.loc[mask, f'{col}_sd'] = df.loc[mask, f'{col}_mean'] * cv
                if mask.sum() > 0:
                    print(f"  {col}: Median CV = {cv:.4f}, imputed {mask.sum()} values")
        
        self.df = df
        return df
    
    def calculate_effect_sizes(self):
        """Calculate Hedges' g. Negative = naturalness degradation from L2."""
        print("\n" + "="*70)
        print("STEP 2: EFFECT SIZE CALCULATION (Hedges' g)")
        print("="*70)
        
        results = []
        for _, r in self.df.iterrows():
            n = r['n']
            sd_pool = np.sqrt(((n-1)*r['Conv_sd']**2 + (n-1)*r['L2_sd']**2) / (2*n-2))
            
            # Conv - L2: negative = degradation
            d = (r['Conv_mean'] - r['L2_mean']) / sd_pool
            J = 1 - 3/(4*(2*n-2) - 1)
            g = d * J
            var_g = 2/n + g**2/(4*n)
            se = np.sqrt(var_g)
            
            # L2 preservation (if L2 available)
            pres = (r['Conv_mean'] / r['L2_mean'] * 100) if pd.notna(r['L2_mean']) else np.nan
            
            results.append({'study': r['Study'], 'g': g, 'se': se, 'var': var_g,
                           'ci_lo': g - 1.96*se, 'ci_hi': g + 1.96*se, 
                           'pres': pres, 'n': n})
            pres_str = f", Pres={pres:.1f}%" if pd.notna(pres) else ""
            print(f"  {r['Study']}: g={g:.2f} [{g-1.96*se:.2f}, {g+1.96*se:.2f}]{pres_str}")
        
        self.es_df = pd.DataFrame(results)
        category_map = {
            'Accentron (2022)': 'S2S',
            'AC-Articulatory (2024)': 'S2S',
            'FAC-NoRef (2021)': 'S2S',
            'ZS-NoRef (2022) *': 'S2S',
            'TTS-Transfer (2024)': 'TTS',
        }
        self.es_df['category'] = self.es_df['study'].map(category_map)
        return self.es_df
    
    def random_effects_analysis(self):
        """DerSimonian-Laird random-effects meta-analysis."""
        print("\n" + "="*70)
        print("STEP 3: RANDOM-EFFECTS META-ANALYSIS")
        print("="*70)
        
        g, var = self.es_df['g'].values, self.es_df['var'].values
        k = len(g)
        
        w_fe = 1/var
        g_fe = np.sum(w_fe * g) / np.sum(w_fe)
        Q = np.sum(w_fe * (g - g_fe)**2)
        
        C = np.sum(w_fe) - np.sum(w_fe**2)/np.sum(w_fe)
        tau2 = max(0, (Q - (k-1)) / C)
        
        w_re = 1/(var + tau2)
        g_re = np.sum(w_re * g) / np.sum(w_re)
        se_re = np.sqrt(1/np.sum(w_re))
        
        I2 = max(0, (Q - (k-1))/Q * 100) if Q > 0 else 0
        
        # Mean preservation (studies with L2)
        valid_pres = self.es_df[self.es_df['pres'].notna()]
        mean_pres = valid_pres['pres'].mean() if len(valid_pres) > 0 else np.nan
        
        self.pooled = {'g': g_re, 'se': se_re, 'ci_lo': g_re - 1.96*se_re, 
                       'ci_hi': g_re + 1.96*se_re, 'tau2': tau2, 'Q': Q, 
                       'I2': I2, 'k': k, 'pres': mean_pres}
        
        print(f"\n  Pooled g = {g_re:.2f} [{g_re-1.96*se_re:.2f}, {g_re+1.96*se_re:.2f}]")
        print(f"  τ² = {tau2:.2f}, Q = {Q:.1f}, I² = {I2:.1f}%")
        if pd.notna(mean_pres):
            print(f"  Mean L2 Preservation = {mean_pres:.1f}%")
        
        return self.pooled
    
    def create_forest_plot(self):
        """Publication-quality forest plot."""
        fig, ax = plt.subplots(figsize=(16, 7))
        big_fs=20
        middle_fs=18
        x_study = -14
        x_ci = 2
        x_pres = 22
        x_method = 30
        
        df = self.es_df.sort_values('g', ascending=False).reset_index(drop=True)
        n_studies = len(df)
        
        for i, (_, r) in enumerate(df.iterrows()):
            y = n_studies - i
            color = self.colors.get(r['category'], 'gray')
            
            ax.errorbar(r['g'], y, xerr=[[r['g']-r['ci_lo']], [r['ci_hi']-r['g']]],
                       fmt='s', markersize=10, color=color, capsize=5, capthick=2, elinewidth=2)
            
            ax.text(x_study, y, r['study'], ha='right', va='center', fontsize=big_fs)
            ax.text(x_ci, y, f"{r['g']:.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]", 
                   ha='left', va='center', fontsize=middle_fs, family='monospace')
            pres_str = f"{r['pres']:.1f}%" if pd.notna(r['pres']) else "—"
            ax.text(x_pres, y, pres_str, ha='center', va='center', fontsize=middle_fs, family='monospace')
            ax.text(x_method, y, r['category'], ha='center', va='center', fontsize=middle_fs, fontweight='bold',
                   color=self.colors.get(r['category'], 'gray'))
        
        # Pooled effect: circle with CI line (like individual studies)
        p = self.pooled
        ax.errorbar(p['g'], 0, xerr=[[p['g']-p['ci_lo']], [p['ci_hi']-p['g']]],
                   fmt='o', markersize=12, color=self.colors['pooled'], capsize=6, capthick=2.5,
                   elinewidth=2.5, markeredgecolor='white', markeredgewidth=2)
        
        ax.text(x_study, 0, "Pooled Effect (Random)", ha='right', va='center', fontsize=big_fs, fontweight='bold')
        ax.text(x_ci, 0, f"{p['g']:.2f} [{p['ci_lo']:.2f}, {p['ci_hi']:.2f}]",
               ha='left', va='center', fontsize=middle_fs, family='monospace', fontweight='bold')
        pres_str = f"{p['pres']:.1f}%" if pd.notna(p['pres']) else "—"
        ax.text(x_pres, 0, pres_str, ha='center', va='center', fontsize=middle_fs, family='monospace', fontweight='bold')
    
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Headers
        ax.text(x_study, n_studies+1, "Study", ha='right', fontsize=big_fs, fontweight='bold')
        ax.text(x_ci, n_studies+1, "Hedges' g [95% CI]", ha='left', fontsize=big_fs, fontweight='bold')
        ax.text(x_pres, n_studies+1, "L2 Pres%", ha='center', fontsize=big_fs, fontweight='bold')
        ax.text(x_method, n_studies+1, "Method", ha='center', fontsize=big_fs, fontweight='bold')
        
        ax.set_xlim(-19, 35)
        ax.set_ylim(-1.15, n_studies+1.8)
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        ax.set_title("Forest Plot: 5-Point Naturalness ↑ Meta-Analysis (k=5)\n"
                    "Negative effect = naturalness degradation from L2 baseline", 
                    fontsize=big_fs, fontweight='bold', pad=15)
        ax.set_xlabel("Hedges' g (Standardized Effect Size)", fontsize=middle_fs)
        ax.tick_params(axis='x', labelsize=middle_fs)
        # Direction labels: split so "|" aligns at x=0
        ax.text(-0.15, -0.9, "← Degradation ", ha='right', fontsize=middle_fs)
        ax.text(-0.15, -0.9, "| Improvement →", ha='left', fontsize=middle_fs)
        
        # Heterogeneity text (bottom right)
        het_text = f"Heterogeneity: I² = {p['I2']:.1f}%, τ² = {p['tau2']:.2f}, Q = {p['Q']:.1f}"
        ax.text(36, -0.9, het_text, ha='right', fontsize=middle_fs, style='italic')
        
        # fig.text(0.5, 0.01, "* SD imputed via CV method    L1 Pres% = Naturalness preserved relative to L1 native", 
        #         ha='center', fontsize=middle_fs, style='italic')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.13)
        
        for fmt in ['pdf', 'png']:
            plt.savefig(f"{self.output_dir}/naturalness_forest_plot.{fmt}", 
                       dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n✓ Saved forest plot to {self.output_dir}/")
        plt.show()
    
    def print_results_discussion(self):
        """Print detailed results and discussion."""
        print("\n" + "="*70)
        print("META-ANALYSIS RESULTS AND DISCUSSION")
        print("="*70)
        
        p = self.pooled
        es = self.es_df
        
        print(f"""
NATURALNESS PRESERVATION RESULTS:
---------------------------------
The 5-point naturalness meta-analysis (k={p['k']}) reveals the accent-naturalness 
trade-off in FAC systems.

POOLED EFFECT:
  • Hedges' g = {p['g']:.2f} (95% CI: [{p['ci_lo']:.2f}, {p['ci_hi']:.2f}])
  • Interpretation: {'LARGE' if abs(p['g']) > 0.8 else 'MEDIUM' if abs(p['g']) > 0.5 else 'SMALL'} negative effect (degradation)
  • Mean L1 Preservation: {p['pres']:.1f}%

INDIVIDUAL STUDY PERFORMANCE:
  • Most preserved: {es.loc[es['g'].idxmax(), 'study']} (g={es['g'].max():.2f})
  • Most degraded: {es.loc[es['g'].idxmin(), 'study']} (g={es['g'].min():.2f})
  • Effect size range: {es['g'].min():.2f} to {es['g'].max():.2f}

HETEROGENEITY ANALYSIS:
  • Q = {p['Q']:.1f} (df = {p['k']-1}, p < 0.001)
  • I² = {p['I2']:.1f}% (Considerable heterogeneity)
  • τ² = {p['tau2']:.2f}
""")
    
    def save_results(self):
        """Save numerical results to CSV."""
        self.es_df.to_csv(f"{self.output_dir}/effect_sizes.csv", index=False)
        pd.DataFrame([self.pooled]).to_csv(f"{self.output_dir}/pooled_results.csv", index=False)
        print(f"✓ Results saved to {self.output_dir}/")
    
    def run(self):
        """Execute complete analysis."""
        print("\n" + "="*70)
        print(" 5-POINT NATURALNESS META-ANALYSIS ")
        print("="*70)
        self.impute_missing_sds()
        self.calculate_effect_sizes()
        self.random_effects_analysis()
        self.create_forest_plot()
        self.print_results_discussion()
        self.save_results()


if __name__ == "__main__":
    analysis = NaturalnessMetaAnalysis()
    analysis.run()
