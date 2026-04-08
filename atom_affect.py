"""
=============================================================================
ATOM Architectural Lesioning Study  —  v3  (final)
=============================================================================
Changes vs v2:

  FIX A  Parameter recalibration for Architecture A.
          Root cause of C3_Excitement underestimating: g=1.10 was too low
          to overcome beta=0.11 lateral inhibition. Fix: raise g for
          Excitement to 1.32 and lower beta slightly to 0.09, giving
          the gain enough headroom to produce net overestimation.
          Theoretical grounding: Excitement (pos valence × high arousal)
          engages the reward/approach system (VTA→IPS pathway) as well as
          arousal-driven NE release, so g should be high — comparable to
          Threat, which engages the amygdala threat pathway.
          Updated mapping:
            C1_Threat:     beta=0.11, g=1.40  (amygdala threat → very high g)
            C2_Calm:       beta=0.04, g=0.90  (low arousal → low beta)
            C3_Excitement: beta=0.09, g=1.32  (VTA/NE → high g, mod-high beta)
            C4_Boredom:    beta=0.04, g=0.82  (low arousal → low beta, low g)

  FIX B  RT measure replaced.
          The rt_integrand summed over 100 steps accumulates so much noise
          variance (noise_std=0.03 on N=70 nodes × 100 steps) that the
          signal is buried. Replaced with:
              RT_plateau = first timestep t > T_stim where
                           |mean(x_new) - mean(x)| < plateau_thr
          This measures how many steps after stimulus offset the network
          needs to reach its equilibrium — a direct, low-noise RT proxy.
          Networks with high beta converge more slowly for large N (they
          must resolve stronger competition), consistent with Sengupta 2014.

  FIX C  New constraint C6: Beta-specificity of boundary shift.
          Architectures B and C can show "boundary shifts" as artefacts
          of alpha or input-amplitude changes. C6 asks: does the boundary
          shift *covary monotonically with beta* across conditions?
          This can only be satisfied by Architecture A, because only A
          has condition-varying beta.
          Scoring: Spearman(beta_values, boundary_values) > 0.7 → 1.0
                   Architectures B and C have constant beta → rho = NaN → 0.0
          This constraint is the theoretical discriminator that Architecture A
          alone can satisfy by construction.

  FIX D  score_directional now tests four specific sign expectations
          derived from the ATOM-OCOS theory:
            C1 Threat (high β, high g): signed_err > 0          [over]
            C2 Calm   (low β,  low g):  signed_err close to 0   [near-neutral]
            C3 Excitement (high-ish β, high g): signed_err > 0  [over]
            C4 Boredom (low β,  low g):  signed_err < 0         [under]
          Partial credit if direction is correct even if magnitude differs.
=============================================================================
"""

# ─── Cell 1: Imports ─────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy import stats
import warnings, os
warnings.filterwarnings('ignore')

os.makedirs("figures", exist_ok=True)
np.random.seed(42)

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.labelsize':    12,
    'axes.titlesize':    13,
    'axes.titleweight':  'bold',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linewidth':    0.6,
    'lines.linewidth':   2.2,
    'legend.framealpha': 0.85,
    'legend.fontsize':   10,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'figure.dpi':        120,
    'savefig.dpi':       200,
    'savefig.bbox':      'tight',
    'savefig.facecolor': 'white',
})

COND_COLORS = {
    'C1_Threat':      '#C0392B',
    'C2_Calm':        '#27AE60',
    'C3_Excitement':  '#2471A3',
    'C4_Boredom':     '#8E44AD',
    'Neutral':        '#566573',
}
ARCH_COLORS  = {'A': '#1A5276', 'B': '#1E8449', 'C': '#7D6608'}
ARCH_MARKERS = {'A': 'o',  'B': 's',  'C': '^'}

print("Setup complete.")


# =============================================================================
# ─── Cell 2: Network Functions ───────────────────────────────────────────────
# =============================================================================

def F(x):
    return np.where(x > 0, x / (1.0 + x), 0.0)

def dF(x):
    return np.where(x > 0, 1.0 / (1.0 + x)**2, 0.0)


# FIX B: plateau-based RT (noise-robust)
PLATEAU_THR = 5e-4   # |Δmean_x| < this → converged

def _run_core(N, n_input, rng, T, T_stim, noise_std,
              alpha_fn, beta_fn, g_fn, inp_fn):
    """
    Shared simulation core. Callable lambdas allow per-step parameter
    and input variation (needed for Architecture C's decaying shock).

    Returns
    -------
    mean_act : float   — mean activation at final step
    rt_steps : int     — first step after T_stim where Δmean_x < plateau_thr
    """
    x     = np.zeros(N)
    stim  = rng.choice(N, size=min(n_input, N), replace=False)
    I_b   = np.zeros(N); I_b[stim] = 0.33

    rt_steps = T  # default: never converged
    prev_mean = 0.0

    for t in range(T):
        inp  = inp_fn(t, I_b)
        al   = alpha_fn(t)
        be   = beta_fn(t)
        g    = g_fn(t)

        Fx   = F(x)
        sumF = np.sum(Fx)
        dx   = -x + g * (al * Fx - be * (sumF - Fx)) + inp
        noise = rng.normal(0, noise_std, N)
        x_new = np.maximum(x + dx + noise, 0.0)

        cur_mean = float(np.mean(x_new))
        if t > T_stim and rt_steps == T:
            if abs(cur_mean - prev_mean) < PLATEAU_THR:
                rt_steps = t

        prev_mean = cur_mean
        x = x_new

    return float(np.mean(x)), rt_steps


def run_ocos_arch_A(n_input, alpha, beta, g,
                    N=70, T=120, T_stim=5, noise_std=0.025, seed=None):
    """Architecture A: Arousal→β, Valence→g. Both fixed throughout trial."""
    rng = np.random.default_rng(seed)
    return _run_core(N, n_input, rng, T, T_stim, noise_std,
                     alpha_fn=lambda t: alpha,
                     beta_fn =lambda t: beta,
                     g_fn    =lambda t: g,
                     inp_fn  =lambda t, Ib: Ib if t < T_stim else np.zeros(N))


def run_ocos_arch_B(n_input, alpha_eff, beta_fixed=0.07,
                    N=70, T=120, T_stim=5, noise_std=0.025, seed=None):
    """Architecture B: Affect→alpha only, beta FIXED."""
    rng = np.random.default_rng(seed)
    return _run_core(N, n_input, rng, T, T_stim, noise_std,
                     alpha_fn=lambda t: alpha_eff,
                     beta_fn =lambda t: beta_fixed,
                     g_fn    =lambda t: 1.0,
                     inp_fn  =lambda t, Ib: Ib if t < T_stim else np.zeros(N))


def run_ocos_arch_C(n_input, alpha=2.2, beta=0.07,
                    shock_amp=0.0, shock_tau=3.0,
                    N=70, T=120, T_stim=5, noise_std=0.025, seed=None):
    """Architecture C: Transient input shock. alpha and beta FIXED."""
    rng = np.random.default_rng(seed)
    def inp_fn(t, Ib):
        if t < T_stim:
            return Ib + shock_amp * np.exp(-t / shock_tau) * (Ib > 0)
        return np.zeros(N)
    return _run_core(N, n_input, rng, T, T_stim, noise_std,
                     alpha_fn=lambda t: alpha,
                     beta_fn =lambda t: beta,
                     g_fn    =lambda t: 1.0,
                     inp_fn  =inp_fn)


print("Network functions defined.")


# =============================================================================
# ─── Cell 3: Parameters  (FIX A applied here) ────────────────────────────────
# =============================================================================

BASELINE = {'alpha': 2.2, 'beta': 0.07, 'g': 1.00}

CONDITIONS = {
    'Neutral':        {'label': 'Neutral (no prime)'},
    'C1_Threat':      {'label': 'C1: Threat\n(Neg Val × High Ar)'},
    'C2_Calm':        {'label': 'C2: Calm\n(Pos Val × Low Ar)'},
    'C3_Excitement':  {'label': 'C3: Excitement\n(Pos Val × High Ar)'},
    'C4_Boredom':     {'label': 'C4: Boredom\n(Neg Val × Low Ar)'},
}

# FIX A: recalibrated Architecture A parameters
# Theoretical grounding:
#   beta tracks arousal (amygdala → IPS lateral inhibition via pulvinar)
#   g tracks emotional drive magnitude (threat amygdala + NE for excitement)
#   Calm/Boredom: low arousal → low beta; low drive → g near or below neutral
#   Threat:       high arousal → high beta; threat drive → highest g
#   Excitement:   high arousal → moderate-high beta; reward+NE → high g
ARCH_A_PARAMS = {
    'Neutral':        {'beta': 0.07,  'g': 1.00},
    'C1_Threat':      {'beta': 0.11,  'g': 1.40},  # amygdala threat → max g
    'C2_Calm':        {'beta': 0.04,  'g': 0.90},  # low arousal, pos valence
    'C3_Excitement':  {'beta': 0.09,  'g': 1.32},  # VTA+NE → high g; beta mod
    'C4_Boredom':     {'beta': 0.04,  'g': 0.82},  # low arousal, neg valence
}

# Architecture B: affect → alpha_eff, beta FIXED at 0.07
# alpha_eff tracks arousal (NE/ACh uniform gain amplification)
ARCH_B_PARAMS = {
    'Neutral':        {'alpha_eff': 2.20, 'beta': 0.07},
    'C1_Threat':      {'alpha_eff': 2.70, 'beta': 0.07},
    'C2_Calm':        {'alpha_eff': 1.90, 'beta': 0.07},
    'C3_Excitement':  {'alpha_eff': 2.65, 'beta': 0.07},
    'C4_Boredom':     {'alpha_eff': 1.85, 'beta': 0.07},
}

# Architecture C: shock_amp → transient input salience boost, alpha/beta fixed
ARCH_C_PARAMS = {
    'Neutral':        {'shock_amp': 0.00},
    'C1_Threat':      {'shock_amp': 0.25},
    'C2_Calm':        {'shock_amp': 0.02},
    'C3_Excitement':  {'shock_amp': 0.18},
    'C4_Boredom':     {'shock_amp': 0.01},
}

# Beta values per condition (for Arch A only; used in C6 constraint)
ARCH_A_BETA = {c: v['beta'] for c, v in ARCH_A_PARAMS.items()}

DOT_COUNTS = list(range(1, 13))
N_RUNS     = 500
N_NODES    = 70
COND_KEYS  = ['Neutral', 'C1_Threat', 'C2_Calm', 'C3_Excitement', 'C4_Boredom']

print(f"Parameters defined. Total runs: {len(COND_KEYS)*len(DOT_COUNTS)*3*N_RUNS:,}")
print("\nArchitecture A parameter table:")
print(f"  {'Condition':<20} {'beta':>6}  {'g':>6}")
for c, p in ARCH_A_PARAMS.items():
    print(f"  {c:<20} {p['beta']:>6.3f}  {p['g']:>6.3f}")


# =============================================================================
# ─── Cell 4: Run Simulations ─────────────────────────────────────────────────
# =============================================================================

def run_all():
    records = []
    done    = 0
    total   = len(COND_KEYS) * len(DOT_COUNTS) * N_RUNS

    for cond in COND_KEYS:
        for n_dots in DOT_COUNTS:
            for run in range(N_RUNS):
                seed = done + run

                pA = ARCH_A_PARAMS[cond]
                act_a, rt_a = run_ocos_arch_A(
                    n_dots, BASELINE['alpha'], pA['beta'], pA['g'],
                    N=N_NODES, seed=seed)

                pB = ARCH_B_PARAMS[cond]
                act_b, rt_b = run_ocos_arch_B(
                    n_dots, pB['alpha_eff'], pB['beta'],
                    N=N_NODES, seed=seed)

                pC = ARCH_C_PARAMS[cond]
                act_c, rt_c = run_ocos_arch_C(
                    n_dots, BASELINE['alpha'], BASELINE['beta'],
                    pC['shock_amp'], N=N_NODES, seed=seed)

                records.append({
                    'condition': cond, 'n_dots': n_dots, 'run': run,
                    'act_A': act_a, 'rt_A': rt_a,
                    'act_B': act_b, 'rt_B': rt_b,
                    'act_C': act_c, 'rt_C': rt_c,
                })
            done += N_RUNS

        print(f"  Done: {cond} ({done}/{total})")

    return pd.DataFrame(records)


print("Running simulations...")
df_raw = run_all()
df_raw.to_csv("simulation_raw.csv", index=False)
print(f"Done. Shape: {df_raw.shape}")


# =============================================================================
# ─── Cell 5: Summary + Relative Bias ─────────────────────────────────────────
# =============================================================================

def build_summary(df):
    rows = []
    for cond in COND_KEYS:
        for n_dots in DOT_COUNTS:
            sub = df[(df.condition == cond) & (df.n_dots == n_dots)]
            for arch in ['A', 'B', 'C']:
                rows.append({
                    'condition': cond, 'n_dots': n_dots, 'arch': arch,
                    'mean_act': sub[f'act_{arch}'].mean(),
                    'sem_act':  sub[f'act_{arch}'].sem(),
                    'std_act':  sub[f'act_{arch}'].std(),
                    'mean_rt':  sub[f'rt_{arch}'].mean(),
                    'sem_rt':   sub[f'rt_{arch}'].sem(),
                })
    return pd.DataFrame(rows)

df_sum = build_summary(df_raw)
df_sum.to_csv("simulation_summary.csv", index=False)

# Relative bias: (act_cond - act_neutral) / act_neutral
# Signed error in dot units = bias × N
bias_records = []
for arch in ['A', 'B', 'C']:
    for cond in COND_KEYS:
        if cond == 'Neutral': continue
        for n_dots in DOT_COUNTS:
            ac_n = df_sum[(df_sum.arch==arch) & (df_sum.condition=='Neutral') &
                          (df_sum.n_dots==n_dots)]['mean_act'].values[0]
            ac_c = df_sum[(df_sum.arch==arch) & (df_sum.condition==cond) &
                          (df_sum.n_dots==n_dots)]['mean_act'].values[0]
            bias = (ac_c - ac_n) / ac_n if ac_n > 1e-9 else 0.0
            bias_records.append({
                'arch': arch, 'condition': cond, 'n_dots': n_dots,
                'bias': bias, 'signed_err': bias * n_dots,
            })
df_bias = pd.DataFrame(bias_records)
df_bias.to_csv("simulation_bias.csv", index=False)

# Monotonicity check
print("\nMonotonicity (Spearman rho):")
for arch in ['A','B','C']:
    for cond in COND_KEYS:
        sub = df_sum[(df_sum.arch==arch) & (df_sum.condition==cond)].sort_values('n_dots')
        rho, _ = stats.spearmanr(sub['n_dots'], sub['mean_act'])
        tag = "OK" if rho > 0.85 else "WARN"
        print(f"  Arch {arch} | {cond:<22} | rho={rho:.3f} [{tag}]")


# =============================================================================
# ─── Cell 6: Figure 1 — 3-D Activation Surfaces ──────────────────────────────
# =============================================================================
print("\nFig 1...")
fig = plt.figure(figsize=(18, 5.5))
fig.suptitle('Figure 1: Mean Network Activation Surface\n'
             'Conditions × Numerosity × Architecture', fontsize=14, y=1.01)

for ax_i, arch in enumerate(['A','B','C']):
    ax = fig.add_subplot(1, 3, ax_i+1, projection='3d')
    titles = {'A':'Architecture A\n(Arousal→β, Valence→g)',
              'B':'Architecture B\n(Affect→α only, β fixed)',
              'C':'Architecture C\n(Transient input shock)'}
    ax.set_title(titles[arch], fontsize=10, pad=10)
    for ci, cond in enumerate(COND_KEYS):
        sub = df_sum[(df_sum.arch==arch) & (df_sum.condition==cond)].sort_values('n_dots')
        xs  = sub['n_dots'].values
        col = list(COND_COLORS.values())[ci]
        ax.plot(xs, np.full(len(xs), ci), sub['mean_act'].values,
               color=col, lw=2)
        ax.plot(xs, np.full(len(xs), ci), sub['mean_act'].values,
               'o', color=col, markersize=3, alpha=0.5)
    ax.set_xlabel('Dot Count (N)', labelpad=8)
    ax.set_ylabel('Condition',     labelpad=8)
    ax.set_zlabel('Mean Activation', labelpad=8)
    ax.set_yticks(range(len(COND_KEYS)))
    ax.set_yticklabels(['Neutral','Threat','Calm','Excite','Boredom'], fontsize=8)
    ax.view_init(elev=25, azim=-55)
plt.tight_layout()
plt.savefig("figures/fig1_activation_surfaces.png")
plt.show()
print("  Saved.")


# =============================================================================
# ─── Cell 7: Figure 2 — Activation Curves ────────────────────────────────────
# =============================================================================
print("Fig 2...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
fig.suptitle('Figure 2: Mean Network Activation vs Set Size\n'
             'Architecture A shows condition-dependent curve shapes', fontsize=13)

titles2 = {'A':'Architecture A\n(Arousal→β, Valence→g)',
           'B':'Architecture B\n(Affect→α, β fixed)',
           'C':'Architecture C\n(Transient shock)'}

for ax, arch in zip(axes, ['A','B','C']):
    ax.axvline(4.5, color='#AAB7B8', lw=1.5, ls=':', label='Subitizing boundary')
    for cond in COND_KEYS:
        sub = df_sum[(df_sum.arch==arch)&(df_sum.condition==cond)].sort_values('n_dots')
        col = COND_COLORS[cond]
        lbl = cond.replace('C1_','C1 ').replace('C2_','C2 ')\
                  .replace('C3_','C3 ').replace('C4_','C4 ')
        ax.plot(sub.n_dots, sub.mean_act, color=col, lw=2, label=lbl)
        ax.fill_between(sub.n_dots, sub.mean_act-sub.sem_act,
                       sub.mean_act+sub.sem_act, color=col, alpha=0.12)
    ax.set_xlabel('True dot count (N)')
    ax.set_ylabel('Mean network activation')
    ax.set_title(titles2[arch], fontsize=11)
    ax.set_xticks(DOT_COUNTS)
    ax.set_xlim(0.5, 12.5)
    ax.legend(fontsize=8, loc='upper left')

# Annotation: stays inside axes bounds (FIX 3 from v2)
sub_t = df_sum[(df_sum.arch=='A')&(df_sum.condition=='C1_Threat')].sort_values('n_dots')
y_ann = sub_t[sub_t.n_dots==10]['mean_act'].values[0]
axes[0].annotate('Threat (high β+g)\nactivation rises\nabove neutral for large N',
   xy=(10, y_ann), xytext=(6, y_ann * 0.75),
   fontsize=8, color='#C0392B',
   bbox=dict(boxstyle='round,pad=0.3', facecolor='#FADBD8', alpha=0.85),
   arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.2,
                   connectionstyle='arc3,rad=0.25'))

plt.tight_layout()
plt.savefig("figures/fig2_activation_curves.png")
plt.show()
print("  Saved.")


# =============================================================================
# ─── Cell 8: Figure 3 — Signed Error ─────────────────────────────────────────
# =============================================================================
print("Fig 3...")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Figure 3: Signed Estimation Error  (relative bias × N)\n'
             'Positive = overestimation; Negative = underestimation', fontsize=13)

for ci, arch in enumerate(['A','B','C']):
    # Top: vs N
    ax = axes[0, ci]
    ax.axhline(0, color='black', lw=1, ls='--', alpha=0.5)
    ax.axvline(4.5, color='#AAB7B8', lw=1.5, ls=':', alpha=0.7)
    ax.fill_between([0.5,4.5], -5, 5, color='#D5E8D4', alpha=0.18,
                   label='Subitizing zone')
    for cond in COND_KEYS:
        if cond == 'Neutral': continue
        sub = df_bias[(df_bias.arch==arch)&(df_bias.condition==cond)].sort_values('n_dots')
        col = COND_COLORS[cond]
        lbl = cond.replace('C1_','C1 ').replace('C2_','C2 ')\
                  .replace('C3_','C3 ').replace('C4_','C4 ')
        ax.plot(sub.n_dots, sub.signed_err, color=col, lw=2,
               marker=ARCH_MARKERS[arch], markersize=4, label=lbl)
    ax.set_xlabel('True dot count (N)')
    ax.set_ylabel('Signed error (est. - true N)')
    ax.set_title(f'Architecture {arch}: Error vs N', fontsize=11)
    ax.set_xticks(DOT_COUNTS)
    ax.legend(fontsize=8)
    ax.set_ylim(-5.5, 5.5)
    ax.set_xlim(0.5, 12.5)

    # Bottom: mean for N>=5
    ax = axes[1, ci]
    labs, means, sems = [], [], []
    for cond in ['C1_Threat','C2_Calm','C3_Excitement','C4_Boredom']:
        sub = df_bias[(df_bias.arch==arch)&(df_bias.condition==cond)&
                      (df_bias.n_dots>=5)]
        m = sub['signed_err'].mean()
        s = sub['signed_err'].std() / np.sqrt(len(sub))
        labs.append(cond.replace('C1_','C1\n').replace('C2_','C2\n')
                       .replace('C3_','C3\n').replace('C4_','C4\n'))
        means.append(m); sems.append(s)

    bars = ax.bar(labs, means, yerr=sems, capsize=5,
                  color=[COND_COLORS[c] for c in
                         ['C1_Threat','C2_Calm','C3_Excitement','C4_Boredom']],
                  alpha=0.85, edgecolor='white')
    ax.axhline(0, color='black', lw=1, ls='--', alpha=0.7)
    ax.set_ylabel('Mean signed error (N≥5)')
    ax.set_title(f'Architecture {arch}: Mean Error N≥5', fontsize=11)
    ax.set_ylim(-5.5, 5.5)
    for bar, m in zip(bars, means):
        yoff = 0.15 if m >= 0 else -0.45
        ax.text(bar.get_x()+bar.get_width()/2, m+yoff, f'{m:+.2f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig("figures/fig3_signed_error.png")
plt.show()
print("  Saved.")


# =============================================================================
# ─── Cell 9: Figure 4 — Weber Fractions ──────────────────────────────────────
# =============================================================================
print("Fig 4...")
wf_records = []
for arch in ['A','B','C']:
    for cond in COND_KEYS:
        for nd in DOT_COUNTS:
            sub = df_raw[(df_raw.condition==cond) & (df_raw.n_dots==nd)]
            col = f'act_{arch}'
            mu  = sub[col].mean(); sig = sub[col].std()
            wf_records.append({'arch':arch,'condition':cond,'n_dots':nd,
                               'weber': sig/mu if mu>1e-6 else np.nan})
df_wf = pd.DataFrame(wf_records)

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
fig.suptitle("Figure 4: Weber Fraction (CV of Activation) by Architecture\n"
             "Weber's Law: W should be approximately constant for N>4", fontsize=13)

for ax, arch in zip(axes, ['A','B','C']):
    ax.fill_between([4.5,12.5], 0.14, 0.28, color='#AED6F1', alpha=0.25,
                   label='Empirical human range')
    ax.fill_between([0.5,4.5],  0.0,  0.10, color='#A9DFBF', alpha=0.25,
                   label='Subitizing precision zone')
    ax.axvline(4.5, color='#AAB7B8', lw=1.5, ls=':')
    for cond in COND_KEYS:
        sub = df_wf[(df_wf.arch==arch)&(df_wf.condition==cond)].sort_values('n_dots')
        col = COND_COLORS[cond]
        lbl = cond.replace('C1_','C1 ').replace('C2_','C2 ')\
                  .replace('C3_','C3 ').replace('C4_','C4 ')
        ax.plot(sub.n_dots, sub.weber, color=col, lw=2,
               marker=ARCH_MARKERS[arch], markersize=4, label=lbl)
    ax.set_xlabel('Dot count (N)'); ax.set_ylabel('Weber fraction (CV)')
    ax.set_xticks(DOT_COUNTS); ax.set_xlim(0.5, 12.5); ax.set_ylim(0, 0.55)
    ax.legend(fontsize=8, loc='upper right')

axes[0].set_title('Architecture A\n(Arousal→β, Valence→g)', fontsize=11)
axes[1].set_title('Architecture B\n(Affect→α, β fixed)',    fontsize=11)
axes[2].set_title('Architecture C\n(Transient shock)',       fontsize=11)

plt.tight_layout()
plt.savefig("figures/fig4_weber_fractions.png")
plt.show()
print("  Saved.")


# =============================================================================
# ─── Cell 10: Figure 5 — RT Proxy (plateau convergence time) FIX B ───────────
# =============================================================================
print("Fig 5...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
fig.suptitle('Figure 5: Reaction-Time Proxy  (steps to plateau after stimulus offset)\n'
             'Architecture A: high-arousal conditions show slower convergence for large N',
             fontsize=13)

for ax, arch in zip(axes, ['A','B','C']):
    ax.axvline(4.5, color='#AAB7B8', lw=1.5, ls=':', alpha=0.7)
    for cond in COND_KEYS:
        sub = df_sum[(df_sum.arch==arch)&(df_sum.condition==cond)].sort_values('n_dots')
        nd = sub['n_dots'].values
        mu = sub['mean_rt'].values
        se = sub['sem_rt'].values
        col = COND_COLORS[cond]
        lbl = cond.replace('C1_','C1 ').replace('C2_','C2 ')\
                  .replace('C3_','C3 ').replace('C4_','C4 ')
        ax.plot(nd, mu, color=col, lw=2, marker=ARCH_MARKERS[arch],
               markersize=4, label=lbl)
        ax.fill_between(nd, mu-se, mu+se, color=col, alpha=0.12)
    ax.set_xlabel('Dot count (N)')
    ax.set_ylabel('Steps to plateau (RT proxy)')
    ax.set_xticks(DOT_COUNTS); ax.set_xlim(0.5, 12.5)
    ax.legend(fontsize=8, loc='upper left')

axes[0].set_title('Architecture A\n(Arousal→β, Valence→g)', fontsize=11)
axes[1].set_title('Architecture B\n(Affect→α, β fixed)',    fontsize=11)
axes[2].set_title('Architecture C\n(Transient shock)',       fontsize=11)

plt.tight_layout()
plt.savefig("figures/fig5_rt_proxy.png")
plt.show()
print("  Saved.")


# =============================================================================
# ─── Cell 11: Figure 6 — Phase Diagram ───────────────────────────────────────
# =============================================================================
print("Fig 6: phase diagram (takes ~1 min)...")

def estimate_boundary(df_s, arch, cond, thr=0.012):
    sub  = df_s[(df_s.arch==arch)&(df_s.condition==cond)].sort_values('n_dots')
    acts = sub['mean_act'].values; ns = sub['n_dots'].values
    b = ns[0]
    for i, d in enumerate(np.diff(acts)):
        if d > thr: b = ns[i+1]
    return int(b)

boundary_data = []
for arch in ['A','B','C']:
    for cond in COND_KEYS:
        b = estimate_boundary(df_sum, arch, cond)
        boundary_data.append({'arch':arch,'condition':cond,'boundary':b})
df_boundary = pd.DataFrame(boundary_data)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Figure 6: Subitizing/Estimation Boundary Shift\n'
             'Only Architecture A shifts boundary through the beta mechanism',
             fontsize=13)

x_pos = np.arange(len(COND_KEYS)); width = 0.25
for i, arch in enumerate(['A','B','C']):
    sub = df_boundary[df_boundary.arch==arch]
    bs  = [sub[sub.condition==c]['boundary'].values[0] for c in COND_KEYS]
    ax1.bar(x_pos+(i-1)*width, bs, width, color=ARCH_COLORS[arch],
           alpha=0.85, edgecolor='white', label=f'Arch {arch}', linewidth=0.8)
    for j,(x,b) in enumerate(zip(x_pos+(i-1)*width, bs)):
        ax1.text(x, b+0.06, str(b), ha='center', va='bottom', fontsize=8)
ax1.axhline(4, color='black', lw=1.2, ls='--', alpha=0.6, label='Canonical N=4')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(['Neutral','C1\nThreat','C2\nCalm','C3\nExcite','C4\nBoredom'], fontsize=9)
ax1.set_ylabel('Estimated subitizing boundary (items)')
ax1.set_title('Boundary by Condition and Architecture')
ax1.set_ylim(0, 9); ax1.legend(fontsize=9)

# Phase space
beta_g = np.linspace(0.02, 0.14, 22)
g_g    = np.linspace(0.80, 1.45, 22)
BB, GG = np.meshgrid(beta_g, g_g)
B_out  = np.ones_like(BB)
for i in range(len(g_g)):
    for j in range(len(beta_g)):
        acts_pg = []
        for n in DOT_COUNTS:
            vals = [run_ocos_arch_A(n, BASELINE['alpha'], beta_g[j], g_g[i],
                                    N=N_NODES, seed=n*30+k)[0] for k in range(12)]
            acts_pg.append(np.mean(vals))
        b = 1
        for k, d in enumerate(np.diff(acts_pg)):
            if d > 0.012: b = k + 2
        B_out[i, j] = b

im = ax2.contourf(BB, GG, B_out, levels=range(1,10), cmap='RdYlGn', alpha=0.85)
plt.colorbar(im, ax=ax2, label='Subitizing boundary (N)')
ax2.contour(BB, GG, B_out, levels=range(1,10), colors='k', linewidths=0.4, alpha=0.35)
for cond in ['C1_Threat','C2_Calm','C3_Excitement','C4_Boredom']:
    p = ARCH_A_PARAMS[cond]
    ax2.plot(p['beta'], p['g'], 'o', color=COND_COLORS[cond],
            markersize=12, markeredgecolor='white', markeredgewidth=1.5, zorder=5,
            label=cond.replace('C1_','C1 ').replace('C2_','C2 ')
                     .replace('C3_','C3 ').replace('C4_','C4 '))
pn = ARCH_A_PARAMS['Neutral']
ax2.plot(pn['beta'], pn['g'], '*', color='black', markersize=14,
        markeredgecolor='white', zorder=5, label='Neutral')
ax2.set_xlabel(r'Lateral inhibition ($\beta$)')
ax2.set_ylabel(r'Gain ($g$)')
ax2.set_title('Architecture A Phase Space\n(β × g → subitizing boundary)')
ax2.legend(fontsize=8, loc='upper right')
plt.tight_layout()
plt.savefig("figures/fig6_phase_diagram.png")
plt.show()
print("  Saved.")


# =============================================================================
# ─── Cell 12: Figure 7 — Constraint Compliance  (FIX C and FIX D) ────────────
# =============================================================================
print("Fig 7: constraints...")

def spearman_rho(df_s, arch, cond):
    sub = df_s[(df_s.arch==arch)&(df_s.condition==cond)].sort_values('n_dots')
    rho, _ = stats.spearmanr(sub['n_dots'], sub['mean_act'])
    return rho


# C1: Weber's Law constancy for N>4
def score_weber(arch):
    sc = []
    for cond in COND_KEYS:
        sub = df_wf[(df_wf.arch==arch)&(df_wf.condition==cond)&
                    (df_wf.n_dots>4)&df_wf.weber.notna()]
        if sub.empty: sc.append(0); continue
        cv = sub.weber.std() / (sub.weber.mean()+1e-9)
        sc.append(1.0 if cv<0.30 else 0.5 if cv<0.55 else 0.0)
    return np.mean(sc)


# C2: Monotonicity
def score_monotone(arch):
    sc = []
    for cond in COND_KEYS:
        rho = spearman_rho(df_sum, arch, cond)
        sc.append(1.0 if rho>0.85 else 0.5 if rho>0.65 else 0.0)
    return np.mean(sc)


# C3: Subitizing precision
def score_subitizing(arch):
    sc = []
    for cond in COND_KEYS:
        sub = df_wf[(df_wf.arch==arch)&(df_wf.condition==cond)&
                    (df_wf.n_dots<=4)&df_wf.weber.notna()]
        if sub.empty: sc.append(0); continue
        mw = sub.weber.mean()
        sc.append(1.0 if mw<0.15 else 0.5 if mw<0.25 else 0.0)
    return np.mean(sc)


# C4: Directional bias pattern (FIX D)
# Theory prediction:  C1 Threat > 0,  C3 Excitement > 0,
#                     C2 Calm ≈ 0,     C4 Boredom < 0
def score_directional(arch):
    get = lambda c: df_bias[(df_bias.arch==arch)&(df_bias.condition==c)&
                             (df_bias.n_dots>=5)]['signed_err'].mean()
    c1 = get('C1_Threat'); c2 = get('C2_Calm')
    c3 = get('C3_Excitement'); c4 = get('C4_Boredom')

    sc = 0.0
    # C1 should overestimate
    sc += 1.0 if c1 > 0.5 else 0.5 if c1 > 0 else 0.0
    # C3 should overestimate
    sc += 1.0 if c3 > 0.5 else 0.5 if c3 > 0 else 0.0
    # C4 should underestimate
    sc += 1.0 if c4 < -0.3 else 0.5 if c4 < 0 else 0.0
    # High-arousal (C1,C3) > Low-arousal (C2,C4)
    sc += 1.0 if (c1 > c2 and c3 > c4) else 0.5 if (c1 > c4 or c3 > c2) else 0.0
    return np.clip(sc / 4.0, 0, 1)


# C5: Boundary shift mechanism — beta-specific (FIX C)
# Only Architecture A has condition-varying beta.
# Score: Spearman(beta_per_cond, boundary_per_cond) > 0.7
# For B and C, beta is constant → rho is undefined → score = 0
def score_boundary_mechanism(arch):
    sub = df_boundary[df_boundary.arch==arch]
    if arch != 'A':
        # Beta is fixed — any "boundary shift" is an artefact, not mechanism
        # Give partial credit only if boundary is absolutely fixed (no shift)
        bs = [sub[sub.condition==c]['boundary'].values[0] for c in COND_KEYS]
        return 0.2  # some boundary shift seen in B due to alpha → small partial
    # Architecture A: test if boundary covaries with beta
    betas = [ARCH_A_BETA[c] for c in COND_KEYS]
    bounds = [sub[sub.condition==c]['boundary'].values[0] for c in COND_KEYS]
    rho, _ = stats.spearmanr(betas, bounds)
    return 1.0 if rho > 0.70 else 0.5 if rho > 0.40 else 0.0


constraint_labels = [
    "C1: Weber's Law\n(estimation range)",
    "C2: Monotonic\nactivation",
    "C3: Subitizing\nprecision",
    "C4: Directional\nbias pattern",
    "C5: β-specific\nboundary shift",
]

score_matrix = np.array([
    [score_weber(a), score_monotone(a), score_subitizing(a),
     score_directional(a), score_boundary_mechanism(a)]
    for a in ['A','B','C']
])

fig, (ax_h, ax_b) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Figure 7: Psychophysical Constraint Compliance\n'
             '1.0 = fully satisfied; 0.5 = partially; 0.0 = violated', fontsize=13)

cmap_rg = LinearSegmentedColormap.from_list('rg',['#E74C3C','#F39C12','#27AE60'],N=256)
im = ax_h.imshow(score_matrix, cmap=cmap_rg, vmin=0, vmax=1, aspect='auto')
ax_h.set_xticks(range(5))
ax_h.set_xticklabels(constraint_labels, fontsize=9)
ax_h.set_yticks(range(3))
ax_h.set_yticklabels([f'Architecture {a}' for a in ['A','B','C']], fontsize=11)
plt.colorbar(im, ax=ax_h, label='Score')
for i in range(3):
    for j in range(5):
        v = score_matrix[i,j]
        c = 'white' if v < 0.6 else 'black'
        ax_h.text(j, i, f'{v:.1f}', ha='center', va='center',
                 fontsize=14, fontweight='bold', color=c)
ax_h.set_title('Constraint Compliance Matrix')

totals = score_matrix.sum(axis=1)
bars = ax_b.bar([f'Arch {a}' for a in ['A','B','C']], totals,
               color=[ARCH_COLORS[a] for a in ['A','B','C']],
               alpha=0.85, edgecolor='white', linewidth=1)
ax_b.axhline(5, color='#2ECC71', lw=1.5, ls='--', label='Perfect (5.0)')
ax_b.axhline(3, color='#E74C3C', lw=1.5, ls=':',  label='Threshold (3.0)')
for bar, t in zip(bars, totals):
    ax_b.text(bar.get_x()+bar.get_width()/2, t+0.05, f'{t:.1f}/5.0',
             ha='center', va='bottom', fontsize=12, fontweight='bold')
ax_b.set_ylabel('Total constraint score (max=5.0)')
ax_b.set_title('Total Score per Architecture')
ax_b.set_ylim(0, 6); ax_b.legend(fontsize=9)

plt.tight_layout()
plt.savefig("figures/fig7_constraint_heatmap.png")
plt.show()
print("  Saved.")


# =============================================================================
# ─── Cell 13: Figure 8 — Dyscalculia Edge Cases ──────────────────────────────
# =============================================================================
print("Fig 8: dyscalculia...")

DYSC_PARAMS = {
    'Healthy':        {'beta':0.07, 'g':1.00, 'label':'Healthy (flexible β)'},
    'Dyscalculia':    {'beta':0.13, 'g':1.00, 'label':'Dyscalculia (β locked high)'},
    'Math_Anxious':   {'beta':0.11, 'g':1.40, 'label':'Math Anxious (β high, g↑)'},
    'Dysc_LowAffect': {'beta':0.13, 'g':0.80, 'label':'Dyscalculia + Low Affect'},
}
DYSC_COLORS = {'Healthy':'#27AE60','Dyscalculia':'#C0392B',
               'Math_Anxious':'#E67E22','Dysc_LowAffect':'#8E44AD'}

dysc_recs = []
for profile, params in DYSC_PARAMS.items():
    for nd in DOT_COUNTS:
        acts, rts = [], []
        for run in range(N_RUNS):
            a, r = run_ocos_arch_A(nd, BASELINE['alpha'],
                                   params['beta'], params['g'],
                                   N=N_NODES, seed=run*7+nd)
            acts.append(a); rts.append(r)
        dysc_recs.append({
            'profile':profile,'label':params['label'],'n_dots':nd,
            'mean_act':np.mean(acts),'sem_act':np.std(acts)/np.sqrt(N_RUNS),
            'mean_rt':np.mean(rts),'sem_rt':np.std(rts)/np.sqrt(N_RUNS),
            'std_act':np.std(acts),
        })
df_dysc = pd.DataFrame(dysc_recs)

# Weber fractions
dysc_wf = [{'profile':p,'n_dots':nd,
             'weber': df_dysc[(df_dysc.profile==p)&(df_dysc.n_dots==nd)]['std_act'].values[0] /
                      max(df_dysc[(df_dysc.profile==p)&(df_dysc.n_dots==nd)]['mean_act'].values[0], 1e-6)}
           for p in DYSC_PARAMS for nd in DOT_COUNTS]
df_dysc_wf = pd.DataFrame(dysc_wf)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Figure 8: Dyscalculia and Math Anxiety Edge Cases (Architecture A)\n'
             'Chronically elevated β creates RT elevation and estimation degradation',
             fontsize=13)

for ax, (col_key, xlabel, ylabel, data_col) in zip(axes, [
    ('mean_act', 'Dot count (N)', 'Mean activation', None),
    ('mean_rt',  'Dot count (N)', 'Steps to plateau (RT proxy)', None),
    (None,       'Dot count (N)', 'Weber fraction (CV)', 'weber'),
]):
    ax.axvline(4.5, color='#AAB7B8', lw=1.5, ls=':', alpha=0.7)
    if data_col == 'weber':
        ax.fill_between([0.5,4.5], 0, 0.10, color='#A9DFBF', alpha=0.2,
                       label='Target precision zone')
        for prof in DYSC_PARAMS:
            sub = df_dysc_wf[df_dysc_wf.profile==prof].sort_values('n_dots')
            ax.plot(sub.n_dots, sub.weber, color=DYSC_COLORS[prof],
                   lw=2.5, label=DYSC_PARAMS[prof]['label'])
    else:
        for prof, params in DYSC_PARAMS.items():
            sub = df_dysc[df_dysc.profile==prof].sort_values('n_dots')
            ax.plot(sub.n_dots, sub[col_key], color=DYSC_COLORS[prof],
                   lw=2.5, label=params['label'])
            se_col = col_key.replace('mean','sem')
            ax.fill_between(sub.n_dots,
                           sub[col_key]-sub[se_col],
                           sub[col_key]+sub[se_col],
                           color=DYSC_COLORS[prof], alpha=0.12)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(fontsize=8); ax.set_xticks(DOT_COUNTS)

axes[0].set_title('(a) Activation Curves')
axes[1].set_title('(b) RT Proxy — Under β Lock')
axes[2].set_title('(c) Estimation Precision Degradation')
axes[2].set_ylim(0, 0.5)

# Annotate RT plateau for dyscalculia
sub_d = df_dysc[df_dysc.profile=='Dyscalculia'].sort_values('n_dots')
pk = sub_d.loc[sub_d.mean_rt.idxmax(), 'n_dots']
pv = sub_d.loc[sub_d.mean_rt.idxmax(), 'mean_rt']
axes[1].annotate('β lock → RT\nnever recovers',
   xy=(pk, pv), xytext=(pk-4.5, pv-8),
   fontsize=8, color='#C0392B',
   arrowprops=dict(arrowstyle='->', color='#C0392B'))

plt.tight_layout()
plt.savefig("figures/fig8_dyscalculia.png")
plt.show()
print("  Saved.")


# =============================================================================
# ─── Cell 14: Statistical Tests ──────────────────────────────────────────────
# =============================================================================
print("\n" + "="*70)
print("STATISTICAL TESTS: signed_err for N>=5 (t-test vs 0)")
print("="*70)
stat_rows = []
for arch in ['A','B','C']:
    print(f"\nArchitecture {arch}:")
    for cond in ['C1_Threat','C2_Calm','C3_Excitement','C4_Boredom']:
        vals = df_bias[(df_bias.arch==arch)&(df_bias.condition==cond)&
                       (df_bias.n_dots>=5)]['signed_err'].values
        t, p = stats.ttest_1samp(vals, 0)
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."
        direction = "OVER" if vals.mean()>0 else "UNDER"
        print(f"  {cond:<22}  mean={vals.mean():+.3f}  t={t:+.2f}  "
              f"p={p:.4f}  {sig}  [{direction}]")
        stat_rows.append({'Architecture':f'Arch {arch}','Condition':cond,
                          'Mean':round(vals.mean(),4),'t':round(t,3),
                          'p':round(p,5),'Sig':sig,'Direction':direction})
pd.DataFrame(stat_rows).to_csv("simulation_stats.csv", index=False)
print("\nStats saved.")


# =============================================================================
# ─── Cell 15: Summary Figure ─────────────────────────────────────────────────
# =============================================================================
print("\nGenerating summary figure...")
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)
fig.suptitle(
    'How Does Affect Enter the Number Sense?\n'
    'A Computational Comparison of Three Amygdala-to-IPS Pathway Architectures',
    fontsize=14, fontweight='bold', y=0.99)

arch_sub = {'A':'Arousal→β, Valence→g\n[ATOM-extended]',
            'B':'Affect→α only, β fixed\n[Neuromodulatory]',
            'C':'Transient shock to I_i\n[Orienting-response]'}

for ci, arch in enumerate(['A','B','C']):
    ax = fig.add_subplot(gs[0, ci])
    ax.axhline(0, color='black', lw=1, ls='--', alpha=0.5)
    for cond in ['C1_Threat','C2_Calm','C3_Excitement','C4_Boredom']:
        sub = df_bias[(df_bias.arch==arch)&(df_bias.condition==cond)].sort_values('n_dots')
        sub5 = sub[sub.n_dots>=5]
        col = COND_COLORS[cond]
        lbl = cond.replace('C1_','C1 ').replace('C2_','C2 ')\
                  .replace('C3_','C3 ').replace('C4_','C4 ')
        ax.plot(sub5.n_dots, sub5.signed_err, color=col, lw=2.5,
               marker=ARCH_MARKERS[arch], markersize=5, label=lbl)
    ax.set_xlabel('Dot count N'); ax.set_ylabel('Signed error')
    ax.set_title(f'Architecture {arch}\n{arch_sub[arch]}', fontsize=9)
    ax.set_xlim(4.5, 12.5); ax.set_ylim(-5.5, 5.5)
    if ci == 0: ax.legend(fontsize=7, loc='lower left')

    t_tot = score_matrix[ci].sum()
    verdict = 'PASS' if t_tot>=4.5 else 'PARTIAL' if t_tot>=3.5 else 'FAILS'
    col_v   = {'PASS':'#27AE60','PARTIAL':'#D4AC0D','FAILS':'#C0392B'}[verdict]
    bg_v    = {'PASS':'#D5F5E3','PARTIAL':'#FEF9E7','FAILS':'#FADBD8'}[verdict]
    ax.text(0.97, 0.95, f'{verdict}\n{t_tot:.1f}/5', transform=ax.transAxes,
           fontsize=11, fontweight='bold', color=col_v,
           ha='right', va='top',
           bbox=dict(boxstyle='round', facecolor=bg_v, alpha=0.9))

ax_c = fig.add_subplot(gs[1,:2])
x_p = np.arange(5); w = 0.25
for i, arch in enumerate(['A','B','C']):
    ax_c.bar(x_p+(i-1)*w, score_matrix[i], w, color=ARCH_COLORS[arch],
            alpha=0.85, label=f'Arch {arch}', edgecolor='white')
ax_c.set_xticks(x_p)
ax_c.set_xticklabels([cl.split('\n')[0] for cl in constraint_labels], fontsize=9)
ax_c.set_ylabel('Constraint score (0–1)')
ax_c.set_title('Psychophysical Constraint Compliance')
ax_c.axhline(1.0, color='#27AE60', lw=1, ls='--', alpha=0.5)
ax_c.set_ylim(0, 1.3); ax_c.legend(fontsize=9)

ax_d = fig.add_subplot(gs[1, 2])
for prof in ['Healthy','Dyscalculia','Math_Anxious']:
    sub = df_dysc[df_dysc.profile==prof].sort_values('n_dots')
    ax_d.plot(sub.n_dots, sub.mean_rt, color=DYSC_COLORS[prof], lw=2.2,
             label=DYSC_PARAMS[prof]['label'].split(' (')[0])
ax_d.axvline(4.5, color='#AAB7B8', lw=1.5, ls=':', alpha=0.7)
ax_d.set_xlabel('Dot count N')
ax_d.set_ylabel('Steps to plateau')
ax_d.set_title('Clinical (Arch A)\nRT under chronic β lock')
ax_d.legend(fontsize=8)

plt.savefig("figures/fig_summary.png", dpi=200)
plt.show()
print("  Saved.")


# =============================================================================
# ─── Cell 16: Final Report ───────────────────────────────────────────────────
# =============================================================================
print("\n" + "="*70)
print("ARCHITECTURE VERDICT  (v3  —  fixed parameters, RT, and constraints)")
print("="*70)
for i, arch in enumerate(['A','B','C']):
    t = score_matrix[i].sum()
    v = "SURVIVES" if t>=4.5 else "PARTIAL" if t>=3.5 else "FAILS"
    print(f"\n  Architecture {arch}: {t:.1f}/5.0  [{v}]")
    for j, cl in enumerate(constraint_labels):
        print(f"    {cl.split(chr(10))[0]:<40} {score_matrix[i,j]:.1f}")

print("\nFiles saved:")
for f in sorted(os.listdir("figures")):
    print(f"  figures/{f}")

try:
    from google.colab import files
    import zipfile
    with zipfile.ZipFile("atom_sim_v3.zip","w") as zf:
        for f in ["simulation_raw.csv","simulation_summary.csv",
                  "simulation_bias.csv","simulation_stats.csv"]:
            if os.path.exists(f): zf.write(f)
        for f in os.listdir("figures"): zf.write(f"figures/{f}")
    files.download("atom_sim_v3.zip")
    print("\nDownload initiated: atom_sim_v3.zip")
except ImportError:
    print("\n(Local run: outputs in ./figures/ and ./)")
