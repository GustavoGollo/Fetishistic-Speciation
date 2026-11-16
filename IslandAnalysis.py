# ============================================================
# IslandAnalysis.py — Dinâmica Clássica + Modelo de Ilhas
# ============================================================
import os
import json
import argparse
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns 
import sys
sys.setrecursionlimit(10000)
from scipy import stats as scipy_stats
import gc 

from spec_core_islands import Simulation, GEN_FIM

DEFAULT_PARAMS = {
    'n_loci': 16, 'n_states': 9, 'init_option': 'a',
    'max_generations': 1000, 'gen_inicio_coleta': 950,
    'n_females': 4000, 'n_males': 4000,
    'FemSort': 80, 'PreSelec': 30, 'n_offspring': 6, 'mutation_rate': 0.005,
    'dbscan_eps': 0.40, 'dbscan_min': 110,
    'score_table': (12.0, 0.0, -2.0, -6.0, -12.0),
    'n_islands': 3, 'migration_rate': 0.002,
}

NEUTRAL_SCORES = (1.0, 1.0, 1.0, 1.0, 1.0)
CONDS = ('spec', 'neutral')
CORES = {'spec': 'C3', 'neutral': 'C0'}


def _add_caption(fig, params, extra_text=""):
    caption_text = (
        f"{extra_text}\n"
        f"Islands={params.get('n_islands', 'N/A')}, "
        f"Migration={params.get('migration_rate', 'N/A')}, "
        f"Loci={params.get('n_loci', 'N/A')}, "
        f"States={params.get('n_states', 'N/A')}\n"
        f"DBSCAN eps={params.get('dbscan_eps', 'N/A')}, min_samples={params.get('dbscan_min', 'N/A')}\n"
        f"Fem={params.get('n_females', 'N/A')}, Mal={params.get('n_males', 'N/A')}, "
        f"Mut={params.get('mutation_rate', 'N/A')}, MaxGen={params.get('max_generations', 'N/A')}\n"
        f"FemSort={params.get('FemSort', 'N/A')}, PreSelec={params.get('PreSelec', 'N/A')}"
    )
    fig.text(0.5, 0.012, caption_text, ha="center", va="top", fontsize=9, style="italic", color="gray")


def run_island_simulation(seed, cond, out_dir, overrides=None):
    params = dict(DEFAULT_PARAMS)
    if overrides: params.update(overrides)
    params['seed'] = seed
    params['save_pedigree'] = True
    params['score_table'] = (DEFAULT_PARAMS['score_table'] if cond == 'spec' else NEUTRAL_SCORES)

    os.makedirs(out_dir, exist_ok=True)
    params['individuals_file'] = os.path.join(out_dir, f'ind_{cond}_{seed}.csv')
    params['pairs_file']       = os.path.join(out_dir, f'pairs_{cond}_{seed}.csv')
    params['clusters_gen_file']= os.path.join(out_dir, f'clusters_{cond}_{seed}.csv')
    params['geno_file']        = os.path.join(out_dir, f'geno_final_{cond}_{seed}.npz')

    sim = Simulation(params)
    while sim.step(): pass
    
    res = sim.collect()
    _persist_metrics(res, seed, cond, out_dir)
    return seed, cond


def _run_one(args):
    seed, cond, out_dir, overrides = args
    run_island_simulation(seed, cond, out_dir, overrides)
    return seed, cond


def _persist_metrics(res, seed, cond, out_dir):
    ml_path = os.path.join(out_dir, f'metrics_long_{cond}_{seed}.csv')
    with open(ml_path, 'w', newline='') as f:
        w = __import__('csv').writer(f)
        w.writerow(['gen', 'replica', 'n_pop', 'var_P', 'n_species', 'frac_nao_atribuidos', 'colapsou'])
        for m in res.get('metricas', []):
            w.writerow([m['gen'], seed, m['n_pop'], round(m['var_P'], 6), m['n_species'], round(m['frac_nao_atribuidos'], 6), m['colapsou']])

    metrics = res.get('metricas', [])
    if metrics:
        gens = [m['gen'] for m in metrics]
        ns, vp, n_pop = [m['n_species'] for m in metrics], [m['var_P'] for m in metrics], [m['n_pop'] for m in metrics]
        mf_path = os.path.join(out_dir, f'metrics_final_{cond}_{seed}.csv')
        with open(mf_path, 'w', newline='') as f:
            w = __import__('csv').writer(f)
            w.writerow(['replica', 'condicao', 'gen_ini', 'gen_fim', 'n_species_medio', 'var_P_final', 'n_pop_final'])
            w.writerow([seed, cond, min(gens), max(gens), round(float(np.mean(ns)), 4), round(float(vp[-1]), 6), int(n_pop[-1])])

    ind = res.get('ind_final')
    if ind is not None:
        P_f, P_m = np.asarray(ind['P_f']), np.asarray(ind['P_m'])
        uids_f = np.asarray(ind.get('uids_f', np.arange(len(P_f))))
        uids_m = np.asarray(ind.get('uids_m', np.arange(len(P_m))))
        ir_path = os.path.join(out_dir, f'ind_final_{cond}_{seed}.csv')
        with open(ir_path, 'w', newline='') as f:
            w = __import__('csv').writer(f)
            w.writerow(['uid', 'sex', 'own_pheno'])
            for u, ph in zip(uids_f, P_f.mean(axis=1)): w.writerow([int(u), 'F', round(float(ph), 4)])
            for u, ph in zip(uids_m, P_m.mean(axis=1)): w.writerow([int(u), 'M', round(float(ph), 4)])

def _read_concat(out_dir, prefix, seeds, max_rows=None):
    frames = []
    for cond in CONDS:
        for s in seeds:
            p = os.path.join(out_dir, f'{prefix}_{cond}_{s}.csv')
            if os.path.exists(p):
                df = pd.read_csv(p)
                df['condicao'] = cond
                df['replica'] = s
                if max_rows and len(df) > max_rows: df = df.sample(n=max_rows, random_state=0)
                frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def _fig_curva_por_gen(df_long, metrica, titulo, ylabel, out_path, params):
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=140)
    for cond in CONDS:
        sub = df_long[df_long["condicao"] == cond]
        if sub.empty: continue
        agg = sub.groupby("gen")[metrica].agg(["mean", "std", "count"]).reset_index()
        mean, cnt = agg["mean"].to_numpy(), agg["count"].to_numpy()
        half = np.where(cnt > 1, 1.96 * agg["std"].to_numpy() / np.sqrt(cnt), 0.0)
        ax.plot(agg["gen"].to_numpy(), mean, lw=1.6, color=CORES[cond], label=cond)
        ax.fill_between(agg["gen"].to_numpy(), mean - half, mean + half, color=CORES[cond], alpha=0.2)
    ax.set_xlabel("Generation"); ax.set_ylabel(ylabel); ax.set_title(titulo); ax.legend(frameon=False); ax.grid(alpha=0.25)
    _add_caption(fig, params, extra_text="Mean ± 95% CI across replicas")
    fig.savefig(out_path, bbox_inches='tight'); plt.close(fig)

def _fig_box_final(df_final, coluna, titulo, ylabel, out_path, params, alpha=0.01):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
    dados = [df_final.loc[df_final["condicao"] == c, coluna].dropna().to_numpy() for c in CONDS]
    for i, d in enumerate(dados):
        if not d.size: continue
        q1, med, q3 = np.percentile(d, [25, 50, 75]); iqr = q3 - q1
        low, high = max(d.min(), q1 - 1.5 * iqr), min(d.max(), q3 + 1.5 * iqr)
        ax.add_patch(plt.Rectangle((i + 0.7, q1), 0.6, q3 - q1, facecolor=CORES[CONDS[i]], alpha=0.5, edgecolor='k'))
        ax.plot([i + 0.7, i + 1.3], [med, med], color='k', lw=1.5)
        ax.plot([i + 1.0, i + 1.0], [low, q1], color='k', lw=1); ax.plot([i + 1.0, i + 1.0], [q3, high], color='k', lw=1)
        ax.plot([i + 0.8, i + 1.2], [low, low], color='k', lw=1); ax.plot([i + 0.8, i + 1.2], [high, high], color='k', lw=1)
        ax.scatter(np.random.default_rng(0).normal(i + 1.0, 0.05, size=d.size), d, s=25, color=CORES[CONDS[i]], alpha=0.8, edgecolor='k', zorder=3)
    p_txt = ""
    if len(dados[0]) and len(dados[1]):
        U, p = scipy_stats.mannwhitneyu(dados[0], dados[1], alternative='two-sided')
        p_txt = f"Mann-Whitney U={U:.1f}, p={p:.4g} ({'sig' if p < alpha else 'n.s.'}, alpha={alpha})"
        titulo += f"\n{p_txt}"
    ax.set_xticks([1, 2]); ax.set_xticklabels([f"spec\n(n={len(dados[0])})", f"neutral\n(n={len(dados[1])})"])
    ax.set_xlim(0.5, 2.5); ax.set_ylabel(ylabel); ax.set_title(titulo, pad=10); ax.grid(alpha=0.25, axis='y')
    _add_caption(fig, params, extra_text="Each dot = one replica")
    fig.savefig(out_path, bbox_inches='tight'); plt.close(fig)

def _fig_hist_pheno(df_ind, out_path, params):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    bins = np.linspace(0, df_ind["own_pheno"].max() if len(df_ind) else 1.0, 40)
    for cond in CONDS:
        sub = df_ind[df_ind["condicao"] == cond]
        if sub.empty: continue
        ax.hist(sub["own_pheno"].to_numpy(), bins=bins, density=True, alpha=0.45, color=CORES[cond], label=cond)
    ax.set_xlabel("Own phenotype (mean of final P)"); ax.set_ylabel("Density"); ax.set_title("Final phenotypic distribution"); ax.legend(frameon=False)
    _add_caption(fig, params, extra_text=f"n_ind sampled = {len(df_ind)}")
    fig.savefig(out_path, bbox_inches='tight'); plt.close(fig)

def gerar_graficos_classicos(out_dir, seeds, params, alpha=0.01, log_func=print):
    df_long = _read_concat(out_dir, 'metrics_long', seeds)
    df_final = _read_concat(out_dir, 'metrics_final', seeds)
    df_ind = _read_concat(out_dir, 'ind_final', seeds, max_rows=20000)
    if not df_long.empty:
        _fig_curva_por_gen(df_long, "n_species", "Number of species", "n_species", os.path.join(out_dir, "fig_n_species_curva.png"), params)
        _fig_curva_por_gen(df_long, "var_P", "Phenotypic variance", "var_P", os.path.join(out_dir, "fig_var_P_curva.png"), params)
        _fig_curva_por_gen(df_long, "frac_nao_atribuidos", "Unassigned fraction", "Fraction", os.path.join(out_dir, "fig_frac_nao_atrib.png"), params)
        _fig_curva_por_gen(df_long, "n_pop", "Population size", "n_pop", os.path.join(out_dir, "fig_n_pop_curva.png"), params)
    if not df_final.empty:
        _fig_box_final(df_final, "n_species_medio", "Mean number of species", "n_species_medio", os.path.join(out_dir, "fig_box_n_species_final.png"), params, alpha)
        _fig_box_final(df_final, "var_P_final", "Final phenotypic variance", "var_P_final", os.path.join(out_dir, "fig_box_var_P_final.png"), params, alpha)
    if not df_ind.empty:
        _fig_hist_pheno(df_ind, os.path.join(out_dir, "fig_hist_own_pheno.png"), params)
    log_func("[plots] Gráficos clássicos gerados.")

def gerar_estatistica_boxplot(out_dir, seeds, params, alpha=0.01, log_func=print):
    dados = {c: [] for c in CONDS}
    max_gen = params.get('max_generations', 400); inicio_janela = max(0, max_gen - 50)
    for cond in CONDS:
        for s in seeds:
            path = os.path.join(out_dir, f'pairs_{cond}_{s}.csv')
            if not os.path.exists(path): continue
            df = pd.read_csv(path); df = df[(df['gen'] >= inicio_janela) & (df['gen'] <= max_gen)]
            df_valid = df[(df['mother_cluster'] >= 0) & (df['father_cluster'] >= 0)]
            dados[cond].append(df_valid['same_cluster'].mean() if len(df_valid) > 0 else 0.0)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
    for i, cond in enumerate(CONDS):
        d = np.array(dados[cond])
        if not d.size: continue
        q1, med, q3 = np.percentile(d, [25, 50, 75]); iqr = q3 - q1
        low, high = max(d.min(), q1 - 1.5 * iqr), min(d.max(), q3 + 1.5 * iqr)
        ax.add_patch(plt.Rectangle((i + 0.7, q1), 0.6, q3 - q1, facecolor=CORES[cond], alpha=0.5, edgecolor='k'))
        ax.plot([i + 0.7, i + 1.3], [med, med], color='k', lw=1.5)
        ax.plot([i + 1.0, i + 1.0], [low, q1], color='k', lw=1); ax.plot([i + 1.0, i + 1.0], [q3, high], color='k', lw=1)
        ax.plot([i + 0.8, i + 1.2], [low, low], color='k', lw=1); ax.plot([i + 0.8, i + 1.2], [high, high], color='k', lw=1)
        ax.scatter(np.random.default_rng(0).normal(i + 1.0, 0.05, size=d.size), d, s=25, color=CORES[cond], alpha=0.8, edgecolor='k', zorder=3)
    p_txt = ""
    if len(dados['spec']) and len(dados['neutral']):
        U, p = scipy_stats.mannwhitneyu(dados['spec'], dados['neutral'], alternative='two-sided')
        p_txt = f"Mann-Whitney U={U:.1f}, p={p:.4g} ({'sig' if p < alpha else 'n.s.'}, alpha={alpha})"
    ax.set_xticks([1, 2]); ax.set_xticklabels([f"spec\n(n={len(dados['spec'])})", f"neutral\n(n={len(dados['neutral'])})"])
    ax.set_xlim(0.5, 2.5); ax.set_ylim(-0.05, 1.05); ax.set_ylabel("Proportion of same-cluster matings")
    title = f"Reproductive isolation (last 50 gens up to {max_gen})"
    if p_txt: title += f"\n{p_txt}"
    ax.set_title(title, pad=10); ax.grid(alpha=0.25, axis='y')
    _add_caption(fig, params, extra_text="Each dot = one replica")
    fig.savefig(os.path.join(out_dir, 'boxplot_same_cluster.png'), bbox_inches='tight'); plt.close(fig)


def analyze_contingency(out_dir, seeds, params, alpha=0.01, log_func=print):
    summary = []
    for cond in CONDS:
        for s in seeds:
            ind_path = os.path.join(out_dir, f'ind_{cond}_{s}.csv')
            if not os.path.exists(ind_path): continue
            df = pd.read_csv(ind_path)
            if df.empty: continue
            df = df[df['gen'] == int(df['gen'].max())]
            if 'island_id' not in df.columns or 'cluster' not in df.columns: continue
            df = df[df['cluster'] >= 0]
            if len(df) == 0: continue
            ct = pd.crosstab(df['island_id'], df['cluster']); n = int(ct.values.sum())
            try: chi2, p, dof, _ = scipy_stats.chi2_contingency(ct)
            except ValueError: chi2, p, dof = float('nan'), float('nan'), 0
            k = min(ct.shape) - 1
            cramers_v = float(np.sqrt(chi2 / (n * k))) if (k > 0 and n > 0) else 0.0
            summary.append(dict(cond=cond, seed=s, chi2=float(chi2), p_value=float(p), dof=int(dof), cramers_v=cramers_v, n=n, significant=bool(p < alpha)))
            ct.to_csv(os.path.join(out_dir, f'contingency_{cond}_{s}.csv'))
    df_sum = pd.DataFrame(summary)
    if df_sum.empty: return df_sum
    df_sum.to_csv(os.path.join(out_dir, 'chi2_summary.csv'), index=False)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
    for cond in CONDS:
        sub = df_sum[df_sum['cond'] == cond]
        if not sub.empty: ax.scatter(sub['seed'], sub['cramers_v'], color=CORES[cond], s=70, edgecolor='k', label=cond, alpha=0.85)
    ax.axhline(0, color='k', lw=0.5); ax.set_xlabel('Seed'); ax.set_ylabel("Cramér's V (island × cluster)"); ax.set_title('Island ↔ cluster dependence (Chi-square)')
    ax.legend(frameon=False); ax.grid(alpha=0.25)
    _add_caption(fig, params, extra_text="Cramér's V per seed and condition")
    fig.savefig(os.path.join(out_dir, 'chi2_cramers_v.png'), bbox_inches='tight'); plt.close(fig)

def _compute_fst(sim, n_states):
    D_all = np.concatenate([sim.D_f, sim.D_m], axis=0); P_all = np.concatenate([sim.P_f, sim.P_m], axis=0)
    isl_all = np.concatenate([sim.island_f, sim.island_m]); islands = np.unique(isl_all); n_loci = D_all.shape[1]
    alleles = {int(isl): np.concatenate([D_all[isl_all == isl], P_all[isl_all == isl]], axis=0) for isl in islands if (isl_all == isl).sum() > 0}
    h_t_arr, h_s_arr = np.zeros(n_loci), np.zeros(n_loci)
    for loc in range(n_loci):
        p_k, n_k = {}, {}
        for isl, arr in alleles.items():
            counts = np.bincount(arr[:, loc], minlength=n_states).astype(float)
            n_k[isl] = counts.sum(); p_k[isl] = counts / counts.sum() if counts.sum() > 0 else np.zeros(n_states)
        total = sum(n_k.values())
        if total == 0: continue
        p_bar = sum(p_k[isl] * n_k[isl] for isl in alleles) / total
        h_t_arr[loc] = 1.0 - np.sum(p_bar ** 2); h_s_arr[loc] = np.mean([1.0 - np.sum(p_k[i] ** 2) for i in alleles])
    return float((h_t_arr - h_s_arr).sum() / h_t_arr.sum()) if h_t_arr.sum() > 0 else 0.0

def compute_fst_analysis(out_dir, seeds, params, n_states=9, log_func=print):
    rows = []
    for cond in CONDS:
        for s in seeds:
            geno_path = os.path.join(out_dir, f'geno_final_{cond}_{s}.npz')
            if not os.path.exists(geno_path): continue
            data = np.load(geno_path)
            class _S: pass
            sim = _S(); sim.D_f, sim.D_m = data['D_f'], data['D_m']; sim.P_f, sim.P_m = data['P_f'], data['P_m']; sim.island_f, sim.island_m = data['island_f'], data['island_m']
            rows.append(dict(cond=cond, seed=s, fst_overall=_compute_fst(sim, n_states)))
    df = pd.DataFrame(rows)
    if df.empty: return df
    df.to_csv(os.path.join(out_dir, 'fst_summary.csv'), index=False)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
    for cond in CONDS:
        sub = df[df['cond'] == cond]
        if not sub.empty: ax.scatter(sub['seed'], sub['fst_overall'], color=CORES[cond], s=70, edgecolor='k', label=cond, alpha=0.85)
    

    ax.axhline(0, color='k', lw=0.5); ax.set_xlabel('Seed'); ax.set_ylabel('F_ST (overall)'); ax.set_title('Genetic differentiation between islands (F_ST)')
    ax.legend(frameon=False); ax.grid(alpha=0.25)
    _add_caption(fig, params, extra_text="Global F_ST per seed and condition")
    fig.savefig(os.path.join(out_dir, 'fst_summary.png'), bbox_inches='tight'); plt.close(fig)

def phenotypic_map_analysis(out_dir, seeds, params, log_func=print):
    for cond in CONDS:
        for s in seeds:
            geno_path = os.path.join(out_dir, f'geno_final_{cond}_{s}.npz')
            if not os.path.exists(geno_path): continue
            data = np.load(geno_path); P = np.concatenate([data['P_f'], data['P_m']], axis=0).astype(np.float32)
            islands = np.concatenate([data['island_f'], data['island_m']])
            if len(P) < 10: continue
            Pc = P - P.mean(axis=0, keepdims=True)
            try: U, S, Vt = np.linalg.svd(Pc, full_matrices=False); proj = U[:, :2] * S[:2]
            except np.linalg.LinAlgError: continue
            fig, ax = plt.subplots(figsize=(7, 6), dpi=140); cmap = plt.get_cmap('tab10')
            for isl in np.unique(islands):
                mask = islands == isl
                ax.scatter(proj[mask, 0], proj[mask, 1], s=4, alpha=0.45, color=cmap(int(isl) % 10), label=f'Island {int(isl)} (n={int(mask.sum())})')
            ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_title(f'Phenotypic space (PCA) — {cond}')
            ax.legend(markerscale=3, frameon=False); ax.grid(alpha=0.2)
            _add_caption(fig, params, extra_text=f"Condition: {cond} | Seed: {s}")
            fig.savefig(os.path.join(out_dir, f'pheno_map_{cond}_{s}.png'), bbox_inches='tight'); plt.close(fig)


def gerar_heatmaps_globais(out_dir, seeds, params, log_func=print):
    max_gen = params.get('max_generations', 400); inicio_janela = max(0, max_gen - 50)
    for cond in CONDS:
        matrices = []
        for s in seeds:
            path = os.path.join(out_dir, f'pairs_{cond}_{s}.csv')
            if not os.path.exists(path): continue
            df = pd.read_csv(path); df = df[(df['gen'] >= inicio_janela) & (df['gen'] <= max_gen)]
            df_valid = df[(df['mother_cluster'] >= 0) & (df['father_cluster'] >= 0)]
            if len(df_valid) == 0: continue
            max_c = int(max(df_valid['mother_cluster'].max(), df_valid['father_cluster'].max())) + 1
            m = np.zeros((max_c, max_c))
            for _, row in df_valid.iterrows(): m[int(row['mother_cluster']), int(row['father_cluster'])] += 1
            matrices.append(m)
        if not matrices: continue
        
        # Junta todas as matrizes
        max_dim = max(m.shape[0] for m in matrices); mat_final = np.zeros((max_dim, max_dim))
        for m in matrices: mat_final[:m.shape[0], :m.shape[1]] += m
        mat_final /= len(matrices)
        
        # CORREÇÃO 1: Cortar eixos vazios (se o max real for 16, mas o eixo ia até 22)
        valid_rows = np.where(mat_final.sum(axis=1) > 0.0)[0]
        if len(valid_rows) > 0:
            max_valid = valid_rows[-1] + 1
            mat_final = mat_final[:max_valid, :max_valid]
            
        # CORREÇÃO 2: Normalizar por linha (proporção de acasalamento para cada mãe)
        row_sums = mat_final.sum(axis=1, keepdims=True)
        mat_final = np.divide(mat_final, row_sums, out=np.zeros_like(mat_final), where=row_sums!=0)
        
        fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
        im = ax.imshow(mat_final, cmap='viridis', origin='lower', vmin=0, vmax=1)
        ax.set_title(f"Mating Matrix — {cond.capitalize()}"); ax.set_xlabel("Father's cluster"); ax.set_ylabel("Mother's cluster")
        fig.colorbar(im, ax=ax, label='Proportion of matings') # Rótulo corrigido
        _add_caption(fig, params, extra_text=f"Window: gen {max_gen-50}–{max_gen} | Mean across replicas")
        fig.savefig(os.path.join(out_dir, f'heatmap_{cond}.png'), bbox_inches='tight'); plt.close(fig)

# ==========================================
# NOVAS FUNÇÕES DE PARENTESCO (KINSHIP)
# ==========================================

def calculate_kinship(uid1, uid2, pedigree, cache=None, depth=0, max_depth=50):
    if cache is None:
        cache = {}
        
    # Trava de segurança: se chegar a 50 gerações de distância, para de buscar
    if depth > max_depth:
        return 0.0
        
    if uid1 == uid2:
        if uid1 not in pedigree or pd.isna(pedigree[uid1]['mother_id']) or pd.isna(pedigree[uid1]['father_id']):
            return 0.5
        return 0.5 * (1 + calculate_kinship(pedigree[uid1]['mother_id'], pedigree[uid1]['father_id'], pedigree, cache, depth+1, max_depth))
        
    if uid1 < uid2:
        uid1, uid2 = uid2, uid1
        
    if (uid1, uid2) in cache:
        return cache[(uid1, uid2)]
        
    if uid1 not in pedigree or pd.isna(pedigree[uid1]['mother_id']) or pd.isna(pedigree[uid1]['father_id']):
        kin = 0.0
    else:
        m = int(pedigree[uid1]['mother_id'])
        f = int(pedigree[uid1]['father_id'])
        kin = 0.5 * (
            calculate_kinship(uid2, m, pedigree, cache, depth+1, max_depth) + 
            calculate_kinship(uid2, f, pedigree, cache, depth+1, max_depth)
        )
        
    cache[(uid1, uid2)] = kin
    return kin

def gerar_graficos_parentesco(out_dir, seeds, params, log_func=print):
    log_func("[plots] Gerando análise de parentesco (Kinship) - Otimizada...")
    cond = 'spec' 
    resultados_box_all = []
    mat_kinship_all = []
    
    max_gen = params.get('max_generations', 1500)
    janela_geracoes = 50
    gen_corte = max(0, max_gen - janela_geracoes)
    
    for s in seeds:
        path = os.path.join(out_dir, f'ind_{cond}_{s}.csv')
        if not os.path.exists(path): continue
        
        # OTIMIZAÇÃO: Lê apenas as colunas necessárias para economizar RAM
        cols = ['uid', 'gen', 'mother_id', 'father_id', 'cluster']
        df = pd.read_csv(path, usecols=cols, dtype={'uid': int, 'gen': int})
        if len(df) == 0: continue
        
        df = df[df['gen'] >= gen_corte]
        if len(df) == 0: continue
        
        df['mother_id'] = pd.to_numeric(df['mother_id'], errors='coerce')
        df['father_id'] = pd.to_numeric(df['father_id'], errors='coerce')
        
        # Constrói o pedigree
        pedigree = df.set_index('uid')[['mother_id', 'father_id']].to_dict('index')
        df_final = df[df['gen'] == int(df['gen'].max())]
        if len(df_final) == 0: continue
        
        cache_kinship = {}
        
        # --- Cálculo do Boxplot ---
        df_sample = df_final.sample(n=min(100, len(df_final)), random_state=42)
        sample_uids = df_sample['uid'].tolist()
        
        for i in range(len(sample_uids)):
            for j in range(i+1, len(sample_uids)):
                u1, u2 = sample_uids[i], sample_uids[j]
                kin = calculate_kinship(u1, u2, pedigree, cache=cache_kinship, max_depth=50)
                c1 = df_sample[df_sample['uid'] == u1]['cluster'].values[0]
                c2 = df_sample[df_sample['uid'] == u2]['cluster'].values[0]
                tipo = 'Same cluster' if c1 == c2 else 'Different Clusters'
                resultados_box_all.append({'Tipo': tipo, 'Kinship': kin})
                
        # --- Cálculo do Heatmap ---
        clusters = sorted([c for c in df_final['cluster'].unique() if c != -1])
        if len(clusters) > 0:
            mat_kinship = np.zeros((len(clusters), len(clusters)))
            for i, c1 in enumerate(clusters):
                for j, c2 in enumerate(clusters):
                    uids_c1 = df_final[df_final['cluster'] == c1]['uid'].tolist()[:10]
                    uids_c2 = df_final[df_final['cluster'] == c2]['uid'].tolist()[:10]
                    kins = []
                    for u1 in uids_c1:
                        for u2 in uids_c2:
                            kins.append(calculate_kinship(u1, u2, pedigree, cache=cache_kinship, max_depth=50))
                    mat_kinship[i, j] = np.mean(kins) if kins else 0
            mat_kinship_all.append(mat_kinship)

        # --- LIMPEZA DE MEMÓRIA (DENTRO DO LOOP!) ---
        del df
        del pedigree
        del df_final
        del df_sample
        del cache_kinship
        gc.collect()

    # --- Plotagem 1: Boxplot Consolidado ---
    if resultados_box_all:
        df_box = pd.DataFrame(resultados_box_all)
        fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
        sns.boxplot(data=df_box, x='Tipo', y='Kinship', palette=['#d62728', '#1f77b4'], ax=ax)
        ax.set_title('Kinship Coefficient — Spec Scenario')
        ax.set_ylabel('Kinship Coefficient'); ax.set_xlabel('')
        _add_caption(fig, params, extra_text=f"Mean across replicas | Kinship calculated over the last {janela_geracoes} generations")
        fig.savefig(os.path.join(out_dir, 'fig_kinship_boxplot.png'), bbox_inches='tight'); plt.close(fig)
    
    # --- Plotagem 2: Heatmap Médio ---
    if mat_kinship_all:
        max_dim = max(m.shape[0] for m in mat_kinship_all)
        mat_final = np.zeros((max_dim, max_dim))
        for m in mat_kinship_all:
            mat_final[:m.shape[0], :m.shape[1]] += m
        
        mat_final = mat_final / len(mat_kinship_all)
        labels = list(range(max_dim))
        
        fig, ax = plt.subplots(figsize=(7, 6), dpi=140)
        sns.heatmap(mat_final, xticklabels=labels, yticklabels=labels, cmap="YlOrRd", annot=False, ax=ax)
        ax.set_title('Mean Kinship Matrix between Clusters (Spec)')
        ax.set_xlabel('Cluster'); ax.set_ylabel('Cluster')
        _add_caption(fig, params, extra_text=f"Mean Kinship coefficient | Last {janela_geracoes} generations")
        fig.savefig(os.path.join(out_dir, 'fig_kinship_heatmap.png'), bbox_inches='tight'); plt.close(fig)
        
    log_func("[plots] Gráficos de parentesco salvos.")


def island_heatmaps_analysis(out_dir, seeds, params, log_func=print):
    n_islands = params.get('n_islands', 3)
    gen_inicio = params.get('gen_inicio_coleta', 350)
    
    for cond in CONDS:
        mats = {i: np.zeros((1, 1)) for i in range(n_islands)}
        has_data = False
        
        for s in seeds:
            p = os.path.join(out_dir, f'pairs_{cond}_{s}.csv')
            if not os.path.exists(p): continue
            df = pd.read_csv(p)
            if df.empty or 'mother_island' not in df.columns: continue
            
            df = df[(df['gen'] >= gen_inicio) & (df['mother_cluster'] >= 0) & (df['father_cluster'] >= 0)]
            if df.empty: continue
            
            for isl in range(n_islands):
                sub = df[df['mother_island'] == isl]
                if sub.empty: continue
                d = int(max(sub['mother_cluster'].max(), sub['father_cluster'].max())) + 1
                m = np.zeros((d, d))
                for _, row in sub.iterrows(): 
                    m[int(row['mother_cluster']), int(row['father_cluster'])] += 1
                if m.sum() == 0: continue
                
                cur = mats[isl]
                nd = max(cur.shape[0], m.shape[0])
                new_m = np.zeros((nd, nd))
                new_m[:cur.shape[0], :cur.shape[1]] += cur
                new_m[:m.shape[0], :m.shape[1]] += m
                mats[isl] = new_m
                has_data = True
                
        if not has_data: continue
        
        for isl, mat in mats.items():
            if mat.size <= 1: continue
            
            # Cortar eixos vazios
            valid_rows = np.where(mat.sum(axis=1) > 0.0)[0]
            if len(valid_rows) == 0: continue
            max_valid = valid_rows[-1] + 1
            mat = mat[:max_valid, :max_valid]
            
            # Normalizar por linha para a barra de cores ir de 0 a 1.0
            row_sums = mat.sum(axis=1, keepdims=True)
            mat_plot = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums!=0)
            
            fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
            im = ax.imshow(mat_plot, cmap='viridis', origin='lower', vmin=0, vmax=1)
            ax.set_title(f'Mating Matrix — Island {isl} ({cond.capitalize()})')
            ax.set_xlabel("Father's cluster")
            ax.set_ylabel("Mother's cluster")
            fig.colorbar(im, ax=ax, label='Proportion')
            _add_caption(fig, params, extra_text=f"Island: {isl} | Condition: {cond}")
            fig.savefig(os.path.join(out_dir, f'heatmap_island{isl}_{cond}.png'), bbox_inches='tight')
            plt.close(fig)




# ==========================================
# ORQUESTRADOR FINAL
# ==========================================
def run_all(out_dir, seeds, n_islands=3, migration_rate=0.002, max_gen=400, start_gen=350, n_cores=1, overrides=None, alpha=0.01, log_func=print):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = os.path.join(out_dir, f'run_{timestamp}'); os.makedirs(run_dir, exist_ok=True)

    parametros_atualizados = {
        'n_islands': n_islands, 'migration_rate': migration_rate, 
        'max_generations': max_gen, 'gen_inicio_coleta': start_gen
    }
    if overrides: parametros_atualizados.update(overrides)
        
    current_params = dict(DEFAULT_PARAMS)
    current_params.update(parametros_atualizados)

    with open(os.path.join(run_dir, 'parameters.json'), 'w') as f:
        to_save = dict(current_params); to_save['seeds'] = list(seeds)
        json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in to_save.items()}, f, indent=4)
    
    tasks = [(s, c, run_dir, parametros_atualizados) for s in seeds for c in CONDS]
    log_func(f"[run] {len(tasks)} simulações em {n_cores} núcleo(s) → {run_dir}")
    
    if n_cores > 1:
        with ProcessPoolExecutor(max_workers=n_cores) as ex:
            for fut in as_completed([ex.submit(_run_one, t) for t in tasks]):
                seed, cond = fut.result(); log_func(f"[sim] OK seed={seed} cond={cond}")
    else:
        for t in tasks:
            seed, cond = _run_one(t); log_func(f"[sim] OK seed={seed} cond={cond}")

    log_func("[plots] Gerando graficos classicos...")
    gerar_graficos_classicos(run_dir, seeds, current_params, alpha, log_func)
    gerar_estatistica_boxplot(run_dir, seeds, current_params, alpha, log_func)
    gerar_heatmaps_globais(run_dir, seeds, current_params, log_func)

    log_func("[plots] Gerando analises de ilhas...")
    analyze_contingency(run_dir, seeds, current_params, alpha, log_func)
    compute_fst_analysis(run_dir, seeds, current_params, log_func=log_func)
    phenotypic_map_analysis(run_dir, seeds, current_params, log_func=log_func)
    island_heatmaps_analysis(run_dir, seeds, current_params, log_func=log_func)
    
    # CHAMADA DE PARENTESCO AQUI
    gerar_graficos_parentesco(run_dir, seeds, current_params, log_func=log_func)
    
    log_func(f"[done] Resultados em {run_dir}")
    return run_dir

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Análise de especiação — Modelo de Ilhas')
    ap.add_argument('--out_dir', type=str, default='island_analysis')
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--n_islands', type=int, default=3)
    ap.add_argument('--migration_rate', type=float, default=0.002) 
    ap.add_argument('--max_gen', type=int, default=400)
    ap.add_argument('--start_gen', type=int, default=350)
    ap.add_argument('--n_cores', type=int, default=4)
    ap.add_argument('--alpha', type=float, default=0.01)
    args = ap.parse_args()
    run_all(args.out_dir, args.seeds, args.n_islands, args.migration_rate, args.max_gen, args.start_gen, args.n_cores)