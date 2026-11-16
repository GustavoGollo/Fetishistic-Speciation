# ============================================================
# spec_core_islands.py — Especiação com Modelo de Ilhas
# ============================================================

import numpy as np
from collections import defaultdict
import os
import csv

try:
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

GEN_FIM = 250
GEN_INICIO_COLETA = 200

def build_score_matrix(score_table, n_states):
    max_dist = n_states // 2
    if len(score_table) < max_dist + 1:
        raise ValueError(f"score_table precisa de >= {max_dist + 1} valores.")
    W = np.zeros((n_states, n_states), dtype=np.float32)
    for d in range(n_states):
        for p in range(n_states):
            diff = abs(d - p)
            dist = min(diff, n_states - diff)
            W[d, p] = score_table[dist]
    return W

def initialize_population(n, n_loci, n_states, option, rng):
    D = np.zeros((n, n_loci), dtype=np.uint8)
    P = np.zeros((n, n_loci), dtype=np.uint8)
    if option == 'a':
        pass
    elif option == 'b':
        mask = rng.random((n, n_loci)) < 0.1
        D[mask] = rng.integers(1, n_states, size=int(mask.sum()))
        mask = rng.random((n, n_loci)) < 0.1
        P[mask] = rng.integers(1, n_states, size=int(mask.sum()))
    elif option == 'c':
        loci_mask = rng.random(n_loci) < 0.1
        D[:, loci_mask] = rng.integers(1, n_states, size=(n, int(loci_mask.sum())))
        P[:, loci_mask] = rng.integers(1, n_states, size=(n, int(loci_mask.sum())))
    elif option == 'd':
        D = rng.integers(0, n_states, size=(n, n_loci)).astype(np.uint8)
        P = rng.integers(0, n_states, size=(n, n_loci)).astype(np.uint8)
    else:
        raise ValueError(f"init_option inválida: {option!r}.")
    return D, P

def assign_islands(n, n_islands, rng):
    if n_islands <= 1:
        return np.zeros(n, dtype=np.int32)
    base = np.arange(n, dtype=np.int32) % n_islands
    return rng.permutation(base).astype(np.int32)

def reproduce(D_f, P_f, D_m, P_m, island_f, island_m,
              target_pop, n_offspring, FemSort, PreSelec, W, 
              mutation_rate, n_loci, n_states, seed, female_ids, 
              male_ids, prop_m_fixed, n_islands, pedigree_log=None):
    rng = np.random.default_rng(seed)
    n_f, n_m = D_f.shape[0], D_m.shape[0]
    
    if n_f == 0 or n_m == 0:
        empty = np.arange(0, dtype=np.int32)
        return (D_f[:0], P_f[:0], D_m[:0], P_m[:0], empty, empty, 0, empty, empty)

    P_f_oh = (P_f[:, :, None] == np.arange(n_states)).astype(np.float32)
    P_f_flat = P_f_oh.reshape(n_f, -1)
    W_Dm = W[D_m].reshape(n_m, -1)

    base_pop = target_pop // n_islands
    rem_pop = target_pop % n_islands
    target_pop_islands = np.full(n_islands, base_pop, dtype=int)
    target_pop_islands[:rem_pop] += 1

    child_D_list, child_P_list, child_island_list = [], [], []
    max_rodadas_global = 0
    
    if pedigree_log is not None: pedigree_log.clear()

    for isl in range(n_islands):
        target_pop_isl = target_pop_islands[isl]
        if target_pop_isl == 0: continue
        
        Nfem_target = int(np.ceil(target_pop_isl / max(1, n_offspring)))
        
        f_idx_isl = np.where(island_f == isl)[0]
        m_idx_isl = np.where(island_m == isl)[0]

        if len(f_idx_isl) == 0 or len(m_idx_isl) == 0:
            continue 

        female_to_males = defaultdict(list)
        rodada = 0
        
        while rodada < 10 and len(female_to_males) < Nfem_target:
            rodada += 1
            for m in m_idx_isl:
                if FemSort >= len(f_idx_isl):
                    valid_f = f_idx_isl
                else:
                    valid_f = rng.choice(f_idx_isl, size=FemSort, replace=False)
                
                if len(valid_f) == 0: continue

                scores = np.einsum('i,ji->j', W_Dm[m], P_f_flat[valid_f])
                pre_e = min(PreSelec, len(valid_f))
                top_idx = np.argpartition(-scores, pre_e - 1)[:pre_e] 
                selected_f = valid_f[top_idx]                
                for f in selected_f:
                    female_to_males[f].append(m)
        
        if len(female_to_males) == 0:
            continue
            
        num_repro_females = len(female_to_males)
        
        if num_repro_females > Nfem_target:
            keys = list(female_to_males.keys())
            rng.shuffle(keys)
            selected_keys = keys[:Nfem_target]
        else:
            selected_keys = list(female_to_males.keys())
            rng.shuffle(selected_keys)

        num_pairs = len(selected_keys)
        if num_pairs == 0: continue

        base_offspring = target_pop_isl // num_pairs
        remainder = target_pop_isl % num_pairs
        
        for i, f in enumerate(selected_keys):
            suitors = female_to_males[f]
            d_f = D_f[f]
            p_m_suitors = P_m[suitors]
            scores = W[d_f, p_m_suitors].sum(axis=1)
            best = int(np.argmax(scores))
            m = suitors[best]

            n_kids = base_offspring + (1 if i < remainder else 0)
            if n_kids == 0: continue

            mask_D = rng.random((n_kids, n_loci)) < 0.5
            mask_P = rng.random((n_kids, n_loci)) < 0.5
            D_m_g, P_m_g = D_m[m], P_m[m]
            D_f_g, P_f_g = D_f[f], P_f[f]
            
            cD = np.where(mask_D, D_m_g, D_f_g).astype(np.uint8)
            cP = np.where(mask_P, P_m_g, P_f_g).astype(np.uint8)
            
            child_D_list.append(cD)
            child_P_list.append(cP)
            child_island_list.append(np.full(n_kids, isl, dtype=np.int32))

            if pedigree_log is not None:
                pedigree_log.append((int(f), int(m), int(n_kids)))
        
        if rodada > max_rodadas_global: max_rodadas_global = rodada

    if len(child_D_list) == 0:
        empty = np.arange(0, dtype=np.int32)
        return (D_f[:0], P_f[:0], D_m[:0], P_m[:0], empty, empty, max_rodadas_global, empty, empty)

    child_D = np.concatenate(child_D_list, axis=0)
    child_P = np.concatenate(child_P_list, axis=0)
    child_island = np.concatenate(child_island_list, axis=0)
    c_idx = len(child_D)

    if mutation_rate > 0 and c_idx > 0:
        mask_D = rng.random((c_idx, n_loci)) < mutation_rate
        mask_P = rng.random((c_idx, n_loci)) < mutation_rate
        rand_D = rng.integers(0, n_states, size=(c_idx, n_loci)).astype(np.uint8)
        rand_P = rng.integers(0, n_states, size=(c_idx, n_loci)).astype(np.uint8)
        child_D = np.where(mask_D, rand_D, child_D)
        child_P = np.where(mask_P, rand_P, child_P)

    n_machos = int(np.floor(c_idx * prop_m_fixed))
    n_machos = min(max(n_machos, 0), c_idx)
    n_femeas = c_idx - n_machos
    perm = rng.permutation(c_idx)
    i_f, i_m = perm[:n_femeas], perm[n_femeas:]

    return (child_D[i_f], child_P[i_f], child_D[i_m], child_P[i_m],
            child_island[i_f], child_island[i_m], max_rodadas_global, i_f, i_m)


class Simulation:
    def __init__(self, params):
        self.p = params
        self.rng = np.random.default_rng(params['seed'])
        self.n_states = int(params.get('n_states', 9))
        self.W = build_score_matrix(params['score_table'], self.n_states)

        self.n_islands = int(params.get('n_islands', 3))
        self.migration_rate = float(params.get('migration_rate', 0.002))

        self.full_registry = {}
        self.cluster_historico = {}
        self.cluster_snapshot_por_gen = {}
        self.island_of_uid = {}

        self.n_loci = params['n_loci']
        self.FemSort = params.get('FemSort', 500)
        self.PreSelec = params.get('PreSelec', 30)
        self.n_offspring = params.get('n_offspring', 6)
        self.dbscan_eps = params.get('dbscan_eps', 0.4)
        self.dbscan_min = params.get('dbscan_min', 120)

        self.target_f = params['n_females']
        self.target_m = params['n_males']
        self.target_pop = self.target_f + self.target_m
        self.prop_m_fixed = self.target_m / self.target_pop

        self.D_f, self.P_f = initialize_population(self.target_f, self.n_loci, self.n_states, params['init_option'], self.rng)
        self.D_m, self.P_m = initialize_population(self.target_m, self.n_loci, self.n_states, params['init_option'], self.rng)

        self._uid_f = np.arange(self.target_f, dtype=np.int64)
        self._uid_m = np.arange(self.target_f, self.target_f + self.target_m, dtype=np.int64)
        self.island_f = assign_islands(self.target_f, self.n_islands, self.rng)
        self.island_m = assign_islands(self.target_m, self.n_islands, self.rng)
        for uid, isl in zip(self._uid_f, self.island_f): self.island_of_uid[int(uid)] = int(isl)
        for uid, isl in zip(self._uid_m, self.island_m): self.island_of_uid[int(uid)] = int(isl)

        self.generation = 0
        self.max_gen = params['max_generations']
        self.gen_inicio_coleta = int(params.get('gen_inicio_coleta', GEN_INICIO_COLETA))
        self.history = {'gen': [], 'mean_P': [], 'var_P': [], 'n_species': [], 'n_rodadas': [], 'n_pop': []}
        self.done = False
        self._next_uid = self.target_f + self.target_m
        self.cluster_of_uid = {}
        self._pedigree_buffer = []
        self._pairs_buffer = []
        self._final_saved = False

        if params.get('save_pedigree', False):
            f_phen, m_phen = self.P_f.mean(axis=1), self.P_m.mean(axis=1)
            for uid, ph, isl in zip(np.arange(self.target_f), f_phen, self.island_f):
                self.full_registry[int(uid)] = dict(gen=0, sex='F', mother=None, father=None, own_pheno=float(ph), island_id=int(isl))
            for uid, ph, isl in zip(np.arange(self.target_f, self.target_f + self.target_m), m_phen, self.island_m):
                self.full_registry[int(uid)] = dict(gen=0, sex='M', mother=None, father=None, own_pheno=float(ph), island_id=int(isl))
            
            path = self.p.get('individuals_file')
            if path:
                with open(path, 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['uid', 'gen', 'sex', 'mother_id', 'father_id', 'own_pheno', 'cluster', 'cluster_gen', 'island_id'])
                    for uid in np.arange(self.target_f):
                        w.writerow([int(uid), 0, 'F', '', '', round(float(f_phen[uid]), 4), -1, 0, int(self.island_f[uid])])
                    for uid in np.arange(self.target_f, self.target_f + self.target_m):
                        w.writerow([int(uid), 0, 'M', '', '', round(float(m_phen[uid - self.target_f]), 4), -1, 0, int(self.island_m[uid - self.target_f])])
            path = self.p.get('pairs_file')
            if path:
                with open(path, 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['gen', 'mother_id', 'father_id', 'mother_cluster', 'father_cluster', 'same_cluster', 'n_offspring', 'mother_island', 'father_island'])

        self._record()
        if HAS_SKLEARN: self._analyze_species()

    def _record(self):
        self.history['gen'].append(self.generation)
        allP = np.concatenate([self.P_f, self.P_m], axis=0).astype(np.float32)
        n_pop = int(allP.shape[0])
        if n_pop > 0:
            self.history['mean_P'].append(allP.mean(axis=0))
            self.history['var_P'].append(float(allP.var()))
        else:
            self.history['mean_P'].append(np.zeros(self.n_loci, dtype=np.float32))
            self.history['var_P'].append(0.0)
        self.history.setdefault('n_pop', []).append(n_pop)

    def _apply_migration(self, islands, n_migrate):
        if n_migrate <= 0 or len(islands) == 0 or self.n_islands <= 1: return islands
        n_migrate = min(n_migrate, len(islands))
        new_islands = islands.copy()
        mig_idx = self.rng.choice(len(islands), size=n_migrate, replace=False)
        for idx in mig_idx:
            cur = int(new_islands[idx])
            others = [j for j in range(self.n_islands) if j != cur]
            if others: new_islands[idx] = self.rng.choice(others)
        return new_islands

    def _save_final_genotypes(self):
        path = self.p.get('geno_file')
        if path:
            np.savez(path, uids_f=self._uid_f, uids_m=self._uid_m, D_f=self.D_f, D_m=self.D_m, P_f=self.P_f, P_m=self.P_m, island_f=self.island_f, island_m=self.island_m)

    def step(self):
        if self.generation >= self.max_gen:
            self.done = True
            return False

        seed_i = int(self.rng.integers(0, 2**31 - 1))
        pedigree_log = []

        (D_f_new, P_f_new, D_m_new, P_m_new, isl_f_new, isl_m_new, n_rodadas, i_f, i_m) = reproduce(
            self.D_f, self.P_f, self.D_m, self.P_m, self.island_f, self.island_m,
            self.target_pop, self.n_offspring, self.FemSort, self.PreSelec, self.W, self.p['mutation_rate'],
            self.n_loci, self.n_states, seed_i, np.arange(self.target_f), np.arange(self.target_m),
            self.prop_m_fixed, self.n_islands, pedigree_log=pedigree_log)

        gen_filhos = self.generation + 1
        c_idx = D_f_new.shape[0] + D_m_new.shape[0]
        if c_idx == 0:
            self.done = True
            return False

        if self.migration_rate > 0 and self.n_islands > 1:
            isl_f_new = self._apply_migration(isl_f_new, int(self.migration_rate * len(isl_f_new)))
            isl_m_new = self._apply_migration(isl_m_new, int(self.migration_rate * len(isl_m_new)))

        uid_f_pais, uid_m_pais = self._uid_f, self._uid_m

        if self.p.get('save_pedigree', False) and c_idx > 0:
            uid_base = self._next_uid
            self._next_uid += c_idx
            mother_of_pos = np.full(c_idx, -1, dtype=np.int64)
            father_of_pos = np.full(c_idx, -1, dtype=np.int64)
            pos = 0
            for (f_local, m_local, nkids) in pedigree_log:
                mother_of_pos[pos:pos + nkids] = int(uid_f_pais[f_local])
                father_of_pos[pos:pos + nkids] = int(uid_m_pais[m_local])
                pos += nkids

            pheno_pos = np.zeros(c_idx, dtype=np.float32)
            if len(i_f) > 0: pheno_pos[i_f] = P_f_new.mean(axis=1)
            if len(i_m) > 0: pheno_pos[i_m] = P_m_new.mean(axis=1)
            
            island_pos = np.zeros(c_idx, dtype=np.int32)
            island_pos[i_f], island_pos[i_m] = isl_f_new, isl_m_new
            
            sex_pos = np.empty(c_idx, dtype='<U1')
            sex_pos[i_f], sex_pos[i_m] = 'F', 'M'

            for k in range(c_idx):
                uid = uid_base + k
                self.full_registry[int(uid)] = dict(gen=gen_filhos, sex=str(sex_pos[k]), mother=int(mother_of_pos[k]) if mother_of_pos[k] >= 0 else None, father=int(father_of_pos[k]) if father_of_pos[k] >= 0 else None, own_pheno=float(pheno_pos[k]), island_id=int(island_pos[k]))
                self._pedigree_buffer.append([uid, gen_filhos, str(sex_pos[k]), '' if mother_of_pos[k] < 0 else int(mother_of_pos[k]), '' if father_of_pos[k] < 0 else int(father_of_pos[k]), round(float(pheno_pos[k]), 4), -1, gen_filhos, int(island_pos[k])])

            self._uid_f, self._uid_m = (uid_base + i_f).astype(np.int64), (uid_base + i_m).astype(np.int64)
        else:
            self._uid_f, self._uid_m = np.arange(len(D_f_new), dtype=np.int64), np.arange(len(D_m_new), dtype=np.int64)

        self.D_f, self.P_f, self.D_m, self.P_m = D_f_new, P_f_new, D_m_new, P_m_new
        self.island_f, self.island_m = isl_f_new.astype(np.int32), isl_m_new.astype(np.int32)

        for uid, isl in zip(self._uid_f, self.island_f): self.island_of_uid[int(uid)] = int(isl)
        for uid, isl in zip(self._uid_m, self.island_m): self.island_of_uid[int(uid)] = int(isl)

        self.generation += 1
        self._record()
        if HAS_SKLEARN: self._analyze_species()

        if self.p.get('save_pedigree', False) and pedigree_log:
            for (f_local, m_local, nkids) in pedigree_log:
                self._pairs_buffer.append((gen_filhos, int(uid_f_pais[f_local]), int(uid_m_pais[m_local]), int(nkids)))

        if self.p.get('save_pedigree', False):
            self._flush_individuals()
            self._flush_pairs()

        if self.generation >= self.max_gen and not self._final_saved:
            self._save_final_genotypes()
            self._final_saved = True
        return True

    def _analyze_species(self):
        if not HAS_SKLEARN or self.generation < self.gen_inicio_coleta: return
        P = np.concatenate([self.P_f, self.P_m], axis=0).astype(np.float32)
        if len(P) < 50: return
        Pc = P - P.mean(axis=0, keepdims=True)
        try: U, S, Vt = np.linalg.svd(Pc, full_matrices=False); proj = U[:, :5] * S[:5]
        except np.linalg.LinAlgError: proj = Pc[:, :5]

        db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min).fit(proj)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        self.history['n_species'].append((self.generation, n_clusters))

        uids = np.concatenate([self._uid_f, self._uid_m])
        for uid, lab in zip(uids, labels): self.cluster_of_uid[int(uid)] = int(lab)
        self.cluster_snapshot_por_gen[self.generation] = {int(uid): int(lab) for uid, lab in zip(uids, labels)}

        if self.p.get('save_pedigree', False):
            path = self.p.get('clusters_gen_file')
            if path:
                file_exists = os.path.isfile(path)
                with open(path, 'a', newline='') as f:
                    w = csv.writer(f)
                    if not file_exists: w.writerow(['gen', 'uid', 'cluster'])
                    for uid, lab in zip(uids, labels): w.writerow([self.generation, int(uid), int(lab)])

    def collect(self):
        gen_fim = int(self.generation)
        ns_por_gen = {int(g): int(n) for g, n in self.history.get('n_species', [])}
        hist = self.history
        metricas = []
        for i, g in enumerate(hist.get('gen', [])):
            g = int(g)
            if g < self.gen_inicio_coleta: continue
            n_pop, var_P = int(hist['n_pop'][i]), float(hist['var_P'][i])
            n_species = ns_por_gen.get(g, 0)
            snap = self.cluster_snapshot_por_gen.get(g)
            frac_na = (sum(1 for c in snap.values() if c == -1) / len(snap)) if snap else 0.0
            colapsou = 1 if (n_pop == 0 or var_P == 0.0) else 0
            metricas.append({'gen': g, 'n_pop': n_pop, 'var_P': var_P, 'n_species': n_species, 'frac_nao_atribuidos': float(frac_na), 'colapsou': colapsou})

        clusters_final = {int(uid): int(c) for uid, c in self.cluster_of_uid.items()}
        ind_final = {'uids_f': np.asarray(self._uid_f, dtype=np.int64), 'uids_m': np.asarray(self._uid_m, dtype=np.int64),
                     'D_f': np.asarray(self.D_f, dtype=np.uint8), 'D_m': np.asarray(self.D_m, dtype=np.uint8),
                     'P_f': np.asarray(self.P_f, dtype=np.uint8), 'P_m': np.asarray(self.P_m, dtype=np.uint8),
                     'island_f': np.asarray(self.island_f, dtype=np.int32), 'island_m': np.asarray(self.island_m, dtype=np.int32)}

        return {'metricas': metricas, 'clusters_final': clusters_final, 'ind_final': ind_final, 'seed': int(self.p['seed']), 'gen_fim': gen_fim, 'replica': None, 'elapsed_s': 0.0}

    def _flush_individuals(self):
        if not self._pedigree_buffer: return
        path = self.p.get('individuals_file')
        if not path:
            self._pedigree_buffer.clear(); return
        file_exists = os.path.isfile(path)
        with open(path, 'a', newline='') as f:
            w = csv.writer(f)
            if not file_exists: w.writerow(['uid', 'gen', 'sex', 'mother_id', 'father_id', 'own_pheno', 'cluster', 'cluster_gen', 'island_id'])
            for r in self._pedigree_buffer:
                r[6] = self.cluster_of_uid.get(int(r[0]), -1)
                r[7] = self.generation
                w.writerow(r)
        self._pedigree_buffer.clear()

    def _flush_pairs(self):
        if not self._pairs_buffer: return
        path = self.p.get('pairs_file')
        if not path:
            self._pairs_buffer.clear(); return
        file_exists = os.path.isfile(path)
        with open(path, 'a', newline='') as f:
            w = csv.writer(f)
            if not file_exists: w.writerow(['gen', 'mother_id', 'father_id', 'mother_cluster', 'father_cluster', 'same_cluster', 'n_offspring', 'mother_island', 'father_island'])
            for (gen, fid, mid, noff) in self._pairs_buffer:
                cf, cm = self.cluster_of_uid.get(int(fid), -1), self.cluster_of_uid.get(int(mid), -1)
                w.writerow([gen, int(fid), int(mid), cf, cm, int(cf == cm and cf >= 0), int(noff), self.island_of_uid.get(int(fid), -1), self.island_of_uid.get(int(mid), -1)])
        self._pairs_buffer.clear()