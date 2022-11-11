import pandas as pd
import numpy as np
import numba as nb
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import math
from scipy.optimize import minimize, minimize_scalar, differential_evolution
import os
import sys
import json
import datetime
import argparse

# PRIVATE CONSTANT PARAMETERS
NBIN_R2_HET_HIST = 64 # 256
THREADS_PER_BLOCK = 128
COL_DTYPE = dict(SNP='U',
                 N='f4',
                 Z='f4', 
                 INFO='f4',
                 A1='U',
                 A2='U') # only SNP and trait-specific columns, other columns (CHR, BP, MAF) are taken from template


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='default_config.py', help="Path to configuration file.")
    return parser.parse_args(args)


def load_config(args):
    with open(args.config) as f:
        config = json.load(f)
    return config

# --- Data loading and processing functions ---

@nb.njit
def get_r2_het(r2, r2_idx, het):
    r2_het = np.empty_like(r2)
    for i in range(len(r2)):
        r2_het[i] = r2[i]*het[r2_idx[i]]
    return r2_het


@nb.njit
def get_ld_n_idx(ld_n):
    # ld_n: ld_n[i] = size of i-th LD block
    # Returns:
    #     ld_n_idx: ld_n_idx[i] = index of the first element of the i-th LD block in corersponding r2 and r2_idx arrays
    ld_n_idx = np.zeros(len(ld_n)+1, dtype='i8') # indices of LD blocks in r2/r2_idx vectors
    for i, n_in_ld in enumerate(ld_n):
        ld_n_idx[i+1] = ld_n_idx[i] + n_in_ld # cumsum
    return ld_n_idx

@nb.njit
def get_r2_het_hist(b2use, r2_het, ld_n):
    n2use = b2use.sum() # number of used SNPs
    r2_het_hist = np.zeros(n2use*NBIN_R2_HET_HIST, dtype='f4')
    # r2_het_hist_edges: max = 0.5, min = 0.5/NBIN_R2_HET_HIST
    r2_het_hist_edges = np.linspace(0.5/NBIN_R2_HET_HIST, 0.5, NBIN_R2_HET_HIST).astype(np.float32)
    #r2_het_hist_edges = np.array([0.5/(NBIN_R2_HET_HIST - i) for i in range(NBIN_R2_HET_HIST)], dtype='f4')
    i_r2orig = 0
    i_template2use = 0
    for i_template, nld in enumerate(ld_n):
        if b2use[i_template]:
            for x_r2_het in r2_het[i_r2orig:i_r2orig+nld]:
                for i, hist_edge in enumerate(r2_het_hist_edges):
                    if x_r2_het <= hist_edge:
                        r2_het_hist[i_template2use*NBIN_R2_HET_HIST + i] += 1
                        break
            i_template2use += 1
        i_r2orig += nld
    return r2_het_hist

@nb.njit
def prune(pval, r2, r2_idx, ld_n, b2use, r2_thresh):
    # b2use: bool vector of SNP indices in template to consider for pruning. All indices not in i2use will not survive.
    # Returns:
    #     is_survived: bool vector, True if SNP survives pruning
    assert len(pval) == len(ld_n)
    assert len(b2use) == len(ld_n)
    isort = np.argsort(pval)
    is_survived = b2use[:]
    ld_n_idx = get_ld_n_idx(ld_n)
    for i in isort:
        if is_survived[i]:
            for j in range(ld_n_idx[i], ld_n_idx[i+1]): # j = index in r2/r2_idx vector
                if r2[j] > r2_thresh:
                    is_survived[r2_idx[j]] = False
            is_survived[i] = True
    return is_survived

@nb.njit
def get_total_het_used_chr(b2use, r2_idx, het, ld_n):
    n_use = np.zeros(len(b2use), dtype='i4') # n_use[i] = number of times i-th SNP on chromosome from template was in LD with any SNP used for fitting
    i_r2orig = 0
    for i_template, nld in enumerate(ld_n):
        if b2use[i_template]:
            n_use[r2_idx[i_r2orig:i_r2orig+nld]] += 1
        i_r2orig += nld
    # (n_use != 0).sum() = number of SNPs from template appeared at least once in LD with any SNP used for fitting
    # (n_use*het).sum() = sum of het of SNPs in all LD of all SNPs used for fiting
    # n_use.sum() == number of SNPs in all LD blocks of all SNPs used for fitting
    total_n_used = (n_use!=0).sum()
    total_het_used = het[n_use!=0].sum()
    return total_het_used, total_n_used


def get_total_het_used(template_dir, snps_df, rand_prune_seed, r2_prune_thresh):
    rng = np.random.default_rng(rand_prune_seed)
    total_het, total_n = 0, 0
    for chrom in snps_df.CHR.unique():
        ld_r2_file = os.path.join(template_dir, f'chr{chrom}.ld_r2')
        ld_idx_file = os.path.join(template_dir, f'chr{chrom}.ld_idx')
        r2 = np.memmap(ld_r2_file, dtype='f4', mode='r')
        r2_idx = np.memmap(ld_idx_file, dtype='i4', mode='r')
        snps_df_chr = snps_df.loc[snps_df.CHR == chrom,:]
        het = 2*snps_df_chr.MAF.values*(1 - snps_df_chr.MAF.values)
        ld_n = snps_df_chr.LD_N.values
            
        b2use = snps_df_chr.IS_VALID.values
        rand_pval = rng.random(snps_df_chr.shape[0])
        bpruned = prune(rand_pval, r2, r2_idx, ld_n, b2use, r2_prune_thresh)
        total_het_chr, total_n_chr = get_total_het_used_chr(bpruned, r2_idx, het, ld_n)
        total_het += total_het_chr
        total_n += total_n_chr
    total_het_used = len(snps_df)*total_het/total_n
    return total_het_used

def swap_z_sign(snps_df, n):
    # Change snps_df inplace. Check allele correspondence between template and sumstats.
    # Set IS_VALID = False for SNPs with allele-mismatch.
    # Swap Z sign for SNPs with swapped alleles.
    # Drop A1_i and A2_i columns (alleles loaded from sumstats).
    
    # Make reverse compliments of reference alleles
    compliments = str.maketrans("ATGC","TACG")
    make_rev_comp = lambda x: x.translate(compliments)[::-1]
    a1_rev_comp = snps_df.A1.apply(make_rev_comp)
    a2_rev_comp = snps_df.A2.apply(make_rev_comp)
    
    for i in range(n):
        z_col, a1_col, a2_col = f"Z_{i}", f"A1_{i}", f"A2_{i}"
        # check if A1 in template is A1 or A2 (both possibly reverse compliment) in sumstats,
        # if nither fits the SNP is invalid.
        i_a1_is_a1 = (snps_df.A1 == snps_df[a1_col]) & (snps_df.A2 == snps_df[a2_col])
        i_a1_is_a1 |= (a1_rev_comp == snps_df[a1_col]) & (a2_rev_comp == snps_df[a2_col])
        i_a1_is_a2 = (snps_df.A1 == snps_df[a2_col]) & (snps_df.A2 == snps_df[a1_col])
        i_a1_is_a2 |= (a1_rev_comp == snps_df[a2_col]) & (a2_rev_comp == snps_df[a1_col])
        # SNP is valid if its alleles match either directly (A1-A1) or swapped (A1-A2), both possibly with reverse compliment
        snps_df.IS_VALID &= i_a1_is_a1 | i_a1_is_a2
        snps_df.loc[i_a1_is_a2, z_col] *= -1 # if A1 in template is A2 in sumstats swap sign of Z
        snps_df.drop(columns=[a1_col, a2_col], inplace=True)
        
        
def load_snps(template_dir, sumstats, *, chromosomes=range(1,23),
              z_thresh=None, info_thresh=None, maf_thresh=None, exclude_regions=[]):
    # Load template SNPs for given chromosomes.
    # Load (multiple) sumstats.
    # Merge each sumstats with template.
    # Allign alleles in sumstats to template and swap effect direction correspondingly.
    # Add IS_VALID column which is True for SNPs for all SNPs passing specified filtering
    if isinstance(sumstats, str):
        sumstats = [sumstats]
        
    # Read template SNPs
    print(f"Reading template SNPs for {len(chromosomes)} chromosomes from {template_dir}")
    snps_df_list = []
    for chrom in chromosomes:
        snp_file = os.path.join(template_dir, f'chr{chrom}.snp.gz')
        df = pd.read_table(snp_file, dtype={"CHR":'i4',"MAF":'f4',"LD_N":'i4'})
        snps_df_list.append(df)
    snps_df = pd.concat(snps_df_list, ignore_index=True)
    print(f"    {snps_df.shape[0]} SNPs")
    snps_df["IS_VALID"] = True
    # Read sumstats
    for i, fname in enumerate(sumstats):
        print(f"Loading sumstats from {fname}")
        cols = pd.read_table(fname, nrows=0).columns
        usecols = [c for c in cols if c in COL_DTYPE]
        df = pd.read_table(fname, usecols=usecols, dtype=COL_DTYPE)
        df.drop_duplicates(subset=["SNP"], keep='first', inplace=True)
        print(f"    {df.shape[0]} SNPs")
        col_rename = {c:f"{c}_{i}" for c in usecols if c!="SNP"}
        df.rename(columns=col_rename, inplace=True)
        snps_df = pd.merge(snps_df, df, on="SNP", how="left")
        snps_df.IS_VALID &= snps_df[f"Z_{i}"].notna() & snps_df[f"N_{i}"].notna()
    print(f"{snps_df.IS_VALID.sum()} common SNPs")
    
    # swap Z_i signs of z-scores and IS_VALID = False when alleles do not correspond to reference
    swap_z_sign(snps_df, len(sumstats))
    print(f"{snps_df.IS_VALID.sum()} SNPs with matched alleles")
    
    # Apply filters
    if z_thresh:
        z_cols = [c for c in snps_df.columns if c.startswith("Z_")]
        snps_df.IS_VALID &= (snps_df[z_cols].abs() < z_thresh).all(axis="columns")
        print(f"{snps_df.IS_VALID.sum()} SNPs with Z < {z_thresh}")
    if info_thresh:
        info_cols = [c for c in snps_df.columns if c.startswith("INFO_")]
        snps_df.IS_VALID &= (snps_df[info_cols] > info_thresh).all(axis="columns")
        print(f"{snps_df.IS_VALID.sum()} SNPs with INFO > {info_thresh}")
    if maf_thresh:
        snps_df.IS_VALID &= snps_df.MAF > maf_thresh
        print(f"{snps_df.IS_VALID.sum()} SNPs with MAF > {maf_thresh}")
    for region in exclude_regions:
        chrom, start_end = region.split(":")
        chrom = int(chrom)
        start, end = map(int, start_end.split("-"))
        i_drop = (snps_df.CHR == chrom) & (snps_df.BP > start) & (snps_df.BP < end)
        snps_df.IS_VALID &= ~i_drop
        print(f"    {i_drop.sum()} SNPs excluded from {region}")
    print(f"{snps_df.IS_VALID.sum()} SNPs after all filters")   
    return snps_df

def load_opt_data(template_dir, snps_df, *, r2_prune_thresh, rand_prune_seed):
    # snps_df is produced by load_snps()
    print("Loading LD data")
    z_cols = [c for c in sorted(snps_df.columns) if c.startswith("Z_")]
    n_cols = [c for c in sorted(snps_df.columns) if c.startswith("N_")]
    assert all(z_col.split('_')[1] == n_col.split('_')[1] for z_col, n_col in zip(z_cols, n_cols))
    z_n_dict = {c:[] for c in z_cols + n_cols}
    r2_het_hist_list = []
    rng = np.random.default_rng(rand_prune_seed)
    for chrom in snps_df.CHR.unique():
        print(f"Processing chr {chrom}")
        # load template
        ld_r2_file = os.path.join(template_dir, f'chr{chrom}.ld_r2')
        ld_idx_file = os.path.join(template_dir, f'chr{chrom}.ld_idx')
        r2 = np.memmap(ld_r2_file, dtype='f4', mode='r')
        r2_idx = np.memmap(ld_idx_file, dtype='i4', mode='r')
        snps_df_chr = snps_df.loc[snps_df.CHR == chrom,:]
        het = 2*snps_df_chr.MAF.values*(1 - snps_df_chr.MAF.values)
        ld_n = snps_df_chr.LD_N.values
        # random prune
        b2use = snps_df_chr.IS_VALID.values
        rand_pval = rng.random(snps_df_chr.shape[0])
        bpruned = prune(rand_pval, r2, r2_idx, ld_n, b2use, r2_prune_thresh)
        print(f"    {bpruned.sum()} SNPs survive pruning")
        print(f"    {ld_n[bpruned].mean():.2f} mean size of LD block of pruned SNPs")
        
        r2_het = get_r2_het(r2, r2_idx, het)
        r2_het_hist = get_r2_het_hist(bpruned, r2_het, ld_n)
        r2_het_hist_list.append(r2_het_hist)
        
        for z_col, n_col in zip(z_cols, n_cols):
            z = snps_df_chr.loc[bpruned,z_col].values
            n = snps_df_chr.loc[bpruned,n_col].values
            z_n_dict[z_col].append(z)
            z_n_dict[n_col].append(n)
    r2_het_hist = np.concatenate(r2_het_hist_list)
    for col, val_list in z_n_dict.items():
        z_n_dict[col] = np.concatenate(val_list)
    print(f"{z_n_dict['Z_0'].size} SNPs loaded")
    print(f"{r2_het_hist.sum()/z_n_dict['Z_0'].size:.2f} mean size of LD block of loaded SNPs")
    return r2_het_hist, z_n_dict


# --- Univariate optimization ---

def cost1x(p, sb2, s02, n_gpu, z_gpu, r2_het_hist_gpu, res_gpu, rng_states, samples_per_thread):
    blockspergrid = z_gpu.size
    cost1x_gpu[blockspergrid, THREADS_PER_BLOCK](p, sb2, s02, n_gpu, z_gpu, r2_het_hist_gpu, res_gpu,
                                                      rng_states, samples_per_thread)
    return res_gpu.copy_to_host().mean()

@cuda.jit
def cost1x_gpu(p, sb2, s02, n, z, r2_het_hist, res, rng_states, samples_per_thread):
    r2_het_hist_shared = cuda.shared.array(NBIN_R2_HET_HIST, dtype='f4')
    res_shared = cuda.shared.array(THREADS_PER_BLOCK, dtype='f8')
    
    pos = cuda.grid(1)
    i_thread_in_block = cuda.threadIdx.x
    i_block = cuda.blockIdx.x # i_block is also index of SNP
    
    L = 1/math.log1p(-p)
    zt, nt = z[i_block], n[i_block]
    
    for i in range(i_thread_in_block, NBIN_R2_HET_HIST, THREADS_PER_BLOCK):
        r2_het_hist_shared[i] = r2_het_hist[i_block*NBIN_R2_HET_HIST + i]
    res_shared[i_thread_in_block] = 0
    
    cuda.syncthreads() # wait for all threads in the block
    
    for i in range(samples_per_thread):
        rand = xoroshiro128p_uniform_float32(rng_states, pos)
        causal_i = math.ceil(L*math.log(1 - rand))
        n_passed = 0
        se2 = 0
        for j,n_in_bin in enumerate(r2_het_hist_shared):
            n_passed += n_in_bin
            while causal_i <= n_passed:
                se2 += (0.5*j + 0.25)/NBIN_R2_HET_HIST # middle of the bin
                rand = xoroshiro128p_uniform_float32(rng_states, pos)
                causal_i += math.ceil(L*math.log(1 - rand)) # math.ceil returns float32 on GPU
        se2 = se2*sb2*nt + s02
        res_shared[i_thread_in_block] += math.exp(-0.5*zt**2/se2) / math.sqrt(2*math.pi*se2)
    
    cuda.syncthreads()
    
    if i_thread_in_block == 0:
        s = 0
        for x in res_shared: s += x
        res[i_block] = -math.log(s/(THREADS_PER_BLOCK*samples_per_thread))
        

def objf1x(par_vec, n_gpu, z_gpu, r2_het_hist_gpu, res_gpu, rng_states, samples_per_thread):
    p, sb2, s02 = par_vec
    p = 10**p
    sb2 = 10**sb2
    
    cost = cost1x(p, sb2, s02, n_gpu, z_gpu, r2_het_hist_gpu, res_gpu, rng_states, samples_per_thread)
    print(f"cost = {cost:.7f}, p = {p:.3e}, sb2 = {sb2:.3e}, s02 = {s02:.4f}", flush=True)
    return cost


def optimize1x(z, n, r2_het_hist, n_samples_grid, n_samples_local, gpu_rng_seed):
    p_lb, p_rb = -5, -2 # on log10 scale
    sb2_lb, sb2_rb = -6, -3 # on log10 scale
    s02_lb, s02_rb = 0.8, 2.5

    r2_het_hist_gpu = cuda.to_device(r2_het_hist)
    n_gpu = cuda.to_device(n)
    z_gpu = cuda.to_device(z)
    res_gpu = cuda.device_array(z.size, dtype='f8')
    rng_states = create_xoroshiro128p_states(z.size*THREADS_PER_BLOCK, seed=gpu_rng_seed)
    bounds = [(p_lb, p_rb), (sb2_lb, sb2_rb), (s02_lb, s02_rb)]
    
    samples_per_thread = int(n_samples_grid/THREADS_PER_BLOCK)
    print(f"Global opt with {samples_per_thread*THREADS_PER_BLOCK} samples per variant.")
    args_opt = (n_gpu, z_gpu, r2_het_hist_gpu, res_gpu, rng_states, samples_per_thread)
    #x0 = [-3, -4.5, 1.0]
    res = differential_evolution(objf1x, bounds, args=args_opt, maxiter=5, popsize=15, polish=False, init='sobol')
    
    
    samples_per_thread = int(n_samples_local/THREADS_PER_BLOCK)
    print(f"Local opt with {samples_per_thread*THREADS_PER_BLOCK} samples per variant.")
    x0 = res.x
    args_opt = (n_gpu, z_gpu, r2_het_hist_gpu, res_gpu, rng_states, samples_per_thread)
    res = minimize(objf1x, x0=x0, args=args_opt, method='Nelder-Mead', bounds=bounds,
            options={'maxiter':50, 'fatol':1E-5, 'xatol':1E-2})
    
    opt_par = [10**res.x[0], 10**res.x[1], res.x[2]]
    opt_res = dict(x=res.x.tolist(), fun=res.fun.tolist())
    for k in ("success", "status", "message", "nfev", "nit"):
        opt_res[k] = res.get(k)
    opt_out = dict(opt_res=opt_res, opt_par=opt_par, grid_cost=None, grid_par=None)
    return opt_out


# --- Bivariate optimization ---

def cost2x(p12, rho, rho0,
           p_1, sb2_1, s02_1, n_gpu_1, z_gpu_1,
           p_2, sb2_2, s02_2, n_gpu_2, z_gpu_2,
           r2_het_hist_gpu, res_gpu, rng_states, samples_per_thread):
    blockspergrid = z_gpu_1.size
    cost2x_gpu[blockspergrid, THREADS_PER_BLOCK](p12, rho, rho0, p_1, sb2_1, s02_1, n_gpu_1, z_gpu_1,
                                                p_2, sb2_2, s02_2, n_gpu_2, z_gpu_2,
                                                r2_het_hist_gpu, res_gpu, rng_states, samples_per_thread)
    return res_gpu.copy_to_host().mean()

@cuda.jit
def cost2x_gpu(p12, rho, rho0, p_1, sb2_1, s02_1, n_1, z_1,p_2, sb2_2, s02_2, n_2, z_2,
              r2_het_hist, res, rng_states, samples_per_thread):
    r2_het_hist_shared = cuda.shared.array(NBIN_R2_HET_HIST, dtype='f4')
    res_shared = cuda.shared.array(THREADS_PER_BLOCK, dtype='f8')
    
    pos = cuda.grid(1)
    i_thread_in_block = cuda.threadIdx.x
    i_block = cuda.blockIdx.x # i_block is also index of SNP
    
    p_causal = p_1 + p_2 - p12 # causal either in 1 or in 2
    p_causal_12 = p12/p_causal # causal in both 1 and 2
    p_causal_1 = p_1/p_causal # causal in 1
    L = 1/math.log1p(-p_causal)

    zt_1, nt_1, zt_2, nt_2 = z_1[i_block], n_1[i_block], z_2[i_block], n_2[i_block]
    
    for i in range(i_thread_in_block, NBIN_R2_HET_HIST, THREADS_PER_BLOCK):
        r2_het_hist_shared[i] = r2_het_hist[i_block*NBIN_R2_HET_HIST + i]
    res_shared[i_thread_in_block] = 0
    
    cuda.syncthreads() # wait for all threads in the block
    
    for i in range(samples_per_thread):
        rand = xoroshiro128p_uniform_float32(rng_states, pos)
        causal_i = math.ceil(L*math.log(1 - rand))
        n_passed = 0
        se2_1, se2_2 = 0, 0
        for j,n_in_bin in enumerate(r2_het_hist_shared):
            n_passed += n_in_bin
            while causal_i <= n_passed:
                r2_het = (0.5*j + 0.25)/NBIN_R2_HET_HIST # middle of the bin
                rand = xoroshiro128p_uniform_float32(rng_states, pos)
                if rand < p_causal_12:
                    se2_1 += r2_het
                    se2_2 += r2_het
                elif rand < p_causal_1:
                    se2_1 += r2_het
                else:
                    se2_2 += r2_het
                rand = xoroshiro128p_uniform_float32(rng_states, pos)
                causal_i += math.ceil(L*math.log(1 - rand)) # math.ceil returns float32 on GPU
        se2_1 = se2_1*sb2_1*nt_1
        se2_2 = se2_2*sb2_2*nt_2
        # covar matrix
        m11 = se2_1 + s02_1
        m22 = se2_2 + s02_2
        m12 = rho*math.sqrt(se2_1*se2_2) + rho0*math.sqrt(s02_1*s02_2)
        det = (m11*m22 - m12**2)
        res_shared[i_thread_in_block] += math.exp(-0.5*(m22*zt_1**2 + m11*zt_2**2 - 2*m12*zt_1*zt_2)/det) / (2*math.pi*math.sqrt(det))
    
    cuda.syncthreads()
    
    if i_thread_in_block == 0:
        s = 0
        for x in res_shared: s += x
        res[i_block] = -math.log(s/(THREADS_PER_BLOCK*samples_per_thread))


def objf2x(par_vec, p_1, sb2_1, s02_1, n_gpu_1, z_gpu_1,
           p_2, sb2_2, s02_2, n_gpu_2, z_gpu_2,
           r2_het_hist_gpu, res_gpu, rng_states, samples_per_thread,
           p12_lb, p12_rb, rho_lb, rho_rb, rho0_lb, rho0_rb):
    p12, rho, rho0 = par_vec
    p12 = 10**p12
    
    cost = cost2x(p12, rho, rho0,
           p_1, sb2_1, s02_1, n_gpu_1, z_gpu_1,
           p_2, sb2_2, s02_2, n_gpu_2, z_gpu_2,
           r2_het_hist_gpu, res_gpu, rng_states, samples_per_thread)
    print(f"cost = {cost:.7f}, p12 = {p12:.3e}, rho = {rho:.4f}, rho0 = {rho0:.4f}", flush=True)
    return cost


def optimize2x(p_1, sb2_1, s02_1, n_1, z_1,
               p_2, sb2_2, s02_2, n_2, z_2,
               r2_het_hist, n_samples_grid, n_samples_local, gpu_rng_seed):
    p12_lb, p12_rb = -5.5, math.log10(min(p_1,p_2)) # on log10 scale
    assert p12_lb < p12_rb
    rho_lb, rho_rb = -1, 1
    rho0_lb, rho0_rb = -1, 1
     
    r2_het_hist_gpu = cuda.to_device(r2_het_hist)
    n_gpu_1, n_gpu_2 = cuda.to_device(n_1), cuda.to_device(n_2)
    z_gpu_1, z_gpu_2 = cuda.to_device(z_1), cuda.to_device(z_2)
    res_gpu = cuda.device_array(z_1.size, dtype='f8')
    rng_states = create_xoroshiro128p_states(z_1.size*THREADS_PER_BLOCK, seed=gpu_rng_seed)
    bounds = [(p12_lb, p12_rb), (rho_lb, rho_rb), (rho0_lb, rho0_rb)]
    
    # rough burn-in opt
    samples_per_thread = int(n_samples_grid/THREADS_PER_BLOCK)
    print(f"Starting burn-in opt with {samples_per_thread*THREADS_PER_BLOCK} samples per variant/thread")
    args_opt = (p_1, sb2_1, s02_1, n_gpu_1, z_gpu_1, p_2, sb2_2, s02_2, n_gpu_2, z_gpu_2, r2_het_hist_gpu,
                res_gpu, rng_states, samples_per_thread, p12_lb, p12_rb, rho_lb, rho_rb, rho0_lb, rho0_rb)
    #x0 = [0.5*(p12_lb + p12_rb), -0.15, 0.0]
    res = differential_evolution(objf2x, bounds, args=args_opt, maxiter=5, popsize=15, polish=False, init='sobol')
    #res = minimize(objf2x, x0=x0, args=args_opt, method='Nelder-Mead', bounds=bounds,
    #        options={'maxiter':50, 'fatol':1E-5, 'xatol':1E-2})
    
    # refined opt
    samples_per_thread = int(n_samples_local/THREADS_PER_BLOCK)
    print(f"Starting refined opt with {samples_per_thread*THREADS_PER_BLOCK} samples per variant/thread")
    args_opt = (p_1, sb2_1, s02_1, n_gpu_1, z_gpu_1, p_2, sb2_2, s02_2, n_gpu_2, z_gpu_2, r2_het_hist_gpu,
                res_gpu, rng_states, samples_per_thread, p12_lb, p12_rb, rho_lb, rho_rb, rho0_lb, rho0_rb)
    x0 = res.x
    res = minimize(objf2x, x0=x0, args=args_opt, method='Nelder-Mead', bounds=bounds,
            options={'maxiter':100, 'fatol':1E-6, 'xatol':1E-3})
    
    opt_par = [10**res.x[0], res.x[1], res.x[2]]
    opt_res = dict(x=res.x.tolist(), fun=res.fun.tolist())
    for k in ("success", "status", "message", "nfev", "nit"):
        opt_res[k] = res.get(k)
    opt_out = dict(opt_res=opt_res, opt_par=opt_par, grid_cost=None, grid_par=None)
    return opt_out


# --- trivariate optimization ---

def cost3x(p_123,
           p_12, rho_12, rho0_12,
           p_13, rho_13, rho0_13,
           p_23, rho_23, rho0_23,
           p_1, sb2_1, s02_1, n_gpu_1, z_gpu_1,
           p_2, sb2_2, s02_2, n_gpu_2, z_gpu_2,
           p_3, sb2_3, s02_3, n_gpu_3, z_gpu_3,
           r2_het_hist_gpu, res_gpu, rng_states, samples_per_thread):
    blockspergrid = z_gpu_1.size
    cost3x_gpu[blockspergrid, THREADS_PER_BLOCK](p_123,
                                                 p_12, rho_12, rho0_12,
                                                 p_13, rho_13, rho0_13,
                                                 p_23, rho_23, rho0_23,
                                                 p_1, sb2_1, s02_1, n_gpu_1, z_gpu_1,
                                                 p_2, sb2_2, s02_2, n_gpu_2, z_gpu_2,
                                                 p_3, sb2_3, s02_3, n_gpu_3, z_gpu_3,
                                                 r2_het_hist_gpu, res_gpu, rng_states, samples_per_thread)
    return res_gpu.copy_to_host().mean()

@cuda.jit
def cost3x_gpu(p_123,
               p_12, rho_12, rho0_12,
               p_13, rho_13, rho0_13,
               p_23, rho_23, rho0_23,
               p_1, sb2_1, s02_1, n_1, z_1,
               p_2, sb2_2, s02_2, n_2, z_2,
               p_3, sb2_3, s02_3, n_3, z_3,
               r2_het_hist, res, rng_states, samples_per_thread):
    r2_het_hist_shared = cuda.shared.array(NBIN_R2_HET_HIST, dtype='f4')
    res_shared = cuda.shared.array(THREADS_PER_BLOCK, dtype='f8')
    
    pos = cuda.grid(1)
    i_thread_in_block = cuda.threadIdx.x
    i_block = cuda.blockIdx.x # i_block is also index of SNP
    
    p_causal = p_1 + p_2 + p_3 - p_12 - p_13 - p_23 + p_123 # causal either in 1 or in 2
    p_causal_123 = p_123/p_causal # causal in all three traits
    p_causal_12 = p_12/p_causal # causal in 1 and 2
    p_causal_13 = p_causal_12 + (p_13 - p_123)/p_causal # causal in 1 and 3
    p_causal_23 = p_causal_13 + (p_23 - p_123)/p_causal # causal in 2 and 3
    p_causal_1 = p_causal_23 + (p_1 - p_12 - p_13 + p_123)/p_causal # causal in 1
    p_causal_2 = p_causal_1 + (p_2 - p_12 - p_23 + p_123)/p_causal # causal in 1
    L = 1/math.log1p(-p_causal)

    zt_1, nt_1, zt_2, nt_2, zt_3, nt_3 = z_1[i_block], n_1[i_block], z_2[i_block], n_2[i_block], z_3[i_block], n_3[i_block]
    
    for i in range(i_thread_in_block, NBIN_R2_HET_HIST, THREADS_PER_BLOCK):
        r2_het_hist_shared[i] = r2_het_hist[i_block*NBIN_R2_HET_HIST + i]
    res_shared[i_thread_in_block] = 0
    
    cuda.syncthreads() # wait for all threads in the block
    
    for i in range(samples_per_thread):
        rand = xoroshiro128p_uniform_float32(rng_states, pos)
        causal_i = math.ceil(L*math.log(1 - rand))
        n_passed = 0
        se2_1, se2_2, se2_3 = 0, 0, 0
        for j,n_in_bin in enumerate(r2_het_hist_shared):
            n_passed += n_in_bin
            while causal_i <= n_passed:
                r2_het = (0.5*j + 0.25)/NBIN_R2_HET_HIST # middle of the bin
                rand = xoroshiro128p_uniform_float32(rng_states, pos)
                if rand < p_causal_123:
                    se2_1 += r2_het
                    se2_2 += r2_het
                    se2_3 += r2_het
                elif rand < p_causal_12:
                    se2_1 += r2_het
                    se2_2 += r2_het
                elif rand < p_causal_13:
                    se2_1 += r2_het
                    se2_3 += r2_het
                elif rand < p_causal_23:
                    se2_2 += r2_het
                    se2_3 += r2_het
                elif rand < p_causal_1:
                    se2_1 += r2_het
                elif rand < p_causal_2:
                    se2_2 += r2_het
                else:
                    se2_3 += r2_het
                rand = xoroshiro128p_uniform_float32(rng_states, pos)
                causal_i += math.ceil(L*math.log(1 - rand)) # math.ceil returns float32 on GPU
        se2_1 = se2_1*sb2_1*nt_1
        se2_2 = se2_2*sb2_2*nt_2
        se2_3 = se2_2*sb2_2*nt_3
        # covar matrix
        m11 = se2_1 + s02_1 
        m22 = se2_2 + s02_2
        m33 = se2_3 + s02_3
        m12 = rho_12*math.sqrt(se2_1*se2_2) + rho0_12*math.sqrt(s02_1*s02_2)
        m13 = rho_13*math.sqrt(se2_1*se2_3) + rho0_13*math.sqrt(s02_1*s02_3)
        m23 = rho_23*math.sqrt(se2_2*se2_3) + rho0_23*math.sqrt(s02_2*s02_3)
        # adjugate covar matrix
        a11 = m33*m22 - m23**2  
        a12 = m13*m23 - m33*m12  
        a13 = m12*m23 - m13*m22  
        a22 = m33*m11 - m13**2  
        a23 = m12*m13 - m11*m23
        a33 = m11*m22 - m12**2
        det = (m11 * a11) + (m12 * a12) + (m13 * a13)
        
        z_invcov_z = (a11*zt_1**2 + a22*zt_2**2 + a33*zt_3**2 + 2*(a12*zt_1*zt_2 + a13*zt_1*zt_3 + a23*zt_2*zt_3))/det
        
        res_shared[i_thread_in_block] += math.exp(-0.5*z_invcov_z) / math.sqrt(det*(2*math.pi)**3)
    
    cuda.syncthreads()
    
    if i_thread_in_block == 0:
        s = 0
        for x in res_shared: s += x
        res[i_block] = -math.log(s/(THREADS_PER_BLOCK*samples_per_thread))

        
def objf3x(p_123,
           p_12, rho_12, rho0_12,
           p_13, rho_13, rho0_13,
           p_23, rho_23, rho0_23,
           p_1, sb2_1, s02_1, n_gpu_1, z_gpu_1,
           p_2, sb2_2, s02_2, n_gpu_2, z_gpu_2,
           p_3, sb2_3, s02_3, n_gpu_3, z_gpu_3,
           r2_het_hist_gpu, res_gpu, rng_states, samples_per_thread):

    p_123 = 10**p_123
    cost = cost3x(p_123,
           p_12, rho_12, rho0_12,
           p_13, rho_13, rho0_13,
           p_23, rho_23, rho0_23,
           p_1, sb2_1, s02_1, n_gpu_1, z_gpu_1,
           p_2, sb2_2, s02_2, n_gpu_2, z_gpu_2,
           p_3, sb2_3, s02_3, n_gpu_3, z_gpu_3,
           r2_het_hist_gpu, res_gpu, rng_states, samples_per_thread)
    print(f"cost = {cost:.7f}, p_123 = {p_123:.3e}", flush=True)
    return cost


def optimize3x(p_12, rho_12, rho0_12,
               p_13, rho_13, rho0_13,
               p_23, rho_23, rho0_23,
               p_1, sb2_1, s02_1, n_1, z_1,
               p_2, sb2_2, s02_2, n_2, z_2,
               p_3, sb2_3, s02_3, n_3, z_3,
               r2_het_hist, n_samples_grid, n_samples_local, gpu_rng_seed):
    p_123_lb, p_123_rb = math.log10(max(5E-6, p_12+p_13-p_1, p_12+p_23-p_2, p_13+p_23-p_3)), math.log10(min(p_12, p_13, p_23))
    print(10**p_123_lb, 10**p_123_rb)
    if p_123_lb >= p_123_rb:
        print("Trivariate optimization is skipped")
        opt_par = [10**p_123_rb]
        opt_res = None
        grid_par = None
        grid_cost = None
    else:
        r2_het_hist_gpu = cuda.to_device(r2_het_hist)
        n_gpu_1, n_gpu_2, n_gpu_3 = cuda.to_device(n_1), cuda.to_device(n_2), cuda.to_device(n_3)
        z_gpu_1, z_gpu_2, z_gpu_3 = cuda.to_device(z_1), cuda.to_device(z_2), cuda.to_device(z_3)
        res_gpu = cuda.device_array(z_1.size, dtype='f8')
        rng_states = create_xoroshiro128p_states(z_1.size*THREADS_PER_BLOCK, seed=gpu_rng_seed)

        n_grid = 16
        grid_par = np.linspace(p_123_lb, p_123_rb, n_grid, dtype='f8')
        grid_cost = np.zeros(grid_par.shape[0], dtype='f8')
        
        samples_per_thread = int(n_samples_grid/THREADS_PER_BLOCK)
        print(f"{samples_per_thread*THREADS_PER_BLOCK} samples on grid")
        print(f"Running grid scan with {grid_par.shape[0]} points")
        for i in range(grid_par.shape[0]):
            p_123 = 10**grid_par[i]
            cost = cost3x(p_123,
                          p_12, rho_12, rho0_12,
                          p_13, rho_13, rho0_13,
                          p_23, rho_23, rho0_23,
                          p_1, sb2_1, s02_1, n_gpu_1, z_gpu_1,
                          p_2, sb2_2, s02_2, n_gpu_2, z_gpu_2,
                          p_3, sb2_3, s02_3, n_gpu_3, z_gpu_3,
                          r2_het_hist_gpu, res_gpu, rng_states, samples_per_thread)
            grid_cost[i] = cost
            if i%4 == 0: print(i, end=" ", flush=True)
        grid_cost=grid_cost.tolist()
        grid_par=grid_par.tolist()
        print("done")
        
        samples_per_thread = int(n_samples_local/THREADS_PER_BLOCK)
        print(f"{samples_per_thread*THREADS_PER_BLOCK} samples in local opt")
        print("Starting local opt")
        args_opt = (p_12, rho_12, rho0_12,
                    p_13, rho_13, rho0_13,
                    p_23, rho_23, rho0_23,
                    p_1, sb2_1, s02_1, n_gpu_1, z_gpu_1,
                    p_2, sb2_2, s02_2, n_gpu_2, z_gpu_2,
                    p_3, sb2_3, s02_3, n_gpu_3, z_gpu_3,
                    r2_het_hist_gpu, res_gpu, rng_states, samples_per_thread)
        
        res = minimize_scalar(objf3x, args=args_opt, method='bounded',
                bounds=(p_123_lb, p_123_rb), options={'maxiter':30, 'xatol':1E-4})
        
        opt_par = [10**res.x]
        opt_res = dict(x=res.x.tolist(), fun=res.fun.tolist())
        for k in ("success", "status", "message", "nfev", "nit"):
            opt_res[k] = res.get(k)
    opt_out = dict(opt_res=opt_res, opt_par=opt_par, grid_cost=grid_cost, grid_par=grid_par)
    return opt_out



if __name__ == "__main__":
    start_time = datetime.datetime.now()
    args = parse_args(sys.argv[1:])
    print(f"Loading config from: {args.config}")
    config = load_config(args)

    snps_df = load_snps(config["template_dir"], config["sumstats"],
            chromosomes=config["snp_filters"]["chromosomes"],
            z_thresh=config["snp_filters"]["z_thresh"],
            info_thresh=config["snp_filters"]["info_thresh"],
            maf_thresh=config["snp_filters"]["maf_thresh"],
            exclude_regions=config["snp_filters"]["exclude_regions"])

    r2_het_hist, z_n_dict = load_opt_data(config["template_dir"], snps_df,
            r2_prune_thresh=config["pruning"]["r2_prune_thresh"],
            rand_prune_seed=config["pruning"]["rand_prune_seed"])

    opt_out_1x_list = []
    for i, sumstats in enumerate(config["sumstats"]):
        print(f"Running univariate optimization for {sumstats}")
        z, n = z_n_dict[f"Z_{i}"], z_n_dict[f"N_{i}"]
        opt_out = optimize1x(z, n, r2_het_hist,
                config["optimization"]["n_samples_grid_1x"],
                config["optimization"]["n_samples_local_1x"],
                config["optimization"]["gpu_rng_seed"])
        opt_out_1x_list.append(opt_out)
        print(opt_out["opt_res"])
        print(f"opt_par = {opt_out['opt_par']}")

    opt_out_2x_list = []
    for i, j in ((0,1), (0,2), (1,2)):
        z_1, n_1 = z_n_dict[f"Z_{i}"], z_n_dict[f"N_{i}"]
        z_2, n_2 = z_n_dict[f"Z_{j}"], z_n_dict[f"N_{j}"]
        p_1, sb2_1, s02_1 = opt_out_1x_list[i]["opt_par"]
        p_2, sb2_2, s02_2 = opt_out_1x_list[j]["opt_par"]
        print(f"Running bivariate optimization for {config['sumstats'][i]} and {config['sumstats'][j]}")
        opt_out = optimize2x(p_1, sb2_1, s02_1, n_1, z_1, p_2, sb2_2, s02_2, n_2, z_2, r2_het_hist,
                config["optimization"]["n_samples_grid_2x"],
                config["optimization"]["n_samples_local_2x"],
                config["optimization"]["gpu_rng_seed"])
        opt_out_2x_list.append(opt_out)
        print(opt_out["opt_res"])
        print(f"opt_par = {opt_out['opt_par']}")

    z_1, n_1 = z_n_dict["Z_0"], z_n_dict["N_0"]
    z_2, n_2 = z_n_dict["Z_1"], z_n_dict["N_1"]
    z_3, n_3 = z_n_dict["Z_2"], z_n_dict["N_2"]
    p_1, sb2_1, s02_1 = opt_out_1x_list[0]["opt_par"]
    p_2, sb2_2, s02_2 = opt_out_1x_list[1]["opt_par"]
    p_3, sb2_3, s02_3 = opt_out_1x_list[2]["opt_par"]
    p_12, rho_12, rho0_12 = opt_out_2x_list[0]["opt_par"]
    p_13, rho_13, rho0_13 = opt_out_2x_list[1]["opt_par"]
    p_23, rho_23, rho0_23 = opt_out_2x_list[2]["opt_par"]
    print(f"Running trivariate optimization of {' and '.join(config['sumstats'])}")
    opt_out = optimize3x(p_12, rho_12, rho0_12, p_13, rho_13, rho0_13, p_23, rho_23, rho0_23,
                      p_1, sb2_1, s02_1, n_1, z_1, p_2, sb2_2, s02_2, n_2, z_2,  p_3, sb2_3, s02_3, n_3, z_3,
                      r2_het_hist, config["optimization"]["n_samples_grid_3x"],
                      config["optimization"]["n_samples_local_3x"], config["optimization"]["gpu_rng_seed"])
    print(opt_out["opt_res"])
    print(f"opt_par = {opt_out['opt_par']}")

    out_dict = dict(config=config, opt_out_1x_list=opt_out_1x_list, opt_out_2x_list=opt_out_2x_list, opt_out_3x=opt_out,
            start_time=str(start_time), end_time=str(datetime.datetime.now()))

    with open(config["out"], 'w') as f:
        json.dump(out_dict, f, indent=4)

    print("Done")
