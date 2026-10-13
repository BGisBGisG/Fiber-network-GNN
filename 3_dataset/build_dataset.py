import argparse
import csv
import os
import random
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

STEP_LIMIT = 10000000
D_FIRST = 0.05
D_STEP = 0.01

def read_dump(path):
    with open(path, 'rb') as fh:
        raw = fh.read()
    pos = 0
    for _ in range(9):
        pos = raw.index(b'\n', pos) + 1
    head = raw[:pos].split(b'\n')
    n = int(head[3])
    box = np.array([[float(v) for v in head[k].split()[:2]] for k in (5, 6, 7)])
    ncol = len(head[8].split()) - 2
    a = np.array(raw[pos:].split(), dtype=np.float64)
    if n == 0 or a.size == 0:
        return box, np.zeros((0, ncol))
    return box, a.reshape(n, -1)

def read_data_file(path):
    with open(path) as fh:
        txt = fh.read()
    na = int(re.search(r'(\d+) atoms', txt).group(1))
    nb = int(re.search(r'(\d+) bonds', txt).group(1))
    ia = txt.index('\nAtoms')
    ib = txt.index('\nBonds')
    ic = txt.find('\nAngles')
    atoms = np.array(txt[txt.index('\n', ia + 1):ib].split(), dtype=np.float64).reshape(na, -1)
    bonds = np.array(txt[txt.index('\n', ib + 1):(ic if ic > 0 else len(txt))].split(), dtype=np.float64).reshape(nb, -1)
    typ = np.zeros(int(atoms[:, 0].max()) + 1, np.int64)
    typ[atoms[:, 0].astype(np.int64)] = atoms[:, 2].astype(np.int64)
    return typ, bonds[:, 2].astype(np.int64), bonds[:, 3].astype(np.int64)

def spans_x(pos, length, edges):
    n = pos.shape[0]
    if len(edges) == 0:
        return False
    parent = list(range(n))
    rank = [0] * n
    pot = np.zeros((n, 3), np.int64)
    wrap = np.zeros((n, 3), bool)

    def find(a):
        path = []
        while parent[a] != a:
            path.append(a)
            a = parent[a]
        for node in reversed(path):
            p = parent[node]
            if p != a:
                pot[node] += pot[p]
            parent[node] = a
        return a

    for i, j in edges:
        i = int(i)
        j = int(j)
        delta = -np.floor((pos[j] - pos[i]) / length + 0.5).astype(np.int64)
        ri = find(i)
        si = pot[i].copy()
        rj = find(j)
        sj = pot[j].copy()
        if ri == rj:
            cyc = (sj - si) - delta
            wrap[ri] |= cyc != 0
        else:
            disp = delta + si - sj
            if rank[ri] < rank[rj]:
                parent[ri] = rj
                pot[ri] = -disp
                wrap[rj] |= wrap[ri]
            else:
                parent[rj] = ri
                pot[rj] = disp
                wrap[ri] |= wrap[rj]
                if rank[ri] == rank[rj]:
                    rank[ri] += 1
    return any(wrap[r, 0] for r in {find(a) for a in range(n)})

def build_case(path):
    name = os.path.basename(os.path.normpath(path))
    nodes = int(name[:-4])
    group = int(name)
    res = {'group': group, 'graphs': [], 'error': ''}
    try:
        info = pd.read_csv(os.path.join(path, 'informationinthesimulation'), sep=' ', skiprows=1, header=None,
                           usecols=[0, 1, 2], dtype=np.float64, engine='c').to_numpy()
        info = info[~np.isnan(info).any(axis=1)]
        stp = info[:, 0].astype(np.int64)
        last = len(stp) - 1
        typ, b1, b2 = read_data_file(os.path.join(path, 'voc2.data'))
        ca = np.nonzero(b1 <= nodes)[0]
        cb = np.nonzero(b2 <= nodes)[0]
        assert len(ca) == len(cb) and np.all(cb >= ca), 'fiber list'
        assert np.all(cb[:-1] + 1 == ca[1:]) and cb[-1] == len(b1) - 1, 'fiber list'
        nf = len(ca)
        fs = b1[ca]
        fe = b2[cb]
        dcls = (typ[b2[ca]] - 1) * D_STEP + D_FIRST
        dcls[typ[b2[ca]] == 0] = D_FIRST
        l0 = (cb - ca + 1) * dcls
        scale = np.int64(10) ** 8
        key0 = np.minimum(b1, b2) * scale + np.maximum(b1, b2)
        ukey, first = np.unique(key0, return_index=True)
        first_of = {}
        for f in range(nf - 1, -1, -1):
            first_of[(int(fs[f]), int(fe[f]))] = f
        cfg = os.path.join(path, 'cfg')
        kept = []
        for fname in sorted(f for f in os.listdir(cfg) if f.endswith('.b')):
            s = int(fname[:-2])
            if s > STEP_LIMIT:
                continue
            if s != 0:
                r = np.nonzero(stp == s)[0]
                if r.size and np.all(r != last) and not np.all(info[r, 1] != info[r + 1, 1]):
                    continue
            kept.append(s)
        labels = {0: info[0, 2]}
        for s in kept:
            if s == 0:
                continue
            r = np.nonzero(stp == s)[0]
            if r.size != 1:
                raise ValueError('step %d appears %d times in informationinthesimulation' % (s, r.size))
            labels[s] = info[r[0], 2]
        alive = True
        stop = -1
        graphs = []
        for s in kept:
            if not alive:
                break
            try:
                _, bd = read_dump(os.path.join(cfg, '%d.b' % s))
                if bd.shape[0] == 0:
                    continue
                keep = np.ones(nf, bool)
                if bd.shape[0] != len(key0):
                    p1 = bd[:, 2].astype(np.int64)
                    p2 = bd[:, 3].astype(np.int64)
                    pkey = np.minimum(p1, p2) * scale + np.maximum(p1, p2)
                    miss = first[~np.isin(ukey, pkey)]
                    keep[np.searchsorted(ca, miss, side='right') - 1] = False
                box, x = read_dump(os.path.join(cfg, '%d.d' % s))
                if x.shape[0] == 0:
                    continue
                length = box[:, 1] - box[:, 0]
                raw = np.zeros((int(x[:, 0].max()) + 1, 3))
                raw[x[:, 0].astype(np.int64)] = x[:, 2:5]
                ks = fs[keep]
                ke = fe[keep]
                if not spans_x(raw[1:nodes + 1], length, np.c_[ks, ke] - 1):
                    stop = s
                    alive = False
                    continue
                pos = raw - box[:, 0]
                dv = pos[b2] - pos[b1]
                dv -= length * np.round(dv / length)
                vk = np.add.reduceat(dv, ca, axis=0)
                cosx = np.clip(vk[:, 0] / np.sqrt((vk ** 2).sum(axis=1)), -1, 1)
                feat = np.c_[l0, vk[:, 0], np.sqrt(vk[:, 1] ** 2 + vk[:, 2] ** 2), cosx, dcls]
                rep = np.array([first_of.get((min(a, b), max(a, b)), first_of.get((max(a, b), min(a, b))))
                                for a, b in zip(ks.tolist(), ke.tolist())], dtype=np.int64)
                ei = np.c_[np.r_[ks, ke], np.r_[ke, ks]].T - 1
                ea = feat[np.r_[rep, rep]].astype(np.float32)
                deg = np.zeros(nodes + 1, np.int64)
                np.add.at(deg, ks, 1)
                np.add.at(deg, ke[ke != ks], 1)
                deg = np.clip(deg[1:], 1, 4)
                graphs.append((s, deg, ei.astype(np.int64), ea))
            except Exception as err:
                print('%s step %d skipped: %s' % (name, s, err), flush=True)
        if stop != -1:
            labels = {s: v for s, v in labels.items() if s < stop}
        res['graphs'] = [(s, deg, ei, ea, float(labels[s])) for s, deg, ei, ea in graphs if s in labels]
    except Exception as err:
        res['error'] = '%s: %s' % (type(err).__name__, err)
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('out_dir')
    ap.add_argument('--keep', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--nproc', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    assert 0 < args.keep <= 1.0
    cases = sorted((d for d in os.listdir(args.run_dir) if d.isdigit() and os.path.isdir(os.path.join(args.run_dir, d, 'cfg'))), key=int)
    paths = [os.path.join(args.run_dir, d) for d in cases]
    print('cases:', len(paths), flush=True)
    results = []
    with Pool(args.nproc) as pool:
        for k, res in enumerate(pool.imap(build_case, paths, chunksize=1)):
            results.append(res)
            print('[%d/%d] group %d: %d graphs %s' % (k + 1, len(paths), res['group'], len(res['graphs']), res['error']), flush=True)
    rng = random.Random(args.seed)
    data_list = []
    rows = []
    for res in sorted(results, key=lambda r: r['group']):
        graphs = res['graphs']
        if not graphs:
            continue
        idx = list(range(len(graphs)))
        if args.keep < 1.0:
            rng.shuffle(idx)
            idx = idx[:max(1, round(len(graphs) * args.keep))]
        for i in idx:
            s, deg, ei, ea, y = graphs[i]
            d = Data(x=torch.from_numpy(deg).long().unsqueeze(1), edge_index=torch.from_numpy(ei).long().contiguous(),
                     edge_attr=torch.from_numpy(ea).contiguous(), y=y)
            d.group = res['group']
            data_list.append(d)
            rows.append((res['group'], s, y, deg.shape[0], ei.shape[1] // 2))
    out = os.path.join(args.out_dir, 'processed')
    os.makedirs(out, exist_ok=True)
    torch.save(InMemoryDataset.collate(data_list), os.path.join(out, 'data.pt'))
    with open(os.path.join(out, 'graphs.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['group', 'step', 'stress_Pa', 'junctions', 'intact_fibers'])
        w.writerows(rows)
    bad = [r for r in results if r['error']]
    print('graphs: %d, networks: %d, cases with errors: %d' % (len(data_list), len({r[0] for r in rows}), len(bad)), flush=True)
    for r in bad:
        print('error in', r['group'], r['error'], flush=True)

if __name__ == '__main__':
    main()
