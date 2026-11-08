"""
P1 s2b: pLM-NN, CryptoBench's own baseline (Skrhak et al., Bioinformatics 2025), on every prepared target.

Recipe (CryptoBench tutorial + github.com/skrhakv/esm2-generator):
  ESM-2 3B (esm2_t36_3B_UR50D), layer 33 (as CryptoBench actually did), on the full UniProt sequence in 1022-residue chunks, cut to the
  residues resolved in the structure; per-residue probabilities from the trained model on OSF
  (benchmark/best_trained, fetched by `p1_s0_data.py --plmnn`).
Deviations, both checked by `validate` against CryptoBench's published 7w19A embedding:
  - Hugging Face weights, stored float16 and run layer by layer in float32 on Apple MPS (fair-esm
    builds the whole model in fp32, ~17 GB RAM).
  - structure -> UniProt cut by our sequence alignment instead of PDBe SIFTS.
Residues with no UniProt counterpart (tags, linkers) get probability 0.

Steps (two environments):
  plm env:            python src/p1_s2b_plmnn.py embed      -> runs/p1/plmnn/x/<target>.npz
                      python src/p1_s2b_plmnn.py validate
  aepocketminer env:  python src/p1_s2b_plmnn.py predict    -> runs/p1/plmnn_out/<target>-preds.npy
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import CRYPTOBENCH, RUNS

P1 = RUNS / "p1"
WORK = P1 / "plmnn"
MODEL = CRYPTOBENCH / "raw" / "benchmark" / "best_trained"
OUT = P1 / "plmnn_out"
CHUNK = 1022                                   # ESM-2 context minus BOS/EOS, as in compute-esm.py
REPR_LAYER = 33                                # layer CryptoBench actually used (see load_esm)
EXTRA_UNIPROT = {"kras_g12d_siip": "P01116-2"}  # 5US4 is KRAS4B (isoform 2B)
THREE = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY",
         "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER",
         "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}
STANDARD = set(THREE.values())


# ---------- embed (plm env) ----------

def uniprot_seq(acc):
    import requests
    cache = WORK / "uniprot" / f"{acc}.fasta"
    if not cache.exists() or not cache.stat().st_size:
        r = requests.get(f"https://rest.uniprot.org/uniprotkb/{acc}.fasta", timeout=60)
        r.raise_for_status()
        text = r.text
        if not text.strip():
            # deleted/demerged since CryptoBench was built: take the entry's last archived version (UniSave)
            hist = requests.get(f"https://rest.uniprot.org/unisave/{acc}?format=json", timeout=60).json()["results"]
            last = max(h["entryVersion"] for h in hist)
            text = requests.get(f"https://rest.uniprot.org/unisave/{acc}?format=fasta&versions={last}", timeout=60).text
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text)
    return "".join(l.strip() for l in cache.read_text().splitlines() if not l.startswith(">"))


def load_esm():
    import torch
    from transformers import AutoTokenizer, EsmModel
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    name = "facebook/esm2_t36_3B_UR50D"
    model = EsmModel.from_pretrained(name, torch_dtype=torch.float16, low_cpu_mem_usage=True,
                                     add_pooling_layer=False).to(dev).eval()
    # Weights stay float16 (5.7 GB, like fair-esm's fp16 checkpoint) but every layer runs in float32: the
    # residual stream reaches ~1300 in a few dimensions, where float16 steps are ~1, and a float16 run
    # drifts (cosine 0.90 vs CryptoBench's 7w19A embedding).
    fl = lambda v: v.float() if torch.is_tensor(v) and v.is_floating_point() else v

    def up(m, args, kwargs):
        m.float()
        return tuple(fl(a) for a in args), {k: fl(v) for k, v in kwargs.items()}

    def down(m, args, out):          # must return None, or the hook replaces the layer output
        m.half()

    for layer in model.encoder.layer:
        layer.register_forward_pre_hook(up, with_kwargs=True)
        layer.register_forward_hook(down)
    # CryptoBench's embeddings are the output of layer 33, not the final layer 36 (cosine 1.0000 vs their
    # 7w19A file at layer 33; 0.90 at layer 36). 33 is the last layer of ESM-2 650M, the setting most likely
    # kept when they moved to 3B.
    model.encoder.emb_layer_norm_after.float()
    captured = {}

    def grab(m, args, out):
        captured["x"] = out[0]

    model.encoder.layer[REPR_LAYER - 1].register_forward_hook(grab)
    return AutoTokenizer.from_pretrained(name), model, dev, captured


def embed_seq(seq, esm):
    """ESM-2 3B layer-33 state per residue, BOS/EOS dropped, chunked like compute-esm.py."""
    import torch
    tok, model, dev, captured = esm
    parts = []
    for i in range(0, len(seq), CHUNK):
        ids = tok(seq[i:i + CHUNK], return_tensors="pt").to(dev)
        with torch.no_grad():
            model(**ids)
        parts.append(captured["x"][0, 1:-1].float().cpu().numpy())
    return np.concatenate(parts).astype(np.float32)


def esm_for(acc, esm):
    cache = WORK / "esm" / f"{acc}.npy"
    if cache.exists():
        return np.load(cache)
    e = embed_seq(uniprot_seq(acc), esm)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, e)
    return e


def structure_residues(pdb, chains=None):
    """(key, residue name) of protein residues with a Cα, file order (= p1_adapters.ca_table order)."""
    import gemmi
    st = gemmi.read_structure(str(pdb))
    return [(f"{c.name}_{r.seqid.num}{r.seqid.icode.strip()}", "MET" if r.name == "MSE" else r.name)
            for c in st[0] if chains is None or c.name in chains
            for r in c if r.name in STANDARD | {"MSE"} and r.find_atom("CA", "*")]


def cut(residues, seqs, chain_order):
    """Embedding row per structure residue: align each chain to its UniProt sequence (mutations kept)."""
    from p1_s1_prep import align_names
    X = np.zeros((len(residues), 2560), np.float32)
    mapped = np.zeros(len(residues), bool)
    for chain, (seq, emb) in zip(chain_order, seqs):
        idx = [i for i, (k, _) in enumerate(residues) if k.split("_")[0] == chain]
        for i, j in align_names([residues[k][1] for k in idx], [THREE.get(a, "UNK") for a in seq], identical=False):
            X[idx[i]] = emb[j]
            mapped[idx[i]] = True
    return X, mapped


def uniprot_for(tid, ds):
    if tid in EXTRA_UNIPROT:
        return [EXTRA_UNIPROT[tid]]
    apo, holo = tid.split("_")
    h = next(h for h in ds[apo] if h["holo_pdb_id"] == holo and h["is_main_holo_structure"])
    return h["uniprot_id"].split("-")


def embed():
    ds = json.loads((CRYPTOBENCH / "raw" / "dataset.json").read_text())
    esm = load_esm()
    (WORK / "x").mkdir(parents=True, exist_ok=True)
    metas = sorted(P1.glob("*/meta.json"))
    log = []
    for n, m in enumerate(metas, 1):
        meta = json.loads(m.read_text())
        tid, chains = meta["target_id"], meta["apo_chains"]
        out = WORK / "x" / f"{tid}.npz"
        if out.exists():
            continue
        try:
            accs = uniprot_for(tid, ds)
            accs = accs * len(chains) if len(accs) < len(chains) else accs   # homomer: one accession
            res = structure_residues(m.parent / f"{tid}.pdb")
            seqs = [(uniprot_seq(a), esm_for(a, esm)) for a in accs]
            X, mapped = cut(res, seqs, chains)
            np.savez(out, X=X, mapped=mapped, keys=np.array([k for k, _ in res]))
            log.append(f"{tid}\tok\t{mapped.mean():.3f} mapped")
        except Exception as e:
            log.append(f"{tid}\tfailed\t{type(e).__name__}: {e}")
        print(f"[{n}/{len(metas)}] {log[-1]}", flush=True)
    (WORK / "embed.log").write_text("\n".join(log) + "\n")


def validate():
    """Our embedding of 7w19 chain A vs the one CryptoBench ships with its tutorial."""
    import requests
    ref = WORK / "validate" / "7w19A_cryptobench.npy"
    ref.parent.mkdir(parents=True, exist_ok=True)
    if not ref.exists():
        url = "https://raw.githubusercontent.com/skrhakv/CryptoBench/master/tutorial/benchmark-tutorial/data/7w19A.npy"
        ref.write_bytes(requests.get(url, timeout=120).content)
    theirs = np.load(ref).astype(np.float32)
    ds = json.loads((CRYPTOBENCH / "raw" / "dataset.json").read_text())
    acc = next(h["uniprot_id"] for h in ds["7w19"] if h["is_main_holo_structure"])
    res = structure_residues(CRYPTOBENCH / "raw" / "cif" / "7w19.cif.gz", chains={"A"})
    X, mapped = cut(res, [(uniprot_seq(acc), esm_for(acc, load_esm()))], ["A"])
    X = X[mapped].astype(np.float32)
    np.save(WORK / "validate" / "7w19A_ours.npy", X)
    print(f"rows: ours {len(X)} (of {len(res)} residues), CryptoBench {len(theirs)}")
    if len(X) != len(theirs):
        print("row counts differ: cannot compare row by row")
        return
    cos = (X * theirs).sum(1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(theirs, axis=1))
    print(f"per-residue cosine similarity: median {np.median(cos):.4f}, min {cos.min():.4f}")
    print("PASS" if np.median(cos) > 0.99 else "CHECK: embeddings differ more than float16 rounding explains")


# ---------- predict (aepocketminer env, TensorFlow) ----------

def load_plmnn():
    import tensorflow as tf
    # tf.saved_model.load avoids keras.models.load_model, which needs tensorflow_addons for a training metric
    f = tf.saved_model.load(str(MODEL)).signatures["serving_default"]
    name = list(f.structured_input_signature[1].keys())[0]

    def run(X):
        out = list(f(**{name: tf.constant(X, tf.float32)}).values())[0].numpy()
        return out[:, 1] if out.ndim == 2 and out.shape[1] == 2 else out.reshape(-1)
    return run


def predict():
    run = load_plmnn()
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted((WORK / "x").glob("*.npz"))
    for z in files:
        d = np.load(z)
        p = np.zeros(len(d["keys"]), np.float32)
        if d["mapped"].any():
            p[d["mapped"]] = run(d["X"][d["mapped"]].astype(np.float32))
        np.save(OUT / f"{z.stem}-preds.npy", p)
    print(f"pLM-NN predictions for {len(files)} targets -> {OUT}")
    ours, theirs = WORK / "validate" / "7w19A_ours.npy", WORK / "validate" / "7w19A_cryptobench.npy"
    if ours.exists() and theirs.exists():
        a, b = run(np.load(ours)), run(np.load(theirs).astype(np.float32))
        if len(a) == len(b):
            print(f"7w19A predictions, ours vs CryptoBench embedding: max |diff| {np.abs(a - b).max():.4f}, "
                  f"r = {np.corrcoef(a, b)[0, 1]:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["embed", "validate", "predict"])
    {"embed": embed, "validate": validate, "predict": predict}[ap.parse_args().step]()
