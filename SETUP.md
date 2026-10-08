# Setup

Developed on an Apple Silicon Mac (osx-arm64). Some steps below are specific to that.

## 1. Conda environments

```bash
conda env create -f envs/structprep.yml      # ProDy, fpocket, Vina, Meeko, RDKit, Java 17
conda env create -f envs/aepocketminer.yml   # AE-PocketMiner, TensorFlow 2.13
```

- [ ] `conda activate structprep`, then check:
  ```bash
  fpocket -h
  vina --version
  java -version        # must show 17.x
  ```

If you'd rather build `structprep` by hand:

```bash
conda create -n structprep python=3.10
conda activate structprep
conda install -c conda-forge pymol-open-source pdb2pqr fpocket rdkit scipy gemmi vina openjdk=17
pip install meeko prody spyrmsd
```

## 2. P2Rank 2.4.2

- [ ] Download from https://github.com/rdk/p2rank/releases and unpack it in the repo root, so `p2rank_2.4.2/prank` exists.
- [ ] **Run it with Java 17 from the activated `structprep` env.** Newer Java fails with "Unsupported class file major version 69". With no Java on the path at all, `phase3_p2rank.py` crashes with a `UnicodeDecodeError` from macOS's "Unable to locate a Java Runtime" message.
- P2Rank often ranks the nucleotide site above the SII-P. That's expected, and the 14 Å-from-His95 rule handles it.

## 3. AE-PocketMiner

- [ ] Clone the Bowman Lab `ae-pocketminer` repo into the repo root.
- [ ] On Apple Silicon, delete the `cudatoolkit` line from its `environment.yml` (there's no NVIDIA GPU). Use `envs/aepocketminer.yml` from this repo instead if you can.
- [ ] Use TensorFlow 2.13 (`pip install "tensorflow==2.13.0"`). The pinned 2.10 doesn't exist for osx-arm64. Also `pip install pyyaml`.
- [ ] **Required code fix** in `ae-pocketminer/src/xtal_predict.py`. Without it the checkpoint won't load on TF 2.13:
  ```python
  opt = tf.keras.optimizers.Adam()          # change this
  opt = tf.keras.optimizers.legacy.Adam()   # to this
  ```
- [ ] Create `ae-pocketminer/my_config.yaml` (regular model, not the attention variant):
  ```yaml
  nn_path: /ABSOLUTE/PATH/TO/KRASG12DBenchmarking/ae-pocketminer/models/pocketminer
  input_pdb_directory: inputs
  output_directory: results/pocketminer
  use_attention: False
  debug: False
  ```
- [ ] Run it on the ladder conformers and the two anchors:
  ```bash
  conda activate aepocketminer
  cd ae-pocketminer
  cp ../runs/ladder_dense/dense_*.pdb ../data/structures/5US4_H.pdb ../data/structures/7RPZ_H.pdb inputs/
  python src/xtal_predict.py my_config.yaml
  ```
  Output is `results/pocketminer/<name>-preds.npy`, one probability per residue.

## 4. Gotchas

- **Check which env you're in.** A new terminal starts in `(base)`. Pocket scripts and P2Rank need `structprep`. AE-PocketMiner needs `aepocketminer`. "command not found" or "No module named X" almost always means the wrong env.
- **Docking RMSD:** strip all hydrogens from both poses, assign bonds from `MRTX1133_ideal.sdf`, and use RDKit's symmetry-aware `GetBestRMS`. A naive PyMOL `rms_cur` mispairs atoms and inflates the number (it reported 5–6 Å; the true value is 0.96 Å). See `src/phase1_validation/rmsd_check.py`.
- **PyMOL vs terminal:** PyMOL commands (`load`, `align`, `iterate`) go in the PyMOL command line. Shell commands (`conda`, `fpocket`, `vina`, `python`) go in the terminal.

## Where things go

Tools (`p2rank_2.4.2/`, `ae-pocketminer/`) and everything generated (`runs/`, fpocket `*_out/` folders, `*.npy`) are gitignored. File locations used by the scripts are set in `src/paths.py`.
