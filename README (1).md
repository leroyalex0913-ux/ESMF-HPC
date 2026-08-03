# ESMFold2 on MSU ICER HPCC

Workflow for running [Biohub's ESMFold2](https://github.com/Biohub/esm) — protein-protein (PPI) and protein-ligand (PL) structure prediction — on ICER's HPC cluster.

## Prerequisites

- An ICER HPCC account with GPU access
- A free [Hugging Face](https://huggingface.co) account
- **Accept the model license** on the [ESMFold2 model page](https://huggingface.co/biohub/ESMFold2) — this is required before you can download the weights, even with a valid token. Do this before generating a token below.

## 1. Set up a clean conda environment

Connect to a GPU dev-node for setup and testing (do not use a login node — it has no GPU):
```bash
ssh dev-amd20-v100
```

Create and activate the environment:
```bash
module purge
module load Miniforge3
conda create --name esmfold2 python=3.10 pip
conda activate esmfold2
module load CUDA/12.9.1
```

Prevent packages from silently installing into your personal `~/.local` instead of this environment (a common source of "module not found" errors that are actually a mismatched-environment problem):
```bash
export PYTHONNOUSERSITE=1
echo 'export PYTHONNOUSERSITE=1' >> ~/.bashrc
```

## 2. Install ESMFold2

```bash
python -m pip install torch torchvision torchaudio
pip install esm@git+https://github.com/Biohub/esm.git@main
```

### Set up the Hugging Face cache

Point the cache at **shared research space**, not scratch — scratch is periodically purged, and the model weights are large enough (6B parameters) that you don't want your lab re-downloading them repeatedly:
```bash
export HF_HOME=/mnt/research/woldring_lab/esmfold2/hf_cache
```
This should be set in your job script (see Step 4) before running your batch command, not just in your interactive shell.

### Authenticate with Hugging Face

1. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). **Read** permission is sufficient — ESMFold2 only downloads weights, it never writes back to Hugging Face.
2. Log in:
   ```bash
   hf auth login
   ```
3. Paste your token when prompted (format: `hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX`).

## 3. Create your input CSV

ESMFold2 supports both protein-protein (PPI) and protein-ligand (PL) interactions from a single combined CSV. See `CSV_input_example.csv` for a template.

**Columns:**

| Column | Description |
|---|---|
| `interaction_id` | Label for this interaction — used to name output files |
| `type` | `PPI` for protein-protein, `PL` for protein-ligand |
| `sequence_1` | Protein amino acid sequence (chain A) |
| `sequence_2` | For `PPI`: second protein sequence. For `PL`: ligand [CCD code](https://www.rcsb.org/ligand) (e.g. `SAH`) |

> CCD codes are more reliable than SMILES strings for ligand input — check the [RCSB Ligand Expo](https://www.rcsb.org/ligand) if you're not sure your ligand has one.

## 4. Configure and submit the batch job

**In `esmfold2_batch.py`, update:**
- `BASE_OUTPUT_DIR` — a unique output directory for this run
- `CSV_PATH` — path to your input CSV from Step 3
- `NUM_SEEDS`, `NUM_DIFFUSION_SAMPLES`, `NUM_LOOPS`, `NUM_SAMPLING_STEPS` — adjust as needed (see note on runtime below)

**In `run_esmfold.sb`, update:**
- `HF_HOME` — should point to your shared research-space cache (see Step 2), not scratch
- `--output` / `--error` — point these at your own logs directory
- `cd` — should point to the directory containing your copy of `esmfold2_batch.py`
- The `python3` command — should point to the same `esmfold2_batch.py`

> The `cd` and the full path in the `python3` line are redundant with each other, but both are needed — SLURM doesn't reliably infer the working directory otherwise.

Submit:
```bash
sbatch run_esmfold.sb
```

**Note on runtime:** more seeds/samples/loops/steps means higher accuracy but longer runtime. For large batches (dozens+ of interactions), estimate your total runtime before submitting — this scales multiplicatively (interactions × seeds × samples).

## 5. Retrieve your results

Each run writes PPI and PL results into separate subfolders, each containing:
- One `.cif` structure file per seed/sample
- A `summary.csv` recording `plddt_mean`, `ptm`, and `iptm` for every structure generated

**ESMFold2's own convention is to rank structures by ipTM score alone** when selecting the best prediction per interaction.

To automatically extract the highest-ipTM structure for each interaction:
```bash
python3 copy_best_cifs_ef2.py /path/to/summary.csv /path/to/output/directory
```
This copies the winning `.cif` file per interaction into your chosen output directory, along with a `summary_scores.csv` log of which seed/sample won and its score. Add `--dry-run` to preview what would be copied without actually copying anything.

## Troubleshooting

- **`ModuleNotFoundError` after a working install** — check `which python` points into your conda environment, not `~/.local` or system Python. Re-run `export PYTHONNOUSERSITE=1` if needed.
- **Import errors mentioning `.local`** — same root cause as above; a package installed outside the active conda environment.
- **`from_pretrained()` fails despite a valid token** — confirm you've accepted the model license on the [ESMFold2 model page](https://huggingface.co/biohub/ESMFold2).
- **`sbatch` job can't find your Python file** — check for stale absolute paths left over from renaming/moving files, and confirm line endings are Unix-style (`file yourscript.sb` should not say "CRLF").
