"""
Batch ESMFold2 protein-protein interaction (PPI) prediction template.

Runs a set of two-chain (A/B) protein complexes through ESMFold2,
each across multiple seeds and diffusion samples, and saves the
predicted structures + confidence metrics.

Fill in PPI_PAIRS below with your real sequences before running.
"""

import os
from esm.models.esmfold2 import (
    ESMFold2InputBuilder,
    ProteinInput,
    StructurePredictionInput,
)
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

# ---------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------

NUM_SEEDS = 5              # independent seeds per pair
NUM_DIFFUSION_SAMPLES = 5  # samples generated per seed
NUM_LOOPS = 20
NUM_SAMPLING_STEPS = 100

OUTPUT_DIR = "/mnt/scratch/ef2/run_label"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# 2. Load protein-protein interaction pairs from a CSV file
#    Expected columns: "PDB id", "Sequence 1", "Sequence 2"
# ---------------------------------------------------------------------

import csv as _csv  # aliased to avoid clashing with the csv import used later

CSV_PATH = "/mnt/sctach/input_csv"

PPI_PAIRS = []
with open(CSV_PATH, newline="") as f:
    reader = _csv.DictReader(f)
    for row in reader:
        pdb_id = row["PDB id"].strip()
        seq_a = row["Sequence 1"].strip()
        seq_b = row["Sequence 2"].strip()
        if not pdb_id or not seq_a or not seq_b:
            continue  # skip any blank/incomplete rows
        PPI_PAIRS.append((pdb_id, seq_a, seq_b))

assert len(PPI_PAIRS) > 0, f"No PPI pairs loaded from {CSV_PATH} - check the file path/format."
print(f"Loaded {len(PPI_PAIRS)} PPI pairs from {CSV_PATH}")

# ---------------------------------------------------------------------
# 3. Load model once, reuse across all runs
# ---------------------------------------------------------------------

print("Loading ESMFold2 model...")
model = ESMFold2Model.from_pretrained("biohub/ESMFold2").cuda().eval()
builder = ESMFold2InputBuilder()

# ---------------------------------------------------------------------
# 4. Run predictions: 5 seeds, each with 5 diffusion samples
# ---------------------------------------------------------------------

summary_rows = []

for pair_id, seq_a, seq_b in PPI_PAIRS:
    spi = StructurePredictionInput(
        sequences=[
            ProteinInput(id="A", sequence=seq_a),
            ProteinInput(id="B", sequence=seq_b),
        ]
    )

    for seed in range(NUM_SEEDS):
        print(f"[{pair_id}] seed={seed} running "
              f"({NUM_DIFFUSION_SAMPLES} diffusion samples)...")

        results = builder.fold(
            model,
            spi,
            num_loops=NUM_LOOPS,
            num_sampling_steps=NUM_SAMPLING_STEPS,
            num_diffusion_samples=NUM_DIFFUSION_SAMPLES,
            seed=seed,
        )

        # When num_diffusion_samples > 1, fold() returns a list of
        # results (one per diffusion sample) rather than a single result.
        if not isinstance(results, list):
            results = [results]

        for sample_idx, result in enumerate(results):
            plddt_mean = float(result.plddt.mean())
            ptm = float(result.ptm)
            iptm = float(result.iptm)

            out_name = f"{pair_id}_seed{seed}_sample{sample_idx}"
            cif_path = os.path.join(OUTPUT_DIR, f"{out_name}.cif")
            with open(cif_path, "w") as f:
                f.write(result.complex.to_mmcif())

            print(f"  -> sample {sample_idx}: pLDDT: {plddt_mean:.3f}  "
                  f"pTM: {ptm:.3f}  ipTM: {iptm:.3f}  saved: {cif_path}")

            summary_rows.append(
                {
                    "pair_id": pair_id,
                    "seed": seed,
                    "sample": sample_idx,
                    "plddt_mean": plddt_mean,
                    "ptm": ptm,
                    "iptm": iptm,
                    "cif_path": cif_path,
                }
            )

# ---------------------------------------------------------------------
# 5. Write a summary CSV of all runs
# ---------------------------------------------------------------------

import csv

summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
with open(summary_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "pair_id", "seed", "sample", "plddt_mean", "ptm", "iptm", "cif_path"
        ],
    )
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"\nDone. {len(summary_rows)} structures written to {OUTPUT_DIR}/")
print(f"Summary: {summary_path}")
