"""
Batch ESMFold2 template: protein-protein (PPI) AND protein-ligand (PL)
predictions from a single combined CSV.

Expected CSV columns:
    interaction_id, type, sequence_1, sequence_2
        - type: "PPI" (sequence_2 = second protein sequence)
                "PL"  (sequence_2 = ligand CCD code, e.g. "SAH", "RIT")

Runs each row through ESMFold2 across multiple seeds and diffusion
samples, and saves predicted structures + confidence metrics.
PPI and PL results are written to separate output subfolders and
separate summary CSVs, since interpreting confidence metrics differs
somewhat between the two interaction types.
"""

import os
from esm.models.esmfold2 import (
    ESMFold2InputBuilder,
    LigandInput,
    ProteinInput,
    StructurePredictionInput,
)
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

# ---------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------

NUM_SEEDS = 5              # independent seeds per pair
NUM_DIFFUSION_SAMPLES = 1  # samples generated per seed
NUM_LOOPS = 20
NUM_SAMPLING_STEPS = 100

BASE_OUTPUT_DIR = "/mnt/scratch/ef2/run_label"
PPI_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "ppi_predictions")
PL_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "ligand_predictions")
os.makedirs(PPI_OUTPUT_DIR, exist_ok=True)
os.makedirs(PL_OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# 2. Load interactions from a combined CSV file
#    Expected columns: "interaction_id", "type", "sequence_1", "sequence_2"
#        type == "PPI": sequence_2 is a second protein sequence
#        type == "PL":  sequence_2 is a ligand CCD code (e.g. "SAH")
# ---------------------------------------------------------------------

import csv as _csv  # aliased to avoid clashing with the csv import used later

CSV_PATH = "/mnt/scratch/input_csv.csv"  # update to your CSV path

ENTRIES = []  # list of (interaction_id, type, seq_1, seq_2)
with open(CSV_PATH, newline="") as f:
    reader = _csv.DictReader(f)
    for row in reader:
        interaction_id = row["interaction_id"].strip()
        row_type = row["type"].strip().upper()
        seq_1 = row["sequence_1"].strip()
        seq_2 = row["sequence_2"].strip()
        if not interaction_id or row_type not in ("PPI", "PL") or not seq_1 or not seq_2:
            continue  # skip blank/incomplete/invalid-type rows
        ENTRIES.append((interaction_id, row_type, seq_1, seq_2))

assert len(ENTRIES) > 0, f"No entries loaded from {CSV_PATH} - check the file path/format."
n_ppi = sum(1 for e in ENTRIES if e[1] == "PPI")
n_pl = sum(1 for e in ENTRIES if e[1] == "PL")
print(f"Loaded {len(ENTRIES)} entries from {CSV_PATH} ({n_ppi} PPI, {n_pl} PL)")

# ---------------------------------------------------------------------
# 3. Load model once, reuse across all runs
# ---------------------------------------------------------------------

print("Loading ESMFold2 model...")
model = ESMFold2Model.from_pretrained("biohub/ESMFold2").cuda().eval()
builder = ESMFold2InputBuilder()

# ---------------------------------------------------------------------
# 4. Run predictions: for each entry, 5 seeds x 5 diffusion samples
# ---------------------------------------------------------------------

ppi_summary_rows = []
pl_summary_rows = []

for interaction_id, row_type, seq_1, seq_2 in ENTRIES:

    if row_type == "PPI":
        spi = StructurePredictionInput(
            sequences=[
                ProteinInput(id="A", sequence=seq_1),
                ProteinInput(id="B", sequence=seq_2),
            ]
        )
        output_dir = PPI_OUTPUT_DIR
        summary_rows = ppi_summary_rows
    else:  # "PL"
        spi = StructurePredictionInput(
            sequences=[
                ProteinInput(id="A", sequence=seq_1),
                LigandInput(id="L", ccd=[seq_2]),
            ]
        )
        output_dir = PL_OUTPUT_DIR
        summary_rows = pl_summary_rows

    for seed in range(NUM_SEEDS):
        print(f"[{row_type}:{interaction_id}] seed={seed} running "
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

            out_name = f"{interaction_id}_seed{seed}_sample{sample_idx}"
            cif_path = os.path.join(output_dir, f"{out_name}.cif")
            with open(cif_path, "w") as f:
                f.write(result.complex.to_mmcif())

            print(f"  -> sample {sample_idx}: pLDDT: {plddt_mean:.3f}  "
                  f"pTM: {ptm:.3f}  ipTM: {iptm:.3f}  saved: {cif_path}")

            summary_rows.append(
                {
                    "interaction_id": interaction_id,
                    "seed": seed,
                    "sample": sample_idx,
                    "plddt_mean": plddt_mean,
                    "ptm": ptm,
                    "iptm": iptm,
                    "cif_path": cif_path,
                }
            )

# ---------------------------------------------------------------------
# 5. Write separate summary CSVs for PPI and PL results
# ---------------------------------------------------------------------

import csv

FIELDNAMES = ["interaction_id", "seed", "sample", "plddt_mean", "ptm", "iptm", "cif_path"]

def write_summary(rows, output_dir):
    summary_path = os.path.join(output_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path

if ppi_summary_rows:
    ppi_summary_path = write_summary(ppi_summary_rows, PPI_OUTPUT_DIR)
    print(f"\nPPI: {len(ppi_summary_rows)} structures written to {PPI_OUTPUT_DIR}/")
    print(f"PPI summary: {ppi_summary_path}")

if pl_summary_rows:
    pl_summary_path = write_summary(pl_summary_rows, PL_OUTPUT_DIR)
    print(f"\nPL: {len(pl_summary_rows)} structures written to {PL_OUTPUT_DIR}/")
    print(f"PL summary: {pl_summary_path}")
