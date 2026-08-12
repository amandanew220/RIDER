"""
validate_true_chain_boundaries.py

Replaces the sequence-midpoint chain-boundary heuristic with a real,
per-residue chain correspondence map, and checks -- BEFORE launching any
new training jobs -- whether the fix actually changes anything for each
of the 6 multi-chain targets.

Pipeline:
  1. Read the raw PDB (crystal_das14/), get ATOM-level chainIDs from
     MDAnalysis, collapse to per-residue chain assignment (checking for
     internal inconsistency rather than assuming residue.chainID).
  2. Build a residue identity list: (chain_id, resid, icode, resname)
     per raw-PDB residue, in file order.
  3. Align the raw-PDB residue sequence against processed.pt's stored
     sequence for that target (global alignment, not a positional
     assumption) to get processed_index -> raw_residue correspondence.
     Any residue that doesn't align unambiguously is flagged, not
     silently assigned.
  4. A target pair (i, j) is retained iff chain[i] == chain[j], using
     the real per-residue chain map -- generalizes to >2 chains,
     unequal chain lengths, and non-midpoint transitions automatically.
  5. Compare the resulting pair set against the ORIGINAL midpoint-based
     pair set. Report the symmetric difference per target. Only targets
     where this difference is nonzero actually need a rerun.

Run on the login node -- no GPU needed, this only reads PDB files and
processed.pt.
"""

import os
import torch
import MDAnalysis as mda
from Bio import Align

DATA_PATH = "/home/anew/scratch/geometric-rna-design/data/"
CRYSTAL_DIR = "/scratch/anew/RIDER_ss_experiment/crystal_das14"

MULTI_CHAIN_TARGETS = {
    "1XPE": 91, "1CSL": 93, "1LNT": 94, "354D": 97, "1Q9A": 76, "1X9C": 1,
}

# Standard RNA residue name -> one-letter code. Covers common terminal /
# alternate PDB naming variants. Anything not here is treated as
# non-standard (matches the project's existing '_' masking convention).
RESNAME_TO_LETTER = {
    "A": "A", "ADE": "A", "A3": "A", "A5": "A", "RA": "A",
    "C": "C", "CYT": "C", "C3": "C", "C5": "C", "RC": "C",
    "G": "G", "GUA": "G", "G3": "G", "G5": "G", "RG": "G",
    "U": "U", "URA": "U", "U3": "U", "U5": "U", "RU": "U",
}


def load_raw_data(sample_index, split_type="das"):
    data_list = list(torch.load(os.path.join(DATA_PATH, "processed.pt"),
                                 weights_only=False).values())

    def index_list_by_indices(lst, indices):
        return [lst[i] for i in indices]

    train_idx, val_idx, test_idx = torch.load(
        os.path.join(DATA_PATH, f"{split_type}_split.pt"), weights_only=False
    )
    test_list = index_list_by_indices(data_list, test_idx)
    idx = sample_index % len(test_list)
    return test_list[idx]


def dotbracket_to_pairs(db_string):
    pairs = set()
    stack = []
    for idx, ch in enumerate(db_string):
        if ch == '(':
            stack.append(idx)
        elif ch == ')':
            if stack:
                pairs.add((stack.pop(), idx))
    return pairs


def get_raw_residue_table(pdb_path):
    """
    Returns a list of dicts, one per residue in file order:
        {"chain_id": str, "resid": int, "icode": str,
         "resname": str, "letter": str or None}
    "letter" is None for non-standard residues (flagged, not guessed).

    Chain assignment is collapsed from ATOM-level chainIDs (MDAnalysis
    exposes ChainIDs per-atom, not per-residue) -- if a residue's atoms
    disagree on chain ID, it is flagged as AMBIGUOUS rather than
    silently assigned to the majority or first value.
    """
    u = mda.Universe(pdb_path)
    table = []
    ambiguous_count = 0

    for residue in u.residues:
        atom_chain_ids = set(residue.atoms.chainIDs) if hasattr(residue.atoms, "chainIDs") else set()
        if len(atom_chain_ids) == 1:
            chain_id = next(iter(atom_chain_ids))
        elif len(atom_chain_ids) == 0:
            chain_id = "UNKNOWN"
            ambiguous_count += 1
        else:
            chain_id = "AMBIGUOUS"
            ambiguous_count += 1

        icode = getattr(residue, "icode", "") or ""
        resname = residue.resname.strip()
        letter = RESNAME_TO_LETTER.get(resname, None)

        table.append({
            "chain_id": chain_id,
            "resid": int(residue.resid),
            "icode": icode,
            "resname": resname,
            "letter": letter,
        })

    if ambiguous_count > 0:
        print(f"  [WARN] {ambiguous_count} residue(s) had ambiguous/missing "
              f"chain ID at the atom level -- flagged, not guessed.")

    return table


def align_raw_to_processed(raw_table, processed_seq):
    """
    Global alignment between the raw-PDB residue sequence (standard
    residues only, letters from raw_table) and processed.pt's stored
    sequence for this target. Returns a dict: processed_index -> raw
    residue dict (chain_id, resid, icode, resname), for residues that
    align unambiguously. Unmatched processed positions are recorded
    separately and NOT silently assigned.
    """
    raw_letters = "".join(r["letter"] if r["letter"] else "?" for r in raw_table)

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -1

    alignment = aligner.align(raw_letters, processed_seq)[0]
    raw_aln, proc_aln = alignment[0], alignment[1]

    correspondence = {}
    unmatched_processed = []

    raw_ptr = 0
    proc_ptr = 0
    for raw_ch, proc_ch in zip(raw_aln, proc_aln):
        raw_is_gap = (raw_ch == "-")
        proc_is_gap = (proc_ch == "-")

        if not proc_is_gap and not raw_is_gap and raw_ch == proc_ch and raw_ch != "?":
            correspondence[proc_ptr] = raw_table[raw_ptr]
        elif not proc_is_gap:
            unmatched_processed.append(proc_ptr)

        if not raw_is_gap:
            raw_ptr += 1
        if not proc_is_gap:
            proc_ptr += 1

    return correspondence, unmatched_processed


def find_pdb_path(pdb_id):
    """Locate the raw crystal PDB file, trying common naming variants."""
    candidates = [
        os.path.join(CRYSTAL_DIR, f"{pdb_id}.pdb"),
        os.path.join(CRYSTAL_DIR, f"{pdb_id}_chainA.pdb"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # fall back: any file starting with the pdb_id
    if os.path.isdir(CRYSTAL_DIR):
        for f in os.listdir(CRYSTAL_DIR):
            if f.startswith(pdb_id):
                return os.path.join(CRYSTAL_DIR, f)
    raise FileNotFoundError(f"No crystal PDB found for {pdb_id} under {CRYSTAL_DIR}")


def midpoint_filter(pairs, seq_len):
    half = seq_len // 2
    return {(i, j) for (i, j) in pairs if (i < half) == (j < half)}


def true_chain_filter(pairs, correspondence):
    """Retain (i, j) iff both residues have a resolved chain AND
    chain[i] == chain[j]."""
    retained = set()
    for (i, j) in pairs:
        if i in correspondence and j in correspondence:
            if correspondence[i]["chain_id"] == correspondence[j]["chain_id"]:
                retained.add((i, j))
    return retained


def validate_target(pdb_id, sample_index):
    print(f"\n{'='*70}\nPDB: {pdb_id}\n{'='*70}")

    raw_data = load_raw_data(sample_index)
    processed_seq = raw_data["sequence"]
    target_dotbracket = raw_data["sec_struct_list"][0]
    all_target_pairs = dotbracket_to_pairs(target_dotbracket)

    pdb_path = find_pdb_path(pdb_id)
    raw_table = get_raw_residue_table(pdb_path)

    correspondence, unmatched = align_raw_to_processed(raw_table, processed_seq)

    # chain breakdown among successfully-mapped residues
    chain_counts = {}
    for entry in correspondence.values():
        chain_counts[entry["chain_id"]] = chain_counts.get(entry["chain_id"], 0) + 1

    midpoint_pairs = midpoint_filter(all_target_pairs, len(processed_seq))
    true_chain_pairs = true_chain_filter(all_target_pairs, correspondence)

    sym_diff = midpoint_pairs.symmetric_difference(true_chain_pairs)

    half = len(processed_seq) // 2
    print(f"raw PDB residues: {len(raw_table)}")
    print(f"processed residues: {len(processed_seq)}")
    print(f"mapped: {len(correspondence)}/{len(processed_seq)}"
          + (f"  (unmatched: {len(unmatched)})" if unmatched else ""))
    print(f"chains: " + ", ".join(f"{k}={v}" for k, v in sorted(chain_counts.items())))
    print(f"midpoint split: {half}/{len(processed_seq) - half}")
    print(f"target pairs before filtering: {len(all_target_pairs)}")
    print(f"retained by midpoint: {len(midpoint_pairs)}")
    print(f"retained by true chains: {len(true_chain_pairs)}")
    print(f"differing pair classifications: {len(sym_diff)}")
    if sym_diff:
        print(f"  differing pairs: {sorted(sym_diff)}")

    return {
        "pdb_id": pdb_id,
        "needs_rerun": len(sym_diff) > 0,
        "sym_diff_count": len(sym_diff),
        "midpoint_pairs": midpoint_pairs,
        "true_chain_pairs": true_chain_pairs,
        "unmatched_count": len(unmatched),
    }


def main():
    results = []
    for pdb_id, sample_index in MULTI_CHAIN_TARGETS.items():
        try:
            results.append(validate_target(pdb_id, sample_index))
        except Exception as e:
            print(f"\n[ERROR] {pdb_id}: {e}")
            results.append({"pdb_id": pdb_id, "needs_rerun": None, "error": str(e)})

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    needs_rerun = [r["pdb_id"] for r in results if r.get("needs_rerun")]
    no_change = [r["pdb_id"] for r in results if r.get("needs_rerun") is False]
    errored = [r["pdb_id"] for r in results if r.get("needs_rerun") is None]

    print(f"Targets needing rerun (pair sets differ): {needs_rerun}")
    print(f"Targets with NO change (midpoint already correct): {no_change}")
    if errored:
        print(f"Targets that errored during validation: {errored}")
    print(f"\n-> Only {len(needs_rerun)} of 6 targets require new jobs, "
          f"not all 6.")


if __name__ == "__main__":
    main()
