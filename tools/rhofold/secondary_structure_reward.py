"""
secondary_structure_reward.py

Full implementation of the R_SS secondary-structure reward term for RIDER,
using RhoFold's own (currently unused) ss_head output.

Pipeline:
    1. ss_head logits -> sigmoid -> pairing probability matrix
    2. Nussinov-style DP decoding -> valid, non-crossing, one-partner-per-
       residue predicted base-pair set (pseudoknots excluded by construction,
       matching the project's stated scope)
    3. Target dot-bracket string (from sec_struct_list) -> target base-pair set
    4. TP/FP/FN -> base-pair F1 (R_SS)
    5. R_combined = R_RIDER + lambda_SS * R_SS

No external dependencies beyond torch and numpy.
"""

import torch
import numpy as np


# ---------------------------------------------------------------------------
# 1. Target parsing: dot-bracket -> base pairs
# ---------------------------------------------------------------------------

def dotbracket_to_pairs(db_string: str, seq_len: int = None) -> set:
    """
    Convert a dot-bracket secondary structure string into a set of
    (i, j) base-pair index tuples, i < j, 0-indexed.

    Only handles simple nested pairs: '(' / ')'. Pseudoknot bracket types
    (e.g. '[', ']', '{', '}') are NOT handled here, matching the project's
    stated scope of excluding pseudoknotted targets (e.g. 1F27, 1L2X) from
    this experiment. If such symbols are present, they are ignored (treated
    as unpaired '.'), and a warning is printed.

    Args:
        db_string: dot-bracket string, possibly padded with trailing '.'
                    characters out to some fixed dataset-wide max length
                    (observed in practice: sec_struct_list entries are
                    padded well past the actual sequence length, e.g. a
                    67nt target's string may be 614 characters long, with
                    the real content ending at index 66).
        seq_len: if given, trims db_string to db_string[:seq_len] before
                  parsing. Pass the actual folded sequence length here to
                  align target pairs with a same-length predicted
                  structure. If None (default), the string is parsed as
                  given -- safe for already-trimmed input, but padded
                  input will still parse correctly since trailing dots
                  contribute no pairs, this is purely for index alignment
                  with a shorter predicted structure, not correctness.

    Example:
        "((..))" -> {(0, 5), (1, 4)}
        "((..))......" with seq_len=6 -> {(0, 5), (1, 4)}  (padding trimmed)
    """
    if seq_len is not None:
        db_string = db_string[:seq_len]

    pairs = set()
    stack = []
    has_pseudoknot_symbols = False

    for idx, ch in enumerate(db_string):
        if ch == '(':
            stack.append(idx)
        elif ch == ')':
            if not stack:
                raise ValueError(
                    f"Unbalanced dot-bracket string at position {idx}: "
                    f"closing bracket with no matching open bracket."
                )
            j = idx
            i = stack.pop()
            pairs.add((i, j))
        elif ch in '[]{}':
            has_pseudoknot_symbols = True
        elif ch != '.':
            raise ValueError(f"Unrecognized character '{ch}' in dot-bracket string.")

    if stack:
        raise ValueError(
            f"Unbalanced dot-bracket string: {len(stack)} unmatched opening "
            f"bracket(s)."
        )

    if has_pseudoknot_symbols:
        print("WARNING: pseudoknot bracket symbols found in target string but "
              "not decoded. This target should likely be excluded from the "
              "non-pseudoknot experiment scope (see 1F27 / 1L2X exclusion).")

    return pairs


def filter_and_remap_pairs(full_dotbracket: str, keep_mask: list) -> set:
    """
    Correctly handle non-standard residues that appear in the MIDDLE of a
    sequence (not just as trailing padding), e.g. a co-crystallized ligand
    parsed as a '_' residue embedded within an RNA chain.

    Naive approaches (strip '_' from seq, then slice sec_struct to the new
    shorter length) misalign indices whenever a removed position isn't at
    the very end, and can orphan a bracket's partner entirely, causing
    dotbracket_to_pairs to raise on unbalanced brackets.

    This function instead: parses ALL pairs from the full, original,
    unmodified dot-bracket string (correct original indexing), keeps only
    pairs where BOTH residues survive the keep_mask filter (both are real,
    standard residues), then remaps each surviving pair's indices to the
    position they'll occupy in the compacted (filtered) sequence.

    Args:
        full_dotbracket: the ORIGINAL, untrimmed dot-bracket string,
                          same length as the original (unfiltered) seq
        keep_mask: list of bool, same length as full_dotbracket, True
                    where the residue at that position should be kept
                    (i.e. seq[i] != '_' there)

    Returns:
        set of (i, j) pairs, indexed against the COMPACTED sequence
        (i.e. ready to compare directly against a prediction made on the
        filtered/compacted sequence)
    """
    all_pairs = dotbracket_to_pairs(full_dotbracket)

    remap = {}
    new_idx = 0
    for old_idx, keep in enumerate(keep_mask):
        if keep:
            remap[old_idx] = new_idx
            new_idx += 1

    remapped_pairs = set()
    dropped = 0
    for (i, j) in all_pairs:
        if i in remap and j in remap:
            remapped_pairs.add((remap[i], remap[j]))
        else:
            dropped += 1

    if dropped > 0:
        print(f"  [INFO] {dropped} target base pair(s) involved a "
              f"non-standard residue and were dropped from the target "
              f"pair set (their partner may still be paired elsewhere).")

    return remapped_pairs


# ---------------------------------------------------------------------------
# 2. Prediction decoding: ss_head logits -> valid base-pair set
# ---------------------------------------------------------------------------

def logits_to_probs(ss_logits: torch.Tensor) -> np.ndarray:
    """
    Convert raw ss_head output (shape (bs, 1, L, L) per rf.py) into a
    symmetric (L, L) numpy probability matrix for a single sequence.
    """
    x = ss_logits
    while x.dim() > 2:
        x = x.squeeze(0)
    probs = torch.sigmoid(x).detach().cpu().numpy()
    probs = 0.5 * (probs + probs.T)
    np.fill_diagonal(probs, 0.0)
    return probs


def nussinov_decode(probs: np.ndarray, threshold: float = 0.5,
                     min_loop_length: int = 3) -> set:
    """
    Nussinov-style dynamic programming decode: finds the set of non-crossing
    base pairs that maximizes total pairing probability.
    """
    L = probs.shape[0]
    score = np.where(probs >= threshold, probs, 0.0)

    dp = np.zeros((L, L))
    backtrace = [[None] * L for _ in range(L)]

    for span in range(min_loop_length + 1, L):
        for i in range(0, L - span):
            j = i + span

            best = dp[i + 1][j]
            best_k = None

            for k in range(i + min_loop_length + 1, j + 1):
                if score[i, k] <= 0.0:
                    continue
                left = dp[i + 1][k - 1] if k - 1 >= i + 1 else 0.0
                right = dp[k + 1][j] if k + 1 <= j else 0.0
                candidate = score[i, k] + left + right
                if candidate > best:
                    best = candidate
                    best_k = k

            dp[i][j] = best
            backtrace[i][j] = best_k

    pairs = set()

    def traceback(i, j):
        if i >= j:
            return
        k = backtrace[i][j]
        if k is None:
            traceback(i + 1, j)
        else:
            pairs.add((i, k))
            traceback(i + 1, k - 1)
            traceback(k + 1, j)

    traceback(0, L - 1)
    return pairs


# ---------------------------------------------------------------------------
# 3. Base-pair F1
# ---------------------------------------------------------------------------

def base_pair_f1(predicted_pairs: set, target_pairs: set) -> dict:
    """
    Compute base-pair F1 between predicted and target pair sets.
    """
    tp = len(predicted_pairs & target_pairs)
    fp = len(predicted_pairs - target_pairs)
    fn = len(target_pairs - predicted_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


# ---------------------------------------------------------------------------
# 4. End-to-end: RhoFold output + target dot-bracket -> R_SS
# ---------------------------------------------------------------------------

def compute_r_ss(ss_logits: torch.Tensor, target_dotbracket: str = None,
                  threshold: float = 0.5, min_loop_length: int = 3,
                  seq_len: int = None, target_pairs: set = None) -> dict:
    """
    Full pipeline from a single RhoFold forward pass's ss_head output and
    the known target secondary structure, to R_SS.

    target_pairs: pre-computed set of (i, j) target base pairs, already
                   aligned to the predicted structure's indexing. Use this
                   instead of target_dotbracket when the target sequence
                   had non-standard residues removed from the MIDDLE of
                   the sequence (not just trailing padding).
    """
    probs = logits_to_probs(ss_logits)

    if seq_len is None:
        seq_len = probs.shape[0]

    predicted_pairs = nussinov_decode(probs, threshold=threshold,
                                       min_loop_length=min_loop_length)
    if target_pairs is None:
        target_pairs = dotbracket_to_pairs(target_dotbracket, seq_len=seq_len)
    stats = base_pair_f1(predicted_pairs, target_pairs)

    return {
        "predicted_pairs": predicted_pairs,
        "target_pairs": target_pairs,
        **stats,
    }


# ---------------------------------------------------------------------------
# 5. Combined reward
# ---------------------------------------------------------------------------

def combined_reward(r_rider: float, r_ss: float, lambda_ss: float) -> float:
    """
    R_combined = R_RIDER + lambda_SS * R_SS   (Eq. 3 in the proposal)
    """
    return r_rider + lambda_ss * r_ss


# ---------------------------------------------------------------------------
# Self-test with synthetic data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    target_db = "((...))."
    target_pairs = dotbracket_to_pairs(target_db)
    print(f"Target dot-bracket: {target_db}")
    print(f"Target pairs: {target_pairs}")
    assert target_pairs == {(0, 6), (1, 5)}, "dotbracket_to_pairs sanity check failed"

    L = 8
    logits = torch.full((L, L), -10.0)
    strong_pairs = [(0, 6), (1, 5)]
    for (i, j) in strong_pairs:
        logits[i, j] = 10.0
        logits[j, i] = 10.0

    result = compute_r_ss(logits, target_db, threshold=0.5, min_loop_length=3)
    print(f"\nPredicted pairs: {result['predicted_pairs']}")
    print(f"TP={result['tp']} FP={result['fp']} FN={result['fn']}")
    print(f"Precision={result['precision']:.3f} Recall={result['recall']:.3f} "
          f"F1 (R_SS)={result['f1']:.3f}")

    assert result["f1"] == 1.0, "Perfect-prediction sanity check failed"
    print("\nSanity check passed: perfect prediction yields R_SS = 1.0")

    r_rider_example = 0.62
    lambda_ss_example = 0.3
    r_combined = combined_reward(r_rider_example, result["f1"], lambda_ss_example)
    print(f"\nExample R_combined = {r_rider_example} + {lambda_ss_example} * "
          f"{result['f1']:.3f} = {r_combined:.3f}")


def filter_intrachain_pairs(pairs: set, seq_len: int) -> set:
    """
    Drop cross-chain base pairs from a pair set, keeping only pairs that
    fall entirely within the first or second "half" of the sequence.
 
    Uses a sequence-midpoint split as an approximation of the actual
    chain boundary (matches the heuristic already validated in
    check_dimer_targets.py against known multi-chain targets). This is
    an approximation, not exact chain-boundary data -- if a target's
    processed.pt entry ever exposes the real per-residue chain ID
    directly, prefer that instead of the midpoint split.
 
    Args:
        pairs: set of (i, j) base-pair tuples, i < j
        seq_len: full sequence length (used to compute the midpoint)
 
    Returns:
        set of (i, j) pairs where both i and j fall on the same side of
        the midpoint (i.e. plausible intra-chain / locally-foldable pairs)
    """
    half = seq_len // 2
    intrachain = {(i, j) for (i, j) in pairs if (i < half) == (j < half)}
    dropped = len(pairs) - len(intrachain)
    if dropped > 0:
        print(f"  [INFO] Dropped {dropped} cross-chain base pair(s) "
              f"(physically unachievable by a single self-folding "
              f"sequence). {len(intrachain)} intra-chain pairs remain "
              f"as the R_SS target.")
    return intrachain
 