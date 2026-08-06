# NOTICE: This file has been modified from the original RIDER codebase
# (https://github.com/COLA-Laboratory/RIDER) as part of research
# conducted at Trinity Western University (2026), under the Apache
# License, Version 2.0.
#
# Modifications: structural evaluation extended with an opt-in
# return_ss flag, capturing RhoFold's secondary-structure (ss_head)
# prediction from the same forward pass used for tertiary (RMSD/GDT_TS)
# scoring, rather than discarding it via a separate predict() call.

import os
import shutil
from datetime import datetime

import numpy as np
import wandb

import torch

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from MDAnalysis.analysis.align import rotation_matrix
from MDAnalysis.analysis.rms import rmsd as get_rmsd

from src.data.data_utils import pdb_to_tensor, get_c4p_coords
from src.constants import NUM_TO_LETTER, PROJECT_PATH, RMSD_THRESHOLD, TM_THRESHOLD, GDT_THRESHOLD
from tools.rhofold.utils.alphabet import get_features


def evaluate(
    rhofold,
    dataset,
    raw_data,
    pred_seq,
    n_samples,
    device,
    save_designs=False,
    parallel_id=None,
    return_ss=False,
    ):
    """Evaluate a predicted sequence with RhoFold-based tertiary metrics.
 
    return_ss: if True, also returns a list of raw ss_head logits (one per
               sample) under results["ss_logits_list"], captured from the
               same forward pass as the coordinates -- no extra fold.
    """
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # include parallel_id to guarantee uniqueness even if two threads land in
    # the same microsecond (a real race observed with return_ss=True, which
    # takes long enough per-sample to make timestamp collisions likely)
    unique_suffix = f"{current_datetime}_{parallel_id if parallel_id is not None else 'main'}"
    results = {"samples_list": []}

    with torch.no_grad():
        data = dataset.featurizer(raw_data).to(device)
        samples = pred_seq.unsqueeze(0)
        results["samples_list"].append(samples.cpu().numpy())

        mask_coords = data.mask_coords.cpu().numpy()
        try:
            output_dir = os.path.join(wandb.run.dir, f"designs_eval/{unique_suffix}/sample0/")
        except AttributeError:
            output_dir = os.path.join(PROJECT_PATH, f"designs_eval/{unique_suffix}/sample0/")
 
        sc_score_rmsd, sc_score_tm, sc_score_gdt, ss_logits_list = self_consistency_score_rhofold(
            samples.cpu().numpy(),
            raw_data,
            mask_coords,
            rhofold,
            output_dir,
            save_designs=save_designs,
            parallel_id=parallel_id,
            return_ss=return_ss,
        )
 
        results["sc_score_rmsd"] = [sc_score_rmsd.mean()]
        results["sc_score_tm"] = [sc_score_tm.mean()]
        results["sc_score_gddt"] = [sc_score_gdt.mean()]
        results["rmsd_within_thresh"] = [(sc_score_rmsd <= RMSD_THRESHOLD).sum() / n_samples]
        results["tm_within_thresh"] = [(sc_score_tm >= TM_THRESHOLD).sum() / n_samples]
        results["gddt_within_thresh"] = [(sc_score_gdt >= GDT_THRESHOLD).sum() / n_samples]
        if return_ss:
            results["ss_logits_list"] = ss_logits_list
 
    return results


def self_consistency_score_rhofold(
    samples,
    true_raw_data,
    mask_coords,
    rhofold,
    output_dir,
    num_to_letter=NUM_TO_LETTER,
    save_designs=False,
    save_pdbs=False,
    use_relax=False,
    parallel_id=None,
    return_ss=False,
    ):
    """Compute RMSD/TM/GDT between RhoFold predictions and the reference structure.
 
    return_ss: if True, also captures raw ss_head logits per sample from the
               same forward pass used for coordinates (replicates
               rhofold.predict()'s internals directly instead of calling
               predict(), which discards output['ss']). When False,
               behavior is identical to the original (calls predict() as
               before) -- zero cost, zero risk for existing callers.
    """
    os.makedirs(output_dir, exist_ok=True)
 
    input_seq = SeqRecord(Seq(true_raw_data["sequence"]), id="input_sequence,", description="input_sequence")
    sequences = [input_seq]
 
    sc_rmsds = []
    sc_tms = []
    sc_gddts = []
    ss_logits_list = [] if return_ss else None
 
    for seq in samples:
        idx = (parallel_id + 1) if parallel_id is not None else 0
 
        sequence_str = "".join([num_to_letter[num] for num in seq])
        seq_record = SeqRecord(
            Seq(sequence_str),
            id=f"sample={idx},",
            description=f"sample={idx}",
        )
        sequences.append(seq_record)
        design_fasta_path = os.path.join(output_dir, f"design{idx}.fasta")
        SeqIO.write(seq_record, design_fasta_path, "fasta")
 
        design_pdb_path = os.path.join(output_dir, f"design{idx}.pdb")
 
        if return_ss:
            # Replicate rhofold.predict()'s internals directly so we can
            # keep output['ss'] before it's discarded -- coordinate
            # extraction path (PDB export, get_c4p_coords) stays IDENTICAL
            # to the non-ss path below, so RMSD/TM/GDT are unaffected.
            device_rho = rhofold.device
            data_dict = get_features(design_fasta_path, design_fasta_path)
            with torch.no_grad():
                outputs = rhofold.forward(
                    tokens=data_dict["tokens"].to(device_rho),
                    rna_fm_tokens=data_dict["rna_fm_tokens"].to(device_rho),
                    seq=data_dict["seq"],
                )
            final_output = outputs[-1]
            node_cords_pred = final_output["cord_tns_pred"][-1].squeeze(0)
            ss_logits_list.append(final_output["ss"])
 
            rhofold.structure_module.converter.export_pdb_file(
                data_dict["seq"],
                node_cords_pred.data.cpu().numpy(),
                path=design_pdb_path,
                chain_id=None,
                confidence=final_output["plddt"][0].data.cpu().numpy(),
                logger=None,
            )
        else:
            rhofold.predict(design_fasta_path, design_pdb_path, use_relax)
 
        _, coords, _, _ = pdb_to_tensor(
            design_pdb_path,
            return_sec_struct=False,
            return_sasa=False,
            keep_insertions=False,
        )
        coords = get_c4p_coords(coords)
        coords = coords - coords.mean(dim=0)
 
        if coords.shape[0] == mask_coords.shape[0]:
            coords = coords[mask_coords, :]
 
        other_coords = true_raw_data["coords_list"][0]
        if other_coords.shape[0] == mask_coords.shape[0]:
            ref = get_c4p_coords(other_coords)[mask_coords, :]
        else:
            ref = get_c4p_coords(other_coords)
        ref = ref - ref.mean(dim=0)
 
        rot = rotation_matrix(ref, coords)[0]
        ref = ref @ rot.T
 
        sc_rmsds.append(get_rmsd(coords, ref, superposition=True, center=True))
        sc_tms.append(get_tmscore(coords, ref))
        sc_gddts.append(get_gddt(coords, ref))
 
        if os.path.exists(design_fasta_path):
            os.unlink(design_fasta_path)
        if not save_pdbs and os.path.exists(design_pdb_path):
            os.unlink(design_pdb_path)
 
    if save_designs:
        SeqIO.write(sequences, os.path.join(output_dir, "all_designs.fasta"), "fasta")
    elif os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
        except OSError:
            pass  # best-effort cleanup; leftover temp dirs are harmless
 
    return np.array(sc_rmsds), np.array(sc_tms), np.array(sc_gddts), ss_logits_list
 


def get_tmscore(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute TM-score from aligned C4' coordinates."""
    l_target = y.shape[0]
    d0_l_target = 1.24 * np.power(l_target - 15, 1 / 3) - 1.8
    di = torch.pairwise_distance(y_hat, y)
    out = torch.sum(1 / (1 + (di / d0_l_target) ** 2)) / l_target
    if torch.isnan(out):
        return torch.tensor(0.0)
    return out


def get_gddt(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute GDT-TS from aligned C4' coordinates."""
    dist = torch.norm(y - y_hat, dim=1)

    count_1 = (dist < 1).sum() / dist.numel()
    count_2 = (dist < 2).sum() / dist.numel()
    count_4 = (dist < 4).sum() / dist.numel()
    count_8 = (dist < 8).sum() / dist.numel()
    out = torch.mean(torch.tensor([count_1, count_2, count_4, count_8]))
    if torch.isnan(out):
        return torch.tensor(0.0)
    return out
