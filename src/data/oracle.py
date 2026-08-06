# NOTICE: This file has been modified from the original RIDER codebase
# (https://github.com/COLA-Laboratory/RIDER) as part of research
# conducted at Trinity Western University (2026), under the Apache
# License, Version 2.0.
#
# Modifications: added a configurable oracle interface (get_oracle())
# supporting both RhoFold and AlphaFold3 as swappable RL reward oracles,
# where the original codebase used RhoFold exclusively.



import os
import subprocess
import json
from abc import ABC, abstractmethod
from datetime import datetime
import shutil

import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from src.data.data_utils import pdb_to_tensor, get_c4p_coords
from src.constants import PROJECT_PATH


class BaseOracle(ABC):
    """Abstract base class for folding oracles.
    All oracles must implement fold() with the same interface.
    """

    @abstractmethod
    def fold(self, sequence: str, output_dir: str, idx: int = 0) -> torch.Tensor:
        """
        Fold a sequence and return C4' coordinates.

        Args:
            sequence: RNA sequence string e.g. "GGGAUCACGGACU"
            output_dir: directory to write temporary files to
            idx: sample index for naming temporary files

        Returns:
            coords: torch.Tensor of shape [N, 3] — C4' coordinates
        """
        pass

    def _pdb_to_c4p_coords(self, pdb_path: str) -> torch.Tensor:
        """Shared utility — read a PDB file and return C4' coordinates."""
        _, coords, _, _ = pdb_to_tensor(
            pdb_path,
            return_sec_struct=False,
            return_sasa=False,
            keep_insertions=False,
        )
        coords = get_c4p_coords(coords)
        coords = coords - coords.mean(dim=0)
        return coords

    def _write_fasta(self, sequence: str, fasta_path: str, idx: int = 0):
        """Shared utility — write a sequence to a FASTA file."""
        record = SeqRecord(
            Seq(sequence),
            id=f"sample={idx},",
            description=f"sample={idx}",
        )
        SeqIO.write(record, fasta_path, "fasta")


class RhoFoldOracle(BaseOracle):
    """Wrapper for the original RhoFold oracle used in RIDER."""

    def __init__(self, rhofold, use_relax: bool = False):
        self.rhofold = rhofold
        self.use_relax = use_relax

    def fold(self, sequence: str, output_dir: str, idx: int = 0) -> torch.Tensor:
        os.makedirs(output_dir, exist_ok=True)
        fasta_path = os.path.join(output_dir, f"design{idx}.fasta")
        pdb_path = os.path.join(output_dir, f"design{idx}.pdb")

        self._write_fasta(sequence, fasta_path, idx)
        self.rhofold.predict(fasta_path, pdb_path, self.use_relax)
        coords = self._pdb_to_c4p_coords(pdb_path)

        # cleanup
        if os.path.exists(fasta_path):
            os.unlink(fasta_path)
        if os.path.exists(pdb_path):
            os.unlink(pdb_path)

        return coords


class AlphaFold3Oracle(BaseOracle):
    """Wrapper for AlphaFold3 running via Apptainer."""

    def __init__(self, sif_path, weights_path, output_base, model_seeds=None, gpu_id=1):
        self.sif_path = sif_path
        self.weights_path = weights_path
        self.output_base = output_base
        self.model_seeds = model_seeds or [1]
        self.gpu_id = gpu_id

    def fold(self, sequence: str, output_dir: str, idx: int = 0) -> torch.Tensor:
        os.makedirs(output_dir, exist_ok=True)

        # write JSON input
        json_path = os.path.join(output_dir, f"design{idx}.json")
        af3_output_dir = os.path.join(output_dir, f"af3_out_{idx}")
        os.makedirs(af3_output_dir, exist_ok=True)

        input_json = {
            "name": f"sample_{idx}",
            "sequences": [{"rna": {"id": "A", "sequence": sequence, "unpairedMsa": ""}}],
            "modelSeeds": self.model_seeds,
            "dialect": "alphafold3",
            "version": 1
        }
        with open(json_path, "w") as f:
            json.dump(input_json, f)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        env["APPTAINERENV_CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)


        # run via apptainer
        cmd = [
            "apptainer", "exec", "--nv",
            "--bind", f"{output_dir}:/root/af_input",
            "--bind", f"{af3_output_dir}:/root/af_output",
            "--bind", f"{self.weights_path}:/root/models",
            self.sif_path,
            "python", "/app/alphafold/run_alphafold.py",
            "--json_path=/root/af_input/" + os.path.basename(json_path),
            "--model_dir=/root/models",
            "--output_dir=/root/af_output",
            "--run_data_pipeline=false",
        ]

        print(f"Launching AlphaFold3 on GPU {self.gpu_id}")
        print(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")

        subprocess.run(cmd, check=True, env=env)

        # find output CIF and convert to coordinates
        cif_path = os.path.join(
            af3_output_dir, f"sample_{idx}", f"sample_{idx}_model.cif"
        )
        coords = self._cif_to_c4p_coords(cif_path)

        # cleanup
        if os.path.exists(json_path):
            os.unlink(json_path)

        if os.path.exists(af3_output_dir):
            shutil.rmtree(af3_output_dir)

        return coords

    def _cif_to_c4p_coords(self, cif_path: str) -> torch.Tensor:
        """Convert AlphaFold3 CIF output to C4' coordinate tensor."""
        # AF3 outputs CIF not PDB — parse with BioPython
        from Bio.PDB import MMCIFParser
        import numpy as np

        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("rna", cif_path)

        c4p_coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        if atom.get_name() == "C4'":
                            c4p_coords.append(atom.get_vector().get_array())
            break  # first model only

        coords = torch.tensor(np.array(c4p_coords), dtype=torch.float32)
        coords = coords - coords.mean(dim=0)
        return coords



def get_oracle(oracle_type: str, cfg) -> BaseOracle:
    """
    Factory function — returns the correct oracle based on config.
    Called once at the start of training/evaluation.

    Usage in evaluator_rl.py:
        oracle = get_oracle(cfg.oracle, cfg)
        coords = oracle.fold(sequence, output_dir, idx)
    """
    if oracle_type == "rhofold":
        from tools.rhofold.config import rhofold_config
        from tools.rhofold.rf import RhoFold
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rhofold = RhoFold(rhofold_config, device)
        rhofold_path = os.path.join(PROJECT_PATH, "tools/rhofold/model_20221010_params.pt")
        rhofold.load_state_dict(torch.load(rhofold_path, map_location=torch.device("cpu"))["model"])
        rhofold = rhofold.to(device)
        rhofold.eval()
        return RhoFoldOracle(rhofold, use_relax=False)

    elif oracle_type == "alphafold3":
        return AlphaFold3Oracle(
            sif_path=cfg.af3_sif_path,
            weights_path=cfg.af3_weights_path,
            output_base=cfg.af3_output_dir,
            model_seeds=getattr(cfg, "af3_model_seeds", [1]),
            gpu_id=cfg.af3_gpu,
        )

    else:
        raise ValueError(f"Unknown oracle type: {oracle_type}. "
                         f"Choose from: rhofold, alphafold3")


