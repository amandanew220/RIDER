# fold(sequence: str) → coords: torch.Tensor

# src/data/oracle.py

import os
import subprocess
import json
from abc import ABC, abstractmethod
from datetime import datetime

import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from src.data.data_utils import pdb_to_tensor, get_c4p_coords
from src.constants import PROJECT_PATH

import pickle
import socket


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
    """Wrapper for AlphaFold3 running via Apptainer on Nibi."""

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


class DRfold2Oracle(BaseOracle):
    """Wrapper for DRfold2."""

    def __init__(self, drfold2_dir: str, env_python: str):
        self.drfold2_dir = drfold2_dir  # path to DRfold2 repo
        self.env_python = env_python    # path to drfold2_env python

    def fold(self, sequence: str, output_dir: str, idx: int = 0) -> torch.Tensor:
        os.makedirs(output_dir, exist_ok=True)
        fasta_path = os.path.join(output_dir, f"design{idx}.fasta")
        pdb_output_dir = os.path.join(output_dir, f"drfold2_out_{idx}")

        self._write_fasta(sequence, fasta_path, idx)

        cmd = [
            self.env_python,
            os.path.join(self.drfold2_dir, "DRfold_infer.py"),
            fasta_path,
            pdb_output_dir,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "1"
        subprocess.run(cmd, check=True, capture_output=False, cwd=self.drfold2_dir, env=env)

        # DRfold2 writes to relax/ subdirectory
        pdb_path = os.path.join(pdb_output_dir, "relax", "model_1.pdb")
        if not os.path.exists(pdb_path):
            # fall back to folds/ if relax failed
            import glob
            candidates = glob.glob(os.path.join(pdb_output_dir, "folds", "*.pdb"))
            if not candidates:
                raise FileNotFoundError(f"DRfold2 produced no PDB output in {pdb_output_dir}")
            pdb_path = candidates[0]

        coords = self._pdb_to_c4p_coords(pdb_path)

        # cleanup
        if os.path.exists(fasta_path):
            os.unlink(fasta_path)

        return coords


class TrRosettaRNAOracle(BaseOracle):
    """Wrapper for trRosettaRNA."""

    def __init__(self, trrosetta_dir: str, env_python: str):
        self.trrosetta_dir = trrosetta_dir
        self.env_python = env_python

    def fold(self, sequence: str, output_dir: str, idx: int = 0) -> torch.Tensor:
        os.makedirs(output_dir, exist_ok=True)
        fasta_path = os.path.join(output_dir, f"design{idx}.fasta")
        pdb_output_dir = os.path.join(output_dir, f"trrosetta_out_{idx}")

        self._write_fasta(sequence, fasta_path, idx)

        cmd = [
            self.env_python,
            os.path.join(self.trrosetta_dir, "predict.py"),
            "--input", fasta_path,
            "--output", pdb_output_dir,
        ]
        subprocess.run(cmd, check=True, capture_output=True, cwd=self.trrosetta_dir)

        import glob
        candidates = glob.glob(os.path.join(pdb_output_dir, "*.pdb"))
        if not candidates:
            raise FileNotFoundError(f"trRosettaRNA produced no PDB output in {pdb_output_dir}")

        coords = self._pdb_to_c4p_coords(candidates[0])

        if os.path.exists(fasta_path):
            os.unlink(fasta_path)

        return coords


class RemoteOracleClient(BaseOracle):
    """Connects to a remote oracle server over a TCP socket."""

    def __init__(self, address_file: str):
        self.address_file = address_file
        self._host = None
        self._port = None

    def _get_address(self):
        if self._host is None:
            with open(self.address_file) as f:
                host, port = f.read().strip().split(":")
            self._host = host
            self._port = int(port)

    def fold(self, sequence: str, output_dir: str, idx: int = 0) -> torch.Tensor:
        self._get_address()

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self._host, self._port))
            s.sendall(pickle.dumps(sequence))
            s.shutdown(socket.SHUT_WR)

            data = b""
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                data += chunk

        coords = pickle.loads(data)
        return torch.tensor(coords, dtype=torch.float32)


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

    elif oracle_type == "drfold2":
        return DRfold2Oracle(
            drfold2_dir=cfg.drfold2_dir,
            env_python=cfg.drfold2_python,
        )

    elif oracle_type == "trrosettarna":
        return TrRosettaRNAOracle(
            trrosetta_dir=cfg.trrosetta_dir,
            env_python=cfg.trrosetta_python,
        )

    elif oracle_type == "remote":
        return RemoteOracleClient(
            address_file=cfg.oracle_address_file
        )

    else:
        raise ValueError(f"Unknown oracle type: {oracle_type}. "
                         f"Choose from: rhofold, alphafold3, drfold2, trrosettarna")


