"""AptaBench-to-DeepRSMA input adaptation.

The adapter consumes Thesis A's processed rows and existing five-fold assignments. RNA
records presented to DeepRSMA use nucleotide categories ``x: [L, 1]``, directed adjacency
``edge_index: [2, E]``, and frozen GB.DNA embeddings ``emb: [L, H]``. ``L`` is the retained
prefix length (at most 511 positions), ``E`` is the number of retained directed backbone or
BPP edges, and ``H`` is 1024. The embedding row and graph node at each position refer to the
same canonical DNA nucleotide. BPP values and graph edge attributes are not passed to the
upstream GAT.
"""

from __future__ import annotations

import __main__
import hashlib
import json
import os
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from rdkit import Chem
from torch import Tensor
from torch_geometric.data import Data
from tqdm import tqdm

_THESIS_ROOT = Path(__file__).resolve().parents[3]
_THESIS_SRC = _THESIS_ROOT / "src"
if str(_THESIS_SRC) not in sys.path:
    sys.path.insert(0, str(_THESIS_SRC))

from data_featurization.split_reader import read_split  # noqa: E402
from dataset_preprocessing.io import sha256_file  # noqa: E402
from dna_feature_encoding.contracts import PreprocessedDNARecord  # noqa: E402
from dna_feature_encoding.structure.graph_store import GraphArtifactStore  # noqa: E402

from .feature import atom_to_feature_vector, bond_to_feature_vector  # noqa: E402
from .process_data_molecule import atom_dict, smilebet  # noqa: E402
from .vocab import WordVocab  # noqa: E402

UPSTREAM_URL = "https://github.com/Hhhzj-7/DeepRSMA.git"
UPSTREAM_COMMIT = "3414ede63efb93fff0bd81029261871dec8c7124"
GB_DNA_MODEL = "GB.DNA-300M"
GB_DNA_REPOSITORY = "genbio-ai/GB.DNA-300M"
GB_DNA_REVISION = "a833c5602b0dad0728c0aeeecaf43086ebeb96ca"
GB_DNA_HIDDEN_SIZE = 1024
MAX_RNA_NODES = 511
EMBEDDING_MANIFEST = "gbdna_manifest.json"
_BASE_INDEX = {"A": 0, "T": 1, "G": 2, "C": 3}
_SPLIT_ORDER = ("unseen_pair", "unseen_aptamer", "unseen_molecule", "unseen_both")


class AptaBenchInputError(RuntimeError):
    """A canonical split row or one of its DeepRSMA inputs is inconsistent."""


@dataclass(frozen=True)
class FoldPartition:
    """pKd-supervised rows for one shared CV fold, in canonical row order."""

    fold_id: int
    train: tuple[Any, ...]
    val: tuple[Any, ...]
    test: tuple[Any, ...]
    assigned_counts: Mapping[str, int]
    pkd_counts: Mapping[str, int]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AptaBenchInputError(f"could not read DNA embedding manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AptaBenchInputError(f"DNA embedding manifest must be a JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_numpy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _embedding_manifest_payload(clean_csv: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "complete",
        "upstream_url": UPSTREAM_URL,
        "upstream_commit": UPSTREAM_COMMIT,
        "model": GB_DNA_MODEL,
        "repository": GB_DNA_REPOSITORY,
        "revision": GB_DNA_REVISION,
        "hidden_size": GB_DNA_HIDDEN_SIZE,
        "max_rna_nodes": MAX_RNA_NODES,
        "canonical_clean_sha256": sha256_file(clean_csv),
        "entries": {},
    }


def _validate_embedding_store(split: Any, embedding_dir: Path) -> dict[str, Any]:
    manifest_path = embedding_dir / EMBEDDING_MANIFEST
    if not manifest_path.is_file():
        raise AptaBenchInputError(
            f"GB.DNA embeddings are not materialized under {embedding_dir}; "
            "run main_aptabench_cv.py --dataset-dir PATH --prepare-dna-embeddings first"
        )
    manifest = _read_json(manifest_path)
    clean_csv = split.processed_root / "clean.csv"
    expected = _embedding_manifest_payload(clean_csv)
    keys = (
        "schema_version",
        "status",
        "upstream_commit",
        "model",
        "repository",
        "revision",
        "hidden_size",
        "max_rna_nodes",
        "canonical_clean_sha256",
    )
    mismatches = [key for key in keys if manifest.get(key) != expected[key]]
    if mismatches:
        raise AptaBenchInputError(
            f"GB.DNA embedding manifest does not match the selected source/configuration: "
            f"{mismatches}"
        )
    if not isinstance(manifest.get("entries"), dict):
        raise AptaBenchInputError("GB.DNA embedding manifest entries must be an object")
    return manifest


def _validate_embedding_file(
    embedding_dir: Path,
    manifest: Mapping[str, Any],
    aptamer_id: str,
    sequence: str,
) -> np.ndarray:
    entries = manifest.get("entries", {})
    entry = entries.get(aptamer_id) if isinstance(entries, Mapping) else None
    if not isinstance(entry, Mapping):
        raise AptaBenchInputError(f"GB.DNA embedding manifest has no aptamer {aptamer_id!r}")
    retained = sequence[:MAX_RNA_NODES]
    if entry.get("sequence_sha256") != _sha256_text(sequence):
        raise AptaBenchInputError(f"GB.DNA embedding sequence hash mismatch for {aptamer_id!r}")
    if entry.get("retained_length") != len(retained):
        raise AptaBenchInputError(f"GB.DNA embedding length mismatch for {aptamer_id!r}")
    expected_name = f"{aptamer_id}.npy"
    if entry.get("file") != expected_name:
        raise AptaBenchInputError(f"unsafe or stale GB.DNA embedding filename for {aptamer_id!r}")
    path = embedding_dir / expected_name
    try:
        values = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise AptaBenchInputError(f"could not load GB.DNA embedding {path}: {exc}") from exc
    if values.shape != (len(retained), GB_DNA_HIDDEN_SIZE):
        raise AptaBenchInputError(
            f"GB.DNA embedding for {aptamer_id!r} must have shape "
            f"[{len(retained)}, {GB_DNA_HIDDEN_SIZE}], got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise AptaBenchInputError(f"GB.DNA embedding for {aptamer_id!r} has non-finite values")
    return np.asarray(values, dtype=np.float32)


class AptaBenchAdapter:
    """Resolve shared CV assignments and assemble upstream-compatible PyG records.

    Assignment membership is read before target filtering. Only rows with a non-null ``pKd``
    are materialized into a model partition, preserving the assigned train, val, or test
    partition while avoiding target imputation.
    """

    def __init__(
        self,
        strategy: str,
        dataset_dir: str | Path,
        embedding_dir: str | Path,
    ) -> None:
        if strategy not in _SPLIT_ORDER:
            raise ValueError(f"unsupported AptaBench split strategy {strategy!r}")
        split_dir = Path(dataset_dir).expanduser().resolve()
        self.split = read_split(split_dir)
        if self.split.split_mode != "cross_validation":
            raise AptaBenchInputError(
                f"DeepRSMA requires a cross-validation dataset directory, got "
                f"{self.split.split_mode!r}"
            )
        if self.split.split_strategy != strategy:
            raise AptaBenchInputError(
                f"dataset directory strategy {self.split.split_strategy!r} does not match "
                f"--split-strategy {strategy!r}"
            )
        if self.split.fold_ids != (0, 1, 2, 3, 4):
            raise AptaBenchInputError(
                f"DeepRSMA requires the existing five folds; got {self.split.fold_ids}"
            )
        self.embedding_dir = Path(embedding_dir).expanduser().resolve()
        self.embedding_manifest = _validate_embedding_store(self.split, self.embedding_dir)
        graph_dir = split_dir / "featurized_inputs" / "aptamer"
        if not graph_dir.is_dir():
            raise AptaBenchInputError(
                f"Thesis A ViennaRNA graph artifacts are missing: {graph_dir}"
            )
        self.graph_store = GraphArtifactStore(graph_dir)
        vocab_path = Path(__file__).resolve().parent / "smiles_vocab.pkl"
        if not vocab_path.is_file():
            raise AptaBenchInputError(f"DeepRSMA SMILES vocabulary is missing: {vocab_path}")
        # The released vocabulary pickle records ``__main__.WordVocab`` because upstream
        # generated it from a script. Recreate that module alias only during deserialization.
        had_main_vocab = hasattr(__main__, "WordVocab")
        previous_main_vocab = getattr(__main__, "WordVocab", None)
        __main__.WordVocab = WordVocab
        try:
            self.drug_vocab = WordVocab.load_vocab(str(vocab_path))
        finally:
            if had_main_vocab:
                __main__.WordVocab = previous_main_vocab
            else:
                delattr(__main__, "WordVocab")
        self._graph_cache: dict[str, Data] = {}
        self._embedding_cache: dict[str, np.ndarray] = {}
        self._molecule_cache: dict[str, Data] = {}

    def fold_partition(self, fold_id: int) -> FoldPartition:
        """Return pKd rows by their already-recorded partition for ``fold_id``."""

        if fold_id not in self.split.fold_ids:
            raise ValueError(f"fold_id must be one of {self.split.fold_ids}, got {fold_id}")
        assigned: dict[str, list[Any]] = {name: [] for name in ("train", "val", "test")}
        supervised: dict[str, list[Any]] = {name: [] for name in assigned}
        for row in self.split.rows:
            membership = [item for item in row.memberships if item.fold_id == fold_id]
            if len(membership) != 1:
                raise AptaBenchInputError(
                    f"example {row.example_id!r} has {len(membership)} assignments for "
                    f"fold {fold_id}"
                )
            partition = membership[0].partition
            assigned[partition].append(row)
            if row.pkd_value is not None:
                supervised[partition].append(row)
        return FoldPartition(
            fold_id=fold_id,
            train=tuple(supervised["train"]),
            val=tuple(supervised["val"]),
            test=tuple(supervised["test"]),
            assigned_counts={name: len(rows) for name, rows in assigned.items()},
            pkd_counts={name: len(rows) for name, rows in supervised.items()},
        )

    def datasets_for_fold(
        self,
        fold_id: int,
        *,
        limit_per_partition: int | None = None,
    ) -> tuple[FoldPartition, dict[str, list[tuple[Data, Data]]]]:
        """Build unchanged DeepRSMA data pairs, optionally limiting smoke-run examples."""

        partition = self.fold_partition(fold_id)
        if limit_per_partition is not None and limit_per_partition < 1:
            raise ValueError("limit_per_partition must be positive when supplied")
        datasets: dict[str, list[tuple[Data, Data]]] = {}
        for name in ("train", "val", "test"):
            rows = getattr(partition, name)
            if limit_per_partition is not None:
                rows = rows[:limit_per_partition]
            datasets[name] = [
                (self._rna_data(row), self._molecule_data(row)) for row in rows
            ]
            if not datasets[name]:
                raise AptaBenchInputError(
                    f"fold {fold_id} has no pKd-available rows in its {name} partition"
                )
        return partition, datasets

    def _rna_data(self, row: Any) -> Data:
        sequence = row.sequence[:MAX_RNA_NODES]
        try:
            nucleotide_indices = [_BASE_INDEX[base] for base in sequence]
        except KeyError as exc:
            raise AptaBenchInputError(
                f"canonical DNA sequence for {row.aptamer_id!r} contains unsupported base {exc}"
            ) from exc
        if row.aptamer_id not in self._graph_cache:
            source = PreprocessedDNARecord(
                aptamer_id=row.aptamer_id,
                canonical_sequence=row.sequence,
                sequence_length=len(row.sequence),
            )
            self._graph_cache[row.aptamer_id] = self.graph_store.load(source)
        graph = self._graph_cache[row.aptamer_id]
        edges = graph.edge_index
        if edges.numel():
            retained_edges = (edges[0] < len(sequence)) & (edges[1] < len(sequence))
            edges = edges[:, retained_edges]
        if row.aptamer_id not in self._embedding_cache:
            self._embedding_cache[row.aptamer_id] = _validate_embedding_file(
                self.embedding_dir,
                self.embedding_manifest,
                row.aptamer_id,
                row.sequence,
            )
        embedding = self._embedding_cache[row.aptamer_id][: len(sequence)]
        if embedding.shape[0] != len(nucleotide_indices):
            raise AptaBenchInputError(
                f"sequence/embedding/graph alignment failed for {row.aptamer_id!r}"
            )
        return Data(
            x=torch.tensor(nucleotide_indices, dtype=torch.float32).reshape(-1, 1),
            edge_index=edges.clone().contiguous(),
            y=torch.tensor(float(row.pkd_value), dtype=torch.float32),
            t_id=row.aptamer_id,
            e_id=row.example_id,
            example_id=row.example_id,
            molecule_id=row.molecule_id,
            emb=torch.from_numpy(embedding.copy()),
            rna_len=len(sequence),
        )

    def _molecule_data(self, row: Any) -> Data:
        if row.molecule_id in self._molecule_cache:
            return self._molecule_cache[row.molecule_id].clone()
        smiles = row.canonical_smiles
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            molecule = Chem.MolFromSmiles(smiles, sanitize=False)
            if molecule is None:
                raise AptaBenchInputError(f"DeepRSMA could not parse canonical SMILES {smiles!r}")
            molecule.UpdatePropertyCache(strict=False)

        atom_features = [atom_to_feature_vector(atom) for atom in molecule.GetAtoms()][:128]
        if not atom_features:
            raise AptaBenchInputError(f"canonical SMILES {smiles!r} has no heavy atoms")
        atom_count = len(atom_features)
        edge_pairs: list[tuple[int, int]] = []
        edge_features: list[list[int]] = []
        for bond in molecule.GetBonds():
            left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if left >= atom_count or right >= atom_count:
                continue
            edge = bond_to_feature_vector(bond)
            edge_pairs.extend(((left, right), (right, left)))
            edge_features.extend((edge, edge))
        if edge_pairs:
            edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 3), dtype=torch.long)

        token_ids: list[int] = []
        cursor = 0
        while cursor < len(smiles) and len(token_ids) < 128:
            token = smiles[cursor : cursor + 2]
            if len(token) == 2 and token in self.drug_vocab.stoi:
                token_ids.append(self.drug_vocab.stoi[token])
                cursor += 2
            else:
                token = smiles[cursor]
                token_ids.append(self.drug_vocab.stoi.get(token, self.drug_vocab.unk_index))
                cursor += 1
        token_ids = token_ids[:128]
        atom_positions = [
            index for index, token_id in enumerate(token_ids) if token_id in atom_dict
        ]
        padded_tokens = token_ids + [self.drug_vocab.pad_index] * (128 - len(token_ids))
        mask = torch.zeros(128, dtype=torch.float32)
        mask[: len(token_ids)] = 1.0
        encoded_smiles = torch.from_numpy(smilebet.encode(smiles.encode("utf-8").upper())).long()
        data = Data(
            x=torch.tensor(atom_features, dtype=torch.long),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(float(row.pkd_value), dtype=torch.float32),
            e_id=row.example_id,
            example_id=row.example_id,
            molecule_id=row.molecule_id,
            smiles_ori=smiles,
            graph_len=atom_count,
            smile_emb=torch.tensor(padded_tokens, dtype=torch.long),
            smile_len=len(token_ids),
            atom_len=atom_positions,
            mask=mask,
            smiles_f=encoded_smiles,
        )
        self._molecule_cache[row.molecule_id] = data.clone()
        return data


def _hidden_states(output: Any) -> Tensor:
    """Return GB.DNA states shaped ``[B, T, H]``.

    ``B`` is the number of aptamers in the current batch, ``T`` is the tokenizer-padded token
    width, and ``H`` is the 1024-wide GB.DNA hidden representation.
    """

    if isinstance(output, Mapping):
        value = output.get("last_hidden_state")
    else:
        value = getattr(output, "last_hidden_state", output)
    if not isinstance(value, Tensor) or value.ndim != 3:
        raise AptaBenchInputError("GB.DNA must return last_hidden_state with shape [B, T, H]")
    return value


def materialize_gbdna_embeddings(
    output_dir: str | Path,
    *,
    dataset_dir: str | Path,
    batch_size: int = 8,
    device: str | torch.device = "cpu",
    overwrite: bool = False,
    embed_task: Any | None = None,
    tokenizer_adapter: Any | None = None,
) -> dict[str, Any]:
    """Save frozen GB.DNA token states once per canonical aptamer.

    Each saved ``[L, H]`` array is in nucleotide order: ``L`` is the retained prefix length
    (at most 511) and ``H`` is the GB.DNA hidden size (1024). BOS/EOS and padded token rows
    are removed using the tokenizer's explicit masks before persistence.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    split = read_split(dataset_dir)
    if split.split_mode != "cross_validation":
        raise AptaBenchInputError(
            f"GB.DNA embedding preparation requires a cross-validation split, "
            f"got {split.split_mode!r}"
        )
    if split.split_strategy not in _SPLIT_ORDER:
        raise AptaBenchInputError(
            f"unsupported AptaBench split strategy {split.split_strategy!r}"
        )
    destination = Path(output_dir).expanduser().resolve()
    clean_csv = split.processed_root / "clean.csv"
    expected = _embedding_manifest_payload(clean_csv)
    aptamers = split.aptamers
    manifest_path = destination / EMBEDDING_MANIFEST

    if destination.exists():
        files = tuple(path for path in destination.iterdir() if path.is_file())
        if manifest_path.is_file():
            existing = _read_json(manifest_path)
            if any(existing.get(key) != expected[key] for key in expected if key != "entries"):
                if not overwrite:
                    raise AptaBenchInputError(
                        f"existing GB.DNA embeddings under {destination} use a different "
                        "dataset or model; pass overwrite=True to replace them"
                    )
            else:
                entries = existing.get("entries")
                if isinstance(entries, Mapping):
                    try:
                        for aptamer_id, sequence in aptamers:
                            _validate_embedding_file(destination, existing, aptamer_id, sequence)
                    except AptaBenchInputError:
                        if not overwrite:
                            raise AptaBenchInputError(
                                f"existing GB.DNA embedding store is incomplete or invalid at "
                                f"{destination}; pass overwrite=True to rebuild it"
                            )
                    else:
                        return existing
                elif not overwrite:
                    raise AptaBenchInputError(
                        f"existing GB.DNA embedding manifest is malformed: {manifest_path}"
                    )
        elif files and not overwrite:
            raise AptaBenchInputError(
                f"unmanifested files already exist under {destination}; "
                "pass overwrite=True to rebuild the embedding store"
            )
        if overwrite:
            for path in destination.iterdir():
                if path.is_file() or path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
    destination.mkdir(parents=True, exist_ok=True)

    if embed_task is None:
        from dna_feature_encoding.batching.tokenizer import (
            GBDNATokenizerAdapter,
            load_gbdna_embed_task,
        )

        embed_task = load_gbdna_embed_task()
        if tokenizer_adapter is None:
            tokenizer_adapter = GBDNATokenizerAdapter(embed_task=embed_task)
    elif tokenizer_adapter is None:
        from dna_feature_encoding.batching.tokenizer import GBDNATokenizerAdapter

        tokenizer_adapter = GBDNATokenizerAdapter(embed_task=embed_task)

    if not hasattr(embed_task, "backbone"):
        raise AptaBenchInputError("configured GB.DNA embed task does not expose a backbone")
    backbone = embed_task.backbone.to(device)
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False

    entries: dict[str, dict[str, Any]] = {}
    offsets = range(0, len(aptamers), batch_size)
    for offset in tqdm(offsets, desc="GB.DNA aptamer batches", unit="batch"):
        batch = aptamers[offset : offset + batch_size]
        aptamer_ids = [item[0] for item in batch]
        sequences = [item[1][:MAX_RNA_NODES] for item in batch]
        token_batch = tokenizer_adapter.tokenize(
            sequences,
            sequence_lengths=[len(sequence) for sequence in sequences],
            aptamer_ids=aptamer_ids,
        )
        input_ids = token_batch.input_ids.to(device)
        attention_mask = token_batch.attention_mask.to(device=device, dtype=torch.bool)
        special_mask = token_batch.special_tokens_mask.to(device=device, dtype=torch.bool)
        with torch.inference_mode():
            output = backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = _hidden_states(output)
        if hidden.shape[:2] != input_ids.shape or hidden.shape[2] != GB_DNA_HIDDEN_SIZE:
            raise AptaBenchInputError(
                f"GB.DNA hidden-state shape must be [B, T, {GB_DNA_HIDDEN_SIZE}], "
                f"got {tuple(hidden.shape)}"
            )
        retained = attention_mask & ~special_mask
        for index, (aptamer_id, full_sequence) in enumerate(batch):
            sequence = full_sequence[:MAX_RNA_NODES]
            values = hidden[index][retained[index]].detach().cpu().to(torch.float32).numpy()
            if values.shape != (len(sequence), GB_DNA_HIDDEN_SIZE) or not np.isfinite(values).all():
                raise AptaBenchInputError(
                    f"GB.DNA token alignment failed for {aptamer_id!r}: got {values.shape}"
                )
            _atomic_numpy(destination / f"{aptamer_id}.npy", values)
            entries[aptamer_id] = {
                "file": f"{aptamer_id}.npy",
                "sequence_sha256": _sha256_text(full_sequence),
                "retained_length": len(sequence),
            }

    expected["entries"] = entries
    _atomic_json(manifest_path, expected)
    return expected


__all__ = [
    "AptaBenchAdapter",
    "AptaBenchInputError",
    "FoldPartition",
    "GB_DNA_HIDDEN_SIZE",
    "GB_DNA_MODEL",
    "GB_DNA_REPOSITORY",
    "GB_DNA_REVISION",
    "MAX_RNA_NODES",
    "UPSTREAM_COMMIT",
    "UPSTREAM_URL",
    "materialize_gbdna_embeddings",
]
