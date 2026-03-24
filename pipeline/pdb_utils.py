"""Minimal PDB utilities that do NOT require ESM-IF1, biotite, or torch_scatter.

Used when --skip-if1 is set: reads native chain sequences directly from a PDB
file using only the standard library, so the pipeline can run BLOSUM +
ESM-Design attacks without any ESM inverse-folding dependencies.
"""

from typing import Dict

# Standard one-letter amino acid codes keyed by PDB three-letter residue name
_THREE_TO_ONE: Dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Common non-standard residues mapped to nearest standard AA
    "MSE": "M",  # selenomethionine
    "HSD": "H", "HSE": "H", "HSP": "H",  # CHARMM histidine variants
    "CSE": "C",  # selenocysteine
}


def read_pdb_sequences(pdb_path: str) -> Dict[str, str]:
    """Read native amino acid sequences for every chain in a PDB file.

    Parses only ATOM records (ignores HETATM). Uses the first alternate
    location indicator where present. Returns one-letter sequences in
    residue order, deduplicating by (chain_id, res_seq, ins_code).

    Unknown residue names are skipped with a warning rather than raising,
    so partially non-standard structures still produce useful sequences.

    Args:
        pdb_path: Path to a .pdb file.

    Returns:
        Dict mapping chain_id (str) -> one-letter sequence (str).
        Chains are ordered by first appearance in the file.
    """
    from collections import OrderedDict

    # chain_id -> ordered dict of (res_seq, ins_code) -> one_letter
    chains: Dict[str, "OrderedDict"] = OrderedDict()
    seen_warnings = set()

    with open(pdb_path, "r") as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec != "ATOM":
                continue

            alt_loc = line[16].strip()
            if alt_loc and alt_loc != "A":
                continue  # keep only first alt location

            chain_id = line[21].strip()
            res_name = line[17:20].strip()
            res_seq = line[22:26].strip()
            ins_code = line[26].strip()
            key = (res_seq, ins_code)

            one = _THREE_TO_ONE.get(res_name)
            if one is None:
                if res_name not in seen_warnings:
                    print(f"[pdb_utils] Unknown residue '{res_name}' in chain {chain_id} — skipping")
                    seen_warnings.add(res_name)
                continue

            if chain_id not in chains:
                chains[chain_id] = OrderedDict()
            chains[chain_id].setdefault(key, one)

    return {cid: "".join(res.values()) for cid, res in chains.items()}


def get_chain_ids(pdb_path: str) -> list:
    """Return chain IDs in order of first appearance in the PDB file."""
    return list(read_pdb_sequences(pdb_path).keys())
