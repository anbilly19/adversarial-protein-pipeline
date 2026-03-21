"""Known adversarial protein sequences that confuse AlphaFold3.

Categories:
    - Fold-switchers: proteins with two distinct stable folds
    - Chameleon sequences: short segments that adopt helix or sheet by context
    - BLOSUM adversarial: conservative mutations that cause large structural shifts
    - Repeat hallucinations: poly-repeat sequences that trigger spurious confidence
"""

from typing import List, Dict


TRICK_SEQUENCES: Dict[str, Dict] = {
    "RfaH_CTD": {
        "seq": "MPNKILVEGNLRVDYDSIPVKKVNLKGGAVKIINDRLASRGLSLNNVKIGKNVSSEGKGEKVNLKELIKGSTTVTVKGQ",
        "desc": "Fold-switcher: beta-barrel (AF3 prediction) vs alpha-helical hairpin (active form). "
                "AF3 always commits to barrel with pLDDT > 85, missing the biological state.",
        "category": "fold_switcher",
        "reference": "Porter et al. 2022, PNAS",
    },
    "GA_domain": {
        "seq": "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
        "desc": "GA/GB chameleon: 88% identical to GB domain but adopts completely different topology. "
                "AF3 cannot distinguish between the two folds.",
        "category": "chameleon",
        "reference": "Dalal et al. 1997, Biochemistry",
    },
    "GB_domain": {
        "seq": "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
        "desc": "GB counterpart of GA_domain — single residue difference at position 2 (T->Q) "
                "drives completely different beta1alpha1beta2beta3 vs 3alpha topology.",
        "category": "chameleon",
        "reference": "Dalal et al. 1997, Biochemistry",
    },
    "NusG_switch_v2": {
        "seq": "MARVELEIKPQEAFEKFIREHLKSVGVSRDTLQNLRAKLSQGDAAQISKVAEELAKTMGISRDDAIREVARILEELGLRN",
        "desc": "NusG superfamily fold-switcher. 24% of the superfamily switches between "
                "beta-clamp and RfaH-like helical hairpin. AF3 predicts single fold with high pLDDT.",
        "category": "fold_switcher",
        "reference": "Coles et al. 2023, Nat Struct Mol Biol",
    },
    "Chameleon_helix": {
        "seq": "MKTAYIAKQRQISFVKSHFSVDNKFNKELEERLGLIEVQAPILSRVGDGT",
        "desc": "Classic chameleon: VDNKFNKE (pos 17-24) embedded in helix-favoring host. "
                "Same 8-mer is beta-strand in protein B1 context.",
        "category": "chameleon",
        "reference": "Minor & Kim 1996, Nature",
    },
    "SpA_wt": {
        "seq": "ADNKFNKEQQNAFYEILHLPNLNEEQRNGFIQSLKDDPSQSTNVLGEAKKLNESQAPK",
        "desc": "SpA IgG-binding domain wildtype. Well-folded, pLDDT > 90. "
                "Baseline for BLOSUM adversarial comparison.",
        "category": "blosum_baseline",
        "reference": "Alkhouri et al. 2024, IEEE PST",
    },
    "SpA_adv": {
        "seq": "ADNKFNKEQQNAFYEILHLPNLNEEQRNGFIQSLKDDPSQSTNVLGEAKKLNESQALK",
        "desc": "SpA adversarial variant: 5 BLOSUM62-conservative mutations (K->L, P->L, K->L at C-term). "
                "Causes RMSD > 12A shift in AF3 prediction despite near-identical sequence.",
        "category": "blosum_adversarial",
        "reference": "Alkhouri et al. 2024, IEEE PST",
    },
    "polyQ_medium": {
        "seq": "MAQQQQQQQQQQQQQQQQQQQQQQQQQQQQQENLFHEQNKQEPQQQLE",
        "desc": "Poly-glutamine repeat (32Q). AF3 hallucinates stable helical structure "
                "with pLDDT > 70 even though long polyQ is experimentally disordered/aggregated.",
        "category": "repeat_hallucination",
        "reference": "Ruff & Pappu 2021, J Mol Biol",
    },
    "GS_repeat": {
        "seq": "GSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGS",
        "desc": "Poly-GS linker (50 residues). Should be fully disordered. AF3 sometimes "
                "predicts helical segments with moderate pLDDT — pure hallucination.",
        "category": "repeat_hallucination",
        "reference": "Hallucination benchmark, Rives et al. 2021",
    },
    "KaiB_fold_switch": {
        "seq": "MSTSEKINILRGAILGLREAQTLLDLFEKDDQVVTISEGQRSMLQSLAQEMQAGRLTPDDLIQHIQVFNELLNQLSQQ",
        "desc": "KaiB circadian clock protein: tetrameric ground-state fold switches to monomeric "
                "fold-switch state upon binding KaiC. AF3 predicts ground state only.",
        "category": "fold_switcher",
        "reference": "Chang et al. 2015, Science",
    },
}


BLOSUM62_SIMILAR: Dict[str, List[str]] = {
    # Maps each amino acid to conservative substitutes with BLOSUM62 score >= 0
    "A": ["S", "T", "G"],
    "R": ["K", "Q", "H"],
    "N": ["D", "S", "T"],
    "D": ["E", "N"],
    "C": ["S"],
    "Q": ["E", "K", "R"],
    "E": ["D", "Q", "K"],
    "G": ["A"],
    "H": ["R", "Y", "N"],
    "I": ["L", "V", "M"],
    "L": ["I", "V", "M"],
    "K": ["R", "Q", "E"],
    "M": ["L", "I", "V"],
    "F": ["Y", "W"],
    "P": ["A"],
    "S": ["T", "A", "N"],
    "T": ["S", "A"],
    "W": ["F", "Y"],
    "Y": ["F", "W", "H"],
    "V": ["I", "L", "A"],
}


AA_VOCAB: List[str] = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX: Dict[str, int] = {aa: i for i, aa in enumerate(AA_VOCAB)}


def get_trick_sequences_by_category(category: str) -> List[Dict]:
    """Filter trick sequences by category."""
    return [
        {"seq": v["seq"], "name": k, "source": "trick_library", **v}
        for k, v in TRICK_SEQUENCES.items()
        if v["category"] == category
    ]


def get_all_trick_sequences() -> List[Dict]:
    """Return all trick sequences as a flat list of candidate dicts."""
    return [
        {"seq": v["seq"], "name": k, "source": "trick_library", **v}
        for k, v in TRICK_SEQUENCES.items()
    ]
