"""SolvaSpec SMARTS-based tautomer and protonation-state preprocessing.

Normalises an input solute SMILES to the physically dominant tautomer /
protonation state in polar protic solvent before it reaches the SolvaSpec
3D-aware pipeline.

Current rule library (documented in the paper, Section 'SMARTS-based
tautomer and protonation-state preprocessing'):

    1. beta-diketones Ar-C(=O)-CH2-C(=O)-Ar -> chelated enol
       Ar-C(=O)-CH=C(OH)-Ar
           relevant to: avobenzone, dibenzoylmethane, curcumin
    2. sulfonic acid -S(=O)(=O)OH -> sulfonate -S(=O)(=O)[O-]
           relevant to: sulisobenzone (benzophenone-4), Ecamsule
    3. ortho-hydroxy-benzophenone: validate only (already correct in
       database-canonical form)
    4. aryl carboxylic acid: leave protonated (pKa ~ 4-5, < 1%
       deprotonated at the effective pH of neutral ethanol)

The library targets UV-filter / cosmetic chemistry. New rules should be
added with (a) a unit test against the appropriate literature reference,
(b) the chemistry rationale documented, and (c) confirmation that the rule
does not mis-fire on the wetlab-16 validation set.

Usage
-----
    from smarts_preprocessor import normalize
    new_smiles, applied_rules = normalize(
        "COc1ccc(C(=O)CC(=O)c2ccc(C(C)(C)C)cc2)cc1",
        solvent="MeOH",
    )
    # -> ('COc1ccc(cc1)C(=O)/C=C(\\O)c2ccc(cc2)C(C)(C)C', ['beta_diketone_enol'])
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions


@dataclass(frozen=True)
class Rule:
    name: str
    pattern_smarts: str
    rxn_smarts: str
    rationale: str


TAUTOMER_RULES: List[Rule] = [
    Rule(
        name="beta_diketone_enol",
        pattern_smarts="[c,C][CX3](=O)[CX4H2][CX3](=O)[c,C]",
        rxn_smarts=(
            "[c,C:1][CX3:2](=[O:3])[CX4H2:4][CX3:5](=[O:6])[c,C:7]"
            ">>[c,C:1][C:2](=[O:3])/[C:4]=[C:5](/[OH])[c,C:7]"
        ),
        rationale=(
            "Ar-C(=O)-CH2-C(=O)-Ar beta-diketones with aryls on both carbonyls "
            "exist >90% as the chelated enol in polar protic solvents; the 6-"
            "membered O-H...O=C intramolecular H-bond gives a bright pi-pi* "
            "transition near 350-360 nm that is absent in the diketo form."
        ),
    ),
]

PROTONATION_RULES: List[Rule] = [
    Rule(
        name="sulfonic_acid_deprotonation",
        pattern_smarts="[SX4](=O)(=O)[OH]",
        rxn_smarts="[SX4:1](=[O:2])(=[O:3])[OH:4]>>[S:1](=[O:2])(=[O:3])[O-:4]",
        rationale=(
            "Sulfonic acids (pKa ~ -2.8) are essentially fully deprotonated "
            "in neutral ethanol or methanol; the deprotonated sulfonate alters "
            "the chromophore electronics (blue-shift of ~10 nm on 4-sulfonate "
            "benzophenones)."
        ),
    ),
]

VALIDATION_RULES: List[Rule] = [
    Rule(
        name="ortho_hydroxy_benzophenone_ok",
        pattern_smarts="[OH]c1ccccc1C(=O)c2ccccc2",
        rxn_smarts="",
        rationale=(
            "ortho-hydroxy-benzophenones (oxybenzone, dioxybenzone, "
            "sulisobenzone framework) already represent the intramolecular "
            "H-bonded enol-like tautomer in their canonical SMILES; no "
            "rewrite needed. Rule exists only to flag matches for audit."
        ),
    ),
    Rule(
        name="aryl_carboxylic_acid_stay_protonated",
        pattern_smarts="c-[CX3](=O)[OH]",
        rxn_smarts="",
        rationale=(
            "Aryl carboxylic acids (cinnamic, coumaric, ferulic, caffeic, "
            "sinapic, PABA) have pKa ~ 4-5. The effective pH of 'neutral' "
            "ethanol is ~7 but on a methanol / ethanol scale these weak "
            "acids remain >99% protonated. No rewrite is applied."
        ),
    ),
]


def _apply_rewrite(mol: Chem.Mol, rule: Rule) -> Tuple[Chem.Mol, bool]:
    """Return (new_mol, changed). Silently returns the input mol if the rule
    does not match or the reaction fails to produce a sanitizable product."""
    if not rule.rxn_smarts:
        return mol, False
    pat = Chem.MolFromSmarts(rule.pattern_smarts)
    if pat is None or not mol.HasSubstructMatch(pat):
        return mol, False
    rxn = rdChemReactions.ReactionFromSmarts(rule.rxn_smarts)
    prods = rxn.RunReactants((mol,))
    if not prods:
        return mol, False
    new_mol = prods[0][0]
    try:
        Chem.SanitizeMol(new_mol)
    except Exception:  # sanitisation failure: keep original
        return mol, False
    return new_mol, True


def normalize(smiles: str, solvent: str = "MeOH") -> Tuple[str, List[str]]:
    """Canonicalise tautomer and protonation state of a solute SMILES.

    Parameters
    ----------
    smiles : str
        Input solute SMILES, in whichever form the user's database reports it.
    solvent : str
        Solvent identifier (currently only used for logging; rule activation
        assumes polar protic / neutral pH, which covers EtOH, MeOH, water).

    Returns
    -------
    (str, list of str)
        The standardised SMILES and the list of rule names that applied.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    applied: List[str] = []
    for rule in list(TAUTOMER_RULES) + list(PROTONATION_RULES):
        mol, changed = _apply_rewrite(mol, rule)
        if changed:
            applied.append(rule.name)
    for rule in VALIDATION_RULES:
        pat = Chem.MolFromSmarts(rule.pattern_smarts)
        if pat is not None and mol.HasSubstructMatch(pat):
            applied.append(f"{rule.name}:matched")
    return Chem.MolToSmiles(mol), applied


def report_rules() -> str:
    """Human-readable summary of the rule library (for documentation / audit)."""
    lines = ["SolvaSpec SMARTS preprocessor rules", "=" * 40, ""]
    for section, rules in [
        ("Tautomer rewrites", TAUTOMER_RULES),
        ("Protonation rewrites", PROTONATION_RULES),
        ("Validation-only", VALIDATION_RULES),
    ]:
        lines.append(f"## {section}")
        for r in rules:
            lines.append(f"- {r.name}: {r.rationale}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    tests = [
        ("Avobenzone (diketo)",
         "COc1ccc(C(=O)CC(=O)c2ccc(C(C)(C)C)cc2)cc1"),
        ("Sulisobenzone (protonated sulfonic acid)",
         "Cc1cc(S(=O)(=O)O)c(C(=O)c2ccccc2)c(O)c1"),
        ("Cinnamic acid (no rewrite expected)",
         "O=C(O)/C=C/c1ccccc1"),
        ("Oxybenzone (validation match only)",
         "COc1ccc(C(=O)c2ccccc2)c(O)c1"),
    ]
    print(report_rules())
    print()
    print("Test results on representative UV-filter inputs:")
    print("-" * 40)
    for label, smi in tests:
        try:
            new_smi, rules = normalize(smi)
            marker = "changed" if rules and not all(r.endswith(":matched") for r in rules) else "unchanged"
            print(f"{label}\n  in  : {smi}\n  out : {new_smi}\n  rules: {rules}  [{marker}]")
        except Exception as exc:
            print(f"{label}\n  ERROR: {exc}")
        print()
