"""
module_deidentification.py
==========================
Demo Piece 2 — k-anonymity de-identification for UCI Diabetes 130-US Hospitals.

The dataset documents ~101,766 inpatient diabetic encounters from 130 US
hospitals over ten years (Strack et al., 2014). It contains a mix of direct
identifiers, quasi-identifiers, sensitive medical attributes, and
non-identifying features.

This module performs four steps:

  1. Attribute classification — every column is assigned one of four roles:
        - IDENTIFIER       : drop (e.g. encounter_id, patient_nbr)
        - QUASI_IDENTIFIER : generalise so groups of records become
                             indistinguishable to k-anonymity (k = 5)
        - SENSITIVE        : keep as-is, since they carry the medical
                             signal we want the model to learn
        - OTHER            : keep as-is (clinical measurements,
                             medications, vitals, etc.)

  2. Generalisation — each quasi-identifier is mapped to a coarser domain:
        - age   : original 10-year bins -> 20-year bins
                  "[0-10)"  -> "[0-20)"   "[10-20)" -> "[0-20)"
                  "[20-30)" -> "[20-40)"  "[30-40)" -> "[20-40)"
                  ... and so on
        - race  : low-frequency categories collapsed into "Other/Unknown"
        - gender: "Unknown/Invalid" collapsed into "Unknown" (rare; ~3 rows)

  3. Suppression — any row that, after generalisation, still lies in a
     quasi-identifier equivalence class smaller than k = 5 is dropped.
     This is what *guarantees* the k-anonymity property, no matter what
     the data distribution looked like to start with.

  4. Verification — an independent function walks the published dataset
     and confirms every equivalence class has at least k rows. This is
     the verifier the supervisor will run live during the demo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd


# ---------------------------------------------------------------------------
# 1) Attribute classification
# ---------------------------------------------------------------------------

# Columns that directly identify a patient or encounter — suppressed entirely.
IDENTIFIERS: tuple[str, ...] = (
    "encounter_id",
    "patient_nbr",
)

# Columns that, in combination, could re-identify a patient via linkage
# to an external source (e.g. an admission registry). These are the
# columns we generalise to satisfy k-anonymity.
QUASI_IDENTIFIERS: tuple[str, ...] = (
    "race",
    "gender",
    "age",
)

# Columns carrying the medical signal. Kept as-is — k-anonymity does not
# require generalising sensitive attributes, only quasi-identifiers.
SENSITIVE_ATTRIBUTES: tuple[str, ...] = (
    "diag_1",
    "diag_2",
    "diag_3",
    "max_glu_serum",
    "A1Cresult",
    "readmitted",        # the prediction target
)


def classify_attributes(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Return a mapping of role -> list of column names present in df.

    Anything not explicitly listed in the constants above is treated as
    OTHER (kept verbatim).
    """
    cols = set(df.columns)
    identifiers       = [c for c in IDENTIFIERS        if c in cols]
    quasi             = [c for c in QUASI_IDENTIFIERS  if c in cols]
    sensitive         = [c for c in SENSITIVE_ATTRIBUTES if c in cols]
    accounted_for     = set(identifiers) | set(quasi) | set(sensitive)
    other             = sorted(cols - accounted_for)
    return {
        "IDENTIFIER":       identifiers,
        "QUASI_IDENTIFIER": quasi,
        "SENSITIVE":        sensitive,
        "OTHER":            other,
    }


# ---------------------------------------------------------------------------
# 2) Generalisation
# ---------------------------------------------------------------------------

# Map UCI Diabetes 10-year age bins -> 20-year bins.
_AGE_GENERALISATION: dict[str, str] = {
    "[0-10)":   "[0-20)",
    "[10-20)":  "[0-20)",
    "[20-30)":  "[20-40)",
    "[30-40)":  "[20-40)",
    "[40-50)":  "[40-60)",
    "[50-60)":  "[40-60)",
    "[60-70)":  "[60-80)",
    "[70-80)":  "[60-80)",
    "[80-90)":  "[80-100)",
    "[90-100)": "[80-100)",
}


def _generalise_age(series: pd.Series) -> pd.Series:
    """Widen 10-year bins to 20-year bins; pass other values through."""
    return series.map(lambda v: _AGE_GENERALISATION.get(v, v))


def _generalise_race(series: pd.Series, min_count: int = 500) -> pd.Series:
    """
    Collapse small-frequency race categories (and explicit unknowns
    encoded as '?') into a single 'Other/Unknown' bucket. The threshold
    `min_count` is conservative — it merges anything that does not
    already have many records, which both improves k-anonymity and
    reduces the chance of suppression dropping rare-race rows later.
    """
    counts = series.value_counts(dropna=False)
    keep = set(counts[counts >= min_count].index) - {"?"}
    return series.map(lambda v: v if v in keep else "Other/Unknown")


def _generalise_gender(series: pd.Series) -> pd.Series:
    """Normalise the rare 'Unknown/Invalid' value to 'Unknown'."""
    return series.replace({"Unknown/Invalid": "Unknown"})


def generalise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply column-specific generalisation rules to all quasi-identifiers
    present in the dataframe. Returns a new dataframe; the input is not
    modified in place.
    """
    out = df.copy()
    if "age" in out.columns:
        out["age"] = _generalise_age(out["age"])
    if "race" in out.columns:
        out["race"] = _generalise_race(out["race"])
    if "gender" in out.columns:
        out["gender"] = _generalise_gender(out["gender"])
    return out


# ---------------------------------------------------------------------------
# 3) Suppression to enforce k-anonymity
# ---------------------------------------------------------------------------

def suppress_small_groups(
    df: pd.DataFrame,
    quasi_identifiers: Iterable[str],
    k: int = 5,
) -> tuple[pd.DataFrame, int]:
    """
    Drop every row whose quasi-identifier equivalence class has fewer than
    k rows. Returns (filtered_df, n_suppressed_rows).

    This is the brute-force, guarantee-correct approach. More sophisticated
    schemes (Mondrian, etc.) preserve more utility but are out of scope for
    the mid-sem deliverable.
    """
    qi = [c for c in quasi_identifiers if c in df.columns]
    if not qi:
        return df.copy(), 0
    group_sizes = df.groupby(qi, dropna=False).transform("size")
    # group_sizes is a DataFrame; every column holds the same value (the
    # group size), so use the first one.
    sizes = group_sizes.iloc[:, 0] if isinstance(group_sizes, pd.DataFrame) else group_sizes
    keep_mask = sizes >= k
    suppressed = int((~keep_mask).sum())
    return df.loc[keep_mask].reset_index(drop=True), suppressed


# ---------------------------------------------------------------------------
# 4) Verification
# ---------------------------------------------------------------------------

@dataclass
class KAnonymityReport:
    """Result of running verify_k_anonymity() against a dataset."""
    k: int
    quasi_identifiers: list[str]
    n_records: int
    n_groups: int
    smallest_group_size: int
    largest_group_size: int
    n_groups_below_k: int
    n_records_in_violating_groups: int
    examples_of_violating_groups: list[dict] = field(default_factory=list)

    @property
    def passes(self) -> bool:
        return self.n_groups_below_k == 0 and self.n_records > 0

    def pretty(self) -> str:
        head = (
            f"k-anonymity verifier (k = {self.k})\n"
            f"  quasi-identifiers   : {self.quasi_identifiers}\n"
            f"  records             : {self.n_records:,}\n"
            f"  equivalence classes : {self.n_groups:,}\n"
            f"  smallest class size : {self.smallest_group_size}\n"
            f"  largest  class size : {self.largest_group_size:,}\n"
            f"  classes below k     : {self.n_groups_below_k}\n"
            f"  records in those    : {self.n_records_in_violating_groups}\n"
        )
        verdict = "  RESULT              : PASS" if self.passes else "  RESULT              : FAIL"
        if self.examples_of_violating_groups:
            examples = "\n  example violations  :"
            for g in self.examples_of_violating_groups[:5]:
                examples += f"\n    {g}"
            return head + verdict + examples
        return head + verdict


def verify_k_anonymity(
    df: pd.DataFrame,
    quasi_identifiers: Iterable[str],
    k: int = 5,
) -> KAnonymityReport:
    """
    Walk every quasi-identifier equivalence class in df and confirm each
    has at least k members. The returned report is suitable to print
    verbatim during a live VIVA demo.
    """
    qi = [c for c in quasi_identifiers if c in df.columns]
    grouped = df.groupby(qi, dropna=False).size().reset_index(name="count")
    violating = grouped[grouped["count"] < k]

    examples = []
    for _, row in violating.head(5).iterrows():
        examples.append({col: row[col] for col in qi} | {"count": int(row["count"])})

    return KAnonymityReport(
        k=k,
        quasi_identifiers=qi,
        n_records=int(len(df)),
        n_groups=int(len(grouped)),
        smallest_group_size=int(grouped["count"].min()) if len(grouped) else 0,
        largest_group_size=int(grouped["count"].max()) if len(grouped) else 0,
        n_groups_below_k=int(len(violating)),
        n_records_in_violating_groups=int(violating["count"].sum()),
        examples_of_violating_groups=examples,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

@dataclass
class DeIdentificationResult:
    classification: dict[str, list[str]]
    n_input_rows: int
    n_output_rows: int
    n_suppressed_rows: int
    verifier_report: KAnonymityReport
    dataframe: pd.DataFrame


def de_identify(df: pd.DataFrame, k: int = 5) -> DeIdentificationResult:
    """
    Run the full de-identification pipeline on the input dataframe:

        classify -> drop identifiers -> generalise -> suppress -> verify

    Returns the cleaned dataframe along with a report describing every
    decision made along the way.
    """
    n_input_rows = len(df)
    classification = classify_attributes(df)

    # Step A: drop direct identifiers entirely.
    df_no_id = df.drop(columns=classification["IDENTIFIER"], errors="ignore")

    # Step B: generalise quasi-identifiers.
    df_gen = generalise(df_no_id)

    # Step C: suppress equivalence classes still below k.
    df_clean, n_suppressed = suppress_small_groups(
        df_gen, classification["QUASI_IDENTIFIER"], k=k
    )

    # Step D: verify the output meets the k-anonymity property.
    report = verify_k_anonymity(df_clean, classification["QUASI_IDENTIFIER"], k=k)

    return DeIdentificationResult(
        classification=classification,
        n_input_rows=n_input_rows,
        n_output_rows=len(df_clean),
        n_suppressed_rows=n_suppressed,
        verifier_report=report,
        dataframe=df_clean,
    )


__all__ = [
    "IDENTIFIERS",
    "QUASI_IDENTIFIERS",
    "SENSITIVE_ATTRIBUTES",
    "classify_attributes",
    "generalise",
    "suppress_small_groups",
    "verify_k_anonymity",
    "de_identify",
    "DeIdentificationResult",
    "KAnonymityReport",
]
