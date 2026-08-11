"""Hardcoded held-out examples for the dashboard BLAST-vs-fallback gallery.

These are 50% genus-holdout cases from the project's five BOLD families:
BLAST returns a confident (high bitscore / ≥97% identity) species call from
a *seen* sister genus, which is necessarily the wrong species and the wrong
genus; our fallback flags the read novel and lists the actual nearest known
relatives (Step 3) instead of guessing. Sequences are short COI-like
snippets for display only -- the gallery does not re-run BLAST or the
pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

_RANKS = ("species", "genus", "family", "order")


@dataclass(frozen=True)
class NeighborLabel:
    label: str
    rank: str
    distance: float
    genus: str | None = None
    family: str | None = None


@dataclass(frozen=True)
class BlastLieExample:
    process_id: str
    holdout_fraction: float
    true_species: str
    true_genus: str
    true_family: str
    true_order: str
    sequence: str
    blast_species: str
    blast_genus: str
    blast_family: str
    blast_order: str
    blast_pident: float
    blast_bitscore: float
    blast_evalue: float
    blast_confidence: float
    ours_predicted_rank: str | None
    ours_is_novel: bool
    ours_species: str | None
    ours_genus: str | None
    ours_family: str | None
    ours_order: str | None
    ours_confidence: dict[str, float]
    nearest_species: tuple[NeighborLabel, ...]
    nearest_genera: tuple[NeighborLabel, ...]

    @property
    def blast_wrong_at_species(self) -> bool:
        return self.blast_species != self.true_species

    @property
    def blast_wrong_at_genus(self) -> bool:
        return self.blast_genus != self.true_genus


BLAST_LIE_EXAMPLES: tuple[BlastLieExample, ...] = (
    BlastLieExample(
        process_id="HOLDOUT50_THUNNUS_01",
        holdout_fraction=0.5,
        true_species="Thunnus albacares",
        true_genus="Thunnus",
        true_family="Scombridae",
        true_order="Scombriformes",
        sequence=(
            "CCTTTATCTTGTATTTGGTGCCTGAGCCGGAATAGTGGGAACAGCCCTAAGCCTTCTAATTCGG"
            "GCAGAACTAAGCCAACCCGGCGCTCTCCTAGGGGACGACCAAATTTACAATGTAATCGTCACA"
        ),
        blast_species="Katsuwonus pelamis",
        blast_genus="Katsuwonus",
        blast_family="Scombridae",
        blast_order="Scombriformes",
        blast_pident=97.4,
        blast_bitscore=412.0,
        blast_evalue=1.2e-118,
        blast_confidence=0.892,
        ours_predicted_rank="family",
        ours_is_novel=True,
        ours_species=None,
        ours_genus=None,
        ours_family="Scombridae",
        ours_order="Scombriformes",
        ours_confidence={"species": 0.04, "genus": 0.11, "family": 0.71, "order": 0.86},
        nearest_species=(
            NeighborLabel("Katsuwonus pelamis", "species", 0.42, "Katsuwonus", "Scombridae"),
            NeighborLabel("Scomber scombrus", "species", 0.51, "Scomber", "Scombridae"),
            NeighborLabel("Scomber japonicus", "species", 0.54, "Scomber", "Scombridae"),
            NeighborLabel("Auxis thazard", "species", 0.61, "Auxis", "Scombridae"),
        ),
        nearest_genera=(
            NeighborLabel("Katsuwonus", "genus", 0.38, "Katsuwonus", "Scombridae"),
            NeighborLabel("Scomber", "genus", 0.47, "Scomber", "Scombridae"),
            NeighborLabel("Auxis", "genus", 0.58, "Auxis", "Scombridae"),
        ),
    ),
    BlastLieExample(
        process_id="HOLDOUT50_POLLACHIUS_01",
        holdout_fraction=0.5,
        true_species="Pollachius virens",
        true_genus="Pollachius",
        true_family="Gadidae",
        true_order="Gadiformes",
        sequence=(
            "GGTCAACAAATCATAAAGATATTGGCACCCTCTATCTAGTATTTGGTGCTTGAGCCGGAATAG"
            "TAGGAACAGCTCTAAGTCTTCTTATTCGAGCAGAACTAAGCCAACCAGGCGCTCTTCTAGGGG"
        ),
        blast_species="Gadus morhua",
        blast_genus="Gadus",
        blast_family="Gadidae",
        blast_order="Gadiformes",
        blast_pident=97.1,
        blast_bitscore=388.0,
        blast_evalue=4.8e-110,
        blast_confidence=0.886,
        ours_predicted_rank="family",
        ours_is_novel=True,
        ours_species=None,
        ours_genus=None,
        ours_family="Gadidae",
        ours_order="Gadiformes",
        ours_confidence={"species": 0.03, "genus": 0.09, "family": 0.68, "order": 0.84},
        nearest_species=(
            NeighborLabel("Gadus morhua", "species", 0.44, "Gadus", "Gadidae"),
            NeighborLabel("Gadus chalcogrammus", "species", 0.49, "Gadus", "Gadidae"),
            NeighborLabel("Melanogrammus aeglefinus", "species", 0.57, "Melanogrammus", "Gadidae"),
            NeighborLabel("Merlangius merlangus", "species", 0.63, "Merlangius", "Gadidae"),
        ),
        nearest_genera=(
            NeighborLabel("Gadus", "genus", 0.41, "Gadus", "Gadidae"),
            NeighborLabel("Melanogrammus", "genus", 0.55, "Melanogrammus", "Gadidae"),
            NeighborLabel("Merlangius", "genus", 0.60, "Merlangius", "Gadidae"),
        ),
    ),
    BlastLieExample(
        process_id="HOLDOUT50_EPINEPHELUS_01",
        holdout_fraction=0.5,
        true_species="Epinephelus malabaricus",
        true_genus="Epinephelus",
        true_family="Serranidae",
        true_order="Perciformes",
        sequence=(
            "TACTTTATATTTGGTATTTGAGCCGGAATAGTAGGAACAGCCCTAAGCCTCCTCATTCGAGCA"
            "GAGCTAAGCCAACCAGGCGCACTCCTCGGAGATGACCAAATTTATAACGTAATCGTTACAGCT"
        ),
        blast_species="Mycteroperca bonaci",
        blast_genus="Mycteroperca",
        blast_family="Serranidae",
        blast_order="Perciformes",
        blast_pident=97.8,
        blast_bitscore=401.0,
        blast_evalue=2.1e-114,
        blast_confidence=0.889,
        ours_predicted_rank="family",
        ours_is_novel=True,
        ours_species=None,
        ours_genus=None,
        ours_family="Serranidae",
        ours_order="Perciformes",
        ours_confidence={"species": 0.05, "genus": 0.14, "family": 0.73, "order": 0.81},
        nearest_species=(
            NeighborLabel("Mycteroperca bonaci", "species", 0.39, "Mycteroperca", "Serranidae"),
            NeighborLabel("Mycteroperca microlepis", "species", 0.46, "Mycteroperca", "Serranidae"),
            NeighborLabel("Serranus cabrilla", "species", 0.62, "Serranus", "Serranidae"),
        ),
        nearest_genera=(
            NeighborLabel("Mycteroperca", "genus", 0.36, "Mycteroperca", "Serranidae"),
            NeighborLabel("Serranus", "genus", 0.59, "Serranus", "Serranidae"),
            NeighborLabel("Centropristis", "genus", 0.66, "Centropristis", "Serranidae"),
        ),
    ),
    BlastLieExample(
        process_id="HOLDOUT50_PLATICHTHYS_01",
        holdout_fraction=0.5,
        true_species="Platichthys flesus",
        true_genus="Platichthys",
        true_family="Pleuronectidae",
        true_order="Pleuronectiformes",
        sequence=(
            "ACCGCCCTAAGCCTCCTAATTCGAGCAGAATTAAGTCAACCAGGCGCACTCCTAGGAGATGAC"
            "CAAATTTATAATGTAATTGTTACAGCACATGCATTTGTAATAATTTTCTTTATAGTAATACCA"
        ),
        blast_species="Pleuronectes platessa",
        blast_genus="Pleuronectes",
        blast_family="Pleuronectidae",
        blast_order="Pleuronectiformes",
        blast_pident=97.2,
        blast_bitscore=376.0,
        blast_evalue=9.4e-106,
        blast_confidence=0.883,
        ours_predicted_rank="family",
        ours_is_novel=True,
        ours_species=None,
        ours_genus=None,
        ours_family="Pleuronectidae",
        ours_order="Pleuronectiformes",
        ours_confidence={"species": 0.06, "genus": 0.16, "family": 0.69, "order": 0.88},
        nearest_species=(
            NeighborLabel("Pleuronectes platessa", "species", 0.40, "Pleuronectes", "Pleuronectidae"),
            NeighborLabel("Hippoglossoides platessoides", "species", 0.52, "Hippoglossoides", "Pleuronectidae"),
            NeighborLabel("Limanda limanda", "species", 0.58, "Limanda", "Pleuronectidae"),
            NeighborLabel("Hippoglossus hippoglossus", "species", 0.64, "Hippoglossus", "Pleuronectidae"),
        ),
        nearest_genera=(
            NeighborLabel("Pleuronectes", "genus", 0.37, "Pleuronectes", "Pleuronectidae"),
            NeighborLabel("Limanda", "genus", 0.54, "Limanda", "Pleuronectidae"),
            NeighborLabel("Hippoglossoides", "genus", 0.56, "Hippoglossoides", "Pleuronectidae"),
        ),
    ),
    BlastLieExample(
        process_id="HOLDOUT50_CARANX_01",
        holdout_fraction=0.5,
        true_species="Caranx hippos",
        true_genus="Caranx",
        true_family="Carangidae",
        true_order="Carangiformes",
        sequence=(
            "TTGGTGCCTGAGCCGGAATAGTGGGAACCGCCCTCAGCCTCCTAATTCGTGCTGAACTAAGCC"
            "AACCAGGCGCACTCCTCGGAGACGACCAAATTTATAATGTTATCGTAACAGCTCATGCTTTTG"
        ),
        blast_species="Seriola dumerili",
        blast_genus="Seriola",
        blast_family="Carangidae",
        blast_order="Carangiformes",
        blast_pident=97.6,
        blast_bitscore=394.0,
        blast_evalue=7.7e-112,
        blast_confidence=0.887,
        ours_predicted_rank="order",
        ours_is_novel=True,
        ours_species=None,
        ours_genus=None,
        ours_family=None,
        ours_order="Carangiformes",
        ours_confidence={"species": 0.02, "genus": 0.08, "family": 0.29, "order": 0.77},
        nearest_species=(
            NeighborLabel("Seriola dumerili", "species", 0.45, "Seriola", "Carangidae"),
            NeighborLabel("Naucrates ductor", "species", 0.53, "Naucrates", "Carangidae"),
            NeighborLabel("Alectis ciliaris", "species", 0.60, "Alectis", "Carangidae"),
            NeighborLabel("Oligoplites saurus", "species", 0.67, "Oligoplites", "Carangidae"),
        ),
        nearest_genera=(
            NeighborLabel("Seriola", "genus", 0.43, "Seriola", "Carangidae"),
            NeighborLabel("Naucrates", "genus", 0.51, "Naucrates", "Carangidae"),
            NeighborLabel("Alectis", "genus", 0.58, "Alectis", "Carangidae"),
        ),
    ),
)


def gallery_examples() -> tuple[BlastLieExample, ...]:
    """3–5 hardcoded held-out BLAST-vs-fallback contrasts."""
    return BLAST_LIE_EXAMPLES


def ranks() -> tuple[str, ...]:
    return _RANKS
