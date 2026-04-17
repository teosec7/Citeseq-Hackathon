from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"

MODEL_WEIGHTS = CACHE_DIR / "clip_cite_v2_best.pt"
RNA_ENCODINGS = CACHE_DIR / "gse_rna_hvg_encodings.npy"

UMAP_COORDS_CACHE = CACHE_DIR / "umap_coords_all.npy"
UMAP_REDUCER_CACHE = CACHE_DIR / "umap_reducer.joblib"
RNA_EMB_CACHE = CACHE_DIR / "all_rna_embeddings.npy"

D_RNA = 2000
D_PROTEIN = 768
D_HIDDEN = 512
D_EMB = 256

BIOBERT_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

STATUS_LABELS = {
    "Healthy": "Healthy control",
    "PSO":     "Psoriasis (cutaneous)",
    "PSA":     "Psoriatic arthritis",
    "PSX":     "Psoriasis, unclear PsA",
}
STATUS_ORDER = ["Healthy", "PSO", "PSA", "PSX"]
STATUS_COLORS = {
    "Healthy": "#2C8EBB",
    "PSO":     "#F0A030",
    "PSA":     "#D94F4F",
    "PSX":     "#7B5EA7",
}

KNOWN_CELL_TYPES = [
    "ASDC", "B intermediate", "B memory", "B naive",
    "CD14 Mono", "CD16 Mono",
    "CD4 CTL", "CD4 Naive", "CD4 Proliferating", "CD4 TCM", "CD4 TEM",
    "CD8 Naive", "CD8 Proliferating", "CD8 TCM", "CD8 TEM",
    "Doublet", "Eryth", "HSPC", "ILC", "MAIT",
    "NK", "NK Proliferating", "NK_CD56bright",
    "Plasmablast", "Platelet", "Treg",
    "cDC1", "cDC2", "dnT", "gdT", "pDC",
]

FALLBACK_PROTEINS = sorted({
    "CD3", "CD4", "CD8", "CD11c", "CD14", "CD16", "CD19", "CD20", "CD25",
    "CD27", "CD34", "CD38", "CD41", "CD45", "CD45RA", "CD45RO", "CD56",
    "CD57", "CD61", "CD64", "CD71", "CD123", "CD127", "CD138", "CD141",
    "CD161", "CD235a", "CD303", "HLA-DR", "IgD", "TCRgamma/delta",
    "TCRValpha7.2", "XCR1", "CD1c",
})


def find_h5mu() -> Path | None:
    candidates = list(ROOT.glob("*.h5mu")) + list(ROOT.glob("**/*.h5mu"))
    candidates = [c for c in candidates if "cache" not in c.parts]
    return candidates[0] if candidates else None
