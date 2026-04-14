# GSE194315 - Hackathon Dataset Description 

## Source

- **GEO**: [GSE194315](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194315) (public since March 2022)
- **Paper**: Liu et al., "Combined Single Cell Transcriptome and Surface Epitope Profiling Identifies Potential Biomarkers of Psoriatic Arthritis and Facilitates Diagnosis via Machine Learning", *Frontiers in Immunology*, 2022 (PMID 35309349)
- **Lab**: Wilson Liao, UCSF Dermatology
- **Tech**: 10x Chromium Single Cell 3' v3.1, TotalSeq-A antibodies, NovaSeq 6000, Cell Ranger 3.1.0 (GRCh38)

---

## What is CITE-Seq

Measures two things from the same cell simultaneously:
1. **RNA** : gene expression via scRNA-seq (33,694 genes)
2. **Surface proteins** : via TotalSeq-A antibody-derived tags (ADTs)

DNA-barcoded antibodies bind surface proteins. During sequencing, both modalities are captured per cell. This natural pairing is what makes the data useful for our task. 

---

## Biological Context

**PBMCs** (Peripheral Blood Mononuclear Cells) are immune cells from blood. Includes T cells, B cells, NK cells, monocytes, dendritic cells, etc.

~30% of psoriasis (PSO) patients eventually develop psoriatic arthritis (PSA), and there's no reliable way to predict who. By the time PSA is diagnosed, joint damage has often already started. This study looked for molecular differences between PSA and PSO in blood that could enable earlier detection.

---

## Patients (105 subjects)

| Group | N | Description |
|-------|---|-------------|
| PSA | 28 | Psoriatic Arthritis |
| PSO | 24 | Cutaneous Psoriasis only |
| PSX | 14 | Psoriasis + joint symptoms
| AS | 10 | Ankylosing Spondylitis |
| Healthy | 29 | No inflammatory/autoimmune disease |

### Two studies from the same data

The GEO entry contains two separate metadata files for two overlapping studies:

| Metadata file | Study | Subjects |
|---|---|---|
| `CellMetadata-PSA_TotalCiteseq_20220103.tsv` | PSA/PSO/PSX/Healthy study (the paper) | 97 unique subjects (95 after QC) |
| `CellMetadata-AS_TotalCiteseq_20220711.tsv` | AS + Healthy study (later revision) | 106 subjects (adds 10 AS patients) |

**We use the PSA file** because it matches the published paper. The AS file is a later revision that re-ran cell type annotation and clustering, resulting in different `CellType`, `Cluster`, and `IncludedInStudy` values for many cells. The 10 AS patients are physically present in the CITE-seq batches (PBMC-02 through PBMC-07, pooled with other patients). They are labeled as "AXI" in the PSA metadata (49,721 cells), but all have `IncludedInStudy == FALSE`, so they are excluded after QC filtering.

**Not all subjects have CITE-seq data.** The full study has 95 subjects after QC, but only **77 of those appear in our CITE-seq files** (PBMC-02 through PBMC-07). The remaining 18 subjects (9 PSA + 9 Healthy) are only in PBMC-01, which is RNA-only and has no protein data. Our hackathon files therefore contain 19 PSA, 24 PSO, 14 PSX, and 20 Healthy subjects.

---

## Batches and Samples

~20 subjects pooled per batch, loaded onto 10x Chromium targeting ~50k cells per lane. 7 batches, 4 lanes each = 28 samples.

| Batch | Modality |
|-------|----------|
| PBMC-01 (4 samples) | RNA only hence **we skip these** (removed from the data) |
| PBMC-02 to PBMC-07 (24 samples) | CITE-Seq (RNA + protein) |

**Demultiplexing**: Demuxlet (SNP-based) assigned cells back to individual patients. DoubletDecon caught doublets. Because patients are pooled across batches, disease groups are mixed within batches.

**All cells in PBMC-02 to PBMC-07 have real measured ADT.** No ADT imputation was applied to these cells (normally since paper is very ambiguous). All cells in PBMC-02 through PBMC-07 were physically stained with TotalSeq-A antibodies, and PBMC-01 has no ADT features in its Cell Ranger output at all (confirmed: Gene Expression only). Our files are loaded directly from the raw Cell Ranger count matrices, the paper's ADT imputation was a downstream Seurat step.

**Note on cell counts vs the paper:** The paper reports 246,762 total cells after QC (matching `IncludedInStudy == True` across all batches) and states that 133,665 (54%) have ADT. However, filtering the metadata to PBMC-02–07 with `IncludedInStudy == True` gives 180,794 cells (the number in our files). The discrepancy with the paper's 133,665 likely reflects additional filtering in the paper's analysis pipeline that is not captured by the `IncludedInStudy` flag alone. 

---

## Features

### RNA
- **33,694 genes** (Gene Expression)

### ADT (Antibody Capture)
- **282 total features** in Cell Ranger output, which break down as:
  - **258 real surface proteins** 
  - **9 isotype controls** : negative controls, removed in our MuData file
  - **15 features flagged as inconsistent** with annotated cell types by the authors. The paper mentions these ("15 features observed to have expression inconsistent with annotated cell types") and references them in Supplementary Table 2, but does not list them by name in the main text. Hence they are still in our data.



---

## Metadata Columns

| Column | Description |
|--------|-------------|
| `Sample` | Sample name based on Chromium batch ID and lane ID (e.g., `PBMC-02-1`). Each batch has 4 lanes. |
| `Run` | Chromium Batch ID (e.g., `PBMC-02`). 6 batches in our data (PBMC-02 through PBMC-07). |
| `Subject` | Subject identity based on Demuxlet demultiplexing.
| `Status` | Clinical status of subject: `PSA`, `PSO`, `PSX`, or `Healthy`. |
| `DemuxletDropletType` | Demuxlet classification (SNG: singlet; DBL: doublet; AMB: ambiguous) we only kept singlets |
| `IncludedInStudy` | `True` if the cell passed all QC criteria (see below). We keep only True. |
| `CellType` | Annotated cell type based on predicted.celltype.l2 of Hao 2021 reference dataset. |
| `Cluster` | Subcluster within cell type from de novo clustering.  |

### QC criteria applied by the authors

| Filter | Threshold |
|--------|-----------|
| RNA UMIs | 500 – 10,000 |
| RNA features | >= 200 |
| Mitochondrial reads | <= 15% |
| Ribosomal reads | <= 60% |
| ADT features | <= 260 |
| Isotype control reads | < 2% |
| Doublets | Removed via Demuxlet + DoubletDecon |


The `IncludedInStudy` flag reflects all of the above. We verified this by computing RNA-only QC metrics on the unfiltered data: excluded cells show thousands of violations (37k high mito, 27k high UMI, 7.5k low genes), while included cells respect the stated thresholds with only minor edge cases
   at the boundaries.

---

## Cell Counts

| Step | Cells |
|------|-------|
| Loaded (PBMC-02 to PBMC-07) | 409,904 |
| Singlets only (SNG) | 277,435 |
| IncludedInStudy == True | **180,794** |

Paper must have used more filtering further down along the pipeline.

### Disease breakdown (after QC)

| Status | Cells |
|--------|-------|
| PSA | 54,385 |
| PSO | 53,654 |
| Healthy | 45,168 |
| PSX | 27,587 |

### Top cell types

| CellType | Cells |
|----------|-------|
| CD14 Mono | ~43k |
| CD4 TCM | ~26k |
| CD8 TEM | ~23k |
| CD4 Naive | ~18k |
| NK | ~15k |
| B naive | ~11k |
| *(30 types total)* | |


---

## What We Provide for the Hackathon

| File | Contents | Shape |
|------|----------|-------|
`GSE194315_raw_mu.h5mu` | MuData combining RNA + ADT, isotype controls removed | 180,794 x (33,694 + 273) |

All files contain raw integer counts from Cell Ranger. No normalization has been applied. All QC filtering (singlets, IncludedInStudy) has been applied. Full metadata is attached to `.obs` in every file.

---

## MuData File Structure

The file is `GSE194315_raw_mu.h5mu`, a [MuData](https://mudata.readthedocs.io/) object loadable with `muon`:

```python
import muon as mu
mdata = mu.read("GSE194315_raw_mu.h5mu")
```

**Two modalities** accessed via `mdata`:

| Key | Content | Shape |
|-----|---------|-------|
| `mdata['rna']` or `mdata.mod['rna']` | Raw RNA counts (Gene Expression) | 180,794 × 33,694 |
| `mdata['protein']` or `mdata.mod['protein']` | Raw ADT counts (Antibody Capture, isotype controls removed) | 180,794 × 273 |

Each modality is an AnnData object with:
- **`.X`** — raw integer count matrix (sparse)
- **`.obs`** — per-cell metadata (shared across both modalities):

| Column | Description |
|--------|-------------|
| `sample` | Sample name (e.g., `PBMC-02-1`) |
| `batch_ID` | Batch ID (e.g., `PBMC-02`) |
| `Sample` | Same as `sample` (from original metadata) |
| `Run` | Same as `batch_ID` (from original metadata) |
| `Subject` | Patient ID from Demuxlet demultiplexing |
| `Status` | Disease group: `PSA`, `PSO`, `PSX`, or `Healthy` |
| `DemuxletDropletType` | Always `SNG` (singlets only after filtering) |
| `IncludedInStudy` | Always `True` (QC-passed cells only) |
| `CellType` | One of 30 annotated cell types |
| `Cluster` | Subcluster within cell type |

- **`.var_names`** — gene names (RNA) or protein names (ADT)

**Note:** `sample`/`batch_ID` and `Sample`/`Run` are duplicate columns, the lowercase versions were added during data loading, the capitalized versions come from the original metadata. They contain the same information.

---
