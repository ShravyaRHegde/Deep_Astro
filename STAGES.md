# Deep_Astro: Project Implementation Stages & Work Breakdown Structure

> **Purpose**: This document outlines the roadmap, completed milestones, and decoupled future work stages for the Deep_Astro project. It serves as an integration guide so that multiple team members and collaborators can work independently on separate Git branches and seamlessly merge their contributions into the `main` branch.

---

## 1. Project Implementation Roadmap Overview

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                DEEP_ASTRO STAGE ROADMAP                                │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ COMPLETED FOUNDATION (v0.1-baseline)                                                   │
│   ├── Stage 1: Architecture, Theory & Documentation Foundation [DONE]                  │
│   ├── Stage 2: Toolchain & Dependency Verification [DONE]                              │
│   ├── Stage 3: Data Ingestion, Cleaning & Preprocessing Pipeline [DONE]                │
│   └── Stage 4: Multi-Model Anomaly Detection & 14-Class Real Benchmark [DONE]          │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ FUTURE COLLABORATIVE WORK (Modular Branches)                                           │
│   ├── Stage 5: Observational Robustness & Stress-Testing Suite (Branch: feature/stage-5) │
│   ├── Stage 6: Scientific Visualization & Plotting Suite (Branch: feature/stage-6)     │
│   ├── Stage 7: Real-Time Alert Stream & GRU Sequential Pipeline (Branch: feature/stage-7)│
│   └── Stage 8: Capstone Report, Presentation Deck & Integration (Branch: feature/stage-8)│
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Completed Stages (Foundational Milestone: `v0.1-baseline`)

The following stages have been implemented, tested, committed, and pushed to `main` under release tag `v0.1-baseline`:

### Stage 1: System Architecture, Theory & Documentation Foundation
* **Status**: **Completed**
* **Key Deliverables**:
  * [docs/SYSTEM_ARCHITECTURE.md](docs/SYSTEM_ARCHITECTURE.md): Complete 6-layer architecture flowchart, plain-English airport scanner analogy, and alert lifecycle sequence diagram.
  * [docs/DATA_STRATEGY.md](docs/DATA_STRATEGY.md): Theoretical proof of subsampling efficacy in Isolation Forests (avoiding swamping/masking) and memory-safe compute boundaries (< 250 MB RAM).
  * [docs/DATASET_SPLIT_GUIDE.md](docs/DATASET_SPLIT_GUIDE.md): The "identical test set" principle for fair algorithm benchmarking.
  * [docs/EXPERIMENT_DESIGN.md](docs/EXPERIMENT_DESIGN.md): Detailed experimental protocols for the 3 project objectives.

### Stage 2: Toolchain & Dependency Setup
* **Status**: **Completed**
* **Key Deliverables**:
  * Configured and validated production environment: Python 3.12, TensorFlow 2.21, Keras 3.15, Scikit-learn 1.8, Astropy 8.0, PyERFA, Astromcad 1.2.7, Pandas 3.0, NumPy 2.4, SciPy 1.17, Gdown.
  * Version-pinned [`requirements.txt`](requirements.txt) and root [`.gitignore`](.gitignore).

### Stage 3: Data Ingestion, Cleaning & Leakage-Free Preprocessing
* **Status**: **Completed**
* **Key Deliverables**:
  * Downloaded official ZTF ALeRCE benchmark dataset (`93,171` training / `23,180` test transients).
  * Implemented automated pipeline in [`src/data_cleaning.py`](src/data_cleaning.py).
  * Removed unclassified `Periodic-Other` contamination (`92,311` train / `22,978` test clean objects).
  * Extracted 186 numerical features; fitted median imputation strictly on training data.
  * Applied `RobustScaler` with $[5.0, 95.0]$ percentile range to normalize extreme astronomical variations.
  * Exported clean master Parquet files (`clean_train.parquet`, `clean_test.parquet`) and laptop-friendly split (`stratified_train_5k.npz`, 4,363 samples, 2.6 MB).
  * Exported metadata summary in `data/processed/dataset_summary.json`.

### Stage 4: Multi-Model Anomaly Detection & 14-Class Real Benchmark
* **Status**: **Completed**
* **Key Deliverables**:
  * Implemented 4 core anomaly algorithms in [`src/anomaly/detectors.py`](src/anomaly/detectors.py): Global IF, MCIF, Global LOF, and Multi-Class LOF (MCLOF).
  * Implemented dynamic class-dependent thresholding in [`src/thresholding/adaptive.py`](src/thresholding/adaptive.py): Cluster-Standardized Scoring and Quantile Normalization (Objective 1).
  * Implemented and executed automated benchmark across all 14 classes in [`experiments/real_data_benchmark.py`](experiments/real_data_benchmark.py).
  * Recorded master results in [`results/benchmark_all_14_classes.csv`](results/benchmark_all_14_classes.csv).
  * Documented full empirical findings in [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md) (MCIF won 6 classes, Global IF won 4 classes, Global LOF won 2 classes, MCIF+Adaptive won 2 classes).

---

## 3. Work Breakdown Structure for Future Stages (Collaborator Guide)

To allow different contributors to work concurrently without merge conflicts or breaking the core baseline, the remaining tasks are split into four decoupled stages:

```
                                    main (v0.1-baseline)
                                             │
         ┌───────────────────┬───────────────┴───────────────┬───────────────────┐
         ▼                   ▼                               ▼                   ▼
    [ Stage 5 ]         [ Stage 6 ]                     [ Stage 7 ]         [ Stage 8 ]
  Stress Testing      Visualizations                  Alert Stream        Final Report
  Branch:             Branch:                         Branch:             Branch:
  feature/stage-5     feature/stage-6                 feature/stage-7     feature/stage-8
         │                   │                               │                   │
         └───────────────────┴───────────────┬───────────────┴───────────────────┘
                                             ▼
                                     Merged into main
```

---

### 🚀 Stage 5: Observational Robustness & Stress-Testing Suite (Objective 3)

* **Objective**: Evaluate how severely missing cadences, photometric noise, and the absence of spectroscopic host redshift degrade anomaly detection performance, and benchmark mitigation techniques.
* **Suggested Assignee**: Collaborator / Team Member A
* **Git Branch**: `feature/stage-5-stress-testing`
* **Input Files Required**:
  * `src/stress_test/degradations.py` (already implemented)
  * `data/processed/clean_train.parquet`, `clean_test.parquet`
  * `src/anomaly/detectors.py`
* **Deliverables**:
  1. Create experiment runner script: `experiments/run_stress_testing.py`.
  2. Implement tests for:
     * **Cadence Dropout**: Mask 20%, 40%, 60%, 80% of observations.
     * **Photometric Noise Escalation**: Add Gaussian error scaling ($\times 1.5, \times 2.0, \times 3.0$).
     * **Redshift Removal**: Test performance when host redshift is zeroed or replaced by noisy photometric redshift.
  3. Generate output results CSV: `results/stress_test_degradation.csv`.
  4. Write documentation report: `docs/STRESS_TEST_RESULTS.md`.
* **Acceptance Criteria**:
  * Script runs via `python experiments/run_stress_testing.py`.
  * Outputs clear degradation tables showing AUROC vs noise level across all 4 models.

---

### 🎨 Stage 6: Scientific Visualization & Plotting Suite

* **Objective**: Create publication-quality plots, ROC/PR curves, and comparison charts for college project presentations, reports, and slides.
* **Suggested Assignee**: Collaborator / Team Member B
* **Git Branch**: `feature/stage-6-visualizations`
* **Input Files Required**:
  * `results/benchmark_all_14_classes.csv`
  * `data/processed/stratified_train_5k.npz`
* **Deliverables**:
  1. Create plotting module: `src/visualization/plots.py`.
  2. Create executable figure generator: `experiments/generate_figures.py`.
  3. Generate and save high-resolution figures (PDF & PNG) in `figures/`:
     * `figures/roc_curves_comparison.png`: ROC curves for SLSN, SNII, SNIa, SNIbc.
     * `figures/model_win_distribution.png`: Bar chart of AUROC across all 14 classes comparing MCIF vs LOF vs Global IF.
     * `figures/adaptive_threshold_lift.png`: Visualization of score separation before and after cluster standardization.
     * `figures/latent_umap_projection.png`: 2D UMAP scatter plot of the clustered latent space.
  4. Add visual documentation: `docs/FIGURES_GUIDE.md`.
* **Acceptance Criteria**:
  * Running `python experiments/generate_figures.py` generates all figures in `figures/` without warnings.
  * Figures follow clean aesthetic styling (Seaborn dark/whitegrid, 300 DPI, readable axis labels).

---

### ⚡ Stage 7: Real-Time Alert Stream Simulator & Sequential GRU Pipeline

* **Objective**: Connect the trained recurrent neural network (GRU) encoder to simulate streaming alerts in real-time and compute early anomaly scores (from $-30$ to $+70$ days relative to trigger).
* **Suggested Assignee**: Collaborator / Team Member C
* **Git Branch**: `feature/stage-7-alert-streaming`
* **Input Files Required**:
  * `src/models/encoder.py`
  * `astromcad` pre-trained weights (`HostClassifier.h5`, `NoHostClassifier.h5`)
* **Deliverables**:
  1. Create alert streaming simulator: `src/streaming/alert_stream.py`.
  2. Create experiment script: `experiments/run_realtime_stream.py`.
  3. Implement temporal anomaly trajectory tracking:
     * Compute $A(z_s, t)$ at 1-day intervals from trigger $-30$ to $+70$ days.
     * Demonstrate early detection of fast transients (Kilonovae / SLSN) *before* or *at* peak brightness.
  4. Output trajectory CSV: `results/realtime_trajectories.csv`.
  5. Write documentation: `docs/REALTIME_STREAMING.md`.
* **Acceptance Criteria**:
  * Successfully loads GRU weights and computes anomaly scores at varying light-curve phases.
  * Proves whether anomalies can be detected within 3–5 days post-trigger.

---

### 📝 Stage 8: Capstone Report, Presentation Deck & Integration

* **Objective**: Consolidate all empirical findings into a final project report (PDF / LaTeX / Markdown) and presentation slide deck.
* **Suggested Assignee**: All Team Members / Lead Integrator
* **Git Branch**: `feature/stage-8-final-report`
* **Input Files Required**:
  * All benchmark CSVs in `results/`
  * All figures in `figures/`
  * Documentation in `docs/`
* **Deliverables**:
  1. Final project report draft: `report/Deep_Astro_Final_Report.md` (or LaTeX).
  2. Presentation outline & slide deck content: `report/presentation_slides.md`.
  3. Executive summary for college capstone evaluators.
  4. Final repository cleanup and release tag `v1.0-final`.
* **Acceptance Criteria**:
  * Report includes introduction, mathematical formulation, base paper comparison, empirical results, and discussion of research gaps.
  * Slides ready for project defense / viva.

---

## 4. Git Collaboration & Integration Workflow

To ensure no collaborator accidentally breaks working code on `main`, adhere to this pull-request workflow:

```bash
# Step 1: Start from the latest clean main branch
git checkout main
git pull origin main

# Step 2: Create your designated stage branch
git checkout -b feature/stage-5-stress-testing

# Step 3: Work, commit, and push to your feature branch
git add .
git commit -m "feat(stage-5): implement cadence dropout simulator and benchmark"
git push -u origin feature/stage-5-stress-testing

# Step 4: Merge back into main when your stage is complete
git checkout main
git pull origin main
git merge feature/stage-5-stress-testing --no-ff -m "merge: integrate Stage 5 stress-testing suite"
git push origin main
```

### Integration Rules:
1. **Never commit large data files directly to Git**: Verify that new data outputs are covered by `.gitignore`.
2. **Every script must run standalone**: Test your script (`python experiments/...`) before opening a pull request.
3. **Update docs alongside code**: Whenever code is added or updated, add the corresponding explanation to `docs/`.
