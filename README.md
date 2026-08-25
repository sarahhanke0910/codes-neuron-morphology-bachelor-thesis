# Scripts for Bachelor Thesis: Stress-Induced Neuronal Remodeling

| File name | Figure | Application |
|---|---|---|
| lif_to_tif.py | — | Convert LIF files to TIF files |
| lif_resolution.py | — | Extract individual pixel resolution for each z-stack |
| swc_metrics.py | — | Extract morphological metrics from SWC reconstruction file |
| validation_plot.py | 3 | Generate descriptive dot plots comparing morphological parameters across brain regions |
| failed_only_plot.py | 5 | Characterize failed reconstructions by extracting morphological metrics and calculating reconstruction success rates |
| Sholl_analyis_RVdG.py | 6 + 7 | Perform Sholl analysis on RVdG-labeled neurons, generating individual and mean ± SEM intersection profiles |
| cortex_vs_secondregion_RVdG.R | 8 + 9 | Compare morphological parameters between Cortex and paired second brain region |
| Golgi_morphology_stats.R | 12 | Compare morphological parameters (total dendrite length, branch points, primary dendrites, max branch order) between Control and Stress |
| Sholl_analysis_Golgi.py / Sholl_curves_control_vs_stress_Golgi.py | 13 | Perform Sholl analysis on Golgi-stained neurons and generate per-region Sholl profile curves |
| Golgi_sholl_stats.R | 14 | Compare Sholl-derived parameters (N_max, r_critical, AUC, enclosing radius, regression coefficient k) |
| spine_classification_Golgi.py / spine_control_vs_stress_plot.py / Golgi_spines_stats.R | 16 | Classify traced dendritic spines into morphological subtypes and compare spine density between Control and Stress |
