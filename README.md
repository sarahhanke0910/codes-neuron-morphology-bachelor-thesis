# Scripts for Bachelor Thesis: Stress-Induced Neuronal Remodeling

| File name | Figure | Application |
|---|---|---|
| lif_to_tif.py | — | Convert lif files to tif files |
| lif_resolution.py | — | Extract individual pixel resolution for each z-stack |
| swc_metrics.py | — | Extract morphological metrics from swc reconstruction file |
| validation_plot.py | 3 | Generated descriptive dot plots comparing morphological parameters (total dendrite length, branch points, primary dendrites, max branch order) across brain regions for RVdG-labeled neurons |
| failed_only_plot.py | 5 | Characterize failed reconstructions by extracting morphological metrics and calculating reconstruction success rates per condition and brain region |
| Sholl_analyis_RVdG.py | 6 + 7 | Perform Sholl analysis on RVdG-labeled neurons, generating individual and mean ± SEM intersection profiles and calculate derived Sholl parameters (N_max, r_critical, AUC, enclosing radius, regression coefficient k) |
| cortex_vs_secondregion_RVdG.R | 8 + 9 | Compare morphological parameters between Cortex and paired second brain region (condition-specific t-tests, pooled comparison, and Kruskal-Wallis test across conditions) |
| Golgi_morpholgoy_stats.R | 12 | Compare morphological parameters (total dendrite length, branch points, primary dendrites, max branch order) between Control and Stress Golgi-stained neurons per brain region using Wilcoxon rank-sum tests |
| Sholl_analysis_Golgi.py / Sholl_curves_control_vs_stress_Golgi.py | 13 | Perform Sholl analysis on Golgi-stained neurons and generate per-region Sholl profile curves overlaying Control and Stress conditions |
| Golgi_sholl_stats.R | 14 | Compare Sholl-derived parameters (N_max, r_critical, AUC, enclosing radius, regression coefficient k) between Control and Stress Golgi-stained neurons per brain region |
| spine_classification_Golgi.py / spine_control_vs_stress_plot.py / Golgi_spines_stats.R | 16 | Classify traced dendritic spines into morphological subtypes, generate spine type distribution and density plots, and statistically compare spine density between Control and Stress per brain region |
