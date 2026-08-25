# ================================================================
# golgi_morphology_stats.R
#
# Compares Control vs. Stress Golgi-stained neurons on four
# morphological parameters (Total Dendrite Length, Branch Points,
# Primary Dendrites, Max Branch Order), separately for each brain
# region (Cortex, Hippocampus, Striatum, Amygdala).
#
# Test: Wilcoxon rank-sum test (Mann-Whitney U), appropriate for a
# two-group comparison with small, non-normally-distributed samples
# (consistent with the non-parametric approach used throughout this
# thesis, see combined_morphology_stats.R for the RABIES/4-condition
# analogue, which additionally required Kruskal-Wallis + Dunn's
# post-hoc since it compared >2 groups).
#
# Multiple comparisons (4 parameters x up to 4 regions = up to 16
# tests) are corrected using Benjamini-Hochberg (BH).
#
# Usage:
#   Rscript golgi_morphology_stats.R \
#       path/to/controls_all_merged_cleaned.csv \
#       path/to/stress_all_merged.csv \
#       path/to/output_folder
#
# Required columns in each input CSV (as produced by swc_metrics.py):
#   TDL_um, branch_points, primaries, max_order, region
#
# Required packages: tidyverse, rstatix, ggpubr
# ================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(rstatix)
  library(ggpubr)
})

# ---- Command-line arguments ----
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript golgi_morphology_stats.R <control_csv> <stress_csv> <output_folder>")
}
control_path <- args[1]
stress_path  <- args[2]
output_dir   <- args[3]

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# ---- Region color scheme (consistent with the Python Golgi plots) ----
region_colors <- c(
  "Cortex"      = "#1f4e9c",
  "Hippocampus" = "#5DA9E9",
  "Striatum"    = "#2EC4B6",
  "Amygdala"    = "#6A4C93"
)
region_order <- names(region_colors)

group_colors <- c("Control" = "#1f4e9c", "Stress" = "#2EC4B6")

# ---- Load data ----
message("Loading Control data from: ", control_path)
control_df <- read_csv(control_path, show_col_types = FALSE) %>% mutate(group = "Control")

message("Loading Stress data from: ", stress_path)
stress_df <- read_csv(stress_path, show_col_types = FALSE) %>% mutate(group = "Stress")

df <- bind_rows(control_df, stress_df) %>%
  mutate(
    group  = factor(group, levels = c("Control", "Stress")),
    region = factor(region, levels = region_order)
  ) %>%
  filter(!is.na(region))

# Normalize the flagged/incomplete-reconstruction column (swc_metrics.py
# calls it 'flag_incomplete'). If absent, treat all neurons as not flagged.
if ("flag_incomplete" %in% names(df)) {
  df <- df %>% mutate(flagged = as.logical(flag_incomplete))
} else if (!"flagged" %in% names(df)) {
  df <- df %>% mutate(flagged = FALSE)
}
df$flagged[is.na(df$flagged)] <- FALSE

message("\nSample sizes per region and group:")
print(df %>% count(region, group) %>% pivot_wider(names_from = group, values_from = n))

n_flagged <- sum(df$flagged, na.rm = TRUE)
if (n_flagged > 0) {
  message("\nNote: ", n_flagged, " neurons are flagged as potentially incomplete reconstructions ",
          "(retained in all statistical tests; shown as open circles in the plots below, ",
          "consistent with the RABIES figures).")
}

# ---- Parameters to test ----
params <- c(
  "TDL_um"        = "Total Dendrite Length (µm)",
  "branch_points" = "Branch Points (n)",
  "primaries"     = "Primary Dendrites (n)",
  "max_order"     = "Max Branch Order (n)"
)

# ---- Run Wilcoxon rank-sum test per region, per parameter ----
all_results <- list()

for (param in names(params)) {
  message("\n=== ", params[param], " ===")

  res <- df %>%
    filter(!is.na(.data[[param]])) %>%
    group_by(region) %>%
    filter(n_distinct(group) == 2) %>%   # only test regions with both groups present
    wilcox_test(as.formula(paste(param, "~ group"))) %>%
    add_significance() %>%
    mutate(parameter = param)

  print(res %>% select(region, group1, group2, n1, n2, statistic, p, p.signif))
  all_results[[param]] <- res
}

results_df <- bind_rows(all_results) %>%
  ungroup() %>%
  mutate(p.adj = p.adjust(p, method = "BH")) %>%
  add_significance("p.adj") %>%
  select(parameter, region, group1, group2, n1, n2, statistic, p, p.adj, p.adj.signif)

message("\n=== Combined results (BH-corrected across all ", nrow(results_df), " tests) ===")
print(results_df)

results_path <- file.path(output_dir, "golgi_morphology_stats_results.csv")
write_csv(results_df, results_path)
message("\nResults table saved to: ", results_path)

# ---- Boxplots: one panel per parameter, combined into a single PNG ----
plot_list <- list()
region_order_num <- setNames(seq_along(region_order), region_order)
bracket_half_width <- 0.1875  # matches position_dodge(0.75) box centers for 2 groups

format_p <- function(p) {
  ifelse(p < 0.001, "p < 0.001", paste0("p = ", sprintf("%.3f", p)))
}

for (param in names(params)) {
  sub <- df %>% filter(!is.na(.data[[param]]))

  stat_sub <- results_df %>%
    filter(parameter == param) %>%
    mutate(y.position = NA_real_)

  # compute a sensible y position for significance brackets (above the max value per region)
  max_by_region <- sub %>% group_by(region) %>% summarise(ymax = max(.data[[param]], na.rm = TRUE))
  stat_sub <- stat_sub %>%
    left_join(max_by_region, by = "region") %>%
    mutate(
      y.position = ymax * 1.08,
      x.num      = region_order_num[as.character(region)],
      p.label    = format_p(p.adj)
    )

  sig_sub    <- stat_sub %>% filter(p.adj < 0.05)
  nonsig_sub <- stat_sub %>% filter(p.adj >= 0.05)

  p <- ggplot(sub, aes(x = region, y = .data[[param]], fill = group)) +
    geom_boxplot(outlier.shape = NA, alpha = 0.5, position = position_dodge(0.75), width = 0.6) +
    geom_point(aes(color = group, shape = flagged, group = group),
               position = position_jitterdodge(jitter.width = 0.12, dodge.width = 0.75, seed = 42),
               size = 2, alpha = 0.8) +
    scale_shape_manual(values = c(`FALSE` = 16, `TRUE` = 1), guide = "none") +
    scale_fill_manual(values = group_colors) +
    scale_color_manual(values = group_colors)

  if (nrow(nonsig_sub) > 0) {
    p <- p + geom_text(data = nonsig_sub, aes(x = region, y = y.position, label = p.adj.signif),
                        inherit.aes = FALSE, size = 5, vjust = 0)
  }

  if (nrow(sig_sub) > 0) {
    p <- p +
      # bracket line under the star, connecting the Control and Stress boxes
      geom_segment(data = sig_sub,
                   aes(x = x.num - bracket_half_width, xend = x.num + bracket_half_width,
                       y = y.position, yend = y.position),
                   inherit.aes = FALSE, linewidth = 0.5) +
      # significance stars above the bracket
      geom_text(data = sig_sub, aes(x = x.num, y = y.position, label = p.adj.signif),
                inherit.aes = FALSE, size = 5, vjust = -0.3) +
      # exact (BH-corrected) p-value below the bracket
      geom_text(data = sig_sub, aes(x = x.num, y = y.position, label = p.label),
                inherit.aes = FALSE, size = 3, vjust = 1.4)
  }

  p <- p +
    labs(title = params[param], x = "Brain Region", y = params[param], fill = "Group", color = "Group") +
    theme_classic(base_size = 13) +
    theme(plot.title = element_text(face = "bold")) +
    # a little extra headroom so stars/p-values/brackets aren't clipped
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.18)))

  plot_list[[param]] <- p
}

combined_plot <- ggarrange(plotlist = plot_list, ncol = 2, nrow = ceiling(length(plot_list) / 2),
                            common.legend = TRUE, legend = "right")

out_path <- file.path(output_dir, "golgi_morphology_all_parameters.png")
ggsave(out_path, combined_plot, width = 12, height = 5 * ceiling(length(plot_list) / 2), dpi = 200)
message("  -> ", out_path)

message("\nDone.")
