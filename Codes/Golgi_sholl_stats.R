# ================================================================
# golgi_sholl_stats.R
#
# Compares Control vs. Stress Golgi-stained neurons on five Sholl-
# derived parameters (N_max, r_critical, AUC, enclosing_r, sholl_k),
# separately for each brain region (Cortex, Hippocampus, Striatum,
# Amygdala).
#
# Test: Wilcoxon rank-sum test (Mann-Whitney U) -- same rationale as
# golgi_morphology_stats.R (two-group comparison, small non-normal
# samples). Multiple comparisons (5 parameters x up to 4 regions) are
# corrected using Benjamini-Hochberg (BH).
#
# Usage:
#   Rscript golgi_sholl_stats.R \
#       path/to/control_sholl_results/sholl_params.csv \
#       path/to/stress_sholl_results/sholl_params.csv \
#       path/to/output_folder
#
# Required columns in each input CSV (as produced by sholl_analysis_golgi.py):
#   region, N_max, r_critical, AUC, enclosing_r, sholl_k, flagged
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
  stop("Usage: Rscript golgi_sholl_stats.R <control_sholl_params_csv> <stress_sholl_params_csv> <output_folder>")
}
control_path <- args[1]
stress_path  <- args[2]
output_dir   <- args[3]

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# ---- Region color scheme (consistent with the Python Golgi/Sholl plots) ----
region_colors <- c(
  "Cortex"      = "#1f4e9c",
  "Hippocampus" = "#5DA9E9",
  "Striatum"    = "#2EC4B6",
  "Amygdala"    = "#6A4C93"
)
region_order <- names(region_colors)

group_colors <- c("Control" = "#1f4e9c", "Stress" = "#2EC4B6")

# ---- Load data ----
message("Loading Control Sholl parameters from: ", control_path)
control_df <- read_csv(control_path, show_col_types = FALSE) %>% mutate(group = "Control")

message("Loading Stress Sholl parameters from: ", stress_path)
stress_df <- read_csv(stress_path, show_col_types = FALSE) %>% mutate(group = "Stress")

df <- bind_rows(control_df, stress_df) %>%
  mutate(
    group  = factor(group, levels = c("Control", "Stress")),
    region = factor(region, levels = region_order)
  ) %>%
  filter(!is.na(region))

if (!"flagged" %in% names(df)) {
  df <- df %>% mutate(flagged = FALSE)
}
df$flagged <- as.logical(df$flagged)
df$flagged[is.na(df$flagged)] <- FALSE

message("\nSample sizes per region and group:")
print(df %>% count(region, group) %>% pivot_wider(names_from = group, values_from = n))

n_flagged <- sum(df$flagged, na.rm = TRUE)
if (n_flagged > 0) {
  message("\nNote: ", n_flagged, " cells are flagged as potentially incomplete reconstructions ",
          "(retained in all statistical tests; shown as open circles in the plots below, ",
          "consistent with the RABIES figures).")
}

# ---- Parameters to test ----
params <- c(
  "N_max"       = "N_max",
  "r_critical"  = "r_critical (µm)",
  "AUC"         = "AUC",
  "enclosing_r" = "Enclosing Radius (µm)",
  "sholl_k"     = "Sholl Regression Coefficient (k)"
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

results_path <- file.path(output_dir, "golgi_sholl_stats_results.csv")
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

  max_by_region <- sub %>% group_by(region) %>% summarise(ymax = max(.data[[param]], na.rm = TRUE))
  stat_sub <- stat_sub %>%
    left_join(max_by_region, by = "region") %>%
    mutate(
      y.position = ymax + abs(ymax) * 0.12 + 0.01,
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
      geom_segment(data = sig_sub,
                   aes(x = x.num - bracket_half_width, xend = x.num + bracket_half_width,
                       y = y.position, yend = y.position),
                   inherit.aes = FALSE, linewidth = 0.5) +
      geom_text(data = sig_sub, aes(x = x.num, y = y.position, label = p.adj.signif),
                inherit.aes = FALSE, size = 5, vjust = -0.3) +
      geom_text(data = sig_sub, aes(x = x.num, y = y.position, label = p.label),
                inherit.aes = FALSE, size = 3, vjust = 1.4)
  }

  p <- p +
    labs(title = params[param], x = "Brain Region", y = params[param], fill = "Group", color = "Group") +
    theme_classic(base_size = 13) +
    theme(plot.title = element_text(face = "bold")) +
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.18)))

  if (param == "sholl_k") {
    p <- p + geom_hline(yintercept = 0, linetype = "dashed", color = "grey60")
  }

  plot_list[[param]] <- p
}

combined_plot <- ggarrange(plotlist = plot_list, ncol = length(plot_list), nrow = 1,
                            common.legend = TRUE, legend = "right")

out_path <- file.path(output_dir, "golgi_sholl_all_parameters.png")
ggsave(out_path, combined_plot, width = 4.5 * length(plot_list), height = 5.5, dpi = 200)
message("  -> ", out_path)

message("\nDone.")
