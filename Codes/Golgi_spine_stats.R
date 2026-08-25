# ================================================================
# golgi_spine_stats.R
#
# Compares Control vs. Stress Golgi-stained neurons on spine density
# (spines / 10 um dendrite), separately for each brain region (Cortex,
# Hippocampus, Striatum, Amygdala).
#
# Test: Wilcoxon rank-sum test (Mann-Whitney U) -- same rationale as
# golgi_morphology_stats.R and golgi_sholl_stats.R (two-group
# comparison, small non-normal samples). Multiple comparisons (up to 4
# regions) are corrected using Benjamini-Hochberg (BH).
#
# Usage:
#   Rscript golgi_spine_stats.R \
#       path/to/control/PyR_csv/summary_per_neuron.csv \
#       path/to/stress/PyR_csv/summary_per_neuron.csv \
#       path/to/output_folder
#
# Required columns in each input CSV (as produced by
# batch_spine_classification.py / batch_spine_classification.R):
#   neuron, region, density_per_10um
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
  stop("Usage: Rscript golgi_spine_stats.R <control_summary_per_neuron_csv> <stress_summary_per_neuron_csv> <output_folder>")
}
control_path <- args[1]
stress_path  <- args[2]
output_dir   <- args[3]

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# ---- Region color scheme (consistent with the other Golgi plots) ----
region_colors <- c(
  "Cortex"      = "#2EC4B6",
  "Hippocampus" = "#5DA9E9",
  "Striatum"    = "#1f4e9c",
  "Amygdala"    = "#6A4C93"
)
region_order <- names(region_colors)

group_colors <- c("Control" = "#1f4e9c", "Stress" = "#2EC4B6")

# ---- Font sizes (substantially enlarged so titles, axis text, tick
# labels, and legend are clearly readable; layout/colors/logic unchanged) ----
FONT_BASE       <- 20
FONT_AXIS_TEXT  <- 18
FONT_AXIS_TICK  <- 16
FONT_TITLE      <- 20
FONT_LEGEND     <- 16
FONT_LEGEND_TIT <- 17
FONT_SIG_LABEL  <- 7    # significance stars (geom_text size, mm-based)
FONT_PLABEL     <- 4.5  # "p = 0.xxx" annotation under stars
FONT_N_LABEL    <- 5    # "n=X" labels on the bar chart

# ---- Load data ----
message("Loading Control spine data from: ", control_path)
control_df <- read_csv(control_path, show_col_types = FALSE) %>% mutate(group = "Control")

message("Loading Stress spine data from: ", stress_path)
stress_df <- read_csv(stress_path, show_col_types = FALSE) %>% mutate(group = "Stress")

df <- bind_rows(control_df, stress_df) %>%
  mutate(
    group  = factor(group, levels = c("Control", "Stress")),
    region = factor(region, levels = region_order)
  ) %>%
  filter(!is.na(region))

message("\nSample sizes per region and group:")
print(df %>% count(region, group) %>% pivot_wider(names_from = group, values_from = n))

# ---- Parameter to test ----
# density_per_10um is the primary (and only) measure -- filopodia are
# NOT excluded from the density calculation.
params <- c(
  "density_per_10um" = "Spine Density (spines / 10 µm)"
)
params <- params[names(params) %in% names(df)]

if (length(params) == 0) {
  stop("'density_per_10um' not found in the input data.")
}

# ---- Run Wilcoxon rank-sum test per region, per parameter ----
all_results <- list()

for (param in names(params)) {
  message("\n=== ", params[param], " ===")

  res <- tryCatch({
    df %>%
      filter(!is.na(.data[[param]])) %>%
      group_by(region) %>%
      filter(n_distinct(group) == 2) %>%   # only test regions with both groups present
      wilcox_test(as.formula(paste(param, "~ group"))) %>%
      add_significance() %>%
      mutate(parameter = param)
  }, error = function(e) {
    message("  WARNING: test failed for '", param, "' (", conditionMessage(e),
            ") -- skipping this parameter. This can happen when a region has too ",
            "little variance or too few complete observations.")
    NULL
  })

  if (!is.null(res)) {
    print(res %>% select(region, group1, group2, n1, n2, statistic, p, p.signif))
    all_results[[param]] <- res
  }
}

if (length(all_results) == 0) {
  stop("No valid statistical comparisons could be computed for any parameter.")
}

results_df <- bind_rows(all_results) %>%
  ungroup() %>%
  mutate(p.adj = p.adjust(p, method = "BH")) %>%
  add_significance("p.adj") %>%
  select(parameter, region, group1, group2, n1, n2, statistic, p, p.adj, p.adj.signif)

message("\n=== Combined results (BH-corrected across all ", nrow(results_df), " tests) ===")
print(results_df)

results_path <- file.path(output_dir, "golgi_spine_stats_results.csv")
write_csv(results_df, results_path)
message("\nResults table saved to: ", results_path)

# ---- Boxplots (Panel C): one panel per parameter, combined into a single PNG ----
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
      y.position = ymax * 1.12 + 0.05,
      x.num      = region_order_num[as.character(region)],
      p.label    = format_p(p.adj)
    )

  sig_sub    <- stat_sub %>% filter(p.adj < 0.05)
  nonsig_sub <- stat_sub %>% filter(p.adj >= 0.05)

  p <- ggplot(sub, aes(x = region, y = .data[[param]], fill = group)) +
    geom_boxplot(outlier.shape = NA, alpha = 0.5, position = position_dodge(0.75), width = 0.6) +
    geom_point(aes(color = group), position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.75),
               size = 2, alpha = 0.8) +
    scale_fill_manual(values = group_colors) +
    scale_color_manual(values = group_colors)

  if (nrow(nonsig_sub) > 0) {
    p <- p + geom_text(data = nonsig_sub, aes(x = region, y = y.position, label = p.adj.signif),
                        inherit.aes = FALSE, size = FONT_SIG_LABEL, vjust = 0)
  }

  if (nrow(sig_sub) > 0) {
    p <- p +
      geom_segment(data = sig_sub,
                   aes(x = x.num - bracket_half_width, xend = x.num + bracket_half_width,
                       y = y.position, yend = y.position),
                   inherit.aes = FALSE, linewidth = 0.6) +
      geom_text(data = sig_sub, aes(x = x.num, y = y.position, label = p.adj.signif),
                inherit.aes = FALSE, size = FONT_SIG_LABEL, vjust = -0.3) +
      geom_text(data = sig_sub, aes(x = x.num, y = y.position, label = p.label),
                inherit.aes = FALSE, size = FONT_PLABEL, vjust = 1.4)
  }

  p <- p +
    labs(title = params[param], x = "Brain Region", y = params[param], fill = "Group", color = "Group") +
    theme_classic(base_size = FONT_BASE) +
    theme(
      plot.title      = element_text(face = "bold", size = FONT_TITLE, margin = margin(b = 14)),
      axis.title.x    = element_text(size = FONT_AXIS_TEXT, margin = margin(t = 14)),
      axis.title.y    = element_text(size = FONT_AXIS_TEXT, margin = margin(r = 14)),
      axis.text.x     = element_text(size = FONT_AXIS_TICK, angle = 30, hjust = 1, margin = margin(t = 6)),
      axis.text.y     = element_text(size = FONT_AXIS_TICK, margin = margin(r = 4)),
      legend.title    = element_text(size = FONT_LEGEND_TIT),
      legend.text     = element_text(size = FONT_LEGEND, margin = margin(t = 4, b = 4)),
      legend.spacing.y = unit(0.3, "cm"),
      plot.margin     = margin(t = 25, r = 25, b = 20, l = 20)
    ) +
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.18)))

  plot_list[[param]] <- p
}

combined_plot <- ggarrange(plotlist = plot_list, ncol = length(plot_list), nrow = 1,
                            common.legend = TRUE, legend = "right",
                            font.label = list(size = FONT_LEGEND_TIT))

out_path <- file.path(output_dir, "golgi_spine_density_all_parameters.png")
ggsave(out_path, combined_plot, width = 8.5 * length(plot_list), height = 7.5, dpi = 300)
message("  -> ", out_path)

# ---- Bar chart version (Panel D, mean +/- SD) of the primary density parameter ----
# Same underlying Wilcoxon results as the boxplot above, just shown as a
# grouped bar chart with n labels instead of individual data points.
bar_param <- "density_per_10um"
if (bar_param %in% names(params)) {
  sub <- df %>% filter(!is.na(.data[[bar_param]]))
  bar_summary <- sub %>%
    group_by(region, group) %>%
    summarise(mean_density = mean(.data[[bar_param]]), sd_density = sd(.data[[bar_param]]),
              n = n(), .groups = "drop")

  stat_sub <- results_df %>%
    filter(parameter == bar_param) %>%
    left_join(bar_summary %>% group_by(region) %>% summarise(ymax = max(mean_density + coalesce(sd_density, 0))),
              by = "region") %>%
    mutate(
      y.position = ymax * 1.1 + 0.1,
      x.num      = region_order_num[as.character(region)],
      p.label    = format_p(p.adj)
    )
  sig_sub    <- stat_sub %>% filter(p.adj < 0.05)
  nonsig_sub <- stat_sub %>% filter(p.adj >= 0.05)

  p_bar <- ggplot(bar_summary, aes(x = region, y = mean_density, fill = group)) +
    geom_col(position = position_dodge(0.75), width = 0.6, color = "black", linewidth = 0.3) +
    geom_errorbar(aes(ymin = mean_density - coalesce(sd_density, 0), ymax = mean_density + coalesce(sd_density, 0)),
                  position = position_dodge(0.75), width = 0.2) +
    geom_text(aes(y = mean_density + coalesce(sd_density, 0) + 0.15, label = paste0("n=", n)),
              position = position_dodge(0.75), size = FONT_N_LABEL, vjust = 0) +
    scale_fill_manual(values = group_colors)

  if (nrow(nonsig_sub) > 0) {
    p_bar <- p_bar + geom_text(data = nonsig_sub, aes(x = region, y = y.position, label = p.adj.signif),
                                inherit.aes = FALSE, size = FONT_SIG_LABEL, vjust = 0)
  }
  if (nrow(sig_sub) > 0) {
    p_bar <- p_bar +
      geom_segment(data = sig_sub,
                   aes(x = x.num - bracket_half_width, xend = x.num + bracket_half_width,
                       y = y.position, yend = y.position),
                   inherit.aes = FALSE, linewidth = 0.6) +
      geom_text(data = sig_sub, aes(x = x.num, y = y.position, label = p.adj.signif),
                inherit.aes = FALSE, size = FONT_SIG_LABEL, vjust = -0.3) +
      geom_text(data = sig_sub, aes(x = x.num, y = y.position, label = p.label),
                inherit.aes = FALSE, size = FONT_PLABEL, vjust = 1.4)
  }

  p_bar <- p_bar +
    labs(title = "Mean Spine Density by Brain Region - Control vs. Stress",
         x = "Brain Region", y = "Mean Spine Density (spines / 10 µm)", fill = "Group") +
    theme_classic(base_size = FONT_BASE) +
    theme(
      plot.title      = element_text(face = "bold", size = FONT_TITLE, margin = margin(b = 14)),
      axis.title.x    = element_text(size = FONT_AXIS_TEXT, margin = margin(t = 14)),
      axis.title.y    = element_text(size = FONT_AXIS_TEXT, margin = margin(r = 14)),
      axis.text.x     = element_text(size = FONT_AXIS_TICK, angle = 30, hjust = 1, margin = margin(t = 6)),
      axis.text.y     = element_text(size = FONT_AXIS_TICK, margin = margin(r = 4)),
      legend.title    = element_text(size = FONT_LEGEND_TIT),
      legend.text     = element_text(size = FONT_LEGEND, margin = margin(t = 4, b = 4)),
      legend.spacing.y = unit(0.3, "cm"),
      plot.margin     = margin(t = 25, r = 25, b = 20, l = 20)
    ) +
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.22)))

  bar_out_path <- file.path(output_dir, "golgi_spine_mean_density_bar_chart.png")
  ggsave(bar_out_path, p_bar, width = 11.5, height = 8, dpi = 300)
  message("  -> ", bar_out_path)
}

message("\nDone.")
