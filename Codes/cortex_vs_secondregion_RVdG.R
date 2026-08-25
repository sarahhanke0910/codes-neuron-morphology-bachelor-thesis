## ============================================================
## rabies_cortex_vs_secondregion_FIXED.R
##
## Produces Panel A and Panel B figures from the RABIES morphology
## dataset, same design/layout/font sizes as before, but with the
## t-test indexing bug fixed (see note below).
##
##   1. cortex_vs_secondregion_combined.png
##      Cortex vs. paired second brain region, within each condition
##      (SOC/EP/CON/Q), bar charts with individual points + t-test.
##
##   2. cortex_vs_secondregion_pooled_combined.png
##      Cortex vs. non-cortical region, pooled across all conditions,
##      bar charts with individual points (colored by condition) + t-test.
##
## ------------------------------------------------------------
## BUG FIX NOTE:
## In the original run_ttests() function, the t.test() call referenced
## `data[[metric_col]]` inside a dplyr::summarise() after group_by(condition).
## Because `data` is the *ungrouped, full* data frame passed into the
## function (not the current group), while `region_role == "Cortex"`
## is evaluated on the *current group only*, R recycled the short
## logical vector to match the full-length vector. This silently mixed
## values from other conditions into each condition's t-test, producing
## incorrect p-values (this is what generated the wrong values in
## Panel A previously).
##
## Fix: use `.data[[metric_col]]` (the dplyr pronoun for "current data
## in context", i.e. the current group), not the raw `data` argument.
## Panel B (pooled) and the Kruskal-Wallis figure were NOT affected,
## since they don't have this group_by/summarise + external-argument
## indexing pattern.
## ------------------------------------------------------------
##
## Input: Metrics_RABIES_all_conditions.xlsx
##   Required columns: Condition, Brain region, TDL_um, primaries,
##   branch_points, max_order
##
## Usage:
##   Rscript rabies_cortex_vs_secondregion_FIXED.R \
##       Metrics_RABIES_all_conditions.xlsx \
##       output_folder
## ============================================================

suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(stringr)
  library(patchwork)
})

# ---- Command-line arguments ----
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript rabies_cortex_vs_secondregion_FIXED.R <Metrics_RABIES_all_conditions.xlsx> <output_folder>")
}
input_path <- args[1]
output_dir <- args[2]
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# ---- Font sizes (unchanged from previous version) ----
FONT_BASE       <- 26
FONT_AXIS_TEXT  <- 22
FONT_AXIS_TICK  <- 20
FONT_TITLE      <- 24
FONT_SUPTITLE   <- 30
FONT_SUBTITLE   <- 20
FONT_STRIP      <- 22
FONT_LEGEND     <- 20
FONT_LEGEND_TIT <- 22
FONT_PVAL       <- 8    # geom_text/annotate uses mm-based size, not pt

# ---- 1. Load and clean data --------------------------------
df <- read_excel(input_path)
names(df) <- str_trim(names(df))

df <- df %>%
  rename(condition = Condition, region = `Brain region`) %>%
  mutate(
    condition = str_trim(condition),
    region = str_trim(region),
    TDL_um = as.numeric(TDL_um),
    primaries = as.numeric(primaries),
    branch_points = as.numeric(branch_points),
    max_order = as.numeric(max_order)
  )

condition_order <- c("SOC", "EP", "CON", "Q")
df$condition <- factor(df$condition, levels = condition_order)

if (!"flag_incomplete" %in% names(df)) {
  df$flag_incomplete <- FALSE
  message("Note: no 'flag_incomplete' column found -- treating all points as complete.")
}
df$flag_incomplete <- as.logical(df$flag_incomplete)
df$flag_incomplete[is.na(df$flag_incomplete)] <- FALSE

df <- df %>%
  mutate(
    region_role = ifelse(region == "Cortex", "Cortex", "Second region"),
    group_label = paste(condition, region, sep = " \u2013 "),
    region_pooled = ifelse(region == "Cortex", "Cortex", "Non-cortical region"),
    region_pooled = factor(region_pooled, levels = c("Cortex", "Non-cortical region"))
  )

group_order <- df %>%
  distinct(condition, region_role, group_label) %>%
  arrange(condition, desc(region_role == "Cortex")) %>%
  pull(group_label)
df$group_label <- factor(df$group_label, levels = group_order)

condition_colors <- c("SOC" = "#2AB7A9", "EP" = "#5AA9E6", "CON" = "#1B3A6B", "Q" = "#7B3FA0")

base_colors <- list(
  SOC = c("#2AB7A9", "#1A7A6E"),
  EP  = c("#8EC7F2", "#3E7FB8"),
  CON = c("#3A4F8C", "#131A3A"),
  Q   = c("#C79EDB", "#7B3FA0")
)
group_colors <- setNames(character(length(levels(df$group_label))), levels(df$group_label))
for (cond in condition_order) {
  labs_for_cond <- grep(paste0("^", cond, " "), levels(df$group_label), value = TRUE)
  cortex_lab <- labs_for_cond[grepl("Cortex$", labs_for_cond)]
  other_lab <- setdiff(labs_for_cond, cortex_lab)
  if (length(cortex_lab) == 1) group_colors[cortex_lab] <- base_colors[[cond]][1]
  if (length(other_lab) == 1) group_colors[other_lab] <- base_colors[[cond]][2]
}

metrics <- c(
  TDL_um         = "Total Dendrite Length (\u00b5m)",
  primaries      = "Primary Dendrites (n)",
  branch_points  = "Branch Points (n)",
  max_order      = "Max Branch Order (n)"
)


# ================================================================
#  FIGURE 1 (Panel A): Cortex vs. paired second region, by condition
# ================================================================

run_ttests <- function(data, metric_col) {
  data %>%
    group_by(condition) %>%
    summarise(
      p_value = tryCatch(
        t.test(
          .data[[metric_col]][region_role == "Cortex"],
          .data[[metric_col]][region_role == "Second region"]
        )$p.value,
        error = function(e) NA_real_
      ),
      .groups = "drop"
    )
}

ttest_results <- bind_rows(lapply(names(metrics), function(m) {
  res <- run_ttests(df, m); res$metric <- m; res
}))
write.csv(ttest_results, file.path(output_dir, "cortex_vs_secondregion_ttests.csv"), row.names = FALSE)

make_bar_plot <- function(data, metric_col, metric_label, pvals, show_legend = FALSE) {
  plot_data <- data %>% select(condition, region_role, group_label, value = all_of(metric_col))
  summary_data <- plot_data %>%
    group_by(condition, region_role, group_label) %>%
    summarise(mean_val = mean(value, na.rm = TRUE), sd_val = sd(value, na.rm = TRUE), .groups = "drop")

  y_max <- max(summary_data$mean_val + summary_data$sd_val, na.rm = TRUE)
  bracket_data <- pvals %>%
    filter(metric == metric_col) %>%
    mutate(
      label = ifelse(is.na(p_value), "n/a", paste0("p = ", signif(p_value, 3))),
      y_line = y_max * 1.28, y_text = y_max * 1.42, tick = y_max * 0.05
    )

  p <- ggplot() +
    geom_col(data = summary_data, aes(x = group_label, y = mean_val, fill = group_label),
             width = 0.7, color = NA) +
    geom_errorbar(data = summary_data, aes(x = group_label, ymin = mean_val - sd_val, ymax = mean_val + sd_val),
                  width = 0.15, color = "grey30", linewidth = 0.8) +
    geom_jitter(data = plot_data, aes(x = group_label, y = value),
                width = 0.1, size = 2.2, color = "black", alpha = 0.6) +
    geom_segment(data = bracket_data, aes(x = 1, xend = 2, y = y_line, yend = y_line),
                 inherit.aes = FALSE, color = "grey20", linewidth = 0.7) +
    geom_segment(data = bracket_data, aes(x = 1, xend = 1, y = y_line - tick, yend = y_line),
                 inherit.aes = FALSE, color = "grey20", linewidth = 0.7) +
    geom_segment(data = bracket_data, aes(x = 2, xend = 2, y = y_line - tick, yend = y_line),
                 inherit.aes = FALSE, color = "grey20", linewidth = 0.7) +
    geom_text(data = bracket_data, aes(x = 1.5, y = y_text, label = label),
              inherit.aes = FALSE, size = FONT_PVAL) +
    coord_cartesian(clip = "off") +
    facet_grid(cols = vars(condition), scales = "free_x", space = "free", switch = "x") +
    scale_fill_manual(values = group_colors, name = "Condition \u2013 Region") +
    scale_x_discrete(labels = function(x) sub(".*\u2013 ", "", x)) +
    scale_y_continuous(expand = expansion(mult = c(0.02, 0.25))) +
    labs(title = metric_label, x = NULL, y = metric_label) +
    theme_minimal(base_size = FONT_BASE) +
    theme(
      axis.text.x = element_text(angle = 40, hjust = 1, size = FONT_AXIS_TICK),
      axis.text.y = element_text(size = FONT_AXIS_TICK),
      axis.title.y = element_text(size = FONT_AXIS_TEXT),
      panel.grid.minor = element_blank(),
      plot.title = element_text(size = FONT_TITLE, face = "bold"),
      strip.placement = "outside",
      strip.background = element_blank(),
      strip.text.x = element_text(size = FONT_STRIP, face = "bold"),
      panel.spacing = unit(1.0, "lines"),
      legend.text = element_text(size = FONT_LEGEND),
      legend.title = element_text(size = FONT_LEGEND_TIT),
      plot.margin = margin(t = 50, r = 20, b = 15, l = 20)
    )
  if (!show_legend) p <- p + guides(fill = "none")
  p
}

plot_list_1 <- lapply(names(metrics), function(m) make_bar_plot(df, m, metrics[[m]], ttest_results, show_legend = TRUE))
combined_1 <- wrap_plots(plot_list_1, nrow = 2, ncol = 2, guides = "collect") +
  plot_annotation(
    title = "Cortex vs. Paired Second Brain Region \u2014 by Condition",
    theme = theme(plot.title = element_text(size = FONT_SUPTITLE, face = "bold", hjust = 0.5))
  ) &
  theme(legend.position = "bottom")

ggsave(file.path(output_dir, "cortex_vs_secondregion_combined.png"),
       plot = combined_1, width = 20, height = 19, dpi = 300)
message("Saved: cortex_vs_secondregion_combined.png")


# ================================================================
#  FIGURE 2 (Panel B): Cortex vs. non-cortical, pooled across conditions
# ================================================================
# (unaffected by the bug -- kept identical to previous version)

run_pooled_ttest <- function(data, metric_col) {
  tryCatch(
    t.test(
      data[[metric_col]][data$region_pooled == "Cortex"],
      data[[metric_col]][data$region_pooled == "Non-cortical region"]
    )$p.value,
    error = function(e) NA_real_
  )
}
pooled_results <- data.frame(
  metric = names(metrics),
  p_value = sapply(names(metrics), function(m) run_pooled_ttest(df, m))
)
write.csv(pooled_results, file.path(output_dir, "cortex_vs_secondregion_pooled_ttests.csv"), row.names = FALSE)

make_pooled_plot <- function(data, metric_col, metric_label, pvals, show_legend = FALSE) {
  plot_data <- data %>% select(condition, region_pooled, value = all_of(metric_col))
  summary_data <- plot_data %>%
    group_by(region_pooled) %>%
    summarise(mean_val = mean(value, na.rm = TRUE), sd_val = sd(value, na.rm = TRUE), .groups = "drop")

  y_max <- max(summary_data$mean_val + summary_data$sd_val, na.rm = TRUE)
  p_val <- pvals$p_value[pvals$metric == metric_col]
  label <- ifelse(is.na(p_val), "n/a", paste0("p = ", signif(p_val, 3)))
  y_line <- y_max * 1.28; y_text <- y_max * 1.42; tick <- y_max * 0.05

  p <- ggplot() +
    geom_col(data = summary_data, aes(x = region_pooled, y = mean_val),
             fill = NA, color = "grey20", linewidth = 1.0, width = 0.6) +
    geom_errorbar(data = summary_data, aes(x = region_pooled, ymin = mean_val - sd_val, ymax = mean_val + sd_val),
                  width = 0.15, color = "grey30", linewidth = 0.8) +
    geom_jitter(data = plot_data, aes(x = region_pooled, y = value, color = condition),
                width = 0.12, size = 2.6, alpha = 0.85) +
    geom_segment(aes(x = 1, xend = 2, y = y_line, yend = y_line), color = "grey20", linewidth = 0.7) +
    geom_segment(aes(x = 1, xend = 1, y = y_line - tick, yend = y_line), color = "grey20", linewidth = 0.7) +
    geom_segment(aes(x = 2, xend = 2, y = y_line - tick, yend = y_line), color = "grey20", linewidth = 0.7) +
    annotate("text", x = 1.5, y = y_text, label = label, size = FONT_PVAL) +
    coord_cartesian(clip = "off") +
    scale_color_manual(values = condition_colors, name = "Condition") +
    scale_y_continuous(expand = expansion(mult = c(0.02, 0.28))) +
    labs(title = metric_label, x = NULL, y = metric_label) +
    theme_minimal(base_size = FONT_BASE) +
    theme(
      axis.text.x = element_text(size = FONT_AXIS_TICK),
      axis.text.y = element_text(size = FONT_AXIS_TICK),
      axis.title.y = element_text(size = FONT_AXIS_TEXT),
      panel.grid.minor = element_blank(),
      plot.title = element_text(size = FONT_TITLE, face = "bold"),
      legend.text = element_text(size = FONT_LEGEND),
      legend.title = element_text(size = FONT_LEGEND_TIT),
      plot.margin = margin(t = 50, r = 20, b = 15, l = 20)
    )
  if (!show_legend) p <- p + guides(color = "none")
  p
}

plot_list_2 <- lapply(names(metrics), function(m) make_pooled_plot(df, m, metrics[[m]], pooled_results, show_legend = TRUE))
combined_2 <- wrap_plots(plot_list_2, nrow = 2, ncol = 2, guides = "collect") +
  plot_annotation(
    title = "Cortex vs. Non-Cortical Region \u2014 Pooled Across Conditions",
    subtitle = "Points colored by condition; bars show pooled mean \u00b1 SD",
    theme = theme(
      plot.title = element_text(size = FONT_SUPTITLE, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = FONT_SUBTITLE, hjust = 0.5, color = "grey30")
    )
  ) &
  theme(legend.position = "bottom")

ggsave(file.path(output_dir, "cortex_vs_secondregion_pooled_combined.png"),
       plot = combined_2, width = 20, height = 19, dpi = 300)
message("Saved: cortex_vs_secondregion_pooled_combined.png")


# ================================================================
#  FIGURE 3: Cortex only, Kruskal-Wallis across conditions
#  (= Figure 11 "Neuronal Morphology -- Cortex, Across Conditions")
#  Unaffected by the t-test bug (operates on cortex_df directly,
#  not via the buggy group_by/summarise + external-data pattern).
# ================================================================

cortex_df <- df %>% filter(region == "Cortex")

run_kruskal <- function(data, metric_col) {
  tryCatch(kruskal.test(data[[metric_col]] ~ data$condition)$p.value, error = function(e) NA_real_)
}
kruskal_results <- data.frame(
  metric = names(metrics),
  p_value = sapply(names(metrics), function(m) run_kruskal(cortex_df, m))
)
write.csv(kruskal_results, file.path(output_dir, "cortex_kruskal_wallis_results.csv"), row.names = FALSE)

make_kruskal_plot <- function(data, metric_col, metric_label, pvals, show_legend = FALSE) {
  plot_data <- data %>% select(condition, flag_incomplete, value = all_of(metric_col))
  summary_data <- plot_data %>%
    group_by(condition) %>%
    summarise(mean_val = mean(value, na.rm = TRUE), sd_val = sd(value, na.rm = TRUE), .groups = "drop")

  y_max <- max(summary_data$mean_val + summary_data$sd_val, na.rm = TRUE)
  p_val <- pvals$p_value[pvals$metric == metric_col]
  label <- ifelse(is.na(p_val), "n/a", paste0("p = ", signif(p_val, 3)))
  y_text <- y_max * 1.35

  p <- ggplot() +
    geom_errorbar(data = summary_data, aes(x = condition, ymin = mean_val - sd_val, ymax = mean_val + sd_val),
                  width = 0.15, color = "grey30", linewidth = 1.0) +
    geom_jitter(data = plot_data %>% filter(!flag_incomplete),
                aes(x = condition, y = value, color = condition),
                width = 0.12, size = 2.6, alpha = 0.85, shape = 16) +
    geom_jitter(data = plot_data %>% filter(flag_incomplete),
                aes(x = condition, y = value, color = condition),
                width = 0.12, size = 2.8, alpha = 0.9, shape = 1, stroke = 1.3) +
    geom_point(data = summary_data, aes(x = condition, y = mean_val, fill = condition),
               shape = 21, size = 6, color = "black", stroke = 0.8) +
    annotate("text", x = 1.5, y = y_text, label = label, size = FONT_PVAL) +
    coord_cartesian(clip = "off") +
    scale_color_manual(values = condition_colors, name = "Condition") +
    scale_fill_manual(values = condition_colors, guide = "none") +
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.32))) +
    labs(title = metric_label, x = NULL, y = metric_label) +
    theme_minimal(base_size = FONT_BASE) +
    theme(
      axis.text.x = element_text(size = FONT_AXIS_TICK),
      axis.text.y = element_text(size = FONT_AXIS_TICK),
      axis.title.y = element_text(size = FONT_AXIS_TEXT),
      panel.grid.minor = element_blank(),
      plot.title = element_text(size = FONT_TITLE, face = "bold"),
      legend.text = element_text(size = FONT_LEGEND),
      legend.title = element_text(size = FONT_LEGEND_TIT),
      plot.margin = margin(t = 50, r = 20, b = 15, l = 20)
    )
  if (!show_legend) p <- p + guides(color = "none")
  p
}

plot_list_3 <- lapply(names(metrics), function(m) make_kruskal_plot(cortex_df, m, metrics[[m]], kruskal_results, show_legend = TRUE))

combined_3 <- wrap_plots(plot_list_3, nrow = 1, ncol = 4, guides = "collect") +
  plot_annotation(
    title = "Neuronal Morphology \u2014 Cortex, Across Conditions",
    subtitle = "Kruskal-Wallis test; n = 3\u20135 neurons per condition. Filled = complete, open = flagged.",
    theme = theme(
      plot.title = element_text(size = FONT_SUPTITLE, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = FONT_SUBTITLE, hjust = 0.5, color = "grey30")
    )
  ) &
  theme(legend.position = "bottom")

ggsave(file.path(output_dir, "cortex_kruskal_wallis.png"),
       plot = combined_3, width = 22, height = 8.5, dpi = 300)
message("Saved: cortex_kruskal_wallis.png")


message("\nDone. All three figures saved to: ", output_dir)
message("Panel A p-values (condition-specific, BUG FIXED):")
print(ttest_results)
message("Panel B p-values (pooled, unchanged):")
print(pooled_results)
message("Figure 11 p-values (Kruskal-Wallis, unchanged):")
print(kruskal_results)
