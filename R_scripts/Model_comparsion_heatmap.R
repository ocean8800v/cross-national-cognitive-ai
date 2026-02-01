library(ComplexHeatmap)
library(circlize)
library(readr)
library(stringr)
library(grid)
library(showtext)

font_add("Arial", "Arial.ttf")
showtext_auto()

auc_data <- read_csv("AUC_Performance.csv", show_col_types = FALSE)
gap_data <- read_csv("Generalisation_Gap.csv", show_col_types = FALSE)
imp_data <- read_csv("AUC_Improvement.csv", show_col_types = FALSE)

auc_data <- auc_data[!grepl("^Group1_", auc_data$Group_Country), ]
auc_data <- auc_data[!grepl("_(HRS|ELSA|LASI|MHAS)$", auc_data$Group_Country), ]

gap_data <- gap_data[!grepl("^Group1_", gap_data$Group_Country), ]
gap_data <- gap_data[!grepl("_(HRS|ELSA|LASI|MHAS)$", gap_data$Group_Country), ]

imp_data <- imp_data[!grepl("^Group1_", imp_data$Group_Country), ]
imp_data <- imp_data[!grepl("_(HRS|ELSA|LASI|MHAS)$", imp_data$Group_Country), ]

reorder_and_add_charls <- function(df) {
  g2_overall <- df[df$Group_Country == "Group2_Overall (4-country)", ]
  g2_charls <- df[df$Group_Country == "Group2_CHARLS (external)", ]
  g2_charls_avg <- g2_charls
  g2_charls_avg$Group_Country <- "Group2_CHARLS_Average (external)"
  
  g3_overall <- df[df$Group_Country == "Group3_Overall (4-country)", ]
  g3_charls <- df[df$Group_Country == "Group3_CHARLS (external)", ]
  g3_charls_avg <- g3_charls
  g3_charls_avg$Group_Country <- "Group3_CHARLS_Average (external)"
  
  rbind(g2_overall, g2_charls_avg, g3_overall, g3_charls_avg)
}

auc_data <- reorder_and_add_charls(auc_data)

gap_data <- reorder_and_add_charls(gap_data)
gap_data[grep("CHARLS_Average", gap_data$Group_Country), -1] <- NA

reorder_imp <- function(df) {
  g2_overall <- df[df$Group_Country == "Group2_Overall (4-country)", ]
  g2_charls_avg <- df[df$Group_Country == "Group2_CHARLS_Average (external)", ]
  
  g3_overall <- df[df$Group_Country == "Group3_Overall (4-country)", ]
  g3_charls_avg <- df[df$Group_Country == "Group3_CHARLS_Average (external)", ]
  
  rbind(g2_overall, g2_charls_avg, g3_overall, g3_charls_avg)
}

imp_data <- reorder_imp(imp_data)

auc_data <- auc_data[, -2]
gap_data <- gap_data[, -2]
imp_data <- imp_data[, -2]

first_num <- function(x){
  if (is.na(x)) return(NA_real_)
  x <- gsub("\u00A0"," ", trimws(as.character(x)))
  m <- str_match(x, "[-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?")
  as.numeric(m[,1])
}

fmt_auc <- function(x){
  if (is.na(x) || trimws(x)=="") return("")
  x <- gsub("\u00A0"," ", trimws(as.character(x)))
  m <- str_match(x, "([-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)\\s*\\(([-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)\\)\\s*[\\[\\(]\\s*([-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)\\s*,\\s*([-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)\\s*[\\]\\)]")
  if (any(is.na(m))) {
    v <- first_num(x)
    if (is.na(v)) return("")
    return(ifelse(round(v, 3) == 0, "0", sprintf("%.2f", v)))
  }
  main_val <- ifelse(round(as.numeric(m[2]), 3) == 0, "0", sub("\\.?0+$", "", sprintf("%.2f", as.numeric(m[2]))))
  sd_val   <- ifelse(round(as.numeric(m[3]), 3) == 0, "0", sub("\\.?0+$", "", sprintf("%.3f", as.numeric(m[3]))))
  min_val  <- ifelse(round(as.numeric(m[4]), 3) == 0, "0", sub("\\.?0+$", "", sprintf("%.2f", as.numeric(m[4]))))
  max_val  <- ifelse(round(as.numeric(m[5]), 3) == 0, "0", sub("\\.?0+$", "", sprintf("%.2f", as.numeric(m[5]))))
  sprintf("%s (%s)\n[%s, %s]", main_val, sd_val, min_val, max_val)
}

fmt_gap <- function(x){
  if (is.na(x) || trimws(x)=="") return("")
  x <- gsub("\u00A0"," ", trimws(as.character(x)))
  m <- str_match(x, "([-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)\\s*\\(([-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)\\)")
  if (any(is.na(m))) {
    v <- first_num(x)
    if (is.na(v)) return("")
    return(ifelse(round(v, 3) == 0, "0", sprintf("%.3f", v)))
  }
  v1 <- ifelse(round(as.numeric(m[2]), 3) == 0, "0", sub("\\.?0+$", "", sprintf("%.3f", as.numeric(m[2]))))
  v2 <- ifelse(round(as.numeric(m[3]), 3) == 0, "0", sub("\\.?0+$", "", sprintf("%.3f", as.numeric(m[3]))))
  sprintf("%s (%s)", v1, v2)
}

fmt_imp <- function(x){
  if (is.na(x) || trimws(x)=="") return("")
  x <- gsub("\u00A0"," ", trimws(as.character(x)))
  m <- str_match(
    x,
    "([-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)\\s*\\(([-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)%,\\s*([-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)%\\)"
  )
  if (any(is.na(m))) {
    v <- first_num(x)
    if (is.na(v)) return("")
    return(ifelse(round(v, 3) == 0, "0", sprintf("%.3f", v)))
  }
  abs_val <- as.numeric(m[2])
  rel_pct <- as.numeric(m[3]) 
  
  format_pct <- function(pct){
    if (abs(pct - round(pct)) < .Machine$double.eps^0.5) {
      sprintf("%.0f%%", pct)
    } else {
      sprintf("%.1f%%", pct)
    }
  }
  
  abs_val_rounded <- sign(abs_val) * floor(abs(abs_val) * 100 + 0.5) / 100
  abs_val_fmt <- ifelse(
    abs_val_rounded == 0,
    "0",
    sub("\\.?0+$", "", as.character(abs_val_rounded))
  )
  
  sprintf("%s (%s)", abs_val_fmt, format_pct(rel_pct))
}

auc_mat <- do.call(cbind, lapply(auc_data[,-1], function(col) sapply(col, first_num)))
gap_mat <- do.call(cbind, lapply(gap_data[,-1], function(col) sapply(col, first_num)))
imp_mat <- do.call(cbind, lapply(imp_data[,-1], function(col) sapply(col, first_num)))

rownames(auc_mat) <- gsub("Group\\d+_", "", auc_data$Group_Country)
rownames(gap_mat) <- gsub("Group\\d+_", "", gap_data$Group_Country)
rownames(imp_mat) <- gsub("Group\\d+_", "", imp_data$Group_Country)

rownames(auc_mat) <- gsub("Overall \\(4-country\\)", "Overall\n(4-country)", rownames(auc_mat))
rownames(gap_mat) <- gsub("Overall \\(4-country\\)", "Overall\n(4-country)", rownames(gap_mat))
rownames(imp_mat) <- gsub("Overall \\(4-country\\)", "Overall\n(4-country)", rownames(imp_mat))

rownames(auc_mat) <- gsub("CHARLS_Average \\(external\\)", "CHARLS\n(external)", rownames(auc_mat))
rownames(gap_mat) <- gsub("CHARLS_Average \\(external\\)", "CHARLS\n(external)", rownames(gap_mat))
rownames(imp_mat) <- gsub("CHARLS_Average \\(external\\)", "CHARLS\n(external)", rownames(imp_mat))

auc_text <- do.call(cbind, lapply(auc_data[,-1], function(col) sapply(col, fmt_auc)))
gap_text <- do.call(cbind, lapply(gap_data[,-1], function(col) sapply(col, fmt_gap)))
imp_text <- do.call(cbind, lapply(imp_data[,-1], function(col) sapply(col, fmt_imp)))

gap_text[is.na(gap_mat)] <- "—"
gap_mat[is.na(gap_mat)] <- NA

imp_text[is.na(imp_mat)] <- "—" 
imp_mat[is.na(imp_mat)] <- NA

valid_auc <- as.numeric(auc_mat[is.finite(auc_mat)])
col_auc <- colorRamp2(
  c(0.5, 0.9),
  c("#f0f0f0", "#E15A56")
)

col_gap <- colorRamp2(
  c(-0.07, -0.03, 0, 0.03, 0.07),
  c("#F19B57", "#FCE0C8", "#f0f0f0", "#FCE0C8", "#F19B57")
)

col_imp <- colorRamp2(
  c(0, 0.20),
  c("#f0f0f0", "#377E8E")
)

group_labels <- gsub("_.*","", auc_data$Group_Country)
group_labels <- gsub("Group2", "Model I", group_labels)
group_labels <- gsub("Group3", "Model II", group_labels)

colnames(auc_mat) <- colnames(gap_mat) <- colnames(imp_mat) <- rep("", 3)
colnames(auc_text) <- colnames(gap_text) <- colnames(imp_text) <- rep("", 3)

ht_opt(
  COLUMN_ANNO_PADDING = unit(0.5, "mm"),
  TITLE_PADDING = unit(1, "mm"),
  ROW_ANNO_PADDING = unit(0.5, "mm")
)

config <- list(
  title_fontsize = 7,
  col_label_fontsize = 5.2,
  row_label_fontsize = 5,
  cell_text_fontsize = 5.5,
  row_title_fontsize = 7,
  legend_fontsize = 5,
  
  col_anno_height = unit(7.5, "mm"),
  row_gap = unit(0, "mm"),
  heatmap_gap = unit(4, "mm"),
  
  width = unit(40, "mm"), 
  row_height = unit(7, "mm")
)

col_labels <- columnAnnotation(
  label = anno_text(
    c("Cognitive\nImpairment", "MCI", "Dementia"),
    just = "center", 
    location = unit(0.3, "npc"),
    rot = 0,
    gp = gpar(fontsize = config$col_label_fontsize, fontfamily = "sans", 
              fontface = c("bold", "plain", "plain"))  
  ),
  show_annotation_name = FALSE,
  height = unit(6, "mm")
)

row_labels <- rowAnnotation(
  labels = anno_text(
    rownames(auc_mat),
    just = "right",
    location = unit(0.99, "npc"),
    gp = gpar(fontsize = config$row_label_fontsize, fontfamily = "sans")
  ),
  show_annotation_name = FALSE,
  width = unit(20, "mm")
)

group_counts <- table(factor(group_labels, levels = c("Model I", "Model II")))
group_positions <- cumsum(group_counts) - group_counts/2 + 0.5
group_text <- rep("", nrow(imp_mat))
group_text[round(group_positions)] <- c("Model I", "Model II")

group_anno <- rowAnnotation(
  Group = anno_text(
    group_text,
    just = "left",
    location = unit(0.1, "npc"),
    rot = 90,
    gp = gpar(fontsize = 6.5, fontfamily = "sans")
  ),
  show_annotation_name = FALSE,
  width = unit(15, "mm")
)

group_counts <- table(factor(group_labels, levels = c("Model I", "Model II")))
group_boundaries <- cumsum(group_counts)[-length(group_counts)]

a_ht <- Heatmap(
  auc_mat, 
  col = col_auc, 
  name = "Mean AUC",
  cluster_rows = FALSE, 
  cluster_columns = FALSE,
  show_row_names = FALSE,
  column_title = "a. AUROC: Mean (SD) [Min, Max]",
  column_title_gp = gpar(fontsize = config$title_fontsize, just = "left", fontfamily = "sans"),
  column_title_side = "top",
  top_annotation = col_labels,
  left_annotation = row_labels,
  width = config$width,
  height = unit(nrow(auc_mat) * as.numeric(config$row_height), "mm"),
  
  cell_fun = function(j,i,x,y,w,h,fill){
    fontface_style <- if(j == 1) "bold" else "plain"  
    grid.text(auc_text[i,j], x, y, gp = gpar(fontsize = config$cell_text_fontsize, 
                                             fontfamily = "sans", fontface = fontface_style))
    if(i %in% group_boundaries) {
      grid.lines(c(0, 1), c(as.numeric(y) - as.numeric(h)/2, as.numeric(y) - as.numeric(h)/2), 
                 gp = gpar(col = "black", lwd = 0.6))
    }
  },
  rect_gp = gpar(col = "transparent", lwd = 0),
  heatmap_legend_param = list(
    direction = "horizontal",
    legend_width = config$width,
    legend_height = unit(3, "mm"),
    title_gp = gpar(fontsize = config$legend_fontsize, fontfamily = "sans"),
    labels_gp = gpar(fontsize = config$legend_fontsize, fontfamily = "sans"),
    grid_height = unit(3, "mm")
  )
)

b_ht <- Heatmap(
  gap_mat, 
  col = col_gap, 
  name = "Mean Gap",
  cluster_rows = FALSE, 
  na_col = "#FFFFFF",
  cluster_columns = FALSE,
  show_row_names = FALSE,
  column_title = "b. Validation-Test Gap: Mean (SD)",
  column_title_gp = gpar(fontsize = config$title_fontsize, just = "left", fontfamily = "sans"),
  column_title_side = "top",
  top_annotation = col_labels,
  width = config$width,
  height = unit(nrow(gap_mat) * as.numeric(config$row_height), "mm"),
  
  cell_fun = function(j,i,x,y,w,h,fill){
    fontface_style <- if(j == 1) "bold" else "plain"  
    grid.text(gap_text[i,j], x, y, gp = gpar(fontsize = config$cell_text_fontsize, 
                                             fontfamily = "sans", fontface = fontface_style))
    if(i %in% group_boundaries) {
      grid.lines(c(0, 1), c(as.numeric(y) - as.numeric(h)/2, as.numeric(y) - as.numeric(h)/2), 
                 gp = gpar(col = "black", lwd = 0.6))
    }
  },
  rect_gp = gpar(col = "transparent", lwd = 0),
  heatmap_legend_param = list(
    direction = "horizontal",
    at = c(-0.03, -0.015, 0, 0.015, 0.03),
    labels = c("-0.03", "-0.015", "0", "0.015", "0.03"),
    legend_width = config$width,
    legend_height = unit(3, "mm"),
    title_gp = gpar(fontsize = config$legend_fontsize, fontfamily = "sans"),
    labels_gp = gpar(fontsize = config$legend_fontsize, fontfamily = "sans"),
    grid_height = unit(3, "mm")
  )
)

c_ht <- Heatmap(
  imp_mat, 
  col = col_imp, 
  name = "Absolute Improvement",
  cluster_rows = FALSE, 
  cluster_columns = FALSE,
  show_row_names = FALSE,
  column_title = "c. Mean Abs. Improvement (Rel.%)",
  column_title_gp = gpar(fontsize = config$title_fontsize, just = "left", fontfamily = "sans"),
  column_title_side = "top",
  top_annotation = col_labels,
  right_annotation = group_anno,
  width = config$width,
  height = unit(nrow(imp_mat) * as.numeric(config$row_height), "mm"),
  
  cell_fun = function(j,i,x,y,w,h,fill){
    fontface_style <- if(j == 1) "bold" else "plain"  
    grid.text(imp_text[i,j], x, y, gp = gpar(fontsize = config$cell_text_fontsize, 
                                             fontfamily = "sans", fontface = fontface_style))
    if(i %in% group_boundaries) {
      grid.lines(c(0, 1), c(as.numeric(y) - as.numeric(h)/2, as.numeric(y) - as.numeric(h)/2), 
                 gp = gpar(col = "black", lwd = 0.6))
    }
  },
  rect_gp = gpar(col = "transparent", lwd = 0),
  heatmap_legend_param = list(
    direction = "horizontal",
    legend_width = config$width,
    at = c(0, 0.05, 0.10, 0.15, 0.20),
    labels = c("0", "0.05", "0.10", "0.15", "0.20"),
    legend_height = unit(3, "mm"),
    title_gp = gpar(fontsize = config$legend_fontsize, fontfamily = "sans"),
    labels_gp = gpar(fontsize = config$legend_fontsize, fontfamily = "sans"),
    grid_height = unit(3, "mm")
  )
)

png("Final_Three_Heatmaps.png", width = 1680, height = 630, res = 300)
ht_list <- draw(
  a_ht + b_ht + c_ht, 
  heatmap_legend_side = "bottom",
  gap = config$heatmap_gap,
  padding = unit(c(2, 2, 2, 2), "mm"),
  auto_adjust = TRUE
)
dev.off()

pdf("Final_Three_Heatmaps.pdf", width = 5.6, height = 2.1)
ht_list <- draw(
  a_ht + b_ht + c_ht, 
  heatmap_legend_side = "bottom",
  gap = config$heatmap_gap,
  padding = unit(c(2, 2, 2, 2), "mm"), 
  auto_adjust = TRUE
)
dev.off()

cat("Packages used:\n")
cat("- ComplexHeatmap version:", as.character(packageVersion("ComplexHeatmap")), "\n")
cat("- circlize version:", as.character(packageVersion("circlize")), "\n")
cat("- readr version:", as.character(packageVersion("readr")), "\n")
cat("- stringr version:", as.character(packageVersion("stringr")), "\n")
cat("- grid version:", as.character(packageVersion("grid")), "\n")