## Path to UTUC GE matrix
GE_path <- file.path("Q:/BLACK LAB UTUC cohort Cedars-Sinai/Data/GE_matrices/",
                     "featureCounts.GRCh38.UTUC_Cedars-Sinai_HF-05-24-2023_with_intronic_reads.GeneListUQ.csv")

## Path to MIBC subtypes
subtypes_path <- file.path("Q:/BLACK LAB UTUC cohort Cedars-Sinai/Data/GE_matrices/Subtypes/",
                           "featureCounts.GRCh38.UTUC_Cedars-Sinai_HF-05-24-2023_with_intronic_reads.GeneListUQ__consensusMIBC_subtypes.csv")

## Gene set (e.g. Oncuria gene panel) as vector of genes
geneset <- c('ANG', 'APOE', 'SERPINA1', 'CA9', 'CXCL8', 'MMP9', 'MMP10', 'SERPINE1', 'SDC1', 'VEGFA')

## Save boxplot figure and if so, where to?
save_fig <- FALSE
save_dir <- 'Q:/BLACK LAB UTUC cohort Cedars-Sinai/Oncuria_genes/'
fig_name <- 'UTUC_Cedars-Sinai_log2_MMP9_expr_boxplot.GeneListUQ'

## Width and height of plot (units = inches)
plot_width <- 6
plot_height <- 4

## Resolution, in ppi, for png, jpeg, or tiff images
save_img_res <- 600


## ========================================================================== ##


## Load GE data and subtype classifications.
GE <- read.csv(GE_path, row.names = NULL, check.names = FALSE)
subtypes <- read.csv(subtypes_path, row.names = 1, check.names = FALSE)

## Subset GE dataframe to just the genes in the gene set and the samples that
## have non-NA subtype classification
samples <- rownames(subtypes)[!is.na(subtypes$consensusClass)]
GE_subset <- GE[GE$Gene %in% geneset, colnames(GE) %in% c("Gene", "Geneid", samples)]
rownames(GE_subset) <- GE_subset$Gene
GE_subset <- GE_subset[, samples]

## Log-transform GE data for detected genes
logExpr <- log2(GE_subset[rowSums(GE_subset) > 0, ] + 1.0)

## Calculate the mean log expression across genes per samples
meanLogExpr <- colMeans(logExpr)

## Get a vector of subtype classes
classes <- subtypes[names(meanLogExpr), "consensusClass"]


## =====  Plot mean log gene expression by subtype class  ===== ##

library(ggplot2)

# df <- data.frame(meanLogExpr = meanLogExpr, classes = classes)
df <- data.frame(meanLogExpr = meanLogExpr, 
                 VEGFA = t(logExpr)[, 'VEGFA'],
                 MMP9 = t(logExpr)[, 'MMP9'],
                 classes = classes)

if (save_fig) {
  dir.create(save_dir, recursive = TRUE)
  img_path <- file.path(save_dir, paste0(fig_name, '.png'))
  png(file = img_path, units = "in", res = save_img_res,
      width = plot_width, height = plot_height)
} else {
  dev.new(width = plot_width, height = plot_height, 
          unit="in", noRStudioGD = TRUE)
}

# Create the box-and-whisker plot with overlaid data points
# h <- ggplot(df, aes(x = classes, y = meanLogExpr)) +
h <- ggplot(df, aes(x = classes, y = MMP9)) +
  geom_boxplot(outlier.shape = NA) +  # Avoid plotting outliers separately
  geom_jitter(width = 0.2, size = 1.5, color = "blue") +  # Overlay data points
  theme_minimal() +  # Use a minimal theme for a clean look
  # labs(x = "Classes", y = "Mean Log Expression")  # Label the axes
  labs(x = "Classes", y = "Log2(MMP9 Expression + 1)")  # Label the axes
print(h)

if(save_fig){dev.off()}
