##  IMPORT PYTHON MODULES  ################################
import os
from pathlib import Path
import glob
import gc # Garbage collection
import numpy as np
import pandas as pd
import mygene
## Shortcut for getting gene info from ID
mg = mygene.MyGeneInfo()

############################################################
#####                                                  #####
#####     CONVERT FEATURECOUNTS OUTPUT MATRIX INTO     #####
#####      NORMALIZED GENE EXPRESSION (GE) MATRIX      #####
#####                                                  #####
############################################################

##  INPUT/OUTPUT DIRECTORIES/FILENAMES  ##################################
## Specify directory path and filenames of count matrix/matrices output
## by featureCounts
featureCounts_dir = ("/DIRECTORY/PATH/FOR/OUTPUT/OF/featureCounts/")
featureCounts_fnames = ["featureCounts*GRCh38*.txt"]

## Specify where (directory path) to save GE matrix
GE_outdir = ("/DIRECTORY/PATH/FOR/GENE_EXPRESSION_MATRICES/")

## Suffix to remove from sample (column) names in featureCounts matrix?
## E.g. If featureCounts was applied to BAM files output by STAR with default
## names, then the suffix "_Aligned.out.bam" should be specified here to
## remove it from sample names in the final gene-expression matrices.
bam_suffix = "_Aligned"

##  DOES DATA INCLUDE ERCC SPIKE-IN READS?  ##############################
includesERCC = True

##   NORMALIZATION OPTIONS   #############################################
##
## Which types of normalization should be performed? One gene expression
## matrix will be generated for each option included in a list here.
##
### These are the available options:
###
###   "TPM"
###
###   "FPKM" (not recommended for anything!)
###
###   "UQ" :            Normalize the upper quartile expression of genes that are
###                     expressed in at least one of the samples.
###
###   "GeneListUQ" :    Normalize to the upper quartile length-normalized expression
###                     of a list of specified genes. Gene list must be provided in a
###                     .txt file with one gene per row.
###
###   "GeneListTotal" : Normalize to the total (summed) expression of a list of
###                     specified genes. Gene list must be provided in a .txt file
###                     with one gene per row.
###
###   "DESeq" :         Normalize by gene lengths and then apply the "median of ratios"
###                     between-sample normalization implemented by DESeq and DESeq2.
###
###   "DESeq_iter":     Repeatedly apply "DESeq" normalization until the scale factors
###                     converge to within some tolerance, up to a maximum number of
###                     iterations.
###
###   "GeneListDEseq":  Same as "DESeq", but restricted to only use genes in a pre-
###                     specified list of genes.
###
###   "GeneListDEseq_iter":  Same as "DESeq_iter", but restricted to only use genes in
###                     a pre-specified list of genes.
###

seq_norm_methods = ["TPM", "GeneListUQ", "GeneListDEseq_iter"]

## Provide a path to a .txt file containing a list of genes if either "GeneListUQ"
## or "GeneListTotal" is included in the list of normalization methods.
GeneList_path = "/groups/zoubeidigrp/RNA-Seq/Stable_genes/StableGenes_Top500_carcinoma.txt"


################################################################################
################################################################################


## Define functions for normalizing RNA-seq feature counts

## Input arguments for all normalization functions:
##   GE_df : A Pandas dataframe generated from the output of featureCounts.
##           Rows correspond to features (genes) and columns to samples, but
##           GE_df should also contain columns for gene names and gene lengths
##           in bases. Sample columns should all be adjacent to each other and
##           should only appear after all non-sample columns (e.g. gene name,
##           gene length, etc). I.e. the last N columns of GE_df should correspond
##           to the N samples.
##   CountsStartCol : Index of the first sample column (remember that Python 
##                    starts indexing at 0, not 1)

def normalize_for_length(GE_df, CountsStartCol):
    ## Divide raw counts by gene length in kb
    lengths_kb = np.divide(np.reshape(np.matrix(GE_df.loc[:,'Length']), (-1,1)), 1.0e3)
    Y = GE_df.copy()
    Y.iloc[:,CountsStartCol:] = np.divide(Y.iloc[:,CountsStartCol:],lengths_kb)
    return Y

def normalize_total(GE_df, CountsStartCol):
    ## Divide by the sum of all counts (e.g. reads per kb after normalization 
    ## for gene length) and x 10^6. Applying normalize_for_length(...) followed
    ## by normalize_total(...) gives TPM values.
    ScaleFactors = np.divide(GE_df.iloc[:,CountsStartCol:].sum(axis=0), 1.0e6)
    Y = GE_df.copy()
    Y.iloc[:,CountsStartCol:] = np.divide(Y.iloc[:,CountsStartCol:], ScaleFactors)
    return Y, ScaleFactors

def normalize_UQ(GE_df, CountsStartCol):
    ## Normalize samples to have equal upper quartile expression after removal
    ## of any genes that have 0 expression in all samples (i.e. normalize UQ
    ## expression of genes that are expressed by at least one sample).
    RowSumExpr = GE_df.iloc[:,CountsStartCol:].sum(axis=1)
    UQ_Expr = GE_df.loc[RowSumExpr > 0].iloc[:,CountsStartCol:].quantile(q=0.75,axis=0)
    print("\nMedian UQ expression = " + str(np.median(UQ_Expr)) + ".\n")
    #ScaleFactors = np.divide(UQ_Expr, np.median(UQ_Expr))
    ScaleFactors = np.divide(UQ_Expr, 25)
    Y = GE_df.copy()
    Y.iloc[:,CountsStartCol:] = np.divide(Y.iloc[:,CountsStartCol:], ScaleFactors)
    return Y, ScaleFactors

# def normalize_GeneListUQ(GE_df, CountsStartCol,GeneList):
#     ## Multiply read/fragment counts by 100, then divide by the upper quartile of the
#     ## read/fragment counts for genes in GeneList (including genes having 0 counts)
#     idx1 = GE_df.loc[:,"Geneid"].isin(GeneList)
#     idx2 = GE_df.loc[:,"Geneid"].map(lambda x: x.split('.')[0]).isin(GeneList)
#     idx3 = GE_df.loc[:,"Gene"].isin(GeneList)
#     genes_idx = np.logical_or(idx1, np.logical_or(idx2, idx3))
#     genes_UQ = GE_df.loc[genes_idx].iloc[:,CountsStartCol:].quantile(q=0.75, axis=0)
#     print("\nMedian UQ expression for listed genes = " + str(np.median(genes_UQ)) + ".\n")
#     #ScaleFactors = np.divide(genes_UQ, np.median(genes_UQ))
#     ScaleFactors = np.divide(genes_UQ, 100)
#     Y = GE_df.copy()
#     Y.iloc[:,CountsStartCol:] = np.divide(Y.iloc[:,CountsStartCol:], ScaleFactors)
#     ## Insert a column to indicate which genes were in the list of genes for normalization
#     new_col_loc = int(np.add(Y.columns.get_loc("Geneid"),1))
#     Y.insert(new_col_loc, "In_GeneList", genes_idx)
#     return Y, ScaleFactors

# def normalize_GeneListTotal(GE_df, CountsStartCol,GeneList):
#     ## Divide by the sum of counts for genes in GeneList and multiply by 1e6 * (N / 2e4),
#     ## where N is the number of genes in GeneList.
#     idx1 = GE_df.loc[:,"Geneid"].isin(GeneList)
#     idx2 = GE_df.loc[:,"Geneid"].map(lambda x: x.split('.')[0]).isin(GeneList)
#     idx3 = GE_df.loc[:,"Gene"].isin(GeneList)
#     genes_idx = np.logical_or(idx1, np.logical_or(idx2, idx3))
#     genes_sum = GE_df.loc[genes_idx].iloc[:,CountsStartCol:].sum(axis=0)
#     ScaleFactors = np.divide(genes_sum, 50 * len(GeneList))
#     Y = GE_df.copy()
#     Y.iloc[:,CountsStartCol:] = np.divide(Y.iloc[:,CountsStartCol:], ScaleFactors)
#     ## Insert a column to indicate which genes were in the list of genes for normalization
#     new_col_loc = int(np.add(Y.columns.get_loc("Geneid"),1))
#     Y.insert(new_col_loc, "In_GeneList", genes_idx)
#     return Y, ScaleFactors

def normalize_GeneListUQ(GE_df, CountsStartCol, GeneList):
    ## Multiply read/fragment counts by 100, then divide by the upper quartile of the
    ## read/fragment counts for genes in GeneList (including genes having 0 counts)
    if "Geneid" in list(GE_df.columns):
        idx1 = np.logical_or(GE_df.loc[:,"Geneid"].isin(GeneList),
                             GE_df.loc[:,"Geneid"].map(lambda x: x.split('.')[0]).isin(GeneList))
    else:
        idx1 = np.zeros(GE_df.shape[0])
    if "Gene" in list(GE_df.columns):
        idx2 = GE_df.loc[:,"Gene"].isin(GeneList)
    else:
        idx2 = np.zeros(GE_df.shape[0])
    if GE_df.index.name=='Gene':
        idx3 = GE_df.index.isin(GeneList)
    elif GE_df.index.name=='Geneid':
        idx3 = np.logical_or(GE_df.index.isin(GeneList),
                             GE_df.index.map(lambda x: x.split('.')[0]).isin(GeneList))
    else:
        idx3 = np.zeros(GE_df.shape[0])
    genes_idx = np.logical_or(idx1, np.logical_or(idx2, idx3))
    genes_UQ = GE_df.loc[genes_idx].iloc[:,CountsStartCol:].quantile(q=0.75, axis=0)
    print("\nMedian UQ expression for listed genes = " + str(np.median(genes_UQ)) + ".\n")
    #ScaleFactors = np.divide(genes_UQ, np.median(genes_UQ))
    ScaleFactors = np.divide(genes_UQ, 100)
    Y = GE_df.copy()
    Y.iloc[:,CountsStartCol:] = np.divide(Y.iloc[:,CountsStartCol:], ScaleFactors)
    ## Insert a column to indicate which genes were in the list of genes for normalization
    new_col_loc = 0
    if "Geneid" in list(GE_df.columns):
        new_col_loc = max(new_col_loc, int(np.add(Y.columns.get_loc("Geneid"),1)))
    if "Gene" in list(GE_df.columns):
        new_col_loc = max(new_col_loc, int(np.add(Y.columns.get_loc("Gene"),1)))
    Y.insert(new_col_loc, "In_GeneList", genes_idx)
    return Y, ScaleFactors
    
def normalize_GeneListTotal(GE_df, CountsStartCol, GeneList):
    ## Divide by the sum of counts for genes in GeneList and multiply by 1e6 * (N / 2e4),
    ## where N is the number of genes in GeneList.
    if "Geneid" in list(GE_df.columns):
        idx1 = np.logical_or(GE_df.loc[:,"Geneid"].isin(GeneList),
                             GE_df.loc[:,"Geneid"].map(lambda x: x.split('.')[0]).isin(GeneList))
    else:
        idx1 = np.zeros(GE_df.shape[0])
    if "Gene" in list(GE_df.columns):
        idx2 = GE_df.loc[:,"Gene"].isin(GeneList)
    else:
        idx2 = np.zeros(GE_df.shape[0])
    if GE_df.index.name=='Gene':
        idx3 = GE_df.index.isin(GeneList)
    elif GE_df.index.name=='Geneid':
        idx3 = np.logical_or(GE_df.index.isin(GeneList),
                             GE_df.index.map(lambda x: x.split('.')[0]).isin(GeneList))
    else:
        idx3 = np.zeros(GE_df.shape[0])
    genes_idx = np.logical_or(idx1, np.logical_or(idx2, idx3))
    genes_sum = GE_df.loc[genes_idx].iloc[:,CountsStartCol:].sum(axis=0)
    ScaleFactors = np.divide(genes_sum, 50 * len(GeneList))
    Y = GE_df.copy()
    Y.iloc[:,CountsStartCol:] = np.divide(Y.iloc[:,CountsStartCol:], ScaleFactors)
    ## Insert a column to indicate which genes were in the list of genes for normalization
    new_col_loc = 0
    if "Geneid" in list(GE_df.columns):
        new_col_loc = max(new_col_loc, int(np.add(Y.columns.get_loc("Geneid"),1)))
    if "Gene" in list(GE_df.columns):
        new_col_loc = max(new_col_loc, int(np.add(Y.columns.get_loc("Gene"),1)))    
    Y.insert(new_col_loc, "In_GeneList", genes_idx)
    return Y, ScaleFactors

def normalize_DESeq(GE_df, CountsStartCol):
    ## Normalize feature counts in each sample using the "median of ratios"
    ## "size factor" implemented by DESeq.
    ### Compute geometric mean across samples for each feature that is expressed in > half
    ### of all samples (compute from nonzero values only):
    N = (GE_df.iloc[:, CountsStartCol:] > 0).sum(axis=1)
    idx = N > (GE_df.shape[1] - CountsStartCol) / 2.0
#     ### Calc. geo. mean as the nth root of the product of n terms:
#     N = np.reshape(np.matrix(N),(-1,1))
#     GeomeanExpr = GE_df.loc[idx].iloc[:, CountsStartCol:].replace(0,np.NaN).prod(axis=1, skipna=True, min_count=1)
#     GeomeanExpr = np.power(np.reshape(np.matrix(GeomeanExpr),(-1,1)), np.divide(1.0,N))
    ### Calc. geo. mean using logs to avoid overflow:
    GeomeanExpr = np.reshape(
        np.matrix(np.exp(np.log(GE_df.loc[idx].iloc[:, CountsStartCol:].replace(0, np.NaN)).mean(axis=1, skipna=True))), 
        (-1,1)
    )
    ### Size factor for sample j is the median of the ratios (counts of feature i
    ### / geometric mean of counts of feature i). Take medians only of nonzeros to
    ### avoid size factors of 0.
    SizeFactors = (np.divide(GE_df.loc[idx].iloc[:, CountsStartCol:], GeomeanExpr)).replace(0, np.NaN).median(axis=0, skipna=True)
    ### Divide all size factors by the geo. mean of all size factors.
    geomean_sf = np.exp(np.mean(np.log(SizeFactors)))
    SizeFactors = np.divide(SizeFactors, geomean_sf)
    ### Scale counts by size factors
    Y = GE_df.copy()
    Y.iloc[:, CountsStartCol:] = np.divide(Y.iloc[:, CountsStartCol:], SizeFactors)
    return Y, SizeFactors

def normalize_DESeq_iter(GE_df, CountsStartCol, maxIters=100, tolerance=1.0e-5):
    ## Iterate DESeq normalization until the SizeFactors converge such that their
    ## range is within tolerance fraction of the mean size factor (up to maxIters
    ## iterations).
    Y = GE_df.copy()
    for it in range(maxIters):
        Y, sf = normalize_DESeq(Y, CountsStartCol)
        if it==0:
            ScaleFactors = sf
        else:
            ScaleFactors *= sf
        if (max(sf) - min(sf)) / np.mean(sf) < tolerance:
            break
    return Y, ScaleFactors, it+1

def normalize_GeneListDESeq(GE_df, CountsStartCol, GeneList, insert_GeneList_col=True):
    ## Apply the DESeq normalization method (as above) using only genes in GeneList.
    ### Get the row indices of genes in GeneList:
    if "Geneid" in list(GE_df.columns):
        idx1 = np.logical_or(GE_df.loc[:,"Geneid"].isin(GeneList),
                             GE_df.loc[:,"Geneid"].map(lambda x: x.split('.')[0]).isin(GeneList))
    else:
        idx1 = np.zeros(GE_df.shape[0])
    if "Gene" in list(GE_df.columns):
        idx2 = GE_df.loc[:,"Gene"].isin(GeneList)
    else:
        idx2 = np.zeros(GE_df.shape[0])
    if GE_df.index.name=='Gene':
        idx3 = GE_df.index.isin(GeneList)
    elif GE_df.index.name=='Geneid':
        idx3 = np.logical_or(GE_df.index.isin(GeneList),
                             GE_df.index.map(lambda x: x.split('.')[0]).isin(GeneList))
    else:
        idx3 = np.zeros(GE_df.shape[0])
    genes_idx = np.logical_or(idx1, np.logical_or(idx2, idx3))
    ### Get the row indices of genes expressed in > half of the samples:
    N = (GE_df.iloc[:, CountsStartCol:] > 0).sum(axis=1)
    exprs_idx = N > (GE_df.shape[1] - CountsStartCol) / 2.0
    ### Indices of genes in GeneList that are also expressed in > half of the samples:
    idx = np.logical_and(genes_idx, exprs_idx)
#     ### Calc. geo. mean as the nth root of the product of n terms:
#     N = np.reshape(np.matrix(N),(-1,1))
#     GeomeanExpr = GE_df.loc[idx].iloc[:, CountsStartCol:].replace(0,np.NaN).prod(axis=1, skipna=True, min_count=1)
#     GeomeanExpr = np.power(np.reshape(np.matrix(GeomeanExpr),(-1,1)), np.divide(1.0,N))
    ### Calc. geo. mean using logs to avoid overflow:
    GeomeanExpr = np.reshape(
        np.matrix(np.exp(np.log(GE_df.loc[idx].iloc[:, CountsStartCol:].replace(0, np.NaN)).mean(axis=1, skipna=True))), 
        (-1,1)
    )
    ### Size factor for sample j is the median of the ratios (counts of feature i
    ### / geometric mean of counts of feature i). Take medians only of nonzeros to
    ### avoid size factors of 0.
    SizeFactors = (np.divide(GE_df.loc[idx].iloc[:, CountsStartCol:], GeomeanExpr)).replace(0, np.NaN).median(axis=0, skipna=True)
    ### Divide all size factors by the geo. mean of all size factors.
    geomean_sf = np.exp(np.mean(np.log(SizeFactors)))
    SizeFactors = np.divide(SizeFactors, geomean_sf)
    ### Scale counts by size factors
    Y = GE_df.copy()
    Y.iloc[:, CountsStartCol:] = np.divide(Y.iloc[:, CountsStartCol:], SizeFactors)
    if insert_GeneList_col:
        ## Insert a column to indicate which genes were in the list of genes for normalization
        new_col_loc = 0
        if "Geneid" in list(GE_df.columns):
            new_col_loc = max(new_col_loc, int(np.add(Y.columns.get_loc("Geneid"),1)))
        if "Gene" in list(GE_df.columns):
            new_col_loc = max(new_col_loc, int(np.add(Y.columns.get_loc("Gene"),1)))
        Y.insert(new_col_loc, "In_GeneList", genes_idx)
    return Y, SizeFactors

def normalize_GeneListDESeq_iter(GE_df, CountsStartCol, GeneList, maxIters=100, tolerance=1.0e-5):
    ## Iterate GeneListDESeq normalization until the SizeFactors converge such that
    ## their range is within tolerance fraction of the mean size factor (up to maxIters
    ## iterations).
    Y = GE_df.copy()
    for it in range(maxIters):
        if it==0:
            Y, sf = normalize_GeneListDESeq(Y, CountsStartCol, GeneList, True)
            CountsStartCol += 1
            ScaleFactors = sf
        else:
            Y, sf = normalize_GeneListDESeq(Y, CountsStartCol, GeneList, False)
            ScaleFactors *= sf
        if (max(sf) - min(sf)) / np.mean(sf) < tolerance:
            break
    return Y, ScaleFactors, it+1


################################################################################
################################################################################


## Ensure correct interpretation of path by OS (e.g. on Windows)
featureCounts_dir = Path(featureCounts_dir)

## Check whether GE_outdir exists and throw error if not
if not os.path.isdir(Path(GE_outdir)):
    raise Exception("GE_outidr does not exist. Please specify a valid existing path.")

## Construct a list of paths to featureCounts files.
featureCounts_paths = []
for fname_pattern in featureCounts_fnames:
    fpath_pattern = os.path.join(featureCounts_dir, fname_pattern)
    featureCounts_paths.extend(list(glob.glob(fpath_pattern)))

for fcounts_path in featureCounts_paths:
    fcounts_file = os.path.basename(fcounts_path)
    print("\nWorking on file " + fcounts_file + "...\n")

    ## Load featureCounts data
    GE_df = pd.read_csv(fcounts_path, header=1, sep='\t')

    print(GE_df.head())  #########
    print("\n")


    ## Remove unneeded columns (Chr, Start, End, Strand)
    GE_df = GE_df.drop(columns=['Chr', 'Start', 'End', 'Strand'], errors='ignore')

    ## Get the column index of the first sample column
    CountsStartCol = np.add(GE_df.columns.get_loc("Length"), 1)

    ## Rename sample columns by removing path and suffix from sample names
    for col in range(CountsStartCol,GE_df.shape[1]):
        samplename = GE_df.columns[col]
        samplename = os.path.basename(samplename).split(bam_suffix)[0]
        GE_df.columns.values[col] = samplename

    print("All features ==> (# rows, # columns) = " + str(GE_df.shape))

#    ## Sum read counts for each gene across all samples
#    GE_df['Sum'] = GE_df.iloc[:, CountsStartCol:].sum(axis=1)
#
#    ## Filter out genes with 0 mapped reads
#    GE_df = GE_df.loc[GE_df.loc[:,'Sum'] > 0, :]
#
#    ## We don't need the 'Sum' column anymore
#    GE_df = GE_df.drop(columns=['Sum'], errors='ignore')
#
#    print("Expressed features only ==> (# rows, # columns) = " + str(GE_df.shape) + "\n")
#    # print(GE_df.head().to_string())  #########
#    # print("\n")

    ## If data includes ERCC spike-in reads, separate ERCC reads.
    if includesERCC:
        ## Find rows for ERCC sequences
        ERCC_rows = GE_df["Geneid"].str.contains("ERCC-").values
        ERCC_df = GE_df.iloc[ERCC_rows, :]
        GE_df = GE_df.iloc[~ERCC_rows, :]

        print(GE_df.head())    #########
        print("\n\n")
        print(ERCC_df.head())  #########
        print("\n\n")

    ## Remove version numbers from Ensembl gene IDs
    ## (MyGene doesn't recognize the gene IDs when version numbers are included)
    # GE_df['Geneid'] = GE_df['Geneid'].map(lambda x: x.split('.')[0])
    GE_df["Geneid_noVersion"] = GE_df['Geneid'].map(lambda x: x.split('.')[0])
    
    ## Convert the gene IDs to gene symbols using MyGene
    # genesym = mg.getgenes(GE_df.loc[:,"Geneid"], 'symbol')
    genesym = mg.getgenes(GE_df["Geneid_noVersion"], 'symbol')
    
    ## Identify gene IDs that did not have a unique symbol
    uniq_queries = []
    dupl_queries = []
    for idx, geneidCall in enumerate(genesym):
        geneidQuery = geneidCall['query']
        if geneidQuery not in uniq_queries:
            ## Add the query to the list of unique queries
            uniq_queries.append(geneidQuery)
        else:
            ## Add the query to the list of duplicated queries
            dupl_queries.append(geneidQuery)

    ## Insert new column into dataframe for gene symbols.
    ### For gene IDs that did not map to a unique symbol, just copy the
    ### Ensembl gene ID (without version number) in place of the gene symbol.
    GE_df.insert(0, "Gene", GE_df.loc[:,"Geneid_noVersion"])

    ## Remove gene IDs that do not have a unique symbol
    for q in dupl_queries:
        genesym = list(filter(lambda i: i['query'] != q, genesym))

    ## Create a dictionary that matches gene IDs to gene symbols
    ## for gene IDs that map to a single symbol
    geneidDict = { genesym[i].get('query') : genesym[i].get('symbol',genesym[i].get('query')) for i in range(len(genesym)) }

    ## Insert gene symbols for gene IDs that mapped to a unique symbol
    GE_df["Gene"] = GE_df["Geneid_noVersion"].map(geneidDict).fillna(GE_df["Gene"])

    ## Drop the "Geneid_noVersion" column now that it's not needed
    GE_df = GE_df.drop(columns=["Geneid_noVersion"], errors='ignore')

    print(GE_df.head())  #########
    print("\n\n")
    
    ## If data contains both human and mouse genes, separate them. 
    human_rows = GE_df["Geneid"].str.startswith("ENSG").values
    mouse_rows = GE_df["Geneid"].str.startswith("ENSMUSG").values
    hs_and_mm = any(human_rows) and any(mouse_rows)
    if hs_and_mm:
        ## Murine genes
        mmGE_df = GE_df.copy().iloc[mouse_rows, :]
        mmGE_CountsStartCol = np.add(mmGE_df.columns.get_loc("Length"), 1)
        ## Sum read counts for each murine gene across all samples
        mmGE_df['Sum'] = mmGE_df.iloc[:, mmGE_CountsStartCol:].sum(axis=1)
        ## Filter out murine genes with 0 mapped reads
        mmGE_df = mmGE_df.loc[mmGE_df.loc[:,'Sum'] > 0, :]
        ## We don't need the 'Sum' column anymore
        mmGE_df = mmGE_df.drop(columns=['Sum'], errors='ignore')
        ## Human genes
        GE_df = GE_df.iloc[human_rows, :]
        print("Human:")
        print(GE_df.head())
        print("\nMurine:")
        print(mmGE_df.head())
        print("\n\n")
        
    ## Save the raw counts matrix prior to normalization
    df_counts = GE_df.copy()
    counts_CSVname = fcounts_file.split('.txt')[0] + '.RawCounts.csv'
    counts_CSVpath = Path(os.path.join(GE_outdir, counts_CSVname))
    print("Saving raw counts dataframe to " + str(counts_CSVpath) + "...\n")
    df_counts.to_csv(path_or_buf=counts_CSVpath, index=False)
    print("Done saving.\n")
    if hs_and_mm:
        mmdf_counts = mmGE_df.copy()
        mmcounts_CSVname = fcounts_file.split('.txt')[0] + '.RawCounts.Murine.csv'
        mmcounts_CSVpath = Path(os.path.join(GE_outdir, mmcounts_CSVname))
        print("Saving murine raw counts dataframe to " + str(mmcounts_CSVpath) + "...\n")
        mmdf_counts.to_csv(path_or_buf=mmcounts_CSVpath, index=False)
        print("Done saving.\n")

    ## Save ERCC_df to CSV file
    if includesERCC:
        df_ERCC_counts = ERCC_df.copy()
        ERCC_counts_CSVname = fcounts_file.split('.txt')[0] + '.RawCounts.ERCC.csv'
        ERCC_counts_CSVpath = Path(os.path.join(GE_outdir, ERCC_counts_CSVname))
        print("Saving ERCC counts dataframe to " + str(ERCC_counts_CSVpath) + "...\n")
        df_ERCC_counts.to_csv(path_or_buf=ERCC_counts_CSVpath, index=False)
        print("Done saving.\n")


    ## Perform each requested normalization and save the normalized
    ## gene expression matrix.

    GE_CountsStartCol = np.add(df_counts.columns.get_loc("Length"), 1)
    if includesERCC:
        ERCC_CountsStartCol = np.add(df_ERCC_counts.columns.get_loc("Length"), 1)

    for M in seq_norm_methods:

        del GE_df
        GE_df = df_counts.copy()
        if hs_and_mm:
            del mmGE_df
            mmGE_df = mmdf_counts.copy()
        if includesERCC:
            del ERCC_df
            ERCC_df = df_ERCC_counts.copy()

        ## TPM
        if M.lower()=="tpm":
            GE_df = normalize_for_length(GE_df, GE_CountsStartCol)
            GE_df, ScaleFactors = normalize_total(GE_df, GE_CountsStartCol)
            GE_CSVname = fcounts_file.split('.txt')[0] + '.TPM.csv'
            if includesERCC:
                ERCC_df = normalize_for_length(ERCC_df, ERCC_CountsStartCol)
                ERCC_df.iloc[:, ERCC_CountsStartCol:] = np.divide(ERCC_df.iloc[:, ERCC_CountsStartCol:], ScaleFactors)
                ERCC_CSVname = fcounts_file.split('.txt')[0] + '.TPM.ERCC.csv'
            if hs_and_mm:
                mmGE_df = normalize_for_length(mmGE_df, GE_CountsStartCol)
                mmGE_df.iloc[:, GE_CountsStartCol:] = np.divide(mmGE_df.iloc[:, GE_CountsStartCol:], ScaleFactors)
                mmGE_CSVname = fcounts_file.split('.txt')[0] + '.TPM.Murine.csv'

        ## UQ
        elif M.lower()=="uq":
            GE_df = normalize_for_length(GE_df, GE_CountsStartCol)
            GE_df, ScaleFactors = normalize_UQ(GE_df, GE_CountsStartCol)
            GE_CSVname = fcounts_file.split('.txt')[0] + '.UQ.csv'
            if includesERCC:
                ERCC_df = normalize_for_length(ERCC_df, ERCC_CountsStartCol)
                ERCC_df.iloc[: ,ERCC_CountsStartCol:] = np.divide(ERCC_df.iloc[:, ERCC_CountsStartCol:], ScaleFactors)
                ERCC_CSVname = fcounts_file.split('.txt')[0] + '.UQ.ERCC.csv'
            if hs_and_mm:
                mmGE_df = normalize_for_length(mmGE_df, GE_CountsStartCol)
                mmGE_df.iloc[:, GE_CountsStartCol:] = np.divide(mmGE_df.iloc[:, GE_CountsStartCol:], ScaleFactors)
                mmGE_CSVname = fcounts_file.split('.txt')[0] + '.UQ.Murine.csv'

        ## GeneListUQ
        elif M.lower()=="genelistuq":
            ## Read genes from text file into list.
            GeneList = []
            with open(GeneList_path) as txt:
                for line in txt:
                    gene = line.split('\n')[0]
                    GeneList.append(gene)
            ## Normalize by gene length first and then by UQ of listed genes
            GE_df = normalize_for_length(GE_df, GE_CountsStartCol)
            GE_df, ScaleFactors = normalize_GeneListUQ(GE_df, GE_CountsStartCol, GeneList)
            GE_CSVname = fcounts_file.split('.txt')[0] + '.GeneListUQ.csv'
            if includesERCC:
                ERCC_df = normalize_for_length(ERCC_df, ERCC_CountsStartCol)
                ERCC_df.iloc[:, ERCC_CountsStartCol:] = np.divide(ERCC_df.iloc[:, ERCC_CountsStartCol:], ScaleFactors)
                ERCC_CSVname = fcounts_file.split('.txt')[0] + '.GeneListUQ.ERCC.csv'
            if hs_and_mm:
                mmGE_df = normalize_for_length(mmGE_df, GE_CountsStartCol)
                mmGE_df.iloc[:, GE_CountsStartCol:] = np.divide(mmGE_df.iloc[:, GE_CountsStartCol:], ScaleFactors)
                mmGE_CSVname = fcounts_file.split('.txt')[0] + '.GeneListUQ.Murine.csv'

        ## GeneListTotal
        elif M.lower()=="genelisttotal":
            ## Read genes from text file into list.
            GeneList = []
            with open(GeneList_path) as txt:
                for line in txt:
                    gene = line.split('\n')[0]
                    GeneList.append(gene)
            ## Normalize by gene length first and then by total of listed genes
            GE_df = normalize_for_length(GE_df, GE_CountsStartCol)
            GE_df, ScaleFactors = normalize_GeneListTotal(GE_df, GE_CountsStartCol, GeneList)
            GE_CSVname = fcounts_file.split('.txt')[0] + '.GeneListTotal.csv'
            if includesERCC:
                ERCC_df = normalize_for_length(ERCC_df, ERCC_CountsStartCol)
                ERCC_df.iloc[:, ERCC_CountsStartCol:] = np.divide(ERCC_df.iloc[:, ERCC_CountsStartCol:], ScaleFactors)
                ERCC_CSVname = fcounts_file.split('.txt')[0] + '.GeneListTotal.ERCC.csv'
            if hs_and_mm:
                mmGE_df = normalize_for_length(mmGE_df, GE_CountsStartCol)
                mmGE_df.iloc[:, GE_CountsStartCol:] = np.divide(mmGE_df.iloc[:, GE_CountsStartCol:], ScaleFactors)
                mmGE_CSVname = fcounts_file.split('.txt')[0] + '.GeneListTotal.Murine.csv'

        ## FPKM
        elif M.lower()=="fpkm":
            GE_df, ScaleFactors = normalize_total(GE_df, GE_CountsStartCol)
            GE_df = normalize_for_length(GE_df, GE_CountsStartCol)
            GE_CSVname = fcounts_file.split('.txt')[0] + '.FPKM.csv'
            if includesERCC:
                ERCC_df.iloc[:, ERCC_CountsStartCol:] = np.divide(ERCC_df.iloc[:, ERCC_CountsStartCol:], ScaleFactors)
                ERCC_df = normalize_for_length(ERCC_df, ERCC_CountsStartCol)
                ERCC_CSVname = fcounts_file.split('.txt')[0] + '.FPKM.ERCC.csv'
            if hs_and_mm:
                mmGE_df.iloc[:, GE_CountsStartCol:] = np.divide(mmGE_df.iloc[:, GE_CountsStartCol:], ScaleFactors)
                mmGE_df = normalize_for_length(mmGE_df, GE_CountsStartCol)
                mmGE_CSVname = fcounts_file.split('.txt')[0] + '.FPKM.Murine.csv'
        
        ## DESeq with normalization for gene length
        elif M.lower()=="deseq" or M.lower()=="deseq2":
            GE_df = normalize_for_length(GE_df, GE_CountsStartCol)
            GE_df, ScaleFactors = normalize_DESeq(GE_df, GE_CountsStartCol)
            GE_CSVname = fcounts_file.split('.txt')[0] + '.DESeq.csv'
            if includesERCC:
                ERCC_df = normalize_for_length(ERCC_df, ERCC_CountsStartCol)
                ERCC_df.iloc[:, ERCC_CountsStartCol:] = np.divide(ERCC_df.iloc[:, ERCC_CountsStartCol:], ScaleFactors)
                ERCC_CSVname = fcounts_file.split('.txt')[0] + '.DESeq.ERCC.csv'
            if hs_and_mm:
                mmGE_df = normalize_for_length(mmGE_df, GE_CountsStartCol)
                mmGE_df.iloc[:, GE_CountsStartCol:] = np.divide(mmGE_df.iloc[:, GE_CountsStartCol:], ScaleFactors)
                mmGE_CSVname = fcounts_file.split('.txt')[0] + '.DESeq.Murine.csv'
        
        ## DESeq iterated until convergence of Size Factors, with additional normalization for gene length
        elif M.lower()=="deseq_iter" or M.lower()=="deseq2_iter":
            GE_df = normalize_for_length(GE_df, GE_CountsStartCol)
            GE_df, ScaleFactors, it = normalize_DESeq_iter(GE_df, GE_CountsStartCol)
            GE_CSVname = fcounts_file.split('.txt')[0] + '.DESeq_' + str(it) + 'iter.csv'
            if includesERCC:
                ERCC_df = normalize_for_length(ERCC_df, ERCC_CountsStartCol)
                ERCC_df.iloc[:, ERCC_CountsStartCol:] = np.divide(ERCC_df.iloc[:, ERCC_CountsStartCol:], ScaleFactors)
                ERCC_CSVname = fcounts_file.split('.txt')[0] + '.DESeq_' + str(it) + 'iter.ERCC.csv'
            if hs_and_mm:
                mmGE_df = normalize_for_length(mmGE_df, GE_CountsStartCol)
                mmGE_df.iloc[:, GE_CountsStartCol:] = np.divide(mmGE_df.iloc[:, GE_CountsStartCol:], ScaleFactors)
                mmGE_CSVname = fcounts_file.split('.txt')[0] + '.DESeq_' + str(it) + 'iter.Murine.csv'
        
        ## GeneListDESeq with normalization for gene length
        elif M.lower()=="genelistdeseq" or M.lower()=="genelistdeseq2":
            ## Read genes from text file into list.
            GeneList = []
            with open(GeneList_path) as txt:
                for line in txt:
                    gene = line.split('\n')[0]
                    GeneList.append(gene)
            GE_df = normalize_for_length(GE_df, GE_CountsStartCol)
            GE_df, ScaleFactors = normalize_GeneListDESeq(GE_df, GE_CountsStartCol, GeneList)
            GE_CSVname = fcounts_file.split('.txt')[0] + '.GeneListDESeq.csv'
            if includesERCC:
                ERCC_df = normalize_for_length(ERCC_df, ERCC_CountsStartCol)
                ERCC_df.iloc[:, ERCC_CountsStartCol:] = np.divide(ERCC_df.iloc[:, ERCC_CountsStartCol:], ScaleFactors)
                ERCC_CSVname = fcounts_file.split('.txt')[0] + '.GeneListDESeq.ERCC.csv'
            if hs_and_mm:
                mmGE_df = normalize_for_length(mmGE_df, GE_CountsStartCol)
                mmGE_df.iloc[:, GE_CountsStartCol:] = np.divide(mmGE_df.iloc[:, GE_CountsStartCol:], ScaleFactors)
                mmGE_CSVname = fcounts_file.split('.txt')[0] + '.GeneListDESeq.Murine.csv'
        
        ## GeneListDESeq iterated until convergence of Size Factors, also with normalization for gene length
        elif M.lower()=="genelistdeseq_iter" or M.lower()=="genelistdeseq2_iter":
            ## Read genes from text file into list.
            GeneList = []
            with open(GeneList_path) as txt:
                for line in txt:
                    gene = line.split('\n')[0]
                    GeneList.append(gene)
            GE_df = normalize_for_length(GE_df, GE_CountsStartCol)
            GE_df, ScaleFactors, it = normalize_GeneListDESeq_iter(GE_df, GE_CountsStartCol, GeneList)
            GE_CSVname = fcounts_file.split('.txt')[0] + '.GeneListDESeq_' + str(it) + 'iter.csv'
            if includesERCC:
                ERCC_df = normalize_for_length(ERCC_df, ERCC_CountsStartCol)
                ERCC_df.iloc[:, ERCC_CountsStartCol:] = np.divide(ERCC_df.iloc[:, ERCC_CountsStartCol:], ScaleFactors)
                ERCC_CSVname = fcounts_file.split('.txt')[0] + '.GeneListDESeq_' + str(it) + 'iter.ERCC.csv'
            if hs_and_mm:
                mmGE_df = normalize_for_length(mmGE_df, GE_CountsStartCol)
                mmGE_df.iloc[:, GE_CountsStartCol:] = np.divide(mmGE_df.iloc[:, GE_CountsStartCol:], ScaleFactors)
                mmGE_CSVname = fcounts_file.split('.txt')[0] + '.GeneListDESeq_' + str(it) + 'iter.Murine.csv'

        print("\nPreview of gene expression matrix after normalization by " + M + " method:\n")
        print(GE_df.head())
        if hs_and_mm:
            print("\nPreview of murine gene expression matrix after normalization of human gene " + 
                  "expression values by " + M + " method:\n")
            print(mmGE_df.head())
        if includesERCC:
            print("\nPreview of ERCC spike-in expression matrix after normalization by " + M + " method:\n")
            print(ERCC_df.head())

        GE_CSVpath = Path(os.path.join(GE_outdir, GE_CSVname))
        GE_df.to_csv(path_or_buf=GE_CSVpath, index=False)
        if includesERCC:
            ERCC_CSVpath = Path(os.path.join(GE_outdir, ERCC_CSVname))
            ERCC_df.to_csv(path_or_buf=ERCC_CSVpath, index=False)
        if hs_and_mm:
            mmGE_CSVpath = Path(os.path.join(GE_outdir, mmGE_CSVname))
            mmGE_df.to_csv(path_or_buf=mmGE_CSVpath, index=False)


    ## Delete data frames etc.
    del GE_df
    del df_counts
    del human_rows
    del mouse_rows
    if hs_and_mm:
        del mmGE_df
        del mmdf_counts
    if includesERCC:
        del ERCC_df
        del df_ERCC_counts
        del ERCC_rows

    gc.collect() ## Garbage collection.


#############
##   END   ##
#############
