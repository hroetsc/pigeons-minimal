### SPLICING PREDICTOR ###
# description:  filter sequences within 5% FDR
# input:        inSPIRE unfiltered identifications
# output:       sequences/sites to remove from negative dataset
# author:       HPR

library(stringr)
library(data.table)
library(dtplyr)
library(dplyr)
library(parallel)
library(eulerr)
source("src/encoding/_invitro_mapping.R")
numCPU = detectCores()

unfiltered_dir = "/data/John/fraggerInVitro/20240211_predictor_unfiltered/"
threshold = 0.05

### INPUT ###
load("data/aSPIRE_240117.RData")

fs = list.files(unfiltered_dir, pattern = "_psms.csv", full.names = T, recursive = T)
allPSMs = lapply(fs, function(x) {
    y = fread(x)
    y = y %>% dplyr::mutate(substrateID = str_remove(basename(x), "_psms.csv")) %>% dplyr::collect() %>% as_tibble()
    return(y)
}) %>%
    rbindlist(use.names=TRUE)


### MAIN PART ###
# ----- filter "almost assigned" sequences -----
allPSMs$qValue %>% summary()
hist(allPSMs$qValue)

negfilter = allPSMs %>%
  as_tibble() %>%
  dplyr::filter(qValue <= threshold) %>%
  dplyr::select(substrateID, peptide, qValue) %>%
  dplyr::rename(pepSeq = peptide) %>%
  mutate(substrateID = ifelse(substrateID == "annexin", "AnnexinA1", substrateID)) %>%
  unique() %>%
  dplyr::filter(substrateID %in% unique(Kinetics$substrateID))

negfilter = Kinetics %>%
  as_tibble() %>%
  dplyr::distinct(substrateID, substrateSeq) %>%
  dplyr::right_join(negfilter) %>%
  dplyr::mutate(pepSeq_IL = gsub("I","L",pepSeq))

negfilter$substrateID %>% unique()

# ----- map to substrate sequence -----
MAP = lapply(unique(negfilter$substrateID), function(ID){
  print(ID)
  locate_peps_substrate(peps = unique(negfilter$pepSeq[negfilter$substrateID == ID]),
                        subSeq = unique(negfilter$substrateSeq[negfilter$substrateID == ID]),
                        numCPU = numCPU)
})

names(MAP) = unique(negfilter$substrateID)

finalfilter = MAP %>%
  plyr::ldply() %>%
  dplyr::rename(substrateID = .id) %>%
  dplyr::mutate(pepSeq_IL = as.character(pepSeq_IL)) %>%
  dplyr::right_join(negfilter)

# sanity check
# list(inSPIRE = unique(finalfilter$pepSeq_IL),
#      aSPIRE = unique(gsub("I","L",Kinetics$pepSeq))) %>%
#   euler(shape = "ellipse") %>%
#   plot(quantities = T)

# ----- extract P1-P1' combinations -----
finallist = finalfilter %>%
  # dplyr::filter(productType == "PSP") %>%
  dplyr::distinct(substrateID, substrateSeq, pepSeq, productType, spliceType, positions, qValue) %>%
  tidyr::separate_rows(positions, sep = ";") %>%
  dplyr::mutate(P1 = str_split_fixed(positions,"_",Inf)[,2],
                P1_ = str_split_fixed(positions,"_",Inf)[,3]) %>%
  unique()


### OUTPUT ###
fwrite(finallist, "data/inSPIRE_20240301_unfiltered.csv")

