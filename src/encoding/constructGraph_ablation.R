### SPLICING PREDICTOR ###
# description:  construct cleavage/splicing graphs - ablation to estimate dataset size
# author:       HPR

# renv::init()
library(stringr)
library(dplyr)
library(data.table)
library(parallel)
theme_set(theme_classic())

source("src/_utils.R")
source("src/encoding/_graphFunctions.R")
source("src/encoding/_nodeEmbedding.R")


# ----- hyperparameters ------
rt_test = 0.15
rt_val = 0.15

sizes_total = c(5,10,20,25)
sizes_test = (sizes_total*rt_test) %>% ceiling()
sizes_val = (sizes_total*rt_val) %>% ceiling()


# ----- INPUT ------
load("data/aSPIRE.RData")
inSilicoSubstrates = Kinetics %>%
  distinct(substrateSeq, substrateID)
set.seed(87)
inSilicoSubstrates = inSilicoSubstrates[sample(1:nrow(inSilicoSubstrates), nrow(inSilicoSubstrates)),]

fs_graph = list.files("data/datasets/", pattern = "graph.csv", full.names = T, recursive = F)
GRAPH_all = lapply(fs_graph, function(f){
  print(f)
  fx = fread(f)
  # remove connections of PCP to PSP
  fx = fx %>%
    dplyr::filter(!(productType == "PSP" & str_detect(educt,"pcp_")))
  return(fx)
}) %>%
  rbindlist() %>%
  dplyr::filter(substrateID != "")

fs_emb = list.files("data/datasets/", pattern = "_embedding.csv", full.names = T, recursive = F)
EMBEDDING_all = lapply(fs_emb, function(f){
  print(f)
  return(fread(f))
}) %>%
  rbindlist() %>%
  dplyr::filter(substrateID != "")


# ----- construct differently sized subsets -----
for (k in 1:length(sizes_total)) {

    print(sizes_total[k])

    substrates = inSilicoSubstrates[1:sizes_total[k], ]
    sizes_train_cnt = sizes_total[k]-sizes_val[k]-sizes_test[k]

    train_subs = substrates$substrateID[1:sizes_train_cnt]
    val_subs = substrates$substrateID[(sizes_train_cnt+1):(sizes_train_cnt+sizes_val[k])]
    test_subs = substrates$substrateID[(sizes_train_cnt+sizes_val[k]+1):(sizes_train_cnt+sizes_val[k]+sizes_test[k])]

    SUB_masks = substrates %>%
        dplyr::mutate(substrateLength = nchar(substrateSeq)) %>%
        dplyr::select(substrateID, substrateLength) %>%
        dplyr::mutate(train_mask = ifelse(substrateID %in% train_subs, T, F),
                        val_mask = ifelse(substrateID %in% val_subs, T, F),
                        test_mask = ifelse(substrateID %in% test_subs, T, F))

    # create indices
    INDEX = EMBEDDING_all %>%
        dplyr::filter(substrateID %in% substrates$substrateID) %>%
        dplyr::distinct(substrateID, product, observed) %>%
        dplyr::left_join(SUB_masks) %>%
        dplyr::mutate(ID = paste0(product,"-",substrateID),
                        index = row_number()-1,
                        pcp_mask = fifelse(str_detect(product, "pcp_"), T, F),
                        psp_mask = fifelse(str_detect(product, "psp_"), T, F),
                        pseudo_mask = fifelse(!pcp_mask & !psp_mask, T, F),
                        pseudo1_mask = fifelse(str_detect(product, "p1_"), T, F),
                        pseudo2_mask = fifelse(str_detect(product, "p1'_"), T, F),
                        subnode_mask = fifelse(product == paste0("pcp_",substrateLength,"_",1), T, F),
                        target = fifelse(pseudo1_mask & observed, 1, 0),
                        target = fifelse(pseudo2_mask & observed, 2, target),
                        target = fifelse(pcp_mask & observed, 3, target),
                        target = fifelse(psp_mask & observed, 4, target))

    
    EMBEDDING_final = INDEX %>%
        dplyr::right_join(EMBEDDING_all %>%
                            dplyr::filter(substrateID %in% substrates$substrateID)) %>%
        dplyr::arrange(index)  # this is important!!!


    GRAPH_final = GRAPH_all %>%
      dplyr::filter(substrateID %in% substrates$substrateID) %>%
      dplyr::mutate(product_ID = paste0(product,"-",substrateID),
                    educt_ID = paste0(educt,"-",substrateID)) %>%
      dplyr::left_join(INDEX %>%
                          dplyr::rename(product_ID = ID,
                                        product_index = index) %>%
                          dplyr::select(product_ID, product_index)) %>%
      dplyr::left_join(INDEX %>%
                          dplyr::rename(educt_ID = ID,
                                        educt_index = index) %>%
                          dplyr::select(educt_ID, educt_index))


  fwrite(GRAPH_final, paste0("data/insilicodata/REAL_GRAPH_",sizes_total[k],".csv"))
  fwrite(EMBEDDING_final, paste0("data/insilicodata/REAL_EMBEDDING_",sizes_total[k],".csv"))
  fwrite(INDEX, paste0("data/insilicodata/REAL_INDICES_",sizes_total[k],".csv"))
}


