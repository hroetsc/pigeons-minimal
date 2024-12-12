### SPLICING PREDICTOR ###
# description:  construct cleavage/splicing graphs
# author:       HPR

# renv::init()
library(stringr)
library(dplyr)
library(data.table)
library(parallel)
library(readxl)

source("src/_utils.R")
source("src/encoding/_graphFunctions_reactionNetwork.R")
source("src/encoding/_nodeEmbedding_reactionNetwork.R")

# as long as pep_size <= Nmin PCP encodings will only contain valid sequences
pep_size = 8
intv_size = 7
multimappers = TRUE

numCPU = 32
Nmin = 7
Nmax = 30
dir.create("data/datasets/",showWarnings = F)
sink("results/constructGraph_log.txt", split = T)

# ----- INPUT -----
load("data/aSPIRE_240117.RData")
# neg_data = fread("data/inSPIRE_negativeSequences.csv")
overview = readxl::read_excel("PCPSpredictor_Datasets_240205.xlsx") # or 240210 for reduced dataset
# NOTE: Atchley (sheet 2)/ Kidera (sheet 3) factors
AAencoding = readxl::read_excel("propensity_encoding.xlsx", sheet=3)
# AAencoding = read.csv("data/AAencoding.csv", stringsAsFactors = F)
AAencoding[,c(2:ncol(AAencoding))] <- apply(AAencoding[,c(2:ncol(AAencoding))],2,function(x){
  as.numeric(gsub("−","-",x))
})

# ----- specify features -----
AA = c("P","G","C","M","A","V","I","L","F","Y","W","H","R","K","D","E","N","Q","S","T","X") %>% sort()

SR1pos = c("P16"=-15,"P15"=-14,"P14"=-13,"P13"=-12,"P12"=-11,"P11"=-10,"P10"=-9,"P9"=-8,
           "P8"=-7,"P7"=-6,"P6"=-5,"P5"=-4,"P4"=-3, "P3"=-2, "P2"=-1, "P1"=0,
           "P-1"=1, "P-2"=2, "P-3"=3, "P-4"=4, "P-5"=5, "P-6"=6, "P-7"=7,"P-8"=8,
           "P-9"=9,"P-10"=10,"P-11"=11,"P-12"=12,"P-13"=13,"P-14"=14,"P-15"=15,"P-16"=16)

SR2pos = c("P-16_"=-16,"P-15_"=-15,"P-14_"=-14,"P-13_"=-13,"P-12_"=-12,"P-11_"=-11,"P-10_"=-10,"P-9_"=-9,
           "P-8_"=-8,"P-7_"=-7,"P-6_"=-6,"P-5_"=-5,"P-4_"=-4, "P-3_"=-3, "P-2_"=-2, "P-1_"=-1,
           "P1_"=0, "P2_"=1, "P3_"=2, "P4_"=3, "P5_"=4, "P6_"=5, "P7_"=6,"P8_"=7,
           "P9_"=8,"P10_"=9,"P11_"=10,"P12_"=11,"P13_"=12,"P14_"=13,"P15_"=14,"P16_"=15)

# introduce size of encoding
SR1pos = SR1pos[names(SR1pos) %in% c(paste0("P-",1:intv_size),paste0("P",1:pep_size))]
SR2pos = SR2pos[names(SR2pos) %in% c(paste0("P-",1:intv_size,"_"),paste0("P",1:pep_size,"_"))]
SRnames = c(names(SR1pos), names(SR2pos))


# ----- get graphs -----
substrates = Kinetics %>%
  distinct(substrateSeq, substrateID) #%>%
  # FIXME this does not include all substrates!
  # dplyr::filter(!substrateID %in% c("MeV-NP","KRASG12C","OPN-C"))  # !!!!! filter out substrates with few products !!!!!

# get masks
train_subs = overview$substrateID[overview$dataset == "training"] %>% na.omit()
test_subs = overview$substrateID[overview$dataset == "testing"] %>% na.omit()
val_subs = overview$substrateID[overview$dataset == "validation"] %>% na.omit()

SUB_masks = substrates %>%
  dplyr::mutate(substrateLength = nchar(substrateSeq)) %>%
  dplyr::select(substrateID, substrateLength) %>%
  dplyr::mutate(train_mask = ifelse(substrateID %in% train_subs, T, F),
                val_mask = ifelse(substrateID %in% val_subs, T, F),
                test_mask = ifelse(substrateID %in% test_subs, T, F))

# FIXME
system("rm -rf data/datasets/*.csv")
missing = substrates$substrateID

# fs = list.files("data/datasets/","_embedding.csv", full.names = T)
# existing = str_split_fixed(basename(fs),"_",Inf)[,1]
# missing = substrates$substrateID[!substrates$substrateID %in% existing]

for (substrate in missing) {

  print("---------------")
  print(substrate)
  print("---------------")
  
  DB = Kinetics %>%
    dplyr::filter(substrateID == substrate) %>%
    disentangleMultimappers.Type() %>%
    dplyr::distinct(substrateID, substrateSeq, pepSeq, productType, spliceType, positions) %>%
    generateILvariants()  # !!!!!!!!!!!!!
  
  # graph
  X = constructGraphNetwork(DB, numCPU, Nmin, Nmax, AA)
  GRAPH = X$ALL %>%
    as_tibble()
  allpos = X$allpos

  # node embedding
  # NOTE: PSP encoding entails AA at P2,P1,P-1',P1'
  # NOTE: remove AA combinations that do not occur in substrate
  NODE_EMBEDDING = nodeEmbedding(DB, SR1pos, SR2pos, allpos, GRAPH, AAencoding, numCPU)
  # combine with labels
  EMBEDDING = GRAPH %>% dplyr::distinct(substrateID, product, product_detected) %>%
    right_join(NODE_EMBEDDING) %>%
    dplyr::rename(observed = product_detected) %>%
    dplyr::filter(!is.na(substrateID)) %>%
    as_tibble()
  
  # save
  fwrite(GRAPH, file = paste0("data/datasets/",substrate,"_graph.csv"))
  fwrite(EMBEDDING, file = paste0("data/datasets/",substrate,"_embedding.csv"))

  gc()

}

# mclapply(missing, function(substrate) {  
# }, mc.cores = 4, mc.preschedule = T, mc.cleanup = T)



# ----- merge datasets -----
fs_graph = list.files("data/datasets/", pattern = "graph.csv", full.names = T, recursive = F)
GRAPH_all = lapply(fs_graph, function(f){
  print(f)
  fx = fread(f) %>% as_tibble()
  return(fx)
}) %>%
  rbindlist() %>%
  as_tibble() %>%
  dplyr::distinct(.keep_all = T) %>%
  dplyr::filter(substrateID != "")

fs_emb = list.files("data/datasets/", pattern = "_embedding.csv", full.names = T, recursive = F)
EMBEDDING_all = lapply(fs_emb, function(f){
  print(f)
  fx = fread(f) %>% as_tibble()
  return(fx)
}) %>%
  rbindlist() %>%
  as_tibble() %>%
  dplyr::distinct(.keep_all = T) %>%
  dplyr::filter(substrateID != "")


# create indices
INDEX = EMBEDDING_all %>%
  dplyr::distinct(substrateID, product, observed) %>%
  dplyr::left_join(SUB_masks) %>%
  dplyr::mutate(ID = paste0(product,"-",substrateID),
                index = row_number()-1,
                pcp_mask = fifelse(str_detect(product, "pcp_"), T, F),
                psp_mask = fifelse(str_detect(product, "psp_"), T, F),
                rcs_mask = fifelse(str_detect(product, "rcs_"), T, F),
                pseudo_mask = fifelse(!pcp_mask & !psp_mask & !rcs_mask, T, F),
                # pseudo_mask = fifelse(!pcp_mask & !psp_mask, T, F),
                pseudo1_mask = fifelse(str_detect(product, "p1_"), T, F),
                pseudo2_mask = fifelse(str_detect(product, "p1'_"), T, F),
                subnode_mask = fifelse(product == paste0("pcp_",substrateLength,"_",1), T, F),
                target = fifelse(pseudo1_mask & observed, 1, 0),
                target = fifelse(pseudo2_mask & observed, 2, target),
                target = fifelse(rcs_mask & observed, 3, target), # !!!
                target = fifelse(pcp_mask & observed, 4, target), # !!!
                target = fifelse(psp_mask & observed, 5, target)) # !!!

table(INDEX$target)
min(INDEX$index)
max(INDEX$index)


# --- join with tables
# embedding
EMBEDDING_final = INDEX %>%
  dplyr::right_join(EMBEDDING_all) %>%
  dplyr::arrange(index)  # this is important!!!

summary(EMBEDDING_final$index)

# adjacency list
GRAPH_final = GRAPH_all %>%
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
  # dplyr::filter(!is.na(educt_index)) # !!!!!!!!!!!!!!!!!!!!!!!!!!!! check everytime !!!!!!!!!!!!!!!!!!!!!!!!!!!!!

summary(GRAPH_final$product_index)
summary(GRAPH_final$educt_index)
nrow(GRAPH_final)

# # TODO: clean up
# tmp = GRAPH_final %>% dplyr::filter(is.na(educt_index))
# NA_educts = tmp$educt_ID %>% unique()
# tmp2 = GRAPH_final %>% dplyr::filter(is.na(product_index))
# NA_products = tmp2$product_ID %>% unique()



# ----- OUTPUT -----
fwrite(GRAPH_final, "data/GRAPH_240403_RCS.csv")
fwrite(EMBEDDING_final, "data/EMBEDDING_240403_RCS.csv")
fwrite(INDEX, "data/INDICES_240403_RCS.csv")

