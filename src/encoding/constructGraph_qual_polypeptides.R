### SPLICING PREDICTOR ###
# description:  construct cleavage/splicing graphs
# author:       HPR

# renv::init()
library(stringr)
library(dplyr)
library(data.table)
library(ggplot2)
library(ggraph)
library(tidygraph)
library(parallel)
library(readxl)
theme_set(theme_classic())

source("src/_utils.R")
source("src/encoding/_graphFunctions.R")
source("src/encoding/_nodeEmbedding.R")

# as long as pep_size <= Nmin PCP encodings will only contain valid sequences
pep_size = 6
intv_size = 2
multimappers = TRUE

numCPU = detectCores()
Nmin = 5
Nmax = 40
dir.create("data/datasets/qual_polypeptides/",showWarnings = F)
sink("results/constructGraph_qual_polypeptides_log.txt", split = T)

# ----- INPUT -----
Kinetics = fread("data/SciData2023_filtered.csv")
overview = readxl::read_excel("datasets_polypeptides.xlsx", sheet = 2)

# ----- specify features -----
AAchar_here = c("P","G","C","M","A","V","I","L","F","Y","W","H","R","K","D","E","N","Q","S","T","X")
AAchar_here_sorted = sort(AAchar_here)

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

# PSP parameters
PSPpos = c(SR1pos, SR2pos)
PSPparam = tidyr::crossing(names(PSPpos),AAchar_here_sorted[-grep("X",AAchar_here_sorted)])
names(PSPparam) = c("pos", "AA")
PSPparam = PSPparam %>% arrange(factor(pos, levels = names(PSPpos)))
PSPparam = do.call(paste, c(PSPparam, sep=";"))

# PCP parameters
PCPpos = c(SR1pos, SR2pos)
PCPparam = tidyr::crossing(names(PCPpos),AAchar_here_sorted[-grep("X",AAchar_here_sorted)])
names(PCPparam) = c("pos", "AA")
PCPparam = PCPparam %>% arrange(factor(pos, levels = names(PCPpos)))
PCPparam = do.call(paste, c(PCPparam, sep=";"))

paste0("number of parameters: ", length(PSPparam)) %>% print()


# ----- get graphs -----
substrates = Kinetics %>%
  distinct(substrateSeq, substrateID)

# get masks
train_subs = overview$substrateID[overview$dataset == "train"] %>% na.omit()
test_subs = overview$substrateID[overview$dataset == "test"] %>% na.omit()
val_subs = overview$substrateID[overview$dataset == "val"] %>% na.omit()

SUB_masks = substrates %>%
  dplyr::select(substrateID) %>%
  dplyr::mutate(train_mask = ifelse(substrateID %in% train_subs, T, F),
                val_mask = ifelse(substrateID %in% val_subs, T, F),
                test_mask = ifelse(substrateID %in% test_subs, T, F))


for (substrate in substrates$substrateID) {
  
  print("---------------")
  print(substrate)
  print("---------------")
  ID = na.omit(overview$dataset[overview$substrateID == substrate])
  
  DB = Kinetics %>%
    dplyr::filter(substrateID == substrate) %>%
    disentangleMultimappers.Type() %>%
    dplyr::distinct(substrateID, substrateSeq, pepSeq, productType, spliceType, positions)
  
  # graph
  X = constructGraph(DB, numCPU, Nmin, Nmax)
  GRAPH = X$ALL
  allpos = X$allpos
  
  # node embedding
  NODE_EMBEDDING = nodeEmbedding(DB, PSPparam, PCPparam, SR1pos, SR2pos, allpos)
  # combine with labels
  EMBEDDING = GRAPH %>% dplyr::distinct(substrateID, product, product_detected) %>%
    right_join(NODE_EMBEDDING) %>%
    dplyr::rename(observed = product_detected)
  
  # save
  fwrite(GRAPH, file = paste0("data/datasets/qual_polypeptides/",substrate,"_graph.csv"))
  fwrite(EMBEDDING, file = paste0("data/datasets/qual_polypeptides/",substrate,"_embedding.csv"))
}




# ----- merge datasets -----
fs_graph = list.files("data/datasets/qual_polypeptides/", pattern = "graph.csv", full.names = T, recursive = F)
GRAPH_all = lapply(fs_graph, function(f){
  fread(f)
}) %>%
  rbindlist() %>%
  dplyr::filter(substrateID != "") %>%
  unique()

fs_emb = list.files("data/datasets/qual_polypeptides/", pattern = "_embedding.csv", full.names = T, recursive = F)
EMBEDDING_all = lapply(fs_emb, function(f){
  fread(f)
}) %>%
  rbindlist() %>%
  dplyr::filter(substrateID != "") %>%
  unique()

# create indices
INDEX = EMBEDDING_all %>%
  dplyr::distinct(substrateID, product, observed) %>%
  dplyr::mutate(ID = paste0(product,"-",substrateID),
                index = row_number()-1,
                pcp_mask = fifelse(str_detect(product, "pcp_"), T, F),
                psp_mask = fifelse(str_detect(product, "psp_"), T, F),
                pseudo_mask = fifelse(!pcp_mask & !psp_mask, T, F),
                target = fifelse(pseudo_mask & observed, 1, 0),
                target = fifelse(pcp_mask & observed, 2, target),
                target = fifelse(psp_mask & observed, 3, target))

table(INDEX$target)
min(INDEX$index)
max(INDEX$index)
table(INDEX$pcp_mask)
table(INDEX$psp_mask)
table(INDEX$pseudo_mask)


# --- join with tables
# embedding
EMBEDDING_final = INDEX %>%
  dplyr::right_join(EMBEDDING_all)

EMBEDDING_final = SUB_masks %>%
  dplyr::right_join(EMBEDDING_final) %>%
  dplyr::arrange(index)

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

summary(GRAPH_final$product_index)
summary(GRAPH_final$educt_index)
nrow(GRAPH_final)


# ----- OUTPUT -----
fwrite(GRAPH_final, "data/GRAPH_polypeptides_qual.csv")
fwrite(EMBEDDING_final, "data/EMBEDDING_polypeptides_qual.csv")
fwrite(INDEX, "data/INDICES_polypeptides_qual.csv")


