### SPLICING PREDICTOR ###
# description:  construct cleavage/splicing graphs
# author:       HPR

library(data.table)
library(dplyr)
library(stringr)

SUFFIX = "240229_5aa"


# ----- INPUT -----
EMBEDDING = fread(paste0("data/EMBEDDING_",SUFFIX,".csv"))
GRAPH = fread(paste0("data/GRAPH_",SUFFIX,".csv"))
INDICES = fread(paste0("data/INDICES_",SUFFIX,".csv"))


# ----- get indiced per node type -----
INDICES_mod = INDICES %>%
    dplyr::arrange(index) %>%
    dplyr::mutate(productType = toupper(str_extract(ID, "^[:graph:]{2,3}(?=_)"))) %>%
    dplyr::group_by(productType) %>%
    dplyr::mutate(index = row_number()-1) %>%
    dplyr::ungroup() %>%
    dplyr::arrange(productType, index)


# ----- add modified indices to embeddings and graph adjacency list -----
# embedding
EMBEDDING_mod = EMBEDDING %>%
    dplyr::select(-index) %>%
    dplyr::mutate(productType = toupper(str_extract(ID, "^[:graph:]{2,3}(?=_)")), .after = product)
EMBEDDING_final = INDICES_mod %>%
  dplyr::right_join(EMBEDDING_mod) %>%
  dplyr::arrange(productType, index)

# adjacency list
GRAPH_final = GRAPH %>%
    dplyr::select(-product_index, -educt_index, -edge_weight,-productType) %>%
    dplyr::left_join(INDICES_mod %>%
                        dplyr::rename(product_ID = ID,
                                      product_index = index) %>%
                        dplyr::select(productType, product_ID, product_index)) %>%
    dplyr::left_join(INDICES_mod %>%
                        dplyr::rename(educt_ID = ID,
                                        educt_index = index,
                                        eductType = productType) %>%
                        dplyr::select(eductType, educt_ID, educt_index)) %>%
    dplyr::mutate(edge_type = fifelse(eductType == "PCP" & productType == "PCP", 0, -1),
                  edge_type = fifelse(eductType == "PCP" & productType == "P1", 1, edge_type),
                  edge_type = fifelse(eductType == "PCP" & productType == "P1'", 2, edge_type),
                  edge_type = fifelse(eductType == "P1" & productType == "PSP", 3, edge_type),
                  edge_type = fifelse(eductType == "P1'" & productType == "PSP", 4, edge_type)) %>%
    dplyr::filter(edge_type != -1) %>%  # NOTE: this removes the decay of PSPs into P1 and P1'
    dplyr::arrange(productType, product_index)

summary(GRAPH_final$product_index)
summary(GRAPH_final$educt_index)
table(GRAPH_final$edge_type)
nrow(GRAPH_final)

# GRAPH_final %>% dplyr::filter(edge_type == -1) %>% dplyr::mutate(types = paste0(eductType,"-",productType)) %>% dplyr::pull(types) %>% table()
# GRAPH_final %>% dplyr::filter(edge_type == 0) %>% dplyr::pull(productType) %>% table()
GRAPH_final %>% dplyr::filter(edge_type == 0) %>% dplyr::pull(product_index) %>% table()
EMBEDDING_final %>% dplyr::filter(productType == "PSP") %>% dplyr::pull(index) %>% summary()

GRAPH_final %>% dplyr::filter(edge_type == 3) %>% dplyr::pull(educt_index) %>% summary()


# ----- transform targets -----
EMBEDDING_final = EMBEDDING_final %>%
    dplyr::mutate(target = fifelse(target != 0, 1, 0))

table(EMBEDDING_final$target)


# ----- OUTPUT -----
fwrite(GRAPH_final, paste0("data/GRAPH_",SUFFIX,"_heterogeneous.csv"))
fwrite(EMBEDDING_final, paste0("data/EMBEDDING_",SUFFIX,"_heterogeneous.csv"))
fwrite(INDICES_mod, paste0("data/INDICES_",SUFFIX,"_heterogeneous.csv"))


