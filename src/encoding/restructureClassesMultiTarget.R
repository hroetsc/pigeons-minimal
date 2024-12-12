### SPLICING PREDICTOR ###
# description:  reconstruct prediction classes to allow for multi-target prediction
# author:       HPR

library(data.table)
library(dplyr)
library(stringr)
library(parallel)
library(eulerr)

numCPU = 32
Nmin = 7
Nmax = 30
SUFFIX = "240305_5aa"

dir.create("results/_analysis/restructureClassesMultiTarget/", recursive = T, showWarnings = F)
AA = c("P","G","C","M","A","V","I","L","F","Y","W","H","R","K","D","E","N","Q","S","T","X") %>% sort()
substr_vec = Vectorize(substr, vectorize.args = c("start","stop"))
substr_vec_all = Vectorize(substr, vectorize.args = c("x","start","stop"))


# ----- INPUT -----
EMBEDDING = fread(paste0("data/EMBEDDING_",SUFFIX,".csv"))
GRAPH = fread(paste0("data/GRAPH_",SUFFIX,".csv"))
INDICES = fread(paste0("data/INDICES_",SUFFIX,".csv"))

load("data/aSPIRE_240117.RData")
inSPIRE_unfiltered = fread("data/inSPIRE_20240301_unfiltered.csv")


# ----- preprocessing -----
DB = Kinetics %>%
    dplyr::filter(productType == "PSP") %>%
    dplyr::distinct(substrateID, substrateSeq, pepSeq, positions) %>%
    tidyr::separate_rows(positions, sep = ";") %>%
    dplyr::mutate(P1 = str_split_fixed(positions, "_", Inf)[,2] %>% as.numeric(),
                  P1_ = str_split_fixed(positions, "_", Inf)[,3] %>% as.numeric(),
                  P1.P1_ = paste(P1, P1_, sep = "_"),
                  start = str_split_fixed(positions, "_", Inf)[,1] %>% as.numeric(),
                  end = str_split_fixed(positions, "_", Inf)[,4] %>% as.numeric(),
                  SR1 = P1-start+1,
                  SR2 = end-P1_+1) %>%
    dplyr::select(-start, -end) %>%
    dplyr::mutate(Spad = paste0("XX",substrateSeq,"XX"),
                  aaP1 = substr_vec_all(Spad,P1,P1+2),  # add +2 to coordinates to account for padded sequence
                  aaP1_ = substr_vec_all(Spad,P1_+2,P1_+3),  # add +2 to coordinates to account for padded sequence
                  aa = paste(aaP1, aaP1_, sep = "_")) %>%
    dplyr::select(-substrateSeq, -Spad) %>%
    dplyr::mutate(ID = paste0("psp_",aa,"-",substrateID))

DB_PCP = Kinetics %>%
    dplyr::filter(productType == "PCP") %>%
    dplyr::distinct(substrateID, positions) %>%
    dplyr::mutate(P1_ = str_extract(positions,"^[:digit:]+(?=_)") %>% as.numeric(),
                  P1 = str_extract(positions,"(?<=_)[:digit:]+$") %>% as.numeric(),
                  product = paste0("pcp_",P1,"_",P1_),
                  ID = paste0(product,"-",substrateID))


# ----- fetch "maybe" class -----
substrates = Kinetics %>%
    dplyr::distinct(substrateID,substrateSeq)
SITES = INDICES %>%
    dplyr::left_join(substrates)
    
# --- ambiguous splice sites
allsites = mclapply(1:nrow(substrates), function(i) {
    
    Spad = paste0("XX",substrates$substrateSeq[i],"XX")
    L = nchar(substrates$substrateSeq[i])
    print(L)

    P1 = c(1:L)
    P1_ = c(1:L)

    P1.P1_ = tidyr::crossing(P1,P1_) %>%
        as_tibble() %>%
        dplyr::mutate(P1.P1_ = paste(P1, P1_, sep = "_"))

    # all amino acid combinations
    aa_5 = P1.P1_ %>% dplyr::mutate(aaP1 = substr_vec(Spad,P1,P1+2),  # add +2 to coordinates to account for padded sequence
                                    aaP1_ = substr_vec(Spad,P1_+2,P1_+3),  # add +2 to coordinates to account for padded sequence
                                    product = paste0("psp_",aaP1,"_",aaP1_),
                                    substrateID = substrates$substrateID[i])

    # flag detected
    D = DB %>%
        dplyr::filter(substrateID == substrates$substrateID[i])
    aa_5 = aa_5 %>%
        dplyr::mutate(detected = P1.P1_ %in% D$P1.P1_) # based on positions !!!!!
    
    return(aa_5)
},
mc.cores = numCPU, mc.cleanup = T, mc.preschedule = T)  
allsites = rbindlist(allsites)

table(allsites$detected)

ambiguous_sites = allsites %>%
    dplyr::group_by(substrateID, product) %>%
    dplyr::summarise(ambiguous = any(detected) & !all(detected),
                     detected = any(detected)) %>%
    dplyr::mutate(ID = paste0(product,"-",substrateID))

table(ambiguous_sites$detected, ambiguous_sites$ambiguous)


# --- sequences at termini
sequences_at_termini = SITES %>%
    dplyr::filter(pcp_mask) %>%
    dplyr::mutate(L = nchar(substrateSeq), 
                  P1 = str_extract(product,"(?<=pcp_)[:digit:]+") %>% as.numeric(),
                  P1_ = str_extract(product,"(?<=[:digit:]_)[:digit:]+$") %>% as.numeric(),
                  at_terminus = (P1-P1_+1) < Nmin)
table(sequences_at_termini$at_terminus)

# --- 1-5% FDR identifications
# pcp and psp
pcp_higherfdr = inSPIRE_unfiltered %>%
    dplyr::filter(productType == "PCP") %>%
    dplyr::mutate(P1_ = str_extract(positions,"^[:digit:]+(?=_)") %>% as.numeric(),
                  P1 = str_extract(positions,"(?<=_)[:digit:]+$") %>% as.numeric(),
                  product = paste0("pcp_",P1,"_",P1_)) %>%
    dplyr::distinct(substrateID,product) %>%
    dplyr::mutate(ID = paste0(product,"-",substrateID))

psp_higherfdr = inSPIRE_unfiltered %>%
    dplyr::filter(productType == "PSP") %>%
    dplyr::distinct(substrateID,substrateSeq,positions,P1,P1_) %>%
    dplyr::mutate(Spad = paste0("XX",substrateSeq,"XX"),
                  aaP1 = substr_vec_all(Spad,P1,P1+2),  # add +2 to coordinates to account for padded sequence
                  aaP1_ = substr_vec_all(Spad,P1_+2,P1_+3),  # add +2 to coordinates to account for padded sequence
                  aa = paste(aaP1, aaP1_, sep = "_")) %>%
    dplyr::distinct(substrateID,aa) %>%
    dplyr::mutate(ID = paste0("psp_",aa,"-",substrateID))


# make sure that it does not occur in the <1% DB
pcp_higherfdr = pcp_higherfdr %>%
    dplyr::filter(!ID %in% DB_PCP$ID)
psp_higherfdr = psp_higherfdr %>%
    dplyr::filter(!ID %in% DB$ID)

nrow(pcp_higherfdr)
nrow(DB_PCP)
nrow(psp_higherfdr)
nrow(DB)


# ----- split target -----
# join info about ambiguity with INDICES table
info = INDICES %>% 
    dplyr::select(substrateID, product, ID, pcp_mask, psp_mask, pseudo_mask, pseudo1_mask, pseudo2_mask, observed) %>%
    dplyr::mutate(ambiguous_sites = ID %in% ambiguous_sites$ID[ambiguous_sites$ambiguous],
                  sequences_at_termini = ID %in% sequences_at_termini$ID[sequences_at_termini$at_terminus],
                  higherfdr = ID %in% c(pcp_higherfdr$ID, psp_higherfdr$ID)) %>%
    dplyr::mutate(maybe = ambiguous_sites | sequences_at_termini | higherfdr)
    # dplyr::mutate(maybe = sequences_at_termini | higherfdr)

# for pseudonodes
# fetch label of PSPs
pseudo = GRAPH %>%
    dplyr::filter(grepl("p1",educt_ID)) %>%
    dplyr::select(product_ID, educt_ID) %>%
    dplyr::rename(ID = product_ID) %>%
    dplyr::left_join(info %>% dplyr::select(ID, maybe, observed)) %>%
    dplyr::group_by(educt_ID) %>%
    dplyr::summarise(maybe_pseudo = (any(maybe) & all(!observed)) | all(maybe)) %>%
    dplyr::ungroup() %>%
    dplyr::filter(maybe_pseudo)

info = info %>%
    dplyr::mutate(maybe = fifelse(ID %in% pseudo$educt_ID, TRUE, maybe))

# --- product type
target_type = rep(0, nrow(info))
target_type[info$pseudo2_mask] = 1
target_type[info$pcp_mask] = 2
target_type[info$psp_mask] = 3


# --- observation target
target_observation = rep(0, nrow(info))
target_observation[info$observed] = 2
target_observation[info$maybe] = 1


# --- combine
info = info %>%
    dplyr::mutate(target_type = target_type,
                  target_observation = target_observation)

table(info$target_type, info$target_observation)

EMBEDDING = info %>%
    dplyr::select(ID, target_type, target_observation) %>%
    dplyr::right_join(EMBEDDING %>% dplyr::select(-target)) %>%
    dplyr::arrange(index)  # this is important!!!


# ----- stats -----
png("results/_analysis/restructureClassesMultiTarget/PSP_space.png", height = 6, width = 6, units = "in", res = 300)
list(allpsp = info$ID[info$psp_mask],
     detectedpsp = info$ID[info$psp_mask & info$observed],
     fdr1to5 = info$ID[info$higherfdr & info$psp_mask],
     ambiguous = info$ID[info$ambiguous_sites]) %>%
    euler(shape = "ellipse") %>%
    plot(quantities = T, main = "PSP space")
dev.off()

png("results/_analysis/restructureClassesMultiTarget/PCP_space.png", height = 6, width = 6, units = "in", res = 300)
list(allpcp = info$ID[info$pcp_mask],
     detectedpcp = info$ID[info$pcp_mask & info$observed],
     fdr1to5 = info$ID[info$higherfdr & info$pcp_mask],
     termini = info$ID[info$sequences_at_termini]) %>%
    euler(shape = "ellipse") %>%
    plot(quantities = T, main = "PCP space")
dev.off()


# ----- OUTPUT -----
fwrite(GRAPH, paste0("data/GRAPH_",SUFFIX,"_restructured.csv"))
fwrite(EMBEDDING, paste0("data/EMBEDDING_",SUFFIX,"_restructured.csv"))
fwrite(info, paste0("data/INDICES_",SUFFIX,"_restructured.csv"))
