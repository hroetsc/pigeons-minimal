### SPLICING PREDICTOR ###
# description:  functions for graph construction from detected products
# author:       HPR

library(tidyr)
library(dplyr)
library(stringr)
library(data.table)

substr_vec = Vectorize(substr, vectorize.args = c("start","stop"))
substr_vec_all = Vectorize(substr, vectorize.args = c("x","start","stop"))


# ----- get all possible non-spliced peptides and splice sites -----
# and amino acids
getAllPositions = function(L, Nmin, Nmax, AA, S) {
  
  Spad = paste0("XX",S,"XX")
  P1 = c(1:L)
  P1_ = c(1:L)

  # splice sites: all possible P1-P1' combinations
  P1.P1_ = tidyr::crossing(P1,P1_) %>%
    as_tibble() %>%
    dplyr::mutate(P1.P1_ = paste(P1,P1_, sep = "_"),
                  aaP1 = substr_vec(Spad,P1,P1+2),  # add +2 to coordinates to account for padded sequence
                  aaP1_ = substr_vec(Spad,P1_+1,P1_+2),  # add +2 to coordinates to account for padded sequence
                  aa = paste(aaP1, aaP1_, sep = "_"))
  
  # filter those that are located at the termini at would be too short
  P1.P1_ = P1.P1_ %>%
    dplyr::mutate(minN = P1+(L-P1_+1)) %>%
    dplyr::mutate(tooShort = minN < Nmin) %>%
    dplyr::filter(!tooShort) %>%
    dplyr::select(-tooShort, -minN)
  
  # 4AA combinations that are possible in substrate
  AA4 = P1.P1_
  
  # non-spliced peptides: valid combinations of P1 (C-term) and P1' (N-term)
  pcp = P1.P1_ %>%
    dplyr::select(-aa) %>%
    dplyr::filter((P1-P1_+1) %in% seq(Nmin, Nmax))
  
  # pad the termini!
  ntermpad = data.frame(P1 = seq(1,Nmin-1), P1_ = 1)%>%
    dplyr::mutate(P1.P1_ = paste(P1,P1_, sep = "_"),
                  aaP1 = substr_vec(Spad,P1,P1+2),  # add +1 to coordinates to account for padded sequence
                  aaP1_ = substr_vec(Spad,P1_+1,P1_+2))  # add +1 to coordinates to account for padded sequence)
  ctermpad = data.frame(P1 = L, P1_ = seq(L-Nmin+1,L)) %>%
    dplyr::mutate(P1.P1_ = paste(P1,P1_, sep = "_"),
                  aaP1 = substr_vec(Spad,P1,P1+2),  # add +1 to coordinates to account for padded sequence
                  aaP1_ = substr_vec(Spad,P1_+1,P1_+2))  # add +1 to coordinates to account for padded sequence)
  # add the substrate as node
  sub = data.frame(P1 = L, P1_ = 1) %>%
    dplyr::mutate(P1.P1_ = paste(P1,P1_, sep = "_"),
                  aaP1 = substr_vec(Spad,P1,P1+2),  # add +1 to coordinates to account for padded sequence
                  aaP1_ = substr_vec(Spad,P1_+1,P1_+2))  # add +1 to coordinates to account for padded sequence)
  
  pcp = rbindlist(list(ntermpad, pcp, ctermpad, sub), use.names = T)
  
  return(list(psp = P1.P1_, pcp = pcp, aa4 = AA4))
}


# ----- get detected P1s, P1's and P1-P1' combinations -----
getDetectedPositions = function(DB) {
  
  substrateSeq = DB$substrateSeq[1]
  substrateSeq_pad = paste0("XX",substrateSeq,"XX")
  
  # --- PCPs
  pcp = DB %>%
    dplyr::filter(productType == "PCP") %>%
    tidyr::separate_rows(positions, sep = ";") %>%
    dplyr::distinct(substrateID, pepSeq, positions) %>%
    dplyr::mutate(P1 = str_split_fixed(positions, "_", Inf)[,2] %>% as.numeric(),
                  P1_ = str_split_fixed(positions, "_", Inf)[,1] %>% as.numeric(),
                  P1.P1_ = paste(P1,P1_, sep = "_")) %>%
    unique()
  
  # --- PSPs
  psp = DB %>%
    dplyr::filter(productType == "PSP") %>%
    tidyr::separate_rows(positions, sep = ";") %>%
    dplyr::distinct(substrateID, pepSeq, positions) %>%
    dplyr::mutate(P1 = str_split_fixed(positions, "_", Inf)[,2] %>% as.numeric(),
                  P1_ = str_split_fixed(positions, "_", Inf)[,3] %>% as.numeric(),
                  aaP1 = substr_vec(substrateSeq_pad,P1,P1+2),
                  aaP1_ = substr_vec(substrateSeq_pad,P1_+1,P1_+2),
                  P1.P1_ = paste(P1,P1_, sep = "_")) %>%
    unique()

  # --- amino acids at splice sites
  aa4 = psp %>%
    dplyr::mutate(aa = paste0(substr_vec(substrateSeq_pad,P1,P1+2),"_",substr_vec(substrateSeq_pad,P1_+1,P1_+2))) %>% # add +1 to coordinates to account for padded sequence
    unique()

  return(list(pcp = pcp, psp = psp, aa4 = aa4))
}


# ----- CONSTRUCT GRAPH -----
constructGraph <- function(DB, numCPU, Nmin, Nmax, AA) {
  
  print("CONSTRUCTING GRAPH")
  
  S = DB$substrateSeq[1]
  L = nchar(S)
  subID = DB$substrateID[1]
  
  # get positions
  allpos = getAllPositions(L, Nmin, Nmax, AA, S)
  detpos = getDetectedPositions(DB)
  
  # ----- PCP graph -----
  # pseudo nodes
  p1 = data.frame(P1 = allpos$psp$P1,
                  aaP1 = allpos$psp$aaP1) %>%  # !!!
    as_tibble() %>%
    dplyr::distinct(aaP1) %>%  # !!! change to AA level instead of position level
    dplyr::mutate(name = paste0("p1_",aaP1),  # !!!
                  detected = fifelse(aaP1 %in% detpos$psp$aaP1, T, F)) # !!! change to label when PSP is detected, and AA level
  
  p1_ = data.frame(P1_ = allpos$psp$P1_,
                   aaP1_ = allpos$psp$aaP1_) %>%  # !!!
    as_tibble() %>%
    dplyr::distinct(aaP1_) %>%  # !!! change to AA level instead of position level
    dplyr::mutate(name = paste0("p1'_", aaP1_),  # !!!
                  detected = ifelse(aaP1_ %in% detpos$psp$aaP1_, T, F)) # !!! change to label when PSP is detected, and AA level
  
  # actual nodes
  pcp = allpos$pcp %>%
    as_tibble() %>%
    dplyr::mutate(name = paste0("pcp_",P1.P1_),
                  aaP1 = allpos$pcp$aaP1,
                  aaP1_ = allpos$pcp$aaP1_,
                  detected = fifelse(P1.P1_ %in% detpos$pcp$P1.P1_, T, F),  # this has explicit information about the length range
                  detected = fifelse(P1.P1_ == paste0(L,"_",1), T, detected))  # substrate node
    
  
  frac_detected = round(100*length(which(pcp$detected))/nrow(pcp),2)
  paste0("percentage of observed non-spliced peptides: ",frac_detected,"%") %>% print()
  frac_detected = round(100*length(which(p1$detected))/nrow(p1),2)
  paste0("percentage of observed P1s: ",frac_detected,"%") %>% print()
  frac_detected = round(100*length(which(p1_$detected))/nrow(p1_),2)
  paste0("percentage of observed P1's: ",frac_detected,"%") %>% print()

  
  # --- PCPs
  COORD_pcp = mclapply(1:nrow(pcp), function(k){
    
    en = pcp$P1[k]
    st = pcp$P1_[k]
    N = en-st+1
    
    prod = pcp$P1.P1_[k]
    prodname = pcp$name[k]
    
    # PCP generated from PCP/substrate through 1 cut
    # NOTE: larger equal/smaller equal introduces self-loops!
    kk = which(pcp$P1_ <= st & pcp$P1 == en |
                 pcp$P1_ == st & pcp$P1 >= en)
    # NOTE: allows only for 1 aa cut
    # kk = which(pcp$P1_ %in% c(st,st-1) & pcp$P1 == en |
    #              pcp$P1_ == st & pcp$P1 %in% c(en,en+1))
    # PCP generated from PSP through cut - omit for now?
    # can only be considered if PSP termini are predicted as well
    
    # add to results
    if(length(kk) > 0) {
      PCP = data.table(product = prod, product_name = prodname, educt = pcp$name[kk],
                       product_detected = pcp$detected[k], educt_detected = pcp$detected[kk])
    } else {
      PCP = data.table(product = prod, product_name = prodname, educt = NA,
                       product_detected = NA, educt_detected = NA)
    }
    
    
    return(PCP)
  }, 
  mc.preschedule = T,
  mc.cleanup = T,
  mc.cores = numCPU)
  
  allPCP = COORD_pcp %>%
    data.table::rbindlist() %>%
    na.omit() %>%
    as_tibble() %>%
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! introduce edge weights !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    dplyr::mutate(edge_weight = 1,
                  edge_weight = fifelse(educt == paste0("pcp_",L,"_1"), 0.5, edge_weight))
  
  
  # --- P1s + P1's (pseudo-nodes)
  # if(nrow(p1) != nrow(p1_)) {
  #   stop("P1 and P1' rows do not match")
  # }
  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! move to AA level !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  # P1
  COORD_p1 = mclapply(1:nrow(p1), function(k){
    
    i = p1$aaP1[k]  # !!!!
    pi = p1$name[k]
    
    # summarising all PCPs with same P1/P1'
    # NOTE: Nmin/Nmax do not need to be introduced (?)
    # P1
    ki = which(pcp$aaP1 == i)  # !!!!
    
    # add to results
    if(length(ki) > 0) {
      P1 = data.table(product = i, product_name = pi, educt = pcp$name[ki],
                      product_detected = p1$detected[k], educt_detected = pcp$detected[ki])
    } else {
      P1 = data.table(product = i, product_name = pi, educt = NA, product_detected = NA, educt_detected = NA)
    }
    
    return(P1)
  }, 
  mc.preschedule = T,
  mc.cleanup = T,
  mc.cores = numCPU)
  
  allP1 = COORD_p1 %>%
    data.table::rbindlist() %>%
    na.omit() %>%
    as_tibble() %>%
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! introduce edge weights !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    dplyr::mutate(edge_weight = 1.5)
  
  # P1'
  COORD_p1_ = mclapply(1:nrow(p1_), function(k){
      
      j = p1_$aaP1_[k]  # !!!!
      pj = p1_$name[k]
      
      # summarising all PCPs with same P1/P1'
      # NOTE: Nmin/Nmax do not need to be introduced (?)
      # P1'
      kj = which(pcp$aaP1_ == j)  # !!!!
      
      # add to results
      if (length(kj) > 0) {
        P1_ = data.table(product = j, product_name = pj, educt = pcp$name[kj],
                        product_detected = p1_$detected[k], educt_detected = pcp$detected[kj])
      } else {
        P1_ = data.table(product = j, product_name = pj, educt = NA, product_detected = NA, educt_detected = NA)
      }
      
      return(P1_)
    }, 
    mc.preschedule = T,
    mc.cleanup = T,
    mc.cores = numCPU)

  allP1_ = COORD_p1_ %>%
    data.table::rbindlist() %>%
    na.omit() %>%
    as_tibble() %>%
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! introduce edge weights !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    dplyr::mutate(edge_weight = 1.5)
  
  
  # ----- PSP graph -----
  psp = allpos$aa4 %>%
    as_tibble() %>%
    unique() %>%
    dplyr::mutate(name = paste0("psp_",aa),
                  detected = ifelse(aa %in% detpos$aa4$aa, T, F))
  
  frac_detected = round(100*length(which(psp$detected))/nrow(psp),2)
  paste0("percentage of observed splice sites (P1-P1' combinations): ",frac_detected,"%") %>% print()
  
  source_for_psp = rbindlist(
    list(p1 %>% mutate(aaP1_ = NA),  # !!!!
         p1_ %>% mutate(aaP1 = NA)),  # !!!!
    use.names = TRUE
  )

  COORD_psp = mclapply(1:nrow(psp), function(k){
    
    i = psp$aaP1[k]
    j = psp$aaP1_[k]
    prod = psp$aa[k]
    prodname = psp$name[k]
    
    # PSP generated from PCP/substrate through splice
    ki = which(source_for_psp$aaP1 == i)
    kj = which(source_for_psp$aaP1_ == j)
    # PSP generated from PSP through cut
    # can only be considered if PSP termini are predicted as well
    
    # add to results
    if(length(ki) > 0 & length(kj) > 0) {
      PSP = data.table(product = prod, product_name = prodname,
                       educt = c(source_for_psp$name[ki], source_for_psp$name[kj]),
                       product_detected = psp$detected[k],
                       educt_detected = c(source_for_psp$detected[ki], source_for_psp$detected[kj]))
      # # --- bidirectional PSP!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      PSPbi = PSP %>%
        dplyr::select(-product) %>%
        as.data.table()
      names(PSPbi) = c("educt","product","educt_detected","product_detected")
      PSPbi = PSPbi %>%
        dplyr::mutate(product_name = product) %>%
        as.data.table()
      PSP = rbindlist(list(PSP,PSPbi), use.names = T)

    } else {
      PSP = data.table(product = prod, product_name = prodname,
                       educt = NA, product_detected = psp$detected[k], educt_detected = NA)
    }
    
    return(PSP)
  }, 
  mc.preschedule = T,
  mc.cleanup = T,
  mc.cores = numCPU)
  
  allPSP = COORD_psp %>%
    data.table::rbindlist() %>%
    na.omit() %>%
    as_tibble() %>%
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! introduce edge weights !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    dplyr::mutate(edge_weight = 2)
  

  # ----- build adjacency list -----
  ALL = rbindlist(
    list(
      allPCP %>% dplyr::mutate(productType = "PCP"),
      allPSP %>% dplyr::mutate(productType = "PSP"),
      allP1 %>% dplyr::mutate(productType = "P1"),
      allP1_ %>% dplyr::mutate(productType = "P1'")
    ),
    use.names = T
  ) %>%
    dplyr::mutate(substrateID = subID) %>%
    dplyr::select(-product) %>%
    dplyr::rename(product = product_name) %>%
     # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! introduce edge weights !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    dplyr::mutate(edge_weight = fifelse(product == educt, 0, edge_weight)) %>%
    as.data.table()
  
  return(list(ALL = ALL, allpos = allpos))
}


