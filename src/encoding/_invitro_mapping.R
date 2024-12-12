### affinitySPI ###
#
# Description - Mapping of peptide sequences to the substrate sequence
# 
# Input       - peptides to be mapped
#             - substrate sequence
# 
# Output      - table with coordinates of all peptides in the substrate sequence
# 
# Authors     HPR, JL


# ----- PCP location -----
locate_PCP_substrate <- function(pep, subSeq){
  
  pos = str_locate_all(subSeq,pep)
  pcp <- data.table(pepSeq = pep, pos1 = pos[[1]][,1], pos2 = pos[[1]][,2])
  
  return(pcp)
}

# ----- PSP location -----
locate_PSP_substrate <- function(pep, subSeq) {
  
  # sort by N-mers
  N = nchar(pep)
  # get all possible splits of N into two splice-reactants
  q = data.table::CJ(c(1:N), c(1:N))
  
  q = q[which(rowSums(q) == N), ] %>%
    as.matrix()
  
  # get all SRs
  P = strsplit(pep,"") %>% unlist()
  PSPpos = lapply(seq(1,nrow(q)), function(i){
    srs = data.frame(pepSeq = pep,
                     SR1 = paste(P[1:q[i,1]], collapse = ""),
                     SR2 = paste(P[(q[i,1]+1):N], collapse = ""))
  }) %>% 
    plyr::ldply()
  
  
  # map SRs as PCP
  sr1_loc = lapply(PSPpos$SR1, function(sr1){
    locate_PCP_substrate(pep = sr1, subSeq)
  }) %>%
    data.table::rbindlist() %>%
    dplyr::rename(SR1 = pepSeq) %>%
    na.omit() %>%
    as.data.frame()
  
  sr2_loc = lapply(PSPpos$SR2, function(sr2){
    locate_PCP_substrate(pep = sr2, subSeq)
  }) %>%
    data.table::rbindlist() %>%
    dplyr::rename(SR2 = pepSeq,
                  pos3 = pos1, pos4 = pos2) %>%
    na.omit() %>%
    as.data.frame()
  
  POS = suppressMessages(dplyr::left_join(PSPpos, sr1_loc)) %>%
    na.omit()
  POS = suppressMessages(dplyr::left_join(POS, sr2_loc)) %>%
    na.omit() %>%
    unique() %>%
    as.data.frame()
  
  # get splice types
  POS$type = NA
  intv = POS$pos3-POS$pos2
  
  POS$type[intv > 0 & POS$pos3 > POS$pos2] = "cis"
  POS$type[intv <= 0 & POS$pos4 < POS$pos1] = "revCis"
  POS$type[intv <= 0 & POS$pos4 >= POS$pos1] = "trans"
  
  # collapse to single assignment per peptide
  POS$allpos = do.call(paste, c(POS[c("pos1","pos2","pos3","pos4")], sep = "_"))
  # add intervening sequence length
  POS$intervSeq = abs(intv) - 1
  
  POS = POS %>%
    lazy_dt() %>%
    dplyr::group_by(pepSeq) %>%
    dplyr::summarise(spliceType = paste(type, collapse = ";"),
                     positions = paste(allpos, collapse = ";"),
                     intervSeq = paste(intervSeq, collapse = ";")) %>%
    as_tibble() %>%
    ungroup()
  
  return(POS)
}


# ----- actual mapping -----
locate_peps_substrate = function(peps, subSeq, numCPU) {
  if (!exists("cl")){
    cl <- makeForkCluster(numCPU)
    stop = T
  } else {
    stop = F
  }
  
  # account for I/L redundancy
  subSeqIL <- gsub("I","L",subSeq)
  pepsIL <- gsub("I","L",peps)
  
  # --- PCP mapping
  pcpMAP = mcmapply(FUN = locate_PCP_substrate,
                    pep = pepsIL,
                    subSeq = subSeqIL,
                    SIMPLIFY = F,
                    mc.cores = numCPU,
                    mc.preschedule = T,
                    mc.cleanup = T)
  
  pcpMAP = rbindlist(pcpMAP) %>%
    na.omit() %>%
    mutate(spliceType = "PCP",
           positions = paste(pos1,pos2, sep = "_"),
           intervSeq = NA,
           productType = "PCP") %>%
    select(-pos1, -pos2) %>%
    as.data.frame()
  
  kk = which(! pepsIL %chin% pcpMAP$pepSeq)
  
  # --- PSP mapping
  if (length(kk) > 0) {
    pspMAP = mcmapply(FUN = locate_PSP_substrate,
                      pep = pepsIL[kk],
                      subSeq = subSeqIL,
                      mc.cores = numCPU,
                      mc.preschedule = T,
                      mc.cleanup = T)
    
    pspMAP = pspMAP %>%
      t() %>%
      as.data.frame() %>%
      na.omit() %>%
      mutate(productType = "PSP")
    
    # --- combine both
    MAP = rbind(pcpMAP, pspMAP) %>%
      rename(pepSeq_IL = pepSeq) %>%
      filter(!is.na(pepSeq_IL))
    
    
  } else {
    # --- combine both
    MAP = pcpMAP %>%
      rename(pepSeq_IL = pepSeq) %>%
      filter(!is.na(pepSeq_IL))
  }
  
  if (stop) { stopCluster(cl) }
  return(MAP)
}



