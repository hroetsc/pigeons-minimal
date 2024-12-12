### SPLICING PREDICTOR ###
# description:  functions for node embedding for peptide graph
# author:       HPR

library(tidyr)
library(dplyr)
library(stringr)
library(data.table)

source("src/encoding/_graphFunctions_pseudopos.R")
substr_vec = Vectorize(substr, vectorize.args = c("start","stop"))

# ----- generate encoding of splice sites or PCPs -----
generateSiteEncoding = function(Seq, SR1pos, SR2pos, param, all, zeroSR1=FALSE, zeroSR2=FALSE, type = NA) {
  
  # --- get all combinations of positions
  residues = tidyr::crossing(c(1:nchar(Seq)), c(1:nchar(Seq)), .name_repair = "minimal") %>%
    as.data.frame()
  names(residues) = c("P1","P1_")
  residues = residues %>%
    dplyr::mutate(P1.P1_ = paste(P1, P1_, sep = "_")) %>%
    dplyr::filter(P1.P1_ %in% all$P1.P1_)
  pos = data.frame(substrateSeq = Seq, P1 = residues$P1, P1_ = residues$P1_)
  
  # --- generate matrices with amino acids
  # valid sequences only!
  SR1TBL = sapply(SR1pos, function(x){
      substr(pos$substrateSeq, start = pos$P1+x, stop = pos$P1+x)
  }) %>%
    as.data.frame()
  SR1TBL[SR1TBL == ""] = "X"
  
  SR2TBL = sapply(SR2pos, function(x){
      substr(pos$substrateSeq, start = pos$P1_+x, stop = pos$P1_+x)
  }) %>%
    as.data.frame()
  SR2TBL[SR2TBL == ""] = "X"
  
  
  
  if (zeroSR1) {
    SR1TBL[,] <- "X"
  }
  if (zeroSR2) {
    SR2TBL[,] <- "X"
  }
  
  
  SRTBL = cbind(SR1TBL, SR2TBL)
  
  # --- get counts
  master = matrix(0, nrow = nrow(SRTBL), ncol = length(param))
  colnames(master) = param
  
  for (j in 1:nrow(SRTBL)) {
    cntN = paste(colnames(SRTBL),SRTBL[j,],sep = ";")
    if (any(grepl("X", cntN))) {
      cntN = cntN[-grep("X",cntN)] 
    }
    master[j,cntN] = 1
  }
  
  
  res = cbind(pos,master) %>% as.data.frame()
  # if (type == "PCP") {
  #   kpad = which(res$P1-res$P1_+1 < pep_size)
  # }
  
  return(res)
}



# ----- generate encoding of splice sites or PCPs using Sarah 5-bit encodings -----
generateSiteEncodingSarah = function(AAencoding, Seq, SR1pos, SR2pos, all, zeroSR1=FALSE, zeroSR2=FALSE, type = "PSP", numCPU) {
  
  # --- get all combinations of positions
  residues = tidyr::crossing(c(1:nchar(Seq)), c(1:nchar(Seq)), .name_repair = "minimal") %>%
    as.data.frame()
  names(residues) = c("P1","P1_")
  residues = residues %>%
    dplyr::mutate(P1.P1_ = paste(P1, P1_, sep = "_")) %>%
    dplyr::filter(P1.P1_ %in% all$P1.P1_)
  pos = data.frame(substrateSeq = Seq, P1 = residues$P1, P1_ = residues$P1_)
  
  # --- generate matrices with amino acids
  # valid sequences only!
  SR1TBL = sapply(SR1pos, function(x){
      substr(pos$substrateSeq, start = pos$P1+x, stop = pos$P1+x)
  }) %>%
    as.data.frame()
  SR1TBL[SR1TBL == ""] = "X"
  
  SR2TBL = sapply(SR2pos, function(x){
      substr(pos$substrateSeq, start = pos$P1_+x, stop = pos$P1_+x)
  }) %>%
    as.data.frame()
  SR2TBL[SR2TBL == ""] = "X"
  

  if (zeroSR1) {
    SR1TBL[,] <- "X"
  }
  if (zeroSR2) {
    SR2TBL[,] <- "X"
  }
  
  if (type == "PCP") {
    SRTBL = cbind(SR2TBL, SR1TBL)
    paramNames = c(paste0(rep(colnames(SR2TBL), each = ncol(AAencoding)),";",rep(colnames(AAencoding), ncol(SR2TBL))),
                    paste0(rep(colnames(SR1TBL), each = ncol(AAencoding)),";",rep(colnames(AAencoding), ncol(SR1TBL))))
  } else {
    SRTBL = cbind(SR1TBL, SR2TBL)
    paramNames = c(paste0(rep(colnames(SR1TBL), each = ncol(AAencoding)),";",rep(colnames(AAencoding), ncol(SR1TBL))),
                    paste0(rep(colnames(SR2TBL), each = ncol(AAencoding)),";",rep(colnames(AAencoding), ncol(SR2TBL))))
  }
  
  
  # --- get counts
  master <- mclapply(1:nrow(SRTBL), function(j) {
    cntN = paste(colnames(SRTBL),SRTBL[j,],sep = ";")
    AA = str_split_fixed(cntN,";",Inf)[,2]
    
    if (any(grepl("X", cntN))) {
      cntN = cntN[-grep("X",cntN)] 
    }
    
    y = AAencoding[AA,] %>%
      t() %>%
      reshape2::melt() %>%
      pull(value)
    return(y)
  }, mc.cores = numCPU, mc.preschedule = TRUE, mc.cleanup = TRUE)

  master <- plyr::ldply(master)
  colnames(master) = paramNames

  res = cbind(pos,master) %>% as.data.frame()
  
  return(res)
}


generateSieEncodingSarah_PSPAA = function(AAencoding, Seq, SR1pos, SR2pos, all, numCPU, zeroSR1=FALSE, zeroSR2=FALSE) {

  all = all %>%
    dplyr::select(aaP1,aaP1_,aa) %>%
    unique()

  paramNames = c(paste0(rep(names(SR1pos), each = ncol(AAencoding)),";",rep(colnames(AAencoding), length(SR1pos))),
                  paste0(rep(names(SR2pos), each = ncol(AAencoding)),";",rep(colnames(AAencoding), length(SR2pos))))


  if (!zeroSR1) {
    master_p1 <- mclapply(1:nrow(all), function(j) {
    AAc = all$aaP1[j] %>% strsplit("") %>% unlist()
    
    y = AAencoding[AAc,] %>%
      t() %>%
      reshape2::melt() %>%
      pull(value)
    return(y)
    }, mc.cores = numCPU, mc.preschedule = TRUE, mc.cleanup = TRUE)
    master_p1 <- plyr::ldply(master_p1)
  }

  if (!zeroSR2) {
    master_p1_ <- mclapply(1:nrow(all), function(j) {
      AA = all$aaP1_[j] %>% strsplit("") %>% unlist()
      
      y = AAencoding[AA,] %>%
        t() %>%
        reshape2::melt() %>%
        pull(value)
      return(y)
    }, mc.cores = numCPU, mc.preschedule = TRUE, mc.cleanup = TRUE)
    master_p1_ <- plyr::ldply(master_p1_)
  }

  master0 = matrix(0, ncol = length(paramNames), nrow = nrow(all))
  colnames(master0) = paramNames
  k_p1 = c(grep("P2;",colnames(master0)), grep("P1;",colnames(master0)))
  k_p1_ = c(grep("P-1_;",colnames(master0)), grep("P1_;",colnames(master0)))
  
  if (!zeroSR1) {
    master0[1:nrow(master0),k_p1] <- as.matrix(master_p1)
  }
  if (!zeroSR2) {
    master0[1:nrow(master0),k_p1_] <- as.matrix(master_p1_)
  }
  
  res = cbind(all, master0) %>% as.data.frame()
  return(res)
}


# ----- Sarah 5-bit encodings and surrounding amino acids -----
generateSiteEncoding_prob = function(AAencoding, Seq, SR1pos, SR2pos, all, numCPU) {

  # amino acids and coordinates
  npad = length(SR1pos)
  cpad = length(SR2pos)
  Seq_padded = paste0(rep("X",npad) %>% paste(collapse = ""), Seq, rep("X",cpad) %>% paste(collapse = ""))
  
  all = all %>%
    dplyr::select(aaP1,aaP1_,aa) %>%
    unique()
  
  residues = tidyr::crossing(c(1:nchar(Seq)), c(1:nchar(Seq)), .name_repair = "minimal") %>%
    as.data.table()
  names(residues) = c("P1","P1_")
  residues = residues %>%
    dplyr::mutate(P1.P1_ = paste(P1, P1_, sep = "_"))
  pos = data.table(substrateSeq = Seq, posP1 = residues$P1, posP1_ = residues$P1_)

  paramNames = c(paste0(rep(names(SR1pos), each = ncol(AAencoding)),";",rep(colnames(AAencoding), length(SR1pos))),
                  paste0(rep(names(SR2pos), each = ncol(AAencoding)),";",rep(colnames(AAencoding), length(SR2pos))))
  SRnames = c(names(SR1pos),names(SR2pos))

  # get amino acids at each of the substrate positions
  AA1 = substr_vec(Seq_padded, pos$posP1+npad+SR1pos[1], pos$posP1+npad+SR1pos[length(SR1pos)]) %>%
    strsplit("") %>%
    plyr::ldply()
  AA2 = substr_vec(Seq_padded, pos$posP1_+npad+SR2pos[1], pos$posP1_+npad+SR2pos[length(SR2pos)]) %>%
    strsplit("") %>%
    plyr::ldply() 
  AA = cbind(AA1,AA2)
  names(AA) = SRnames

  # combine with all aa combinations
  pos = cbind(pos, AA)
  pos = pos %>%
    dplyr::mutate(aaP1 = paste(P2,P1,sep = ""),
                  aaP1_ = paste(`P-1_`,P1_,sep = ""),
                  aa = paste(aaP1,aaP1_, sep = "_"))
  all = all %>% dplyr::left_join(pos)

  # get encoding
  SRTBL = all[,SRnames]
  master <- mclapply(1:nrow(SRTBL), function(j) {
    cntN = paste(colnames(SRTBL),SRTBL[j,],sep = ";")
    AA = str_split_fixed(cntN,";",Inf)[,2]
    
    if (any(grepl("X", cntN))) {
      cntN = cntN[-grep("X",cntN)] 
    }
    
    y = AAencoding[AA,] %>%
      t() %>%
      reshape2::melt() %>%
      pull(value)
    return(y)
  }, mc.cores = numCPU, mc.preschedule = TRUE, mc.cleanup = TRUE)

  master <- plyr::ldply(master)
  colnames(master) = paramNames

  # summarise on AA node level
  master = cbind(all %>% select(aaP1,aaP1_,aa,substrateSeq,posP1,posP1_), master) %>%
    as_tibble()
  res = master %>%
    dplyr::mutate(P1.P1_ = paste0(posP1,"_",posP1_)) %>%
    dplyr::group_by(aa) %>%
    dplyr::summarise_at(paramNames, mean)

  return(res)
}


# ----- NODE EMBEDDING -----
nodeEmbedding <- function(DB, SR1pos, SR2pos, allpos, AAencoding, numCPU) {

    print("GET EMBEDDING OF NODES")

    Seq = DB$substrateSeq[1]
    subID = DB$substrateID[1]
    AAencoding = AAencoding %>%
    tibble::column_to_rownames("amino_acid")
    AA = rownames(AAencoding) %>% sort()

    paramNames = c(paste0(rep(names(SR1pos), each = ncol(AAencoding)),";",rep(colnames(AAencoding), length(SR1pos))),
                    paste0(rep(names(SR2pos), each = ncol(AAencoding)),";",rep(colnames(AAencoding), length(SR2pos))))


    # for old one-hot encoding
    # PSP parameters
    PSPpos = c(SR1pos, SR2pos)
    PSPparam = tidyr::crossing(names(PSPpos),AA[-grep("X",AA)])
    names(PSPparam) = c("pos", "AA")
    PSPparam = PSPparam %>% arrange(factor(pos, levels = names(PSPpos)))
    PSPparam = do.call(paste, c(PSPparam, sep=";"))

    # PCP parameters
    PCPpos = c(SR1pos, SR2pos)
    PCPparam = tidyr::crossing(names(PCPpos),AA[-grep("X",AA)])
    names(PCPparam) = c("pos", "AA")
    PCPparam = PCPparam %>% arrange(factor(pos, levels = names(PCPpos)))
    PCPparam = do.call(paste, c(PCPparam, sep=";"))

    # --- PSP features
    # Sarah-1 probabilistic encoding
    PSPraw = generateSiteEncoding_prob(AAencoding = AAencoding, Seq = Seq, SR1pos = SR1pos, SR2pos = SR2pos,
                                        all = allpos$psp, numCPU=numCPU)
    FEAT_psp = PSPraw %>%
    dplyr::rename(product = aa) %>%
    dplyr::mutate(product = paste0("psp_",product), # class is the P1
                    class = 1) %>%
    dplyr::select(product, class, all_of(paramNames))


    # --- PCP features
    # Sarah-1
    FEAT_pcp = generateSiteEncodingSarah(AAencoding = AAencoding, Seq = Seq, SR1pos = SR1pos, SR2pos = SR2pos,
                                        type = "PCP", all = allpos$pcp, numCPU=numCPU) %>%
    dplyr::rename(product = substrateSeq,
                    class = P1) %>%
    dplyr::mutate(product = paste0("pcp_",class,"_",P1_), # class is the P1
                    class = -1) %>%
    dplyr::select(-P1_)

    # --- P1/P1' features
    # Sarah-1
    FEAT_p1 = generateSiteEncodingSarah(AAencoding = AAencoding, Seq = Seq, SR1pos = SR1pos, SR2pos = SR2pos,
                                      all = allpos$pcp, zeroSR2 = TRUE, numCPU=numCPU) %>%
      dplyr::rename(product = substrateSeq,
                    class = P1) %>%
      dplyr::mutate(product = paste0("p1_",class),
                    class = 0) %>%
      dplyr::select(-P1_) %>%
      unique()
    
    FEAT_p1_ = generateSiteEncodingSarah(AAencoding = AAencoding, Seq = Seq, SR1pos = SR1pos, SR2pos = SR2pos,
                                          all = allpos$pcp, zeroSR1 = TRUE, numCPU=numCPU) %>%
      dplyr::rename(product = substrateSeq,
                    class = P1) %>%
      dplyr::mutate(product = paste0("p1'_",P1_),
                    class = 0) %>%
      dplyr::select(-P1_) %>%
      unique()



    # --- COMBINE ALL
    FEATURES = rbindlist(
    list(FEAT_psp, FEAT_pcp, FEAT_p1, FEAT_p1_),
    use.names=TRUE
    )

    return(FEATURES)
}

