### SPLICING PREDICTOR ###
# description:  functions for node embedding for peptide graph
# author:       HPR

library(tidyr)
library(dplyr)
library(stringr)
library(data.table)

source("src/encoding/_graphFunctions_reactionNetwork.R")
substr_vec = Vectorize(substr, vectorize.args = c("start","stop"))


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
    # SRTBL = cbind(SR1TBL, SR2TBL)
    # paramNames = c(paste0(rep(colnames(SR1TBL), each = ncol(AAencoding)),";",rep(colnames(AAencoding), ncol(SR1TBL))),
    #                 paste0(rep(colnames(SR2TBL), each = ncol(AAencoding)),";",rep(colnames(AAencoding), ncol(SR2TBL))))
    
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



# ----- generate encoding relative cleavage sites -----
generateSiteEncodingRCS = function(AAencoding, Seq, SR1pos, SR2pos, sites, paramNames, numCPU) {

    # --- get all combinations of positions
    pos = data.frame(substrateSeq = Seq, c = sites$cleavage_site, P1 = sites$cleavage_site_abs, Nterm = sites$cleavage_site_abs-sites$cleavage_site+1)

    # --- generate matrices with amino acids
    # valid sequences only!
    # TODO: N-terminal sites only (?)
    SR1TBL = sapply(SR1pos, function(x){
        substr(pos$substrateSeq, start = pos$P1-pos$Nterm+1+x, stop = pos$P1-pos$Nterm+1+x) 
    }) %>%
        as.data.frame()
    SR1TBL[SR1TBL == ""] = "X"
    
    # --- get counts
    master <- mclapply(1:nrow(SR1TBL), function(j) {
        cntN = paste(colnames(SR1TBL),SR1TBL[j,],sep = ";")
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
    master_sr2 = master
    master_sr2[,] <- 0

    mastermaster = cbind(master, master_sr2)
    colnames(mastermaster) = paramNames

    res = cbind(pos,mastermaster) %>% as.data.frame()
    
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
        dplyr::mutate(P1.P1_ = paste(P1, P1_, sep = "_")) %>%
        as.data.table()
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
    # FIXME selection of relevant positions !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    pos = cbind(pos, AA)
    pos = pos %>%
        dplyr::mutate(aaP1 = paste(P3,P2,P1,sep = ""),
                    aaP1_ = paste(`P-1_`,P1_,sep = ""),
                    aa = paste(aaP1,aaP1_, sep = "_")) %>%
        as.data.table()
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
        as.data.table()

    res = master %>%
        dtplyr::lazy_dt() %>%
        dplyr::mutate(P1.P1_ = paste0(posP1,"_",posP1_)) %>%
        dplyr::group_by(aa) %>%
        dplyr::summarise_at(paramNames, mean) %>%
        as_tibble()

    return(res)
}



# ----- NODE EMBEDDING -----
nodeEmbedding <- function(DB, SR1pos, SR2pos, allpos, GRAPH, AAencoding, numCPU) {
    
    print("GET EMBEDDING OF NODES")
    
    Seq = DB$substrateSeq[1]
    subID = DB$substrateID[1]
    AAencoding = AAencoding %>%
      tibble::column_to_rownames("amino_acid")
    AA = rownames(AAencoding) %>% sort()

    paramNames = c(paste0(rep(names(SR1pos), each = ncol(AAencoding)),";",rep(colnames(AAencoding), length(SR1pos))),
                    paste0(rep(names(SR2pos), each = ncol(AAencoding)),";",rep(colnames(AAencoding), length(SR2pos))))

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


    # --- relative cleavage site
    # !!!!!
    RCS = GRAPH %>%
        dplyr::filter(productType == "RCS") %>%
        dplyr::distinct(product, cleavage_site, cleavage_site_abs) %>%
        as_tibble()
    
    FEAT_rcs = generateSiteEncodingRCS(AAencoding = AAencoding, Seq = Seq, SR1pos = SR1pos,
                                       SR2pos = SR2pos, sites = RCS, paramNames = paramNames, numCPU = numCPU) %>%
        dplyr::select(-substrateSeq, -Nterm) %>%
        dplyr::rename(product = P1,
                      class = c) %>%
        dplyr::mutate(product = paste0("rcs_",product,"_",class),
                      class = -1)
    

    # --- P1/P1' features
    # probabilistic
    FEAT_p1 = PSPraw
    k = which(names(FEAT_p1) %in% outer(names(SR2pos),names(AAencoding), paste, sep=";"))
    FEAT_p1[,k] <- 0  
    FEAT_p1 = FEAT_p1 %>%
        as_tibble() %>%
        dplyr::mutate(aa = str_split_fixed(aa,"_",Inf)[,1]) %>%
        unique() %>%
        dplyr::rename(product = aa) %>%
        dplyr::mutate(product = paste0("p1_",product), # class is the P1
                        class = 0) %>%
        dplyr::select(product, class, all_of(paramNames))

    
    FEAT_p1_ = PSPraw
    m = which(names(FEAT_p1_) %in% outer(names(SR1pos),names(AAencoding), paste, sep=";"))
    FEAT_p1_[,m] <- 0  
    FEAT_p1_ = FEAT_p1_ %>%
        as_tibble() %>%
        dplyr::mutate(aa = str_split_fixed(aa,"_",Inf)[,2]) %>%
        unique() %>%
        dplyr::rename(product = aa) %>%
        dplyr::mutate(product = paste0("p1'_",product), # class is the P1
                        class = 0) %>%
        dplyr::select(product, class, all_of(paramNames))
    
    
    # --- COMBINE ALL
    FEATURES = rbindlist(
      list(FEAT_psp, FEAT_pcp, FEAT_rcs, FEAT_p1, FEAT_p1_),
    #   list(FEAT_psp, FEAT_pcp, FEAT_p1, FEAT_p1_),
      use.names=TRUE
    ) %>%
      as_tibble()
  
  return(FEATURES)
}

