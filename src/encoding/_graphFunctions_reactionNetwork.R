### SPLICING PREDICTOR ###
# description:  functions for graph construction (restructured using reaction networks)
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
        
    # 4AA combinations that are possible in substrate
    # filter those PSPs that are located at the termini at would be too short
    AA4 = P1.P1_ %>%
        dplyr::mutate(minN = P1+(L-P1_+1)) %>%
        dplyr::mutate(tooShort = minN < Nmin) %>%
        dplyr::filter(!tooShort) %>%
        dplyr::select(-tooShort, -minN)
    
    # non-spliced peptides: valid combinations of P1 (C-term) and P1' (N-term)
    pcp = P1.P1_ %>%
        dplyr::select(-aa) %>%
        # dplyr::filter((P1-P1_+1) >= Nmin)
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
    # # add first cleavage products of PCPs
    # pad = rbind(data.frame(P1 = L, P1_ = seq(Nmin+1,L)),
    #             data.frame(P1 = L, P1_ = seq(2,L))) %>%
    #     dplyr::mutate(P1.P1_ = paste(P1,P1_, sep = "_"),
    #                     aaP1 = substr_vec(Spad,P1,P1+2),  # add +1 to coordinates to account for padded sequence
    #                     aaP1_ = substr_vec(Spad,P1_+1,P1_+2))  # add +1 to coordinates to account for padded sequence)

    pcpall = rbindlist(list(ntermpad, pcp, ctermpad, sub), use.names = T) %>%
        as_tibble() %>%
        unique()
    
    #   pcp = rbindlist(list(pad, pcp), use.names = T) %>%
    #     unique()
  
  return(list(psp = P1.P1_, pcp = pcpall, aa4 = AA4))
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


# ----- get PCP clevage templates -----
cleavageTemplate = function(Nmin, Nmax, L) {
    
    # 'normal'
    cleave = lapply((2*Nmin):L, function(N) {
        seq(Nmin,N-Nmin)
    })
    names(cleave) = (2*Nmin):L
    # cleave[[length(cleave)+1]] = 1:L
    # names(cleave) = c((2*Nmin):Nmax, L)

    return(cleave)
}


# ----- CONSTRUCT GRAPH -----
constructGraphNetwork <- function(DB, numCPU, Nmin, Nmax, AA) {
    
    print("CONSTRUCTING NETWORK GRAPH")
    
    S = DB$substrateSeq[1]
    L = nchar(S)
    subID = DB$substrateID[1]
    
    # get positions
    allpos = getAllPositions(L, Nmin, Nmax, AA, S)
    detpos = getDetectedPositions(DB)

    # ----- PCP graph -----
    # pseudo nodes
    p1 = data.table(P1 = allpos$psp$P1,
                    aaP1 = allpos$psp$aaP1) %>%  # !!!
        as_tibble() %>%
        dplyr::distinct(aaP1) %>%  # !!! change to AA level instead of position level
        dplyr::mutate(name = paste0("p1_",aaP1),  # !!!
                      detected = fifelse(aaP1 %in% detpos$psp$aaP1, T, F)) # !!! change to label when PSP is detected, and AA level
    
    p1_ = data.table(P1_ = allpos$psp$P1_,
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
                        detected = fifelse(P1.P1_ == paste0(L,"_",1), T, detected)) %>% # substrate node
        dplyr::mutate(N = P1-P1_+1,
                      validLength = N >= Nmin & N <= Nmax)
    
    frac_detected = round(100*length(which(pcp$detected))/nrow(pcp),2)
    paste0("percentage of observed non-spliced peptides: ",frac_detected,"%") %>% print()
    frac_detected = round(100*length(which(unlist(pcp$detected)[pcp$validLength]))/length(which(pcp$validLength)),2)
    paste0("percentage of observed non-spliced peptides in valid length range: ",frac_detected,"%") %>% print()
    frac_detected = round(100*length(which(p1$detected))/nrow(p1),2)
    paste0("percentage of observed P1s: ",frac_detected,"%") %>% print()
    frac_detected = round(100*length(which(p1_$detected))/nrow(p1_),2)
    paste0("percentage of observed P1's: ",frac_detected,"%") %>% print()

    
    # --- PCPs
    longpcp = pcp %>%
        dplyr::filter(N >= (2*Nmin) & (N <= (2*Nmax) | N == L)) %>%
        dplyr::arrange(N)  # important!!!
    cleavage_template = cleavageTemplate(Nmin,Nmax,L)

    # search for products that could result from cleavage of parental (long) PCPs
    COORD_pcp = mclapply(1:nrow(longpcp), function(k){
        
        en = longpcp$P1[k]
        st = longpcp$P1_[k]
        N = longpcp$N[k]
        
        eductname = longpcp$name[k]
        educt_detected = longpcp$detected[k]

        cleave = cleavage_template[[as.character(N)]]

        # get products resulting from cleavage of current PCP
        PCP = lapply(cleave, function(c) {
            
            abs_site = st+c-1
            k1 = which(pcp$P1_ == st & pcp$P1 == abs_site)
            k2 = which(pcp$P1_ == abs_site+1 & pcp$P1 == en)

            # update label
            # TODO?
            product_detected = sapply(c(k1,k2), function(j) {
                detected = pcp$detected[j]
                # if(!detected & !pcp$validLength[j] & educt_detected) TRUE else educt_detected
            }) %>% as.vector()
            # pcp$detected[c(k1,k2)] = product_detected
            # educt_detected = if(!educt_detected & !longpcp$validLength[k] & any(product_detected)) TRUE else educt_detected

            # relative cleavage site (rcs)
            rcs = paste("rcs", abs_site, c, sep = "_")
            rcs_detected = educt_detected | any(pcp$detected[c(k1,k2)]) # !!!!

            # --- with RCS
            # reactant --> rcs, rcs --> product
            PCPd = data.table(educt = c(eductname,rcs,rcs), educt_detected = c(educt_detected,rcs_detected,rcs_detected),
                             product = c(rcs, pcp$name[c(k1,k2)]), product_detected = c(rcs_detected,product_detected),
                             cleavage_site = c, cleavage_site_abs = abs_site, product_site = c("N+C","N","C"),
                             validLength = c(longpcp$validLength[k], pcp$validLength[c(k1,k2)]))
            # add self-loops
            PCP_self = data.table(educt = c(eductname, pcp$name[c(k1,k2)]), educt_detected = c(educt_detected,product_detected),
                                  product = c(eductname, pcp$name[c(k1,k2)]), product_detected = c(educt_detected,product_detected),
                                  cleavage_site = c, cleavage_site_abs = abs_site, product_site = rep("self",3), validLength = c(longpcp$validLength[k], pcp$validLength[c(k1,k2)]))
            PCPf = rbindlist(list(PCPd,PCP_self), fill=TRUE, use.names=TRUE)

            # # --- without RCS
            # # reactant --> product
            # PCPd = data.table(educt = eductname, educt_detected = educt_detected,
            #                  product = pcp$name[c(k1,k2)], product_detected = product_detected,
            #                  cleavage_site = c, cleavage_site_abs = abs_site, product_site = c("N","C"),
            #                  validLength = pcp$validLength[c(k1,k2)])
            # # add self-loops
            # PCP_self = data.table(educt = c(eductname, pcp$name[c(k1,k2)]), educt_detected = c(educt_detected,product_detected),
            #                       product = c(eductname, pcp$name[c(k1,k2)]), product_detected = c(educt_detected,product_detected),
            #                       cleavage_site = c, cleavage_site_abs = abs_site, product_site = rep("self",3), validLength = c(longpcp$validLength[k], pcp$validLength[c(k1,k2)]))
            # PCPf = rbindlist(list(PCPd,PCP_self), fill=TRUE, use.names=TRUE)

            return(PCPf)
        }) %>%
            rbindlist()

        # # TODO?
        # # update also in pcp table
        # kk = which(pcp$name == eductname)
        # pcp$detected[kk] = any(PCP$educt_detected)
        # longpcp$detected[k] = any(PCP$educt_detected)
        
        return(PCP)
    }, 
    mc.preschedule = T,
    mc.cleanup = T,
    mc.cores = numCPU)
    
    allPCP = COORD_pcp %>%
        data.table::rbindlist() %>%
        as_tibble()
    
    m = which(sapply(allPCP$product_detected, length) == 0)
    if (length(m) > 0) {
        allPCP = allPCP[-m,] %>%
            dplyr::mutate(product_detected = unlist(product_detected),
                          educt_detected = unlist(educt_detected))
    } else {
       allPCP = allPCP %>%
            dplyr::mutate(product_detected = unlist(product_detected),
                          educt_detected = unlist(educt_detected))
    }
    
    # add the too short sequences at the termini (they can't be educts)
    shortpcp = pcp %>%
        dplyr::filter(N < Nmin)
    SHORT = data.table(educt = shortpcp$name, educt_detected = shortpcp$detected,
                       product = shortpcp$name, product_detected = shortpcp$detected,
                       cleavage_site = NA, cleavage_site_abs = NA, product_site = "self", validLength = FALSE)
    allPCP = rbindlist(list(allPCP, SHORT)) %>%
        as_tibble()
    
    # connect the longest educts (those that do not occur as product to the substrate node)
    # !!!!
    # educts = allPCP$educt[allPCP$product_site != "self"] %>% unique()
    # products = allPCP$product[allPCP$product_site != "self"] %>% unique()
    educts = allPCP$educt[allPCP$product_site == "N+C"] %>% unique()
    products = allPCP$product[! allPCP$product_site %in% c("self", "N+C")] %>% unique()
    long_products = pcp %>%
        dplyr::filter(name %in% educts[!educts %in% products])
    # !!!
    nlong = nrow(long_products)
    rcs_long = paste("rcs", long_products$P1, long_products$P1-long_products$P1_+1, sep = "_")
    LONG = data.table(educt = c(rep(paste0("pcp_",L,"_",1), nlong), rcs_long), educt_detected = TRUE,
                      product = c(rcs_long,long_products$name), product_detected = c(rep(TRUE, nlong), long_products$detected),
                      cleavage_site = long_products$P1-long_products$P1_+1, cleavage_site_abs = long_products$P1,
                      product_site = NA, validLength = c(rep(NA, nlong), long_products$validLength))
    # new self-loops
    LONGSELF = data.table(educt = rcs_long, educt_detected = TRUE, product = rcs_long, product_detected = TRUE, 
                          cleavage_site = long_products$P1-long_products$P1_+1, cleavage_site_abs = long_products$P1,
                          product_site = NA, validLength = NA)
    # LONG = data.table(educt = paste0("pcp_",L,"_",1), educt_detected = TRUE,
    #                   product = long_products$name, product_detected = long_products$detected,
    #                   cleavage_site = NA, cleavage_site_abs = NA, product_site = NA, validLength = long_products$validLength)

    allPCP = rbindlist(list(allPCP, LONG, LONGSELF)) %>%
        as_tibble()

    # paste0("number of relative cleavage sites: ", length(unique(allPCP$product[allPCP$product_site == "N+C"]))) %>% print()
    
    # C-terminal fragment is oberved more often than N-terminal fragment
    # is that because of splicing?

    # FIXME: OPEN QUESTIONS
    # current implementation: omit substrate node, allow for 1-level cleavage
    # add first two hops from substrate irregardless of length?
    # what to do with too long or too short products? wrt labelling and keeping them?
    # analyse the distance from substrate node - should increase with decreasing product length?
    # precursors have to be long enough to result in cleavage of minimal length product (???)
    # what about too long/too short products that are included here for sake of completeness?

    # --- P1s + P1's (pseudo-nodes)
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
        P1 = data.table(product = pi, educt = pcp$name[ki],
                        product_detected = p1$detected[k], educt_detected = pcp$detected[ki])
        } else {
        P1 = data.table(product = pi, educt = NA, product_detected = NA, educt_detected = NA)
        }
        
        return(P1)
    }, 
    mc.preschedule = T,
    mc.cleanup = T,
    mc.cores = numCPU)
    
    allP1 = COORD_p1 %>%
        data.table::rbindlist() %>%
        na.omit() %>%
        as_tibble()
    
    
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
            P1_ = data.table(product = pj, educt = pcp$name[kj],
                            product_detected = p1_$detected[k], educt_detected = pcp$detected[kj])
        } else {
            P1_ = data.table(product = pj, educt = NA, product_detected = NA, educt_detected = NA)
        }
        
        return(P1_)
        }, 
        mc.preschedule = T,
        mc.cleanup = T,
        mc.cores = numCPU)

    allP1_ = COORD_p1_ %>%
        data.table::rbindlist() %>%
        na.omit() %>%
        as_tibble()
    
    
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
        PSP = data.table(product = prodname,
                            educt = c(source_for_psp$name[ki], source_for_psp$name[kj]),
                            product_detected = psp$detected[k],
                            educt_detected = c(source_for_psp$detected[ki], source_for_psp$detected[kj]))
        
        } else {
        PSP = data.table(product = prodname,
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
        as_tibble()
    

    # ----- build adjacency list -----
    ALL = rbindlist(
        list(
        allPCP %>% dplyr::mutate(productType = ifelse(str_detect(product,"pcp_"),"PCP","RCS")),
        allPSP %>% dplyr::mutate(productType = "PSP"),
        allP1 %>% dplyr::mutate(productType = "P1"),
        allP1_ %>% dplyr::mutate(productType = "P1'")
        ),
        use.names = T,
        fill = T
    ) %>%
        dplyr::mutate(substrateID = subID) %>%
        as.data.table()
    
    return(list(ALL = ALL, allpos = allpos))
}

