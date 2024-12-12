### SPLICING PREDICTOR ###
# description:  functions for quantitative node features for peptide graph
# author:       HPR

source("src/analysis/_analysis-utils.R")
dir.create("results/_analysis/quantitative/", showWarnings = F, recursive = T)


nodeFeatures = function(DB, tpoints = c(1,2,3,4)) {
  
  substrate = DB$substrateID[1]
  
  # ----- get detected positions -----
  # --- PCPs
  pcp = DB %>%
    dplyr::filter(productType == "PCP") %>%
    tidyr::separate_rows(positions, sep = ";") %>%
    dplyr::distinct(substrateID, pepSeq, positions) %>%
    dplyr::mutate(P1 = str_split_fixed(positions, "_", Inf)[,2],
                  P1_ = str_split_fixed(positions, "_", Inf)[,1],
                  P1.P1_ = paste(P1,P1_, sep = "_")) %>%
    unique()
  
  # --- PSPs
  psp = DB %>%
    dplyr::filter(productType == "PSP") %>%
    tidyr::separate_rows(positions, sep = ";") %>%
    dplyr::distinct(substrateID, pepSeq, positions) %>%
    dplyr::mutate(P1 = str_split_fixed(positions, "_", Inf)[,2],
                  P1_ = str_split_fixed(positions, "_", Inf)[,3],
                  P1.P1_ = paste(P1,P1_, sep = "_")) %>%
    unique()
  
  QUAL = rbind(pcp, psp)
  
  # --- join with intensities ---
  QUANT = DB %>%
    dplyr::filter(!is.na(substrateID)) %>%
    tidyr::separate_rows(digestTimes, intensities, sep=";") %>%
    dplyr::rename(digestTime = digestTimes,
                  intensity = intensities) %>%
    dplyr::mutate(intensity = as.numeric(intensity),
                  digestTime = as.numeric(digestTime)) %>%
    dplyr::mutate(intensity = log10(intensity),
                  intensity = fifelse(!is.finite(intensity),0,intensity)) %>%
    dplyr::group_by(substrateID, pepSeq, digestTime) %>%
    dplyr::mutate(intensity = mean(intensity, na.rm = T)) %>%
    dplyr::ungroup() %>%
    dplyr::mutate(intensity = fifelse(!is.finite(intensity), 0, intensity)) %>%
    resolve_multimapper() %>%
    dplyr::select(substrateID, productType, pepSeq, digestTime, intensity) %>%
    dplyr::ungroup() %>%
    dplyr::filter(digestTime %in% seq(0,4))
  
  MASTER = right_join(QUAL, QUANT) %>%
    dplyr::mutate(product = paste0(tolower(productType),"_",P1,"_",P1_)) %>%
    # for spliced peptides does not make a lot of sense! do not predict on spliced peptides!
    dplyr::group_by(product, digestTime) %>%
    dplyr::summarise(intensity = sum(intensity)) %>%
    tidyr::spread(digestTime, intensity, fill = NA) %>%
    dplyr::select(-`0`)
  
  # NOTE: if predicted for spliced peptides, would have to aggregate spliced peptides per splice site
  # at first sight, they look quite different per splice site - think about this later!
  
  # ----- SCS- and PSP-P1 -----
  # SCS for now (?)
  # res = lapply(tpoints, function(time){
  #   p1 = SCSandPSP_overtime(DB, target = "P1", time = time, SR2forSCS = F)
  #   p1_ = SCSandPSP_overtime(DB, target = "P1_", time = time, SR2forSCS = F)
  #   
  #   # plot(x = p1$scs_mean, y = p1$psp_mean)
  #   
  #   res1 = p1 %>%
  #     dplyr::mutate(product = paste0("p1_",residue),
  #                   time = time) %>%
  #     dplyr::select(product, time, scs_mean)
  #   
  #   res2 = p1_ %>%
  #     dplyr::mutate(product = paste0("p1'_",residue),
  #                   time = time) %>%
  #     dplyr::select(product, time, scs_mean)
  #   
  #   return(rbind(res1,res2))
  # }) %>%
  #   rbindlist()
  # 
  # res = res %>%
  #   tidyr::spread(time, scs_mean, fill = NA)
  
  # --- join with master
  # JOINED = rbind(MASTER, res)
  JOINED = MASTER %>%
    dplyr::filter(!grepl("psp",product))
  
  # --- data imputation, normalisation
  
  # TODO: think about imputation
  
  # k = which(JOINED$product == "psp_14_9")
  # v = JOINED[k,as.character(tpoints)] %>% as.numeric()
  # iszero = any(-1*diff(v) == v[1:(length(v)-1)])
  # if (iszero) {
  #   kk = which(v == 0)
  #   v[kk] = v[kk-1]+0.5*(v[kk+1]-v[kk-1])
  # }
  
  cn = as.character(tpoints)
  JOINED[,cn] = (JOINED[,cn] - min(JOINED[,cn]))/(max(JOINED[,cn]) - min(JOINED[,cn]))
  
  # --- plot
  PLOT = JOINED %>%
    dplyr::mutate(type = str_extract(product, "^[^_]+") %>% toupper(),
                  P1 = str_extract(product,"(?<=_)[:digit:]+(?=_|$)"),
                  P1_ = str_extract(product,"[:digit:]+$"))
  
  
  # pdf(paste0("results/_analysis/quantitative/",substrate,"_kinetics.pdf"), height = 12, width = 18)
  # par(mfrow = c(4,6))
  # L = nchar(DB$substrateSeq[1])
  # for (i in 1:L) {
  #   cntp1 = PLOT %>% filter(type == "P1" & P1 == i)
  #   
  #   if (nrow(cntp1) > 0) {
  #     
  #     plot(x = c(tpoints), y = c(cntp1[,cn]), type = "b",
  #          main = paste0("P1: ", i),
  #          ylab = "normalised intensity", xlab = "time [hrs]",
  #          lwd = 2, pch = 16)
  #     
  #     pcp = PLOT %>% filter(type == "PCP" & P1 == i)
  #     if (nrow(pcp) > 0) {
  #       
  #       # tt = apply(pcp[,cn], 2, sum)
  #       # lines(tt, col = "grey", lwd = 2, pch = 16, type = "b")
  #       
  #       pcp[,cn][pcp[,cn] == 0] = NA
  #       l = c(min(pcp[,cn], na.rm = T), max(pcp[,cn], na.rm = T))
  #       
  #       plot(x = c(tpoints), y = c(pcp[1,cn]), type = "b",
  #            main = paste0("non-spliced peptides with P1 = ", i),
  #            ylab = "normalised intensity", xlab = "time [hrs]",
  #            lwd = 2, pch = 16, col = plottingCols["PCP"],
  #            ylim = l)
  #       
  #       for (ii in 2:nrow(pcp)) {
  #         lines(x = c(tpoints), y = c(pcp[ii,cn]), type = "b",
  #               lwd = 2, pch = 16, col = plottingCols["PCP"])
  #       }
  #       
  #     } else {
  #       plot.new()
  #     }
  #     
  #     psp = PLOT %>% filter(type == "PSP" & P1 == i)
  #     if (nrow(psp) > 0) {
  #       
  #       # tt = apply(psp[,cn], 2, sum)
  #       # lines(tt, col = "grey", lwd = 2, pch = 16, type = "b")
  #       
  #       psp[,cn][psp[,cn] == 0] = NA
  #       l = c(min(psp[,cn], na.rm = T), max(psp[,cn], na.rm = T))
  #       
  #       plot(x = c(tpoints), y = c(psp[1,cn]), type = "b",
  #            main = paste0("spliced peptides with P1 = ", i),
  #            ylab = "normalised intensity", xlab = "time [hrs]",
  #            lwd = 2, pch = 16, col = plottingCols["PSP"],
  #            ylim = l)
  #       
  #       for (ii in 2:nrow(psp)) {
  #         lines(x = c(tpoints), y = c(psp[ii,cn]), type = "b",
  #               lwd = 2, pch = 16, col = plottingCols["PSP"])
  #       }
  #       
  #     } else {
  #       plot.new()
  #     }
  #     
  #   }
  #   
  #   cntp1_ = PLOT %>% filter(type == "P1'" & P1 == i)
  #   if (nrow(cntp1_) > 0) {
  #     
  #     plot(x = c(tpoints), y = c(cntp1_[,cn]), type = "b",
  #          main = paste0("P1': ", i),
  #          ylab = "normalised intensity", xlab = "time [hrs]",
  #          lwd = 2, pch = 16)
  #     
  #     pcp = PLOT %>% filter(type == "PCP" & P1_ == i)
  #     if (nrow(pcp) > 0) {
  #       
  #       # tt = apply(pcp[,cn], 2, sum)
  #       # lines(tt, col = "grey", lwd = 2, pch = 16, type = "b")
  #       
  #       pcp[,cn][pcp[,cn] == 0] = NA
  #       l = c(min(pcp[,cn], na.rm = T), max(pcp[,cn], na.rm = T))
  #       
  #       plot(x = c(tpoints), y = c(pcp[1,cn]), type = "b",
  #            main = paste0("non-spliced peptides with P1' = ", i),
  #            ylab = "normalised intensity", xlab = "time [hrs]",
  #            lwd = 2, pch = 16, col = plottingCols["PCP"],
  #            ylim = l)
  #       
  #       for (ii in 2:nrow(pcp)) {
  #         lines(x = c(tpoints), y = c(pcp[ii,cn]), type = "b",
  #               lwd = 2, pch = 16, col = plottingCols["PCP"])
  #       }
  #       
  #     } else {
  #       plot.new()
  #     }
  #     
  #     psp = PLOT %>% filter(type == "PSP" & P1_ == i)
  #     if (nrow(psp) > 0) {
  #       
  #       # tt = apply(psp[,cn], 2, sum)
  #       # lines(tt, col = "grey", lwd = 2, pch = 16, type = "b")
  #       
  #       psp[,cn][psp[,cn] == 0] = NA
  #       l = c(min(psp[,cn], na.rm = T), max(psp[,cn], na.rm = T))
  #       
  #       plot(x = c(tpoints), y = c(psp[1,cn]), type = "b",
  #            main = paste0("spliced peptides with P1' = ", i),
  #            ylab = "normalised intensity", xlab = "time [hrs]",
  #            lwd = 2, pch = 16, col = plottingCols["PSP"],
  #            ylim = l)
  #       
  #       for (ii in 2:nrow(psp)) {
  #         lines(x = c(tpoints), y = c(psp[ii,cn]), type = "b",
  #               lwd = 2, pch = 16, col = plottingCols["PSP"])
  #       }
  #       
  #     } else {
  #       plot.new()
  #     }
  #     
  #   }
  #   
  # }
  # dev.off()
  
  
  return(JOINED)
}

