### in vitro analysis ###
# description:    utility function for all analyses
# author:         HPR

library(ggplot2)
library(ggseqlogo)

# ----- PLOTTING -----
# ----- colours -----
plottingCols = c(
  PCP = "#EC9A56",
  PSP = "#7B80C7",
  cis = "darkslateblue",
  revCis = "lightskyblue",
  trans = "#BA69BE",
  allcis = "#9BBFE5",
  multimapper = "gray",
  
  randomDB = "gray80"
)

# create alphabet of difference logo
cs = data.frame(chars = c("P",# special case
                          "M", "A","V","I","L","F","Y","W", # hydrophobic
                          "H","R","K",  # basic
                          "D","E",  # acidic
                          "N","Q","S","T","G","C"),  
                cols = c(rep("deeppink",1),
                         rep("orange", 8),
                         rep("seagreen3",3),
                         rep("firebrick",2),
                         rep("dodgerblue",6)),
                groups = c(rep("special",1),
                           rep("hydrophobic", 8),
                           rep("basic",3),
                           rep("acidic",2),
                           rep("polar",6)))

csggplot = make_col_scheme(chars = c("P",# special case
                                     "M", "A","V","I","L","F","Y","W", # hydrophobic
                                     "H","R","K",  # basic
                                     "D","E",  # acidic
                                     "N","Q","S","T","G","C"),  # nucleophilic/polar
                           cols = c(rep("deeppink",1),
                                    rep("orange", 8),
                                    rep("seagreen3",3),
                                    rep("firebrick",2),
                                    rep("dodgerblue",6)),
                           groups = c(rep("special",1),
                                      rep("hydrophobic", 8),
                                      rep("basic",3),
                                      rep("acidic",2),
                                      rep("polar",6)))


quartzFonts(helvetica = c("Helvetica Neue Light","Helvetica Neue Bold",
                          "Helvetica Neue Light Italic", 
                          "Helvetica Neue Bold Italic"))

add.alpha <- function(col, alpha=1){
  if(missing(col))
    stop("Please provide a vector of colours.")
  apply(sapply(col, col2rgb)/255, 2, 
        function(x) 
          rgb(x[1], x[2], x[3], alpha=alpha))  
}




# ----- splitted violin -----
# modified from: https://stackoverflow.com/questions/35717353/split-violin-plot-with-ggplot2
GeomSplitViolin <- ggproto("GeomSplitViolin", GeomViolin, 
                           draw_group = function(self, data, ..., draw_quantiles = NULL) {
                             
                             data <- transform(data, xminv = x - violinwidth * (x - xmin), xmaxv = x + violinwidth * (xmax - x))
                             grp <- data[1, "group"]
                             newdata <- plyr::arrange(transform(data, x = if (grp %% 2 == 1) xminv else xmaxv), if (grp %% 2 == 1) y else -y)
                             # newdata <- plyr::arrange(transform(data, x = if (grp %% 2 == 1) xminv else xmaxv), -y)
                             newdata <- rbind(newdata[1, ], newdata, newdata[nrow(newdata), ], newdata[1, ])
                             newdata[c(1, nrow(newdata) - 1, nrow(newdata)), "x"] <- round(newdata[1, "x"])
                             
                             if (length(draw_quantiles) > 0 & !scales::zero_range(range(data$y))) {
                               stopifnot(all(draw_quantiles >= 0), all(draw_quantiles <=
                                                                         1))
                               quantiles <- ggplot2:::create_quantile_segment_frame(data, draw_quantiles)
                               aesthetics <- data[rep(1, nrow(quantiles)), setdiff(names(data), c("x", "y")), drop = FALSE]
                               aesthetics$alpha <- rep(1, nrow(quantiles))
                               both <- cbind(quantiles, aesthetics)
                               quantile_grob <- GeomPath$draw_panel(both, ...)
                               ggplot2:::ggname("geom_split_violin", grid::grobTree(GeomPolygon$draw_panel(newdata, ...), quantile_grob))
                             }
                             else {
                               ggplot2:::ggname("geom_split_violin", GeomPolygon$draw_panel(newdata, ...))
                             }
                           })

# from: https://stackoverflow.com/questions/35717353/split-violin-plot-with-ggplot2
geom_split_violin <- function(mapping = NULL, data = NULL, stat = "ydensity", position = "identity", ..., 
                              draw_quantiles = NULL, trim = TRUE, scale = "area", na.rm = FALSE, 
                              show.legend = NA, inherit.aes = TRUE) {
  layer(data = data, mapping = mapping, stat = stat, geom = GeomSplitViolin, 
        position = position, show.legend = show.legend, inherit.aes = inherit.aes, 
        params = list(trim = trim, scale = scale, draw_quantiles = draw_quantiles, na.rm = na.rm, ...))
}


# ----- DATA HANDLING -----
# ----- generate all I/L variants of peptides -----
generateILvariants = function(DB) {
  # disentangle positions
  S = DB$substrateSeq %>% unique()
  Q = DB %>%
    tidyr::separate_rows(positions, sep = ";")
  pos = str_split_fixed(Q$positions, "_", Inf)
  pos = apply(pos,2,as.numeric)

  pepSeq = apply(pos,1,function(x) {
    if(!is.na(x[3])) {
      p = paste(substr(S,x[1],x[2]),substr(S,x[3],x[4]),sep = "")
    } else {
      p = substr(S,x[1],x[2])
    }
    return(p)
  })

  F = Q %>%
    dplyr::select(-pepSeq) %>%
    dplyr::mutate(pepSeq = pepSeq) %>%
    dplyr::group_by(substrateID,substrateSeq,productType,spliceType,pepSeq) %>%
    dplyr::reframe(positions = paste(positions,collapse = ";"))

  return(F)
}

# ----- disentangle multi-mappers -----
disentangleMultimappers.Type = function(DB, silent=T) {
  
  pepTable = DB %>%
    filter(productType == "PSP") %>%
    select(pepSeq, spliceType) %>%
    unique() %>%
    mutate(spliceType = sapply(spliceType, function(x){
      
      cnt_types = str_split(x, coll(";"), simplify = T) %>%
        paste() %>%
        unique()
      if (length(cnt_types) == 1) {
        return(cnt_types)
      } else if (any(cnt_types == "trans")) {
        cnt_types = cnt_types[-which(cnt_types == "trans")]
        
        if (length(cnt_types) == 1) {
          return(cnt_types)
        } else {
          return("cis_multi-mapper")
        }
        
      }  else {
        return("cis_multi-mapper")
      }
      
      
    })) %>%
    as.data.frame()
  
  
  DB = left_join(DB %>% select(-spliceType), pepTable) %>%
    mutate(spliceType = ifelse(productType == "PCP", "PCP", spliceType))
  
  return(DB)
}

removeMultimappers.Type = function(DB) {
  k = which(DB$spliceType == "type_multi-mapper")
  
  if (length(k) > 0) {
    DB = DB[-k, ]
  }
  
  return(DB)
}




disentangleMultimappers.AA = function(DB, retSinglePos = T) {
  
  print("DISENTANGLE MULTI-MAPPERS FOR AA AT sP1 AND sP1'")
  
  DB$AAmultimapper = "no"
  
  # only considering PSP multi-mappers
  k = which(str_detect(DB$positions, coll(";")) & !str_detect(DB$productType, "PCP"))
  
  if (length(k) > 0) {
    DB_mm = DB[k, ]
    DB_nomm = DB[-k, ]
    
    pb = txtProgressBar(min = 0, max = nrow(DB_mm), style = 3)
    for (r in 1:nrow(DB_mm)) {
      
      setTxtProgressBar(pb, r)
      
      cnt_pos = strsplit(DB_mm$positions[r], ";") %>%
        unlist() %>%
        str_split_fixed(pattern = coll("_"), n = Inf)
      
      sP1 = str_sub(DB_mm$substrateSeq[r], start = cnt_pos[, 2], end = cnt_pos[, 2])
      sP1d = str_sub(DB_mm$substrateSeq[r], start = cnt_pos[, 3], end = cnt_pos[, 3])
      
      if ((length(unique(sP1)) > 1) | (length(unique(sP1d)) > 1)) {
        DB_mm$AAmultimapper[r] = "yes"
        
      } else if (retSinglePos) {
        DB_mm$positions[r] = paste(cnt_pos[1, c(1:4)], collapse = "_")
      }
      
    }
    
    DB = rbind(DB_nomm, DB_mm) %>%
      as.data.frame()
  }
  
  
  # beep("mario")
  return(DB)
}

removeMultimappers.AA = function(DB) {
  print("REMOVE PEPTIDES THAT ARE MULTI-MAPPERS IN TERMS OF AA AT sP1 OR sP1'")
  
  k = which(DB$AAmultimapper == "yes")
  
  if (length(k) > 0) {
    DB = DB[-k, ]
  }
  
  return(DB)
}

# ----- extract amino acids -----
extract_aminoacids = function(tbl, onlyValidSeq = F, coordinates = F){
  
  tbl$spliceType[(tbl$spliceType == "") | (is.na(tbl$spliceType))] = "PCP"
  
  # table with position indices
  pos = str_split_fixed(tbl$positions, coll("_"), Inf) %>% as.data.frame()
  pos = apply(pos, 2, function(x){as.numeric(as.character(x))})
  
  pcp = which(tbl$spliceType == "PCP")
  psp = which(tbl$spliceType != "PCP")
  
  
  # PCPs
  pcpTBL = sapply(PCPpos, function(x){
    if (onlyValidSeq) {
      substr(tbl$pepSeq[pcp], start = pos[pcp,2]-pos[pcp,1]+1+x, stop = pos[pcp,2]-pos[pcp,1]+1+x) 
    } else if (coordinates) {
      pos[pcp,2]+x
    } else {
      substr(tbl$substrateSeq[pcp], start = pos[pcp,2]+x, stop = pos[pcp,2]+x)
    }
  }) %>%
    as.data.frame() %>%
    mutate(spliceType = tbl$spliceType[pcp],
           positions = tbl$positions[pcp],
           pepSeq = tbl$pepSeq[pcp],
           substrateID = tbl$substrateID[pcp],
           substrateSeq = tbl$substrateSeq[pcp])
  
  
  # PSPs
  pspSR1TBL = sapply(SR1pos, function(x){
    if (onlyValidSeq & "sr1" %in% names(tbl)) {
      substr(tbl$sr1[psp], start = pos[psp,2]-pos[psp,1]+1+x, stop = pos[psp,2]-pos[psp,1]+1+x)
    } else if (coordinates) {
      pos[psp,2]+x
    } else {
      substr(tbl$substrateSeq[psp], start = pos[psp,2]+x, stop = pos[psp,2]+x)
    }
  }) %>%
    as.data.frame() %>%
    mutate(spliceType = tbl$spliceType[psp],
           positions = tbl$positions[psp],
           pepSeq = tbl$pepSeq[psp],
           substrateID = tbl$substrateID[psp],
           substrateSeq = tbl$substrateSeq[psp])
  
  
  pspSR2TBL = sapply(SR2pos, function(x){
    if (onlyValidSeq & "sr2" %in% names(tbl)) {
      substr(tbl$sr2[psp], start = x+1, stop = x+1)
    } else if (coordinates) {
      pos[psp,3]+x
    } else {
      substr(tbl$substrateSeq[psp], start = pos[psp,3]+x, stop = pos[psp,3]+x)
    }
  }) %>%
    as.data.frame() %>%
    mutate(spliceType = tbl$spliceType[psp],
           positions = tbl$positions[psp],
           pepSeq = tbl$pepSeq[psp],
           substrateID = tbl$substrateID[psp],
           substrateSeq = tbl$substrateSeq[psp])
  
  # merge all tables
  pspTBL = cbind(pspSR1TBL[,names(SR1pos)], pspSR2TBL)
  
  pcpPlaceholder = matrix("", length(pcp), length(SR2pos)) %>%
    as.data.frame()
  names(pcpPlaceholder) = SRnames[! SRnames %in% names(PCPpos)]
  pcpTBL2 = cbind(cbind(pcpTBL[,names(PCPpos)], pcpPlaceholder)[, c(names(SR1pos), names(SR2pos))],
                  pcpTBL[,which(!names(pcpTBL) %in% names(PCPpos))])
  
  TBL = rbind(pspTBL, pcpTBL2) %>% as.data.frame()
  return(TBL)
}




# ----- I/L redundancy -----
ILredundancy = function(DB) {
  
  print("REPLACE ALL ISOLEUCINS BY LEUCINS")
  
  DB$pepSeq = str_replace_all(DB$pepSeq, "I", "L")
  DB$substrateSeq = str_replace_all(DB$substrateSeq, "I", "L")
  
  return(DB)
}

# ----- unique peptides -----
uniquePeptides = function(DB) {
  print("UNIQUE PEPTIDES")
  
  DB = DB %>%
    distinct(substrateID, substrateSeq, pepSeq, productType,
             #spliceType,
             positions, .keep_all = T)
  return(DB)
}

# ----- MAPPING OF PEPTIDE SEQUENCES -----
# ----- PCP location -----
locate_PCP <- function(peps, subSeq){
  
  # check which peptides are detected as PCP
  k = sapply(peps, function(x){
    grepl(pattern=x,x=subSeq)
  }) %>% which()
  
  # for those that are, do mapping
  if(length(k)>0){
    
    pcp = lapply(k, function(j){
      
      # get all positions of a PCP to account foe multi-mapping
      pos = str_locate_all(subSeq,peps[j]) %>%
        plyr::ldply() %>%
        as.data.frame()
      id = data.frame(pepSeq = peps[j], pos1 = pos$start, pos2 = pos$end)
      
      return(id)
    }) %>%
      plyr::ldply()
    
  } else {
    pcp <- NULL
  }
  return(pcp %>% select(-.id))
}

# ----- PSP location -----
locate_PSP <- function(PSPpeps, subSeq) {
  
  # sort by N-mers
  Nmers = nchar(PSPpeps) %>% unique() %>% sort()
  
  allPSPpos = list()
  for (k in 1:length(Nmers)) {
    N = Nmers[k]
    
    # get all possible splits of N into two splice-reactants
    q = suppressMessages(tidyr::crossing(c(1:N), c(1:N), .name_repair = "unique"))
    q = q[which(rowSums(q) == N), ] %>%
      as.matrix()
    
    # get all PSP candidates of length N and split them
    cntCand = PSPpeps[nchar(PSPpeps) == N]
    
    PSPpos = lapply(cntCand, function(s){
      
      # get all SRs
      P = strsplit(s,"") %>% unlist()
      cntSRs = sapply(seq(1,nrow(q)), function(i){
        srs = data.frame(pepSeq = s,
                         SR1 = paste(P[1:q[i,1]], collapse = ""),
                         SR2 = paste(P[(q[i,1]+1):N], collapse = ""))
      }) %>% 
        t() %>% 
        as.data.frame()
      
      return(cntSRs)
    }) %>%
      plyr::ldply()
    
    PSPpos = PSPpos %>%
      mutate(pepSeq = unlist(pepSeq),
             SR1 = unlist(SR1),
             SR2 = unlist(SR2))
    
    # map SRs as PCP
    sr1_loc = locate_PCP(peps = PSPpos$SR1, subSeq = subSeq) %>%
      rename(SR1 = pepSeq)
    
    sr2_loc = locate_PCP(peps = PSPpos$SR2, subSeq = subSeq) %>%
      rename(SR2 = pepSeq,
             pos3 = pos1, pos4 = pos2)
    
    POS = suppressMessages(left_join(PSPpos, sr1_loc)) %>%
      na.omit()
    POS = suppressMessages(left_join(POS, sr2_loc)) %>%
      na.omit() %>%
      unique()
    
    # get splice types
    POS$type = NA
    intv = POS$pos3-POS$pos2
    
    POS$type[intv > 0 & POS$pos3 > POS$pos2] = "cis"
    POS$type[intv <= 0 & POS$pos4 < POS$pos1] = "revCis"
    POS$type[intv <= 0 & POS$pos4 >= POS$pos1] = "trans"
    
    # collapse to single assignment per peptide
    POS$allpos = do.call(paste, c(POS[c("pos1","pos2","pos3","pos4")], sep = "_"))
    
    POS = POS %>%
      group_by(pepSeq) %>%
      summarise(spliceType = paste(type, collapse = ";"),
                positions = paste(allpos, collapse = ";"))
    
    allPSPpos[[k]] = POS
  }
  
  pspMAP = plyr::ldply(allPSPpos) %>% as.data.frame()
  return(pspMAP)
}


# ----- actual mapping -----
locate_peps = function(subSeq, peps) {
  
  # account for I/L redundancy
  subSeq = gsub("I","L",subSeq)
  peps = gsub("I","L",peps)
  
  # PCP mapping
  pcpMAP = locate_PCP(peps, subSeq)
  pcpMAP = pcpMAP %>%
    mutate(positions = do.call(paste, c(pcpMAP[c("pos1","pos2")], sep = "_")),
           spliceType = "PCP",
           productType = "PCP") %>%
    select(-pos1, -pos2)
  
  # remove all PSPs from pool of spliced peptides
  PSPpeps = peps[!peps %in% pcpMAP$pepSeq]
  pspMAP = locate_PSP(PSPpeps, subSeq) %>%
    mutate(productType = "PSP")
  
  MAP = rbind(pcpMAP, pspMAP)
  return(MAP)
}


# re-map peptides
mappingHR = function(DB) {
  
  target = c("positions", "spliceType", "productType")
  target = target[target %in% names(DB)]
  if (length(target) > 0) {
    DB = DB[, -which(colnames(DB) %in% target)]
  }
  
  subSeqs = DB$substrateSeq %>% unique()
  d = list()
  
  pb = txtProgressBar(min = 0, max = length(subSeqs), style = 3)
  for(i in 1:length(subSeqs)){
    setTxtProgressBar(pb, i)
    
    MAP = locate_peps(subSeq = subSeqs[i],
                      peps = unique(DB$pepSeq[DB$substrateSeq == subSeqs[i]])) %>%
      mutate(substrateSeq = subSeqs[i])
    
    d[[i]] = MAP
  }
  d = plyr::ldply(d)
  
  DB = suppressMessages(left_join(DB, d)) %>%
    as.data.frame()
  
  return(DB)
}

getPositions <- function(seq,substrate){
  
  
  
  #########################
  # PCP
  #########################
  
  
  l = nchar(seq)
  
  k = which((grepl(pattern=seq,x=substrate)==TRUE))
  if(length(k)>0){
    
    pcp = numeric()
    
    for(j in 1:length(k)){
      a = substrate
      x = strsplit(a,split=seq)[[1]]
      nn = nchar(x)
      n1 = rep(NA,(length(nn)-1))
      n2 = rep(NA,(length(nn)-1))
      for(r in 1:(length(x)-1)){
        n1[r] = sum(nn[1:r])+(r-1)*nchar(seq)+1
        n2[r] = n1[r]+nchar(seq)-1
      }
      pcp = rbind(pcp,cbind(n1,n2))
    }
    return(pcp)
  }
  
  
  
  #########################
  # PSP
  #########################
  
  
  
  
  ll = nchar(seq)
  
  
  pept = unlist(seq)
  N = nchar(seq)
  
  # split peptides to P matrix
  P = strsplit(pept,split="")[[1]]
  
  # get permutations of length N
  x = c(1:N)
  y = c(1:N)
  z = as.vector(outer(x,y,paste,sep="_"))
  q = matrix(NA,length(z),2)
  for(i in 1:length(z)){
    q[i,] = as.numeric(strsplit(z[i],split="_")[[1]])
  }
  
  qs = apply(q,1,sum)
  k = which(qs==N)
  q = q[k,]
  
  # loop over all peptides
  res2 <- list()
  res1 <- list()
  
  psp <- list()
  
  psp <- list()
  res1 <- list()
  res2 <- list()
  
  # generate all strings for searches
  S = matrix(NA,dim(q)[1],2)
  for(i in 1:dim(q)[1]){
    S[i,1] = paste(P[1:q[i,1]],sep="",collapse="")
    S[i,2] = paste(P[(q[i,1]+1):N],sep="",collapse="")
  }
  
  # search each entry in prot for the two corresponding fragments and extract acc and positions
  
  for(i in 1:dim(S)[1]){
    
    psp[[i]] <- list()
    res1[[i]] = which((grepl(pattern=S[i,1],x=substrate)==TRUE))
    res2[[i]] = which((grepl(pattern=S[i,2],x=substrate)==TRUE))
    
    kk = which(res1[[i]]%in%res2[[i]])
    k = res1[[i]][kk]
    if(length(k)>0){
      
      for(j in 1:length(k)){
        
        a = substrate
        
        
        x = strsplit(a,split=S[i,1])[[1]]
        nn = nchar(x)
        n1 = rep(NA,(length(nn)-1))
        n2 = rep(NA,(length(nn)-1))
        for(r in 1:(length(x)-1)){
          n1[r] = sum(nn[1:r])+(r-1)*nchar(S[i,1])+1
          n2[r] = n1[r]+nchar(S[i,1])-1
        }
        #check if substrate Cterm==S[i,1]
        len = nchar(S[i,1])
        y = paste(strsplit(a,split="")[[1]][(nchar(a)-len+1):nchar(a)],collapse="")
        if(S[i,1]==y){
          n1 = c(n1,nchar(a)-len+1)
          n2 = c(n2,nchar(a))
        }
        tmp = unique(apply(cbind(n1,n2),1,paste,collapse="_"))
        tmp2 = matrix(as.numeric(unlist(strsplit(tmp,split="_"))),length(tmp),2,byrow=TRUE)
        n1 = tmp2[,1]
        n2 = tmp2[,2]
        
        x = strsplit(a,split=S[i,2])[[1]]
        nn = nchar(x)
        n3 = rep(NA,(length(nn)-1))
        n4 = rep(NA,(length(nn)-1))
        for(r in 1:(length(x)-1)){
          n3[r] = sum(nn[1:r])+(r-1)*nchar(S[i,2])+1
          n4[r] = n3[r]+nchar(S[i,2])-1
        }
        #check if substrate Cterm==S[i,2]
        len = nchar(S[i,2])
        y = paste(strsplit(a,split="")[[1]][(nchar(a)-len+1):nchar(a)],collapse="")
        if(S[i,2]==y){
          n3 = c(n3,nchar(a)-len+1)
          n4 = c(n4,nchar(a))
        }
        tmp = unique(apply(cbind(n3,n4),1,paste,collapse="_"))
        tmp2 = matrix(as.numeric(unlist(strsplit(tmp,split="_"))),length(tmp),2,byrow=TRUE)
        n3 = tmp2[,1]
        n4 = tmp2[,2]
        
        # get all internal combinations and keep only those with intv<=25
        
        z = as.vector(outer(n2,n3,paste,sep="_"))
        y = matrix(NA,length(z),2)
        for(zz in 1:length(z)){
          y[zz,] = as.numeric(strsplit(z[zz],split="_")[[1]])
        }
        intv = y[,2]-y[,1]-1
        x = which(intv<0)
        if(length(x)>0){ intv[x] = y[x,1]-y[x,2]+1-nchar(S[i,1])-nchar(S[i,2]) }
        x = which(intv<0)
        #    if(length(x)>0){ intv[x] = 1000 }
        
        select = which(intv<=5000)
        
        nnn = length(select)
        if(nnn>0){
          psp[[i]][[j]] = matrix(NA,nnn,5)
          
          for(j2 in 1:nnn){
            
            psp[[i]][[j]][j2,] = c(pept,y[select[j2],1]-nchar(S[i,1])+1,y[select[j2],1],y[select[j2],2],y[select[j2],2]+nchar(S[i,2])-1)
          }
        }
        
      }
      
    }
    
    
  }
  
  # unlist results and return as unique matrix with all possible explanations as rows
  x = unlist(psp)
  #  psp = matrix(x,length(x)/5,5,byrow=FALSE)
  
  res = numeric()
  for(i in 1:length(psp)){
    if(length(psp[[i]])>0){
      for(j in 1:length(psp[[i]])){
        res = rbind(res,psp[[i]][[j]])
      }
    }
  }
  
  
  # print(res)
  
  return(res)
  
  
}


# re-map peptides
mapping = function(DB) {
  
  d = DB %>%
    dplyr::select(substrateSeq, productType, spliceType, pepSeq) %>%
    dplyr::distinct()
  
  # pb = txtProgressBar(min = 0, max = dim(d)[1], style = 3, file="tmp")
  
  for(i in 1:dim(d)[1]){
    # setTxtProgressBar(pb, i)
    
    if(!(d$productType[i]=="CONT" | d$productType[i]=="CONT_synError")){
      s = gsub("I","L",as.vector(d$pepSeq[i]))
      substrate = gsub("I","L",as.vector(d$substrateSeq[i]))
      x = getPositions(s,substrate)
      
      
      #PCP
      if(dim(x)[2]==2){
        d$positions[i] = paste(apply(x,1,paste,collapse="_"),collapse=";")
      }
      
      
      #PSP
      if(dim(x)[2]>2){
        # print(i)
        if(dim(x)[2]==5 & dim(x)[1]>1){
          d$positions[i] = paste(apply(x[,-1],1,paste,collapse="_"),collapse=";")
        }
        if(dim(x)[2]==5 & dim(x)[1]==1){
          d$positions[i] = paste(x[,-1],collapse="_")
        }
        
        types = rep("cis",dim(x)[1])
        
        intv = as.numeric(x[,4])-as.numeric(x[,3])
        k = which(intv<=0)
        if(length(k)>0){
          types[k] = "revCis"
          
          k2 = which(as.numeric(x[k,2])<=as.numeric(x[k,5]) & as.numeric(x[k,3])>=as.numeric(x[k,4]))
          if(length(k2)>0){
            types[k[k2]] = "trans"
          }
          
        }
        
        d$spliceType[i] = paste(types,collapse=";")
      }
    }
    
  }
  
  DB$productType = NULL
  DB$spliceType = NULL
  DB$positions = NULL
  
  
  DB = left_join(DB, d) %>%
    as.data.frame()
  
  # re-assign the product type
  pos = strsplit(DB$positions, "_")
  wr = which(sapply(pos, length) == 2 & str_detect(DB$productType, "PSP"))
  if (length(wr) > 0) {
    DB$productType[wr] = str_replace(DB$productType[wr], "PSP", "PCP")
  }
  
  xr = which(sapply(pos, length) == 4 & str_detect(DB$productType, "PCP"))
  if (length(xr) > 0) {
    DB$productType[xr] = str_replace(DB$productType[xr], "PCP", "PSP")
  }
  
  return(DB)
}


