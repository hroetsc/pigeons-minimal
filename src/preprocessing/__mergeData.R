### protein PCPS ###
# description:  merge aSPIre outputs and produce basic statistic
# input:        aSPIre output
# output:       table with all peptide sequences
# author:       HPR

library(dplyr)
library(stringr)
library(data.table)
source("src/_utils.R")


### INPUT ###
finalK = list.files("/Volumes/DATA16040/DATA/SPIce_QSB/QUANTITATIVE/aSPIRE-proteins/PCPSpredictor_240103/", pattern = "Kinetics.csv",
                    recursive = T, full.names = T)

# finalK = finalK[-grep("FFH_6p25", finalK)]
# finalK = finalK[-grep("FFH_12p5", finalK)]
finalK


### MAIN PART ###
Kinetics = lapply(finalK, function(x){
  fread(x)
}) %>%
  plyr::ldply() %>%
  as.data.frame()


Kinetics = na.omit(Kinetics)
Kinetics = Kinetics %>%
  disentangleMultimappers.Type()
Kinetics$spliceType[is.na(Kinetics$spliceType)] = "PCP"


# remove peptides that have 0 intensity @ 4 hours
RemPeps = Kinetics %>%
  tidyr::separate_rows(digestTimes, intensities, sep=";") %>%
  rename(digestTime = digestTimes,
         intensity = intensities) %>%
  mutate(intensity = as.numeric(intensity),
         digestTime = as.numeric(digestTime)) %>%
  filter((digestTime == 4 & substrateID != "aSyn") | (digestTime == 3 & substrateID == "aSyn")) %>%
  group_by(pepSeq, digestTime) %>%
  summarise(remove = ifelse(all(is.na(intensity)), T, F)) %>%
  filter(remove)
RemPeps = RemPeps$pepSeq %>% unique()
print(length(RemPeps))

Kinetics = Kinetics[-which(Kinetics$pepSeq %in% RemPeps), ]


KU = Kinetics %>%
  dplyr::distinct(substrateID, pepSeq, .keep_all = T)

table(KU$productType)
table(KU$substrateID,KU$productType)


### OUTPUT ###
save(Kinetics, file = "data/aSPIRE.RData")
write.csv(Kinetics, "data/allPeptides_aSPIRE.csv", row.names = F)


