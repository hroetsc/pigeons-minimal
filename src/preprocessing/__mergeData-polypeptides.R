### protein PCPS ###
# description:  merge aSPIre outputs and produce basic statistic
# input:        aSPIre output
# output:       table with all peptide sequences
# author:       HPR

library(dplyr)
library(stringr)

source("src/invitroSPI_utils.R")


### INPUT ###
overview = readxl::read_excel("data/polypeptides_datasets.xlsx")
finalK = list.files("/Volumes/DATA16040/DATA/SPIce_QSB/QUANTITATIVE/aSPIRE-polypeptides/", pattern = "Kinetics.csv",
                    recursive = T, full.names = T)

finalK


### MAIN PART ###
protNames = str_extract(finalK, "[^/]+(?=_finalKinetics.csv)")
# finalK = finalK[protNames %in% overview$protein_name]

Kinetics = lapply(finalK, function(x){
  read.csv(x,stringsAsFactors = F)
})
names(Kinetics) = protNames
Kinetics = Kinetics %>%
  plyr::ldply() %>%
  as.data.frame() %>%
  rename(protein_name = .id)

Kinetics$spliceType[is.na(Kinetics$spliceType)] = "PCP"
Kinetics = na.omit(Kinetics)
Kinetics = Kinetics %>%
  disentangleMultimappers.Type()


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
  dplyr::distinct(substrateID, pepSeq, .keep_all = T) %>%
  left_join(overview)

table(KU$productType)
table(KU$protein_name,KU$productType)


### OUTPUT ###
save(Kinetics, file = "data/polypeptide_aSPIRE.RData")
write.csv(Kinetics, "data/polypeptide_allPeptides_aSPIRE.csv", row.names = F)

