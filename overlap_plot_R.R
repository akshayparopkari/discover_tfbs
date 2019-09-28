library("ggplot2")

orfs <- round((28689/42385) * 100, digits = 2)
orfs.ntars <- round((31360/42385) * 100, digits = 2)

overlap.data <- data.frame("Feature" = c("ORFs", "ORFs_nTARs"),
                           "Overlap.Percent" = c(orfs, orfs.ntars))

ggplot(data=overlap.data, aes(x=Feature, y=Overlap.Percent, fill=Feature)) +
  geom_bar(colour="black", stat="identity", width = 0.25) +
  geom_text(aes(label=Overlap.Percent), vjust=-0.4, size=5) +
  theme(axis.text = element_blank(), axis.ticks = element_blank(),
        legend.position="right", legend.title = element_blank())
