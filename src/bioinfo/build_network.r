library(bio3d)
library(bio3d.cna)
library(bio3d.nma)
library(igraph)
pdb <- read.pdb("3FJQ")
modes <- nma(pdb)
cij <- dccm(modes)
net <- cna(cij, cutoff.cij=0.3)
print(net)
# Summary information as a table 
x <- summary(net)
# Plot both the ‘full’ all-residue network and simplified community network
plot(net, pdb, full = TRUE, vertex.label.cex=0.7)
plot(net, pdb)
write.matrix(net$cij, file="net.csv")
write.matrix(net$community.cij, file="communities.csv")
