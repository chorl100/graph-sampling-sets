import warnings

from src.gsp.metrics import sampling_centrality, z_score

warnings.simplefilter(action='ignore', category=FutureWarning)

import networkx as nx
import pandas as pd


def graph_from_rin(rin_file: str) -> nx.Graph:
    # load RIN
    edgelist_df = pd.read_csv(f"./data/RINs/{rin_file}.pdb.rin", sep="\t")
    # create graph
    return nx.from_pandas_edgelist(edgelist_df, "Residue1", "Residue2", "Distance")


def calculate_sampling_centralities(mutants, k, p):
    print(mutants)
    for pdb in mutants:
        print("Mutant:", pdb)
        G = graph_from_rin(pdb)
        # relabel nodes from residues to IDs
        G_relabelled = nx.convert_node_labels_to_integers(G)
        W = nx.adjacency_matrix(G_relabelled)
        centralities = sampling_centrality(G_relabelled, W, k, p)
        # compute Z-values
        z_scores = z_score(centralities)
        # write results to .csv file
        centrality_df = pd.DataFrame({
            "NodeID": range(len(G.nodes)),
            "Residue": G.nodes,
            "Centrality_SCA": centralities,
            "Z_SCA": z_scores
        })
        centrality_df.to_csv(f"./data/RIN_SC/{pdb}.pdb.sc", sep="\t", index=False)


def main():
    with open("./data/list_PDB_single_mutations.lst", "r") as f:
        mutants = f.read().splitlines()
    k = 100
    p = 0.4
    calculate_sampling_centralities(mutants, k, p)


if __name__ == '__main__':
    main()
