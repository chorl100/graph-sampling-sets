import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from venn import pseudovenn

from src.bioinfo.utils import precision, npv, recall, specificity, accuracy, f1_score


def main():
    # get mapping from 1-letter to 3-letter amino acid names
    amino_acids_3_letters = ["Ala", "Arg", "Asp", "Asn", "Cys", "Glu", "Gln", "Gly", "His", "Ile", "Leu", "Lys", "Met",
                             "Phe", "Pro", "Ser", "Thr", "Trp", "Tyr", "Val"]
    amino_acids_1_letter = ["A", "R", "D", "N", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W",
                            "Y", "V"]
    amino_acids = pd.DataFrame({"3_letters": amino_acids_3_letters, "1_letter": amino_acids_1_letter})

    # read dataset
    print("Reading SKEMPI 2...")
    skempi_df = read_skempi2()
    mutants_df = get_mutants(skempi_df)
    mutants_df["mut_cleaned"] = reformat_skempi(mutants_df, amino_acids)

    print("Creating results table...")
    rows = list()
    # for each mutated residue
    for i in range(len(mutants_df)):
        # get the mutation, location, centralities, ddG and add them all to a result table
        row = mutants_df.iloc[i]
        pdb = row["PDB"][:4]
        residue = row["mut_cleaned"]
        # add centrality value and Z-score
        df_sc = pd.read_csv(f"./data/bio/RIN_SC/{pdb}.pdb.sc", sep="\t")
        idx = df_sc["Residue"] == residue
        current_centrality = df_sc.loc[idx]["Centrality_SCA"].iloc[0]
        current_z_score = df_sc.loc[idx]["Z_SCA"].iloc[0]
        rows.append([row["PDB"], residue, row["mut_location"], current_centrality, current_z_score, row["ddG"]])
    results_dir = "./data/bio/results"
    os.makedirs(results_dir, exist_ok=True)
    results = pd.DataFrame(rows, columns=["PDB", "Residue", "Location", "centrality_SCA", "Z_sca", "ddG"])
    results.to_csv(f"{results_dir}/centralities_Z_ddG_5A_water3.5_single_false.csv", sep="\t", index=False)

    print("Calculating centrality statistics...")
    stats = calculate_statistics(results)
    save_statistics(stats, results_dir)

    print("Storing list of found residues...")
    # get central residues
    found_residues_df = get_found_residues(results)
    found_residues_df = found_residues_df["PDB"] + "__" + found_residues_df["Residue"]
    found_residues_df.to_csv(f"{results_dir}/sca_5A_water3.5_single_false_ddG_supeq0.592__found_by_centrality.lst",
                             sep="\t", index=False, header=False)
    # store residues in dict (as a set because venn requires a dict of sets)
    found_residues = {"SCA": set(found_residues_df)}
    # load residues found by other centrality measures and add them to the dict
    found_residues.update(load_found_residues(results_dir))
    # delete page rank centrality
    del found_residues["PRA"]
    print("Plotting venn diagram...")
    plot_venn_diagram(found_residues, results_dir)


def read_skempi2() -> pd.DataFrame:
    # load SKEMPI 2 data
    skempi2 = pd.read_csv("./data/bio/skempi_v2+ddG.csv", sep="\t")
    return skempi2


def get_mutants(skempi2_df) -> pd.DataFrame:
    # keep only single mutations
    single_mutants = skempi2_df.apply(lambda row: ',' not in row["Mutation(s)_PDB"], axis=1)
    mutations_df = skempi2_df.loc[single_mutants]

    # keep only relevant columns
    mutations_df = mutations_df[["#Pdb", "Mutation(s)_cleaned", "iMutation_Location(s)", "ddG (kcal mol^(-1))"]]
    # rename columns
    mapper = {"#Pdb": "PDB", "Mutation(s)_cleaned": "mut_cleaned",
              "iMutation_Location(s)": "mut_location", "ddG (kcal mol^(-1))": "ddG"}
    mutations_df = mutations_df.rename(columns=mapper)

    # remove duplicate residues; keep residue with highest ddG
    pdb_with_residue = mutations_df["PDB"] + "__" + mutations_df["mut_cleaned"]
    mutations_df["PDBwithRes"] = pdb_with_residue
    pdb_with_residues_sets = {"PDBwithRes": list(), "ddG": list()}
    for i in mutations_df.index:
        residue = mutations_df["mut_cleaned"].loc[i]
        # remove the mutated residue
        mut_wo_residue = residue[:-1]
        pdb_with_residues_sets_candidate = mutations_df["PDB"].loc[i] + "__" + mut_wo_residue
        current_ddG = mutations_df["ddG"].loc[i]
        try:
            occurence = pdb_with_residues_sets["PDBwithRes"].index(pdb_with_residues_sets_candidate)
        except ValueError:
            occurence = None
        if occurence is not None:
            old_ddG = pdb_with_residues_sets["ddG"][occurence]
            if abs(current_ddG) > abs(old_ddG):
                pdb_with_residues_sets["ddG"][occurence] = current_ddG
                orig_idx = mutations_df["PDBwithRes"].str.startswith(pdb_with_residues_sets_candidate)
                # drop old mutant
                mutations_df.drop(mutations_df.index[orig_idx][0], inplace=True)
            else:
                # drop new mutant
                mutations_df.drop(i, inplace=True)
        else:
            pdb_with_residues_sets["PDBwithRes"].append(pdb_with_residues_sets_candidate)
            pdb_with_residues_sets["ddG"].append(current_ddG)
    return mutations_df


def skempi_to_central(mutation: str, amino_acids):
    """
    Convert format for a residue from SKEMPI 2 to centralities format.
    """
    AA1Letter = mutation[0]
    AA3Letters = amino_acids["3_letters"].loc[amino_acids["1_letter"] == AA1Letter].iloc[0]
    chain_id = mutation[1]
    index = mutation[2:-1]
    return AA3Letters + index + "." + chain_id


def central_to_skempi(mutation: str, amino_acids):
    """
    Convert format for a residue from centralities format to SKEMPI 2 format.
    """
    AA3Letters = mutation[:3]
    index = mutation.split(".")[0][3:]
    chain_id = mutation[-1]
    AA1Letter = amino_acids["1_letter"].loc[amino_acids["3_letters"] == AA3Letters].iloc[0]
    return AA1Letter + chain_id + index


# reformat data
def reformat_skempi(skempi, amino_acids):
    """Convert skempi mutation column to new format."""
    mutations = skempi["mut_cleaned"]
    return mutations.apply(lambda x: skempi_to_central(x, amino_acids))


def get_found_residues(results, thres_ddg=0.592, thres_z_score=2.) -> pd.DataFrame:
    central_residues = (abs(results["ddG"]) >= thres_ddg) & (results["Z_sca"] >= thres_z_score)
    return results[["PDB", "Residue"]].loc[central_residues]


def calculate_statistics(results, thres_ddg=0.592, thres_z_score=2.):
    tp = np.count_nonzero((abs(results["ddG"]) >= thres_ddg) & (results["Z_sca"] >= thres_z_score))
    tn = np.count_nonzero((abs(results["ddG"]) < thres_ddg) & (results["Z_sca"] < thres_z_score))
    fp = np.count_nonzero((abs(results["ddG"]) < thres_ddg) & (results["Z_sca"] >= thres_z_score))
    fn = np.count_nonzero((abs(results["ddG"]) >= thres_ddg) & (results["Z_sca"] < thres_z_score))

    prec = precision(tp, fp)
    negprec = npv(tn, fn)
    rec = recall(tp, fn)
    spec = specificity(tn, fp)
    acc = accuracy(tp, tn, fp, fn)
    f1 = f1_score(tp, fp, fn)
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "Precision": prec, "negprec": negprec, "Recall": rec, "Specificity": spec, "Accuracy": acc, "F1 score": f1}


def save_statistics(stats: dict, dir: str):
    df = pd.DataFrame.from_dict(stats, orient="index", columns=["sca"])
    df.to_csv(f"{dir}/centralities_statistics_5A_water3.5_single_false__ddG_0.592.csv", sep="\t")


def load_found_residues(dir: str):
    centralities = ["bca", "cca", "dca", "eca", "pra", "rca"]
    found_residues = dict()
    for centrality in centralities:
        file = f"{dir}/{centrality}_5A_water3.5_single_false_ddG_supeq0.592__found_by_centrality.lst"
        with open(file, "r") as f:
            residues = f.read().splitlines()
        found_residues[centrality.upper()] = set(residues)
    return found_residues


def plot_venn_diagram(found_residues: dict, dir: str):
    filename = "venn_5A_water3.5_single_false_ddG_supeq0.592.pdf"
    plt.figure()
    pseudovenn(found_residues, cmap="plasma", hint_hidden=False)
    plt.savefig(f"{dir}/{filename}")


if __name__ == '__main__':
    main()
