import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sspa

class SimulateData:
    """
    Class for creation of semi-synthetic multi-omics data with known pathway signals
    """    
    def __init__(self, input_data, metadata, pathways, enriched_paths):
        """_summary_

        Args:
            input_data (list): List of omics data as pandas DataFrames.
            metadata (list): List of metadata corresponding to each omics dataset.
            pathways (dict): Dictionary of pathways.
            enriched_paths (list): List of pathways to enrich.
        """
        self.input_data = input_data
        self.input_data_filt = []
        self.metadata = metadata
        self.metadata_filt = []
        self.metadata_perm = []
        self.pathways = pathways
        self.enriched_paths = enriched_paths
        self.enriched_mols = []
        self.sim_data = []

    def sample_permutation(self, n_clusters):
        """
        Permute sample class labels to create synthetic clusters

        Args:
            n_clusters (int): Number of primary clusters to create
        """
        
        # only filter if multi-omics data supplied
        if len(self.input_data) > 1:
            # get sample intersection between n omics dataframes
            intersect_samples = list(set.intersection(*[set(df.index.tolist()) for df in self.input_data]))

            # filter each dataframe to contain same samples
            self.input_data_filt = [df.loc[intersect_samples, :] for df in self.input_data]

            n_samples = len(intersect_samples)

            # filter metadata
            self.metadata_filt = [i[i.index.isin(intersect_samples)] for i in self.metadata]
        else:
            self.input_data_filt = [df for df in self.input_data]
            self.metadata_filt = [i for i in self.metadata]

        # Generate synthetic primary cluster labels
        primary_labels = np.tile(np.arange(n_clusters), int(np.ceil(len(self.metadata_filt[0]) / n_clusters)))[:len(self.metadata_filt[0])]
        
        # permute sample metadata
        rng = np.random.default_rng()
        self.metadata_perm = rng.permutation(primary_labels)

    def enrich_paths_base(self, effect_sizes, effect_type='var', input_type='log'):
        """
        Enrich specified pathways in n omics datasets 

        Args:
            effect_sizes (list): List of effect sizes for each primary cluster
        """
        n_clusters = len(effect_sizes)

        # fill in the data with permuted samples
        self.sample_permutation(n_clusters)

        # get metabolites and proteins to be enriched from each pathway 
        enriched_mols = list(sum([self.pathways[i] for i in self.enriched_paths], []))
        self.enriched_mols = enriched_mols
        enriched_proteins = [i for i in enriched_mols if i.startswith("P|O|Q")]
        enriched_metabs = np.setdiff1d(enriched_mols, enriched_proteins)

        for df in self.input_data_filt:
            df_enriched = df.copy()

            for cluster_id, effect in enumerate(effect_sizes):
                indices = np.argwhere(self.metadata_perm == cluster_id).ravel()
                
                if input_type == 'zscore':
                    if effect_type == 'constant':
                        df_enriched.iloc[indices, df_enriched.columns.isin(enriched_mols)] *= (1 + effect)
                    if effect_type == 'var':
                        sd = df_enriched.iloc[:, df_enriched.columns.isin(enriched_mols)].std()
                        alpha = effect / sd
                        df_enriched.iloc[indices, df_enriched.columns.isin(enriched_mols)] *= (1 + alpha)

                if input_type == 'log':
                    if effect_type == 'constant':
                        df_enriched.iloc[indices, df_enriched.columns.isin(enriched_mols)] += effect
                    if effect_type == 'var':
                        sd = df_enriched.iloc[:, df_enriched.columns.isin(enriched_mols)].std()
                        alpha = effect / sd
                        df_enriched.iloc[indices, df_enriched.columns.isin(enriched_mols)] += alpha

            # adding group labels
            df_enriched["Group"] = self.metadata_perm
            self.sim_data.append(df_enriched)
        
        return self.sim_data

# Example usage:
# simulated_dset = SimulateData(
#     input_data=[metab.iloc[:, :-1], prot.iloc[:, :-1]],
#     metadata=[metab['Group'], prot['Group']],
#     pathways=mo_paths_dict,
#     enriched_paths=['R-HSA-112316']
# ).enrich_paths_base(effect_sizes=[1, 2, 3])

# metab_sim = simulated_dset[0]  # Enriched metabolomics data
# prot_sim = simulated_dset[1]   # Enriched proteomics data

# print(metab_sim.head())
# print(prot_sim.head())