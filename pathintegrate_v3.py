import pandas as pd
import numpy as np
import sklearn.decomposition
import sspa
import sklearn
from mbpls.mbpls import MBPLS
from pathintegrate.app import launch_network_app
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import json
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go

class PathIntegrate:
    '''PathIntegrate class for multi-omics pathway integration.

    Args:
        omics_data (dict): Dictionary of omics data. Keys are omics names, values are pandas DataFrames containing omics data where rows contain samples and columns reprsent features.
        metadata (pandas.Series): Metadata for samples. Index is sample names, values are class labels.
        pathway_source (pandas.DataFrame): GMT style pathway source data. Must contain column 'Pathway_name'. 
        sspa_scoring (object, optional): Scoring method for ssPA. Defaults to sspa.sspa_SVD. Options are sspa.sspa_SVD, sspa.sspa_ssGSEA, sspa.sspa_KPCA, sspa.sspa_ssClustPA, sspa.sspa_zscore.
        min_coverage (int, optional): Minimum number of molecules required in a pathway. Defaults to 3.

    Attributes:
        omics_data (dict): Dictionary of omics data. Keys are omics names, values are pandas DataFrames.
        omics_data_scaled (dict): Dictionary of omics data scaled to mean 0 and unit variance. Keys are omics names, values are pandas DataFrames.
        metadata (pandas.Series): Metadata for samples. Index is sample names, values are class labels.
        pathway_source (pandas.DataFrame): Pathway source data.
        pathway_dict (dict): Dictionary of pathways. Keys are pathway names, values are lists of molecules.
        sspa_scoring (object): Scoring method for SSPA.
        min_coverage (int): Minimum number of omics required to cover a pathway.
        sspa_method (object): SSPA scoring method.
        sspa_scores_mv (dict): Dictionary of SSPA scores for each omics data. Keys are omics names, values are pandas DataFrames.
        sspa_scores_sv (pandas.DataFrame): SSPA scores for all omics data concatenated.
        coverage (dict): Dictionary of pathway coverage. Keys are pathway names, values are number of omics covering the pathway.
        mv (object): Fitted MultiView model.
        sv (object): Fitted SingleView model.
        labels (pandas.Series): Class labels for samples. Index is sample names, values are class labels.
    '''

    def __init__(self, omics_data:dict, metadata, pathway_source, sspa_scoring=sspa.sspa_SVD, min_coverage=3):
        self.omics_data = omics_data
        self.omics_data_scaled = {k: pd.DataFrame(StandardScaler().fit_transform(v), columns=v.columns, index=v.index) for k, v in self.omics_data.items()}
        self.metadata = metadata
        self.pathway_source = pathway_source
        self.pathway_dict = sspa.utils.pathwaydf_to_dict(pathway_source)
        self.sspa_scoring = sspa_scoring
        self.min_coverage = min_coverage

        # sspa_methods = {'svd': sspa.sspa_SVD, 'ssGSEA': sspa.sspa_ssGSEA, 'kpca': sspa.sspa_KPCA, 'ssClustPA': sspa.sspa_ssClustPA, 'zscore': sspa.sspa_zscore}
        self.sspa_method = self.sspa_scoring
        self.sspa_scores_mv = None
        self.sspa_scores_sv = None
        self.coverage = self.get_multi_omics_coverage()

        self.mv = None
        self.sv = None
        self.sv_us = None

        self.labels = pd.factorize(self.metadata)[0]
    
    def get_multi_omics_coverage(self):
        all_molecules = sum([i.columns.tolist() for i in self.omics_data.values()], [])
        coverage = {k: len(set(all_molecules).intersection(set(v))) for k, v in self.pathway_dict.items()}
        return coverage

    def MultiView(self, ncomp=2):
        """Fits a PathIntegrate MultiView model using MBPLS.

        Args:
            ncomp (int, optional): Number of components. Defaults to 2.

        Returns:
            object: Fitted PathIntegrate MultiView model.
        """

        print('Generating pathway scores...')
        sspa_scores_ = [self.sspa_method(self.pathway_source, self.min_coverage) for i in self.omics_data_scaled.values()]
        sspa_scores = [sspa_scores_[n].fit_transform(i) for n, i in enumerate(self.omics_data_scaled.values())]
        # sspa_scores = [self.sspa_method(self.pathway_source, self.min_coverage).fit_transform(i) for i in self.omics_data_scaled.values()]
        # sspa_scores = [self.sspa_method(i, self.pathway_source, self.min_coverage, return_molecular_importance=True) for i in self.omics_data.values()]

        self.sspa_scores_mv = dict(zip(self.omics_data.keys(), sspa_scores))
        print('Fitting MultiView model')
        mv = MBPLS(n_components=ncomp)
        mv.fit([i.copy(deep=True) for i in self.sspa_scores_mv.values()], self.labels)

        # compute VIP and scale VIP across omics
        vip_scores = VIP_multiBlock(mv.W_, mv.Ts_, mv.P_, mv.V_)
        vip_df = pd.DataFrame(vip_scores, index=sum([i.columns.tolist() for i in self.sspa_scores_mv.values()], []))
        vip_df['Name'] = vip_df.index.map(dict(zip(self.pathway_source.index, self.pathway_source['Pathway_name'])))
        vip_df['Source'] = sum([[k] * v.shape[1] for k, v in self.sspa_scores_mv.items()], [])
        vip_df['VIP_scaled'] = vip_df.groupby('Source')[0].transform(lambda x: StandardScaler().fit_transform(x.values[:,np.newaxis]).ravel())
        vip_df['VIP'] = vip_scores
        mv.name = 'MultiView'

        # only some sspa methods can return the molecular importance
        if hasattr(sspa_scores_[0], 'molecular_importance'):
            mv.molecular_importance = dict(zip(self.omics_data.keys(), [i.molecular_importance for i in sspa_scores_]))
        mv.beta = mv.beta_.flatten()
        mv.vip = vip_df
        mv.omics_names = list(self.omics_data.keys())
        mv.sspa_scores = self.sspa_scores_mv
        mv.coverage = self.coverage
        self.mv = mv

        return self.mv

    def SingleView(self, model=sklearn.linear_model.LogisticRegression, model_params=None):
        """Fits a PathIntegrate SingleView model using an SKLearn-compatible predictive model.

        Args:
            model (object, optional): SKlearn prediction model class. Defaults to sklearn.linear_model.LogisticRegression.
            model_params (_type_, optional): Model-specific hyperparameters. Defaults to None.

        Returns:
            object: Fitted PathIntegrate SingleView model.
        """

        concat_data = pd.concat(self.omics_data_scaled.values(), axis=1)
        print('Generating pathway scores...')

        sspa_scores = self.sspa_method(self.pathway_source, self.min_coverage)
        self.sspa_scores_sv = sspa_scores.fit_transform(concat_data)
       
        if model_params:
            sv = model(**model_params) # ** this is inputed into the scikit learn model
        else:
            sv = model()
        print('Fitting SingleView model')

        # fitting the model

        sv.fit(X=self.sspa_scores_sv, y=self.labels) 
        sv.sspa_scores = self.sspa_scores_sv
        sv.name = 'SingleView'
        sv.coverage = self.coverage

        # only some sspa methods can return the molecular importance
        if hasattr(sspa_scores, 'molecular_importance'):
            sv.molecular_importance = sspa_scores.molecular_importance
        self.sv = sv

        return self.sv
    
    # no cross validation in unsupervised (but can bootstrap)

    # cross-validation approaches

    def SingleViewClust(self, model=sklearn.cluster.KMeans, model_params=None, use_pca=True, pca_params=None, consensus_clustering=False, n_runs=10, auto_n_clusters=False, subsample_fraction=0.8, return_plot=False, return_ground_truth_plot=False, return_comparison_plot=False, return_metrics_table=False):
        """
        Fits a PathIntegrate SingleView Unsupervised model using an SKLearn-compatible KMeans model.

        Args:
            model (object, optional): SKLearn clustering model class. Defaults to sklearn.cluster.KMeans.
            model_params (dict, optional): Model-specific hyperparameters. Defaults to None.
            use_pca (bool, optional): Whether to perform PCA before clustering. Defaults to False.
            pca_params (dict, optional): PCA-specific hyperparameters. Defaults to None.
            consensus_clustering (bool, optional): Whether to perform consensus clustering. Defaults to False.
            n_runs (int, optional): Number of runs for consensus clustering. Defaults to 10.

        Returns:
            object: Fitted PathIntegrate SingleView Clustering model with various clustering evaluation metrics.
        """
        
        def normalize_score(score, score_min, score_max):
            return (score - score_min) / (score_max - score_min)

        concat_data = pd.concat(self.omics_data_scaled.values(), axis=1)
        print('Generating pathway scores...')
        
        sspa_scores = self.sspa_method(self.pathway_source, self.min_coverage)
        self.sspa_scores_sv = sspa_scores.fit_transform(concat_data)

        combined_data_scaled = StandardScaler().fit_transform(self.sspa_scores_sv)
        combined_data_final = pd.DataFrame(combined_data_scaled, index=self.sspa_scores_sv.index, columns=self.sspa_scores_sv.columns) 
        self.sspa_scores_sv = combined_data_final

        if use_pca:
            print('Performing PCA...')
            if pca_params is None:
                pca_params = {'n_components': min(concat_data.shape[1], 50)}
            
            pca = sklearn.decomposition.PCA(**pca_params)
    
            # Perform PCA and keep the result as a DataFrame
            pca_components = pca.fit_transform(self.sspa_scores_sv)
    
            # Create a DataFrame with the PCA components, keeping the same index
            component_names = [f'PC{i+1}' for i in range(pca_components.shape[1])]
            self.sspa_scores_sv = pd.DataFrame(data=pca_components, columns=component_names, index=self.sspa_scores_sv.index)
        else:
            print('Not Using PCA...')

                
        if auto_n_clusters:
            print('Determining optimal number of clusters...')
            best_score = -1
            best_n_clusters = None
            silhouette_scores = []
            for n_clusters in range(*n_clusters_range):
                sv_clust = model(n_clusters=n_clusters, **(model_params or {}))
                labels = sv_clust.fit_predict(self.sspa_scores_sv)
                silhouette_avg = sklearn.metrics.silhouette_score(self.sspa_scores_sv, labels)
                silhouette_scores.append(silhouette_avg)
                if silhouette_avg > best_score:
                    best_score = silhouette_avg
                    best_n_clusters = n_clusters

            model_params['n_clusters'] = best_n_clusters
            print(f'Optimal number of clusters determined: {best_n_clusters}')

        if consensus_clustering and n_runs > 0:
            n_samples = self.sspa_scores_sv.shape[0]
            consensus_matrix = np.zeros((n_samples, n_samples))

            for run in range(n_runs):
                print(f'Run {run + 1}/{n_runs}')
                subsample_idx = np.random.choice(n_samples, int(subsample_fraction * n_samples), replace=False)
                subsample_data = self.sspa_scores_sv.iloc[subsample_idx]  # Use iloc to select rows by index
                sv_clust = model(**(model_params or {}))
                labels = sv_clust.fit_predict(subsample_data)
                
                for i in range(len(subsample_idx)):
                    for j in range(i + 1, len(subsample_idx)):
                        if labels[i] == labels[j]:
                            consensus_matrix[subsample_idx[i], subsample_idx[j]] += 1
                            consensus_matrix[subsample_idx[j], subsample_idx[i]] += 1

            consensus_matrix /= n_runs
            consensus_labels = model(n_clusters=model_params['n_clusters']).fit_predict(consensus_matrix)
        else:
            sv_clust = model(**(model_params or {}))
            consensus_labels = sv_clust.fit_predict(self.sspa_scores_sv)
        

        self.sv_clust = sv_clust
        self.sv_clust.sspa_scores = self.sspa_scores_sv
        self.sv_clust.labels_ = consensus_labels
        self.sv_clust.name = 'SingleViewClust'

        print('Calculating clustering metrics...')
        
        # Calculate clustering metrics
        silhouette_avg = sklearn.metrics.silhouette_score(self.sspa_scores_sv, consensus_labels)
        calinski_harabasz = sklearn.metrics.calinski_harabasz_score(self.sspa_scores_sv, consensus_labels)
        davies_bouldin = sklearn.metrics.davies_bouldin_score(self.sspa_scores_sv, consensus_labels)
        
        # Normalize the scores
        silhouette_norm = normalize_score(silhouette_avg, -1, 1)
        calinski_harabasz_norm = normalize_score(calinski_harabasz, 0, np.max([calinski_harabasz]))
        davies_bouldin_norm = normalize_score(davies_bouldin, 0, np.max([davies_bouldin]))
        davies_bouldin_norm = 1 - davies_bouldin_norm  # Invert Davies-Bouldin index since lower is better

        # Compute combined score (weighted average)
        combined_score = (silhouette_norm + calinski_harabasz_norm + davies_bouldin_norm) / 3
        self.sv_clust.metrics = {
            'Silhouette_Score': silhouette_avg,
        }

        if return_plot:
            consensus_labels_series = pd.Series(consensus_labels, index=self.sspa_scores_sv.index, name='Consensus_Cluster')
            sspa_scores_labels = self.sspa_scores_sv
            sspa_scores_labels['Consensus_Cluster'] = consensus_labels_series

            plt.figure(figsize=(10, 8))
            sns.scatterplot(x=sspa_scores_labels.iloc[:, 0], y=sspa_scores_labels.iloc[:, 1], hue=sspa_scores_labels['Consensus_Cluster'], palette='tab10', s=100, edgecolor='black')
            plt.title('Clustering Results')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.legend(title='Cluster')
            plt.grid(True)
            plt.show()
        
        if return_ground_truth_plot:
            metadata_pca = self.metadata.to_frame(name='Who_Group')
            sspa_scores_meta = pd.concat([self.sspa_scores_sv, metadata_pca], axis=1)
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x=sspa_scores_meta.iloc[:, 0], y=sspa_scores_meta.iloc[:, 1], hue=sspa_scores_meta['Who_Group'], palette='tab10', s=100, edgecolor='black')
            plt.title('Ground Truth Labels')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.legend(title='True Label')
            plt.grid(True)
            plt.show()

        if return_comparison_plot:
            metadata_pca = self.metadata.to_frame(name='Who_Group')
            sspa_scores_meta = pd.concat([self.sspa_scores_sv, metadata_pca], axis=1)
            
            # Convert the consensus labels to a pandas Series and align it with the DataFrame
            consensus_labels_series = pd.Series(consensus_labels, index=sspa_scores_meta.index, name='Consensus_Cluster')

            # Append the consensus labels to the sspa_scores_meta DataFrame
            sspa_scores_meta['Consensus_Cluster'] = consensus_labels_series

            # Now generate the confusion matrix using the Who_Group and Consensus_Cluster columns
            confusion_df = pd.crosstab(sspa_scores_meta['Who_Group'], sspa_scores_meta['Consensus_Cluster'])

            # Normalize the confusion matrix by dividing by the total number of samples in each ground truth label
            normalized_confusion_df = confusion_df.div(confusion_df.sum(axis=1), axis=0)

            # Plot the confusion matrix as a Plotly heatmap
            fig = go.Figure(data=go.Heatmap(
                z=normalized_confusion_df.values,
                x=normalized_confusion_df.columns,
                y=normalized_confusion_df.index,
                colorscale='Blues',
                text=confusion_df.values,
                texttemplate="%{text}",
                hovertemplate="True Label: %{y}<br>Predicted: %{x}<br>Count: %{text}<extra></extra>",
                colorbar=dict(title="Normalized Value")
            ))

            fig.update_layout(
                title="Confusion Matrix: Cluster vs Ground Truth Label",
                xaxis_title="Predicted Cluster",
                yaxis_title="Ground Truth Label",
                xaxis=dict(tickmode='array', tickvals=list(range(len(confusion_df.columns))), ticktext=confusion_df.columns),
                yaxis=dict(tickmode='array', tickvals=list(range(len(confusion_df.index))), ticktext=confusion_df.index),
                height=600,
                width=800
            )

            fig.show()           

            # Calculate the Adjusted Rand Index (ARI) as a metric for the quality of clustering
            ari_score = sklearn.metrics.adjusted_rand_score(sspa_scores_meta['Who_Group'], sspa_scores_meta['Consensus_Cluster'])


            # Append ARI to metrics
            self.sv_clust.metrics['Adjusted_Rand_Index'] = ari_score
            # Print the ARI score
            print(f"Adjusted Rand Index (ARI) Score: {ari_score:.4f}")


        if return_metrics_table:
            metrics_df = pd.DataFrame(self.sv_clust.metrics, index=[0])
            
            # Transpose the DataFrame for better visualization
            metrics_df = metrics_df.T.reset_index()
            metrics_df.columns = ['Metric', 'Value']
            
            plt.figure(figsize=(8, 10))
            sns.barplot(x='Metric', y='Value', data=metrics_df, palette='tab10')
            plt.title('Clustering Metrics')
            plt.xlabel('Metric')
            plt.ylabel('Value')
            plt.xticks(rotation=45, ha='right')  # Rotate metric names for better readability
            plt.grid(True)
            plt.show()

        print('Finished')


        self.sv_clust.sspa_scores_clusters = self.sspa_scores_sv

        return self.sv_clust


    def SingleViewDimRed(self, model=sklearn.decomposition.PCA, model_params=None, return_pca_plot=False, return_biplot=False, return_loadings_plot=False):
        """
        Applies a dimensionality reduction technique to the input data.

        Args:
            model (object, optional): The dimensionality reduction model to use. Defaults to sklearn.decomposition.PCA.
            model_params (dict, optional): Model-specific hyperparameters. Defaults to None.

        Returns:
            object: Fitted dimensionality reduction model with reduced data.
        """
        concat_data = pd.concat(self.omics_data_scaled.values(), axis=1)
        print('Generating pathway scores...')

        sspa_scores = self.sspa_method(self.pathway_source, self.min_coverage)
        self.sspa_scores_sv = sspa_scores.fit_transform(concat_data)

        if return_biplot or return_loadings_plot:
            if model_params is None:
                model_params = {}
            if model_params.get('n_components', 2) != 2:
                print("Warning: n_components has been set to 2 for the biplot.")
            
            model_params['n_components'] = 2


        sv_dim = model(**model_params) if model_params else model()
        print('Fitting SingleView Dimensionality Reduction model')

        reduced_data_scaled = StandardScaler().fit_transform(self.sspa_scores_sv)
        reduced_data_sspa = pd.DataFrame(reduced_data_scaled, columns=self.sspa_scores_sv.columns)

        reduced_data = sv_dim.fit_transform(reduced_data_sspa)
        
        explained_variance = sv_dim.explained_variance_ratio_
        n_components = model_params['n_components'] if model_params and 'n_components' in model_params else default_value


        sv_dim.reduced_data = reduced_data
        sv_dim.explained_variance = explained_variance
        sv_dim.n_components = n_components
        sv_dim.sspa_scores_pca = self.sspa_scores_sv
        sv_dim.name = 'SingleViewDimRed'

        if return_pca_plot:
            pca_df = pd.DataFrame(data=reduced_data[:, :2], columns=['PC1', 'PC2'])
            metadata_pca = self.metadata.to_frame(name='Who_Group').reset_index(drop=True)
            pca_df_named = pd.concat([pca_df, metadata_pca], axis=1)
            
            sns.set_style("white")
            sns.scatterplot(data=pca_df_named, x='PC1', y='PC2', hue='Who_Group', s=100, edgecolor='black')

            plt.title('PCA of Integrated Data (First 2 Principal Components)')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.legend(title='Metadata Group')
            plt.grid(True)

            plt.show()
        
        if return_biplot or return_loadings_plot:
            pca_df = pd.DataFrame(data=reduced_data[:, :2], columns=['PC1', 'PC2'])
            metadata_pca = self.metadata.to_frame(name='Who_Group').reset_index(drop=True)
            pca_df_named = pd.concat([pca_df, metadata_pca], axis=1)

            loadings_df = pd.DataFrame(sv_dim.components_.T, columns=['PC1', 'PC2'], index=reduced_data_sspa.columns)

            # url for pathway hierarchy in JSON format
            url = "https://rest.kegg.jp/get/br:br08901/json"
            response = requests.get(url)
            if response.status_code == 200:
                hierarchy_json = response.json()
                with open('br08901.json', 'w') as f:
                    json.dump(hierarchy_json, f, indent=4)
            else:
                print("Failed to retrieve data. Status code:", response.status_code)

            def create_id_name_mapping(node):
                id_name_mapping = {}
                
                if 'children' in node:
                    for child in node['children']:
                        id_name_mapping.update(create_id_name_mapping(child))
                else:
                    # Extract the ID and name from the node
                    pathway_id, pathway_name = node['name'].split('  ', 1)
                    id_name_mapping[pathway_id] = pathway_name.strip()
                
                return id_name_mapping

            # Create the ID-to-name mapping
            pathway_mapping = create_id_name_mapping(hierarchy_json)
            

            # Create the bar plot for top loadings
            plt.figure(figsize=(15, 6))
            
            # Identify the top 10 loadings for PC1 and PC2 separately
            top_loadings_pc1 = loadings_df['PC1'].sort_values(key=abs, ascending=False).head(10)
            top_loadings_pc2 = loadings_df['PC2'].sort_values(key=abs, ascending=False).head(10)

            # Combine the top loadings to ensure uniqueness
            top_loadings = pd.concat([top_loadings_pc1, top_loadings_pc2])

        if return_biplot:
            # Set the scaling factor for the arrows
            scaling_factor = 200

            # Create the biplot with adjusted scaling
            sns.set_style("white")
            plt.figure(figsize=(10, 8))

            # Plot the PCA scores (scatter plot of the samples)
            sns.scatterplot(data=pca_df_named, x='PC1', y='PC2', hue='Who_Group', s=100, edgecolor='black', alpha=0.2, legend=False)

            # Add the top loadings as vectors (arrows)
            for variable in top_loadings.index:
                color = 'red' if variable in top_loadings_pc1.index else 'blue'
                plt.arrow(0, 0, loadings_df.loc[variable, 'PC1'] * scaling_factor,
                        loadings_df.loc[variable, 'PC2'] * scaling_factor,
                        color=color, alpha=0.8, head_width=0.5, linewidth=2)
                plt.text(loadings_df.loc[variable, 'PC1'] * scaling_factor * 1.15,
                        loadings_df.loc[variable, 'PC2'] * scaling_factor * 1.15,
                        variable, color='black', ha='center', va='center', fontsize=10)
            
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.title('Biplot of PCA with Top 10 Loadings Highlighted')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.grid(True)

            sv_dim.biplot = plt.gcf()  # Capture the current figure as biplot

        if return_loadings_plot:
            
            index_mapping = self.pathway_source['Pathway_name'].to_dict()
            
            # Rename the indices in loadings_df using this mapping
            def rename_index(index):
                if index.startswith('R-HSA'):
                    return index_mapping.get(index, index)  
                else:
                    return pathway_mapping.get(index, index)  

            loadings_df.index = [rename_index(idx) for idx in loadings_df.index]

            top_loadings_pc1 = loadings_df['PC1'].sort_values(key=abs, ascending=False).head(15)
            top_loadings_pc2 = loadings_df['PC2'].sort_values(key=abs, ascending=False).head(15)

            # Combine the top loadings to ensure uniqueness
            top_loadings = pd.concat([top_loadings_pc1, top_loadings_pc2])

            # Color the bars red for PC1 and blue for PC2
            colors = ['red' if variable in top_loadings_pc1.index else 'blue' for variable in top_loadings.index]

            # Bar plot with PC1 on the left and PC2 on the right
            top_loadings.plot(kind='barh', color=colors)

            # Customize plot appearance
            plt.title('Top 5 Loadings for PC1 and PC2')
            plt.xlabel('Loading Value')
            plt.ylabel('Variables')
            plt.axvline(0, color='black', linewidth=0.5)  # Line at the center
            plt.xticks(rotation=45)
            plt.grid(True)

            # Show the bar plot
            plt.show()

            sv_dim.loadings_plot = plt.gcf()   
            
        sv_dim.pca_plot = plt.gcf()  
        sv_dim.metadata_pca = metadata_pca
        sv_dim.sspa_scores_met = pca_df_named
        self.sv_dim = sv_dim

        return self.sv_dim

    # wont have area under curve for crossvalidation - need to ARI/ASW

    def SingleViewGridSearchCV(self, param_grid, model=sklearn.linear_model.LogisticRegression, grid_search_params=None):
        '''Grid search cross-validation for SingleView model.
        
        Args:
            param_grid (dict): Grid search parameters.
            model (object, optional): SKlearn prediction model class. Defaults to sklearn.linear_model.LogisticRegression.
            grid_search_params (dict, optional): Grid search parameters. Defaults to None.
            
        Returns:
            object: GridSearchCV object.
                
        '''
        # concatenate omics - unscaled to avoid data leakage
        concat_data = pd.concat(self.omics_data.values(), axis=1)

        # Set up sklearn pipeline
        pipe_sv = sklearn.pipeline.Pipeline([
            ('Scaler', StandardScaler().set_output(transform="pandas")),
            ('sspa', self.sspa_method(self.pathway_source, self.min_coverage)),
            ('model', model())
        ])

        # Set up cross-validation
        grid_search = GridSearchCV(pipe_sv, param_grid=param_grid, **grid_search_params)
        grid_search.fit(X=concat_data, y=self.labels)
        return grid_search
    
    # only 1 model so one parameter way to grid search pca components
    # advantage of multi view is interpretation of contribution

    def MultiViewCV(self):
        '''Cross-validation for MultiView model.

        Returns:
            object: Cross-validation results.
        '''

        # Set up sklearn pipeline
        pipe_mv = sklearn.pipeline.Pipeline([
            ('sspa', self.sspa_method(self.pathway_source, self.min_coverage)),
            ('mbpls', MBPLS(n_components=2))
        ])

        # Set up cross-validation
        cv_res = cross_val_score(pipe_mv, X=[i.copy(deep=True) for i in self.omics_data.values()], y=self.labels)
        return cv_res

    def MultiViewGridSearchCV(self):
        pass



def VIP_multiBlock(x_weights, x_superscores, x_loadings, y_loadings):
    """Calculate VIP scores for multi-block PLS.

    Args:
        x_weights (list): List of x weights.
        x_superscores (list): List of x superscores.
        x_loadings (list): List of x loadings.
        y_loadings (list): List of y loadings.

    Returns:
        numpy.ndarray: VIP scores for each feature across all blocks. Features are in original order.
    """
    # stack the weights from all blocks 
    weights = np.vstack(x_weights)
    # calculate product of sum of squares of superscores and y loadings
    sumsquares = np.sum(x_superscores**2, axis=0) * np.sum(y_loadings**2, axis=0)
    # p = number of variables - stack the loadings from all blocks
    p = np.vstack(x_loadings).shape[0]
    
    # VIP is a weighted sum of squares of PLS weights 
    vip_scores = np.sqrt(p * np.sum(sumsquares*(weights**2), axis=1) / np.sum(sumsquares))
    return vip_scores


