#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 2025

@author: victor_m@cimat.mx
"""

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, RFE
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD, NMF, PCA
import umap.umap_ as umap
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

import pickle
from sklearn.metrics.pairwise import pairwise_distances

import skbio
from skbio import Table
from biom import load_table
from skbio.diversity import beta_diversity


def get_data(data_path, data = 1):
    """
    Retrieves and processes microbiome data and associated metadata (in TSV format) from specified paths. 
    It handles two datasets: a Latin America-specific dataset (latam) and the global EMP500 dataset, 
    depending on the parameter provided.

    Parameters
    ---------- 
    data_path: str
            Base path to the directory where the BIOM and metadata files are stored.

    data:  int, optional, default=1
            Flag to determine which dataset to load. 
            1 Latam Shotgun (default), loads the Latin America (filtered) dataset. 
            2 EMP 500 Shotgun, loads the global dataset
            3 Latam Amplicon, loads the Latin America (filtered) 16S dataset.

    Returns
    -------
        biom_df : pandas.DataFrame
            A DataFrame containing the microbiome data (features as columns, samples as rows).
        valid_metadata_df: pandas.DataFrame 
            A DataFrame containing the metadata for samples that exist in both the microbiome data 
            and metadata files
    """
    if data==1:
        # generate the metadata path by joining github_data_path with "emp500/sample.tsv"
        biom_path = data_path + "Data_Latam/shotgun/latam_ogu.biom"
        metadata_path = data_path + "Data_Latam/latam_samples.tsv"
    elif data==2:
        biom_path = data_path + "Data_emp500/shotgun/ogu.biom"
        metadata_path = data_path + "Data_emp500/sample.tsv"
    else:
        biom_path = data_path + "Data_Latam/amplicon/latam_16s.biom"
        metadata_path = data_path + "Data_Latam/latam_samples.tsv"

    # shotgun data
    #biom= Table.read(biom_path)
    biom= load_table(biom_path)
    biom_df = biom.to_dataframe()
    biom_df = biom_df.T

    # metadata
    # Read the TSV file directly from the URL into a pandas DataFrame
    metadata_df = pd.read_csv(metadata_path, sep='\t', low_memory=False, index_col='sample_name')
    valid_sample_names = biom_df.index.intersection(metadata_df.index)
    valid_metadata_df = metadata_df.loc[valid_sample_names]

    return biom_df, valid_metadata_df

def scale_matrix(matrix):
    """Scales the values of a matrix between 0 and 1.

    Args:
        matrix: A NumPy array representing the matrix.

    Returns:
        A NumPy array with scaled values.
    """
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    scaled_matrix = (matrix - min_val) / (max_val - min_val)
    return scaled_matrix


def plot_3d(data_df):
    # Create a 3D scatter plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for each sample
    ax.scatter(data_df.iloc[:,0], data_df.iloc[:,1], data_df.iloc[:,2], c='red')

    # Set labels for the axes
    ax.set_xlabel(data_df.columns[0])
    ax.set_ylabel(data_df.columns[1])
    ax.set_zlabel(data_df.columns[2])

    # Set title
    ax.set_title('3D Scatter Plot')

    # Add sample names as labels
    for i, sample_name in enumerate(data_df.index):
        ax.text(data_df.iloc[i, 0], data_df.iloc[i, 1], data_df.iloc[i, 2], sample_name, size=10, zorder=1, color='k')

    plt.show()

def get_tfidf(data):
    """
    This function computes the Term Frequency-Inverse Document Frequency (TF-IDF) representation of input data.
    TF-IDF is a numerical statistic that reflects how important a feature (a gene in our case) is in 
    a sample relative to a collection of samples (dataset). This representation is inspired by text mining, 
    and natural language processing (NLP) applications where feature weighting is required in sparse count
    data.

    Parameters
    ---------- 
    data : array-like or sparse matrix
        Input data to transform, where rows represent samples and columns represent genes.

    Returns
    -------
    tfidf_transformer: TfidfTransformer object
        The fitted TfidfTransformer instance, which can be reused to transform new data using the same scaling.
    tfidf: numpy.ndarray
        The TF-IDF transformed data as a dense array. Each element represents the TF-IDF weight of a gene 
        in a sample.
    """
    vectorizer = TfidfTransformer()
    tfidf_transformer = vectorizer.fit(data)
    X = tfidf_transformer.transform(data)
    tfidf = X.toarray()
    
    return  tfidf_transformer, tfidf

# realiza factorizacion SVD o NMF
def get_factorization(data, n_comp=100, nmf=True):
    """
    The get_factorization function performs dimensionality reduction on input data using either 
    Non-Negative Matrix Factorization (NMF) or Truncated Singular Value Decomposition (TruncatedSVD).

    Parameters
    ---------- 
    data: array-like or pandas.DataFrame
        The input dataset to be factorized. Must be a 2D array or DataFrame where rows represent samples 
        and columns represent features.

    n_comp: int, optional, default=100
        The number of components (latent features) to extract from the data.

    nmf: bool, optional, default=True
        If True, uses Non-Negative Matrix Factorization (NMF).
        If False, uses Truncated Singular Value Decomposition (TruncatedSVD).

    Returns
    -------
    fact_model: sklearn.decomposition.NMF or sklearn.decomposition.TruncatedSVD
        A fitted dimensionality reduction model (either NMF or TruncatedSVD object) that can be used to 
        transform new data.
    
    Notes
    -----
    NMF requires the input data to be non-negative. If negative values are present, consider using 
        TruncatedSVD instead.

    TruncatedSVD is similar to PCA but works on sparse matrices efficiently.
    """
    if nmf==False:
        fact_model = TruncatedSVD(n_components=n_comp)
        fact_model.fit(data)
    else:
        fact_model = NMF(n_components = n_comp, init=None, max_iter=12000)
        fact_model.fit(data)
    
    return fact_model

def save_pickle_model(model_obj, file_path, model_name):
    """
    This function serializes a Python object (typically a machine learning model) and saves it to disk 
    in pickle format (.pkl). This allows for persistent storage and later retrieval of the object.
    
    Parameters
    ---------- 
    model_obj: any Python object
        The object to be serialized and saved. This is typically a trained machine learning model, 
        but can be any Python object that is pickle-compatible.

    file_path: str
        he directory path where the pickle file will be saved.

    model_name: str
        The name to be given to the saved pickle file (should not include extension).
    
    Returns
    -------
    None (saves file to disk)
    """    
    pkl_name = os.path.join(file_path, model_name)
    with open(pkl_name,'wb') as file:
        pickle.dump(model_obj,file)

def load_from_pickle(file_path, model_name):
    """
    This function loads and deserializes a Python object (typically a machine learning model or data structure) 
    that was previously saved in pickle format (.pkl). This function is the counterpart to save_pickle_model, 
    enabling retrieval of persisted objects.

    Parameters
    ---------- 
    file_path: str
        The directory path where the pickle file is stored.

    model_name: str
        The name of the pickle file to load (with or without .pkl extension).

    Returns
    -------
    fact_model: any Python object
        The deserialized object that was stored in the pickle file
    """
    pkl_name = os.path.join(file_path, model_name)
    with open(pkl_name,'rb') as file:
        fact_model = pickle.load(file)
    
    return fact_model


def get_distance_matrix(X, metric = 'euclidean'):
    """
    Computes a pairwise distance matrix between all samples in the input data 
    using a specified distance metric. The resulting matrix is returned as a pandas DataFrame 
    with row and column labels matching the input data's index

    Parameters
    ----------
    X : pandas.DataFrame or array-like
        Input data where rows represent samples and columns represent features. 
        If a DataFrame is provided, the index will be used to label the output matrix.
    metric : str or callable, optional (default='euclidean')
        The distance metric to use. This can be any metric supported by 
        sklearn.metrics.pairwise.pairwise_distances, such as: 'euclidean' (default), 'cosine',
        'manhattan', 'correlation', etc.

    Returns
    -------
    dist_matrix : pandas.DataFrame
        A square distance matrix where each element (i, j) represents the distance between 
        sample i and sample j. Row and column labels match the index of input X.
    """
    dist_matrix = pairwise_distances(X, X, metric)
    dist_matrix = pd.DataFrame(dist_matrix)
    dist_matrix.columns = X.index
    dist_matrix.index = X.index
    return dist_matrix

def get_lsa(data_table, exist_fact = False, fact_path = os.getcwd(), n_file = 'svd', n_comp = 100, save=True):
    """
    This function performs dimensionality reduction via Truncated Singular Value Decomposition (SVD). 
    It provides options to either create and fit a new dimensionality reduction model, or load an existing 
    pre-trained model. 
    This representation is inspired by Latent Semantic Analysis (LSA) on texts represented by sparse 
    counting matrices such as BOW or TFIDF.
    The function returns the transformed data in a lower-dimensional space while 
    preserving the original index structure.

    Parameters
    ---------- 
    Parameter	    Type	        Default	        Description
    _______________________________________________________________________________________
    data_table	    pd.DataFrame	Required	    Input data matrix (samples x features)
    exist_fact	    bool	        False	        Whether to use an existing factorization model
    fact_path	    str	            os.getcwd()	    Directory path for loading/saving models
    n_file	        str	            'svd'	        Base filename for model storage (without extension)
    n_comp	        int	            100	            Number of components for dimensionality reduction
    save	        bool	        True	        Whether to save newly created models

    Returns
    -------
    fact_lsa: pd.DataFrame 
        Dataframe containing the LSA-transformed data with rows matching original data index and 
        columns representing the n_comp latent dimensions
    """
    # si no hay un modelo de factorización guardado, obtiene uno y lo guarda
    if exist_fact is False:
        svd_fact = get_factorization(data=data_table, n_comp=n_comp, nmf=False)
        if save is True:
            pkl_file = n_file + '.pkl'
            save_pickle_model(svd_fact, fact_path, pkl_file)
    else: 
        # lsa
        pkl_file = n_file + '.pkl'
        svd_fact = load_from_pickle(fact_path, pkl_file)

    fact_lsa = svd_fact.transform(data_table)
    fact_lsa = pd.DataFrame(fact_lsa, index=data_table.index)
    
    return fact_lsa

def get_nmf(data_table, exist_fact = False, fact_path = os.getcwd(), n_file = 'nmf', n_comp = 100, save=True):
    """
    This function performs Non-Negative Matrix Factorization (NMF) on input data, 
    providing options to either create a new factorization model or load an existing one. 
    It returns the transformed data in a lower-dimensional space while preserving the original index structure.

    Parameters
    ---------- 
    Parameter	    Type	        Default	        Description
    _______________________________________________________________________________________
    data_table	    pd.DataFrame	Required	    Input data matrix (samples x features)  - must be non-negative
    exist_fact	    bool	        False	        Whether to use an existing factorization model
    fact_path	    str	            os.getcwd()	    Directory path for loading/saving models
    n_file	        str	            'nmf'	        Base filename for model storage (without extension)
    n_comp	        int	            100	            Number of components for dimensionality reduction
    save	        bool	        True	        Whether to save newly created models

    Returns
    -------
    fact_lsa: pd.DataFrame 
        Dataframe containing the NMF-transformed data with rows matching original data index and 
        columns representing the n_comp latent dimensions
    """
    if exist_fact is False:
        nmf_fact = get_factorization(data=data_table, n_comp=n_comp, nmf=True)
        if save is True:
            pkl_file = n_file + '.pkl'
            save_pickle_model(nmf_fact, fact_path, pkl_file)
    else: 
        pkl_file = n_file + '.pkl'
        nmf_fact = load_from_pickle(fact_path, pkl_file)

    fact_nmf = nmf_fact.transform(data_table)
    fact_nmf = pd.DataFrame(fact_nmf, index=data_table.index)
    
    return fact_nmf

def get_reduced_table(data_table, only_counts = False, exist_fact = False, exist_vect = False,
                    fact_path = os.getcwd(), n_comp = 100):
    """
    This function performs dimensionality reduction on input data through two pathways: 
    (1) TF-IDF transformation followed by both NMF and SVD factorization (default) and (2) Direct NMF and SVD 
    factorization on count data (when only_counts=True).
    The function provides flexible options for using existing models or creating new ones, with automatic 
    model persistence capabilities.

    Parameters
    ---------- 
    Parameter	Type	        Default	        Description
    _____________________________________________________________________________________
    data_table	pd.DataFrame	Required	    Input data matrix (samples x features)
    only_counts	bool	        False	        Bypass TF-IDF transformation when True
    exist_fact	bool	        False	        Use existing factorization models if available
    exist_vect	bool	        False	        Use existing TF-IDF vectorizer if available
    fact_path	str	            os.getcwd()	    Directory path for model retrieve/save operations
    n_comp	    int	            100	            Number of components for dimensionality reduction

    Returns
    -------
    For only_counts=False:
        tfidf: TF-IDF transformed matrix (pd.DataFrame)
        tfidf_vect: TF-IDF vectorizer object
        fact_lsa: SVD-transformed data (pd.DataFrame)
        fact_nmf: NMF-transformed data (pd.DataFrame)
    For only_counts=True:
        fact_lsa: SVD-transformed count data (pd.DataFrame)
        fact_nmf: NMF-transformed count data (pd.DataFrame)
    """
    if only_counts is False:
        # realiza TF-IDF y factorizacion
        if exist_vect is False:
            tfidf_vect, tfidf = get_tfidf(data_table)
            # save tfidf model
            pkl_file = 'tfidf.pkl'
            save_pickle_model(tfidf_vect, fact_path, pkl_file)
        else:
            pkl_file = 'tfidf.pkl'
            tfidf_vect = load_from_pickle(fact_path, pkl_file)
            X = tfidf_vect.transform(data_table)
            tfidf = X.toarray()

        tfidf = pd.DataFrame(tfidf,columns=tfidf_vect.get_feature_names_out(),index=data_table.index)

        # si no hay un modelo de factorización guardado, obtiene uno y lo guarda
        if exist_fact is False:
            svd_fact = get_factorization(data=tfidf, n_comp=n_comp, nmf=False)
            nmf_fact = get_factorization(data=tfidf, n_comp=n_comp, nmf=True)
            pkl_file = 'nmf.pkl'
            save_pickle_model(nmf_fact, fact_path, pkl_file)
            pkl_file = 'svd.pkl'
            save_pickle_model(svd_fact, fact_path, pkl_file)
        else: 
            # lsa
            pkl_file = 'svd.pkl'
            svd_fact = load_from_pickle(fact_path, pkl_file)
            # nmf
            pkl_file = 'nmf.pkl'
            nmf_fact = load_from_pickle(fact_path, pkl_file)

        fact_lsa = svd_fact.transform(tfidf)
        fact_nmf = nmf_fact.transform(tfidf)
        fact_lsa = pd.DataFrame(fact_lsa,index=data_table.index)
        fact_nmf = pd.DataFrame(fact_nmf,index=data_table.index)
        return tfidf, tfidf_vect, fact_lsa, fact_nmf
    else:
        # si no hay un modelo de factorización guardado, obtiene uno y lo guarda
        if exist_fact is False:
            svd_fact = get_factorization(data=data_table, n_comp=n_comp, nmf=False)
            nmf_fact = get_factorization(data=data_table, n_comp=n_comp, nmf=True)
            pkl_file = 'nmf_count.pkl'
            save_pickle_model(nmf_fact, fact_path, pkl_file)
            pkl_file = 'svd_count.pkl'
            save_pickle_model(svd_fact, fact_path, pkl_file)
        else: 
            # lsa
            pkl_file = 'svd_count.pkl'
            svd_fact = load_from_pickle(fact_path, pkl_file)
            # nmf
            pkl_file = 'nmf_count.pkl'
            nmf_fact = load_from_pickle(fact_path, pkl_file)

        fact_lsa = svd_fact.transform(data_table)
        fact_nmf = nmf_fact.transform(data_table)
        fact_lsa = pd.DataFrame(fact_lsa,index=data_table.index)
        fact_nmf = pd.DataFrame(fact_nmf,index=data_table.index)
        return fact_lsa, fact_nmf



def split_stratified_into_train_val_test(X, y, frac_train=0.6, frac_val=0.15, frac_test=0.25, std = True, 
                                         two_subsets=False, random_state=None):
    '''
    Splits a dataset into three subsets (train, val, and test)
    following fractional ratios provided by the user, where each subset is
    stratified by the values in y (that is, each subset has
    the same relative frequency of the values in the column). It performs this
    splitting by running train_test_split() twice.

    Parameters
    ----------
    X : numpy dataframe of covariates
    y : numpy array of responses
    frac_train : float
    frac_val   : float
    frac_test  : float
        The ratios with which the dataframe will be split into train, val, and
        test data. The values should be expressed as float fractions and should
        sum to 1.0.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().

    Returns
    -------
    df_train, df_val, df_test :
        Dataframes containing the three splits.
    '''
    
    if round(frac_train + frac_val + frac_test,10) != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                         (frac_train, frac_val, frac_test))

    # Split original dataframe into temp and test dataframes.
    #x_train, x_temp, y_train, y_temp = train_test_split(X, y, stratify=y, test_size=(1.0 - frac_train), random_state=random_state)
    x_temp, x_test, y_temp, y_test = train_test_split(X, y, stratify=y, test_size=(1.0 - (frac_train+frac_val)), random_state=random_state)
    scaler = None
    if std:
        # standardize train_val (temp) and test data
        scaler = StandardScaler()
        x_temp = scaler.fit_transform(x_temp)
        x_test = scaler.transform(x_test)
        
    # weights for class imbalance (https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)
    class_w = compute_class_weight('balanced',classes=np.unique(y_temp),y=y_temp)
    # the latter is equivalent to:
    # unique, class_counts = np.unique(y_temp, return_counts=True)
    # class_w = sum(class_counts)/(len(unique)*class_counts)    
    if two_subsets:        
        x_train = x_temp
        y_train = y_temp
        x_val = None
        y_val = None
        #return x_train, y_train, x_test, y_test, class_w, scaler
    else:
        # Split the temp dataframe into train and val dataframes.
        relative_frac_val = frac_val / (frac_train + frac_val)
        x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, stratify=y_temp, 
                                                          test_size=relative_frac_val, random_state=random_state)
        #assert len(df_input) == len(df_train) + len(df_val) + len(df_test)
    
    return x_train, y_train, x_val, y_val, x_test, y_test, class_w, scaler


def hierarchical_cluster(X, link = 'ward', labels = None, plot_dist_matrix = True, metric_d = 'euclidean'):
    """
    This function performs hierarchical clustering on input data and visualizes the results either 
    as a distance matrix heatmap with clustering (clustermap) or as a dendrogram. 
    The function provides flexibility in clustering methods and visualization options.

    Parameters
    ---------- 
    Parameter	        Type	        Default	        Description
    _____________________________________________________________________________________
    X	                array-like	    Required	    Input data matrix (n_samples × n_features)
    link	            str	            'ward'	        Linkage method for clustering: 'ward', 'complete', 
                                                        'average', 'single'
    labels	            list/array	    None	        Labels for data points (for visualization)
    plot_dist_matrix	bool	        True	        Visualization type:
                                                            True: Distance matrix clustermap. 
                                                            False: Dendrogram
    metric_d	        str	            'euclidean'	    Distance metric for distance matrix

    Returns
    -------
    None (displays visualization)
    """
    link_mat = linkage(X, method=link, metric=metric_d)
    
    if plot_dist_matrix:
        dd = get_distance_matrix(X,metric_d)

        g = sns.clustermap(dd, row_linkage=link_mat, col_linkage=link_mat, xticklabels=labels, yticklabels=labels)
        # Set the new x-tick labels with desired font size
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=6)  # Adjust fontsize as needed
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=6)  # Adjust fontsize as needed
        # Display the plot
        plt.show()
    else:
        # Plot the dendrogram with labels
        plt.figure(figsize=(10, 5))
        dendrogram(link_mat,
                    orientation='top',
                    labels=labels,
                    distance_sort='descending',
                    show_leaf_counts=True)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Data Points')
        plt.ylabel('Distance')
        plt.show()


def elbow_kmeans(X, nclust = 2, plot = True):
    """
    This function performs the elbow method analysis for K-means clustering, which helps determine the 
    optimal number of clusters by evaluating the distortion (inertia) for different cluster counts. 
    The function can visualize the elbow curve and returns distortion values for further analysis.

    Parameters
    ---------- 
    X: 	array-like	
        Input data matrix (n_samples, n_features)
    nclust: int, default 2
    	List of cluster numbers to evaluate
    plot: bool, default	True
    	Whether to plot the elbow curve
    Returns
    -------
    distortions: list 
        Distortion values (inertia) for each cluster number in nclusters
    """
    distortions = []
    nclusters = range(1,nclust)
    for i in nclusters:
        km = KMeans(
            n_clusters=i, init='random',
            n_init=10, max_iter=300,
            tol=1e-04, random_state=0
        )
        km.fit(X)
        distortions.append(km.inertia_)
    if plot:
        # plot
        plt.figure(figsize=(5, 3))
        plt.plot(nclusters, distortions, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.show()

    return distortions


def silhouette_metric(X, range_n_clusters = [2,3], plot_sil = True, plot_scatter = True):
    """
    This function evaluates K-means clustering quality using silhouette analysis for different numbers of 
    clusters. It provides both numerical metrics and visual diagnostics to help determine the optimal 
    number of clusters.

    Parameters
    ---------- 
    X:  DataFrame/array	
        Input data (n_samples, n_features)
    range_n_clusters: list, default	[2]
        List of cluster numbers to evaluate
    plot_sil: 	bool, default True	
        Whether to plot silhouette diagrams
    plot_scatter:   bool, default True	
        Whether to plot cluster visualizations (using the first two columns of X)

    Returns
    -------
    sil_avg: list 
        Average silhouette scores for each cluster number in range_n_clusters
    """
    sil_avg = []
    for n_clusters in range_n_clusters:
        # instanciamos un objeto de KMeans, especificando n_clusters en el constructor
        # fijamos la semilla (random_state) para poder reproducir el resultado
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # el silhouette_score es el valor promedio para cada muestra ($\bar{s}_K$ en la notación de clase)
        silhouette_avg = silhouette_score(X, cluster_labels)
        sil_avg.append(silhouette_avg)
        #print("For n_clusters =", n_clusters,
        #      "The average silhouette_score is :", silhouette_avg)

        if plot_sil:
            # Subplot (1 row, 2 columns)
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

            # 1st subplot es el grafico de silueta
            # Observa que, el coeficiente de silueta está en [-1,1], pero para visualizar mejor los datos
            # lo ponemos en [-0.1, 1], ya que en este ejemplo todos caen en ese rango
            ax1.set_xlim([-1, 1])
            
            # ponemos un margen de (n_clusters+1)*10 entre cada silueta individual para
            # cada cluster 
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

            # valor de silueta para cada observacion
            sample_silhouette_values = silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # agregar el score silhouette scores para las observaciones que caen en el cluster i
                # y se ordenan
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                0, ith_cluster_silhouette_values,
                                facecolor=color, edgecolor=color, alpha=0.7)

                # etiquetas de los silhouette plots
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # pone el score average silhouette como una linea puteada en rojo
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            #ax1.set_yticks([])  
            #ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # El segundo grafico muestra los clusters formados 
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(X[:,0], X[:,1], marker='.', s=50, lw=0, alpha=0.5,
                        c=colors, edgecolor='k')

            centers = clusterer.cluster_centers_
            # grafica centroides
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                            s=50, edgecolor='k')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                        "with n_clusters = %d" % n_clusters),
                        fontsize=14, fontweight='bold')

        plt.show()
    return sil_avg


