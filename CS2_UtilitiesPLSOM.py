from minisomPLSOM import MiniSom_PLSOM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import altair as alt
from collections import Counter
from matplotlib.patches import Rectangle
import hyperopt
from joblib import dump, load

from plotly import express as px
from plotly import graph_objects as go
from plotly import offline as pyo
import warnings
warnings.filterwarnings('ignore')
import os
from hyperopt import Trials, STATUS_OK, hp, fmin, tpe

from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist

from doepy import build
import networkx as nx



#remove scientific notation
np.set_printoptions(precision=3, suppress = True)


#_________________________
# General helper functions
#_________________________

def cleanData(data, mode, start = 1, *end):
    """Loads the full factorial DF from makeMatrix.ipynb and formats it for SOM input
    data : csv file expected with a column for each receipe component,
    mode is a string if only formulation or also performance "recipe" or "recipe+performance"
    start: column id of reiepe start
    end: optional column id of recipe end
    returns clean pandas DF """
    

    if mode == "recipe":
        #loadData
        data = data.iloc[: , start:]
        data.columns = ["Ma", "BF", "WF", "SG", "CT"]
        data_ori = np.asarray(data)
        #remove the matrix column
        data_noMat = data.drop(columns = "Ma")

        return data_noMat

    elif mode == "recipe+performance":

        end = end[0]+1
        data_r = data.iloc[: , start:end]
        data_r.columns = ["Ma", "BF", "WF", "SG", "CT"]
        #remove the matrix column
        data_noMat = data_r.drop(columns = "Ma")

        data_performance = data.iloc[:, end:]
        return data_noMat, data_performance

    else:
        print("double check mode")


def scaleData(data, scalingType, saveModel,*fpath):
    """Applies SKlearn minMax or StandardScaling to dataframe for the first time
    data: pandas df
    scalingType: string "MinMax or StandardScaler supported
    returns scaled pandas DF """
    
    if scalingType == "MinMax":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    elif scalingType == "StandardScaler":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    else:
        print("scaler type not supported")

    data_scaled = scaler.fit_transform(data)
    data_scaled = pd.DataFrame(data_scaled, columns = data.columns.tolist())

    if saveModel == True:
        if fpath:
            fname = "%s/StandardScaler.pkl" % fpath
            pickle.dump(scaler, open(fname,'wb'))
        pickle.dump(scaler, open('StandardScaler.pkl','wb'))


    return data_scaled, scaler

def loadScaler(path):
    scaler = pickle.load(open(path,'rb'))

    return scaler



def applyScaling (data, scaler, df):
    """Applies transform to data given a saved scaling model
    data: array
    scaler: previously fit to data model
    returns unclaed data as DF if df == True or as NumpyArray if df == False """
    #OMG 
    data_scaled = scaler.transform(data)

    if df == True:
        data_scaled_df = pd.DataFrame(data_scaled, columns = data.columns.tolist())
        return data_scaled_df

    else:
        return data_scaled



def applyInverseScaling(data, scaler):
    """Applies inverse transform to data given a saved scaling model
    data: array
    scaler: previously fit to data model
    returns unclaed data """
    
    data_unscaled = scaler.inverse_transform(data)
    
    return data_unscaled



def trainPCA(data, save, saveName):
    """trains and saves SKlearn PCA model
    returns the pca model and transformed data as DF """

    from sklearn.decomposition import PCA 
    
    model = PCA()
    model.fit(data)
    
    print("Explained varience ratio a.k.a Eigen Values", model.explained_variance_ratio_)
    print("Components a.k.a Eigen Vectors")
    print(model.components_)
     
    data_reduced = model.transform(data)
    data_num_reduced_df = pd.DataFrame(data_reduced)

    if save:
        dump(model, "%s.joblib" %saveName)
        print("model is saved")

    return model, data_num_reduced_df



def createCategory(df):
    """Creates a concatenated string of the present fillers per receipe
    (could be improved to generate formulation tag as Arianna does)
    takes df
    returns df with category column """

    df = df.drop(['summ'], axis=1, errors='ignore')
    df['summ'] = df.sum(axis = 1)
    df.round(decimals = 2)
    
    df['category'] = np.where((df.BF > 0) & (df.BF >= df.summ), 'BF',
                        np.where((df.WF > 0) & (df.WF >= df.summ), 'WF',
                        np.where((df.SG > 0) & (df.SG >= df.summ), 'SG',
                        np.where((df.CT > 0) & (df.CT >= df.summ), 'CT',
                        np.where((df.BF > 0) & (df.WF) > 0 , 'BFWF',
                        np.where((df.BF > 0) & (df.SG) > 0 , 'BFSG',
                        np.where((df.BF > 0) & (df.CT) > 0 , 'BFCT',
                        np.where((df.WF > 0) & (df.SG) > 0 , 'WFSG',
                        np.where((df.WF > 0) & (df.CT) > 0 , 'WFCT',
                        np.where((df.SG > 0) & (df.CT) > 0 , 'SGCT', "none"))))))))))
    
    return df

            
def findMapSize(data):
    """returns the smallest square-ish map given an array size
    data: DF of inputs
    returns sizeX and sizeY"""
    from math import sqrt

    matrixLen = data.shape[0]
    prelim = sqrt(matrixLen)
    if round(prelim)**2 < matrixLen:
        sizeX = round(prelim) +1
        sizeY = round(prelim) 
    else:
        sizeX = round(prelim)
        sizeY = sizeX
    print("difference between datalength and map nodes is: ", sizeX*sizeY - matrixLen , "nodes")
    return sizeX, sizeY



def trainPLSOM(data, sig, max_iter, sizeX, sizeY, s, save):
    """Initializes and trains a PLSOM parametreless Self Organinzing Map
    data: Dataframe of inputs
    sig: neighbourhood upper range
    max_iter: training iterations
    sizeX: map size X
    sizeY: map size Y
    s: random seed
    save: if true saves model for later use
    
    returns trained model instance"""


    #Cast dataframe as array
    data_arr = np.asarray(data)
    
    #initialize map
    som = MiniSom_PLSOM(sizeX, sizeY, data_arr.shape[1], sigma=sig,
                  neighborhood_function='gaussian', random_seed=s)
    som.pca_weights_init(data_arr)
    som.train(data_arr, max_iter, True)
    
    round_s = round(sig, 3)
    
    
    if save:
        with open('./Search/%s x %s som %s %s iter%s.p' % (sizeX, sizeY, round_s, max_iter), 'wb') as outfile:
            pickle.dump(som, outfile)
    
    return som

def findWinners(data, som):
    """returns a dataframe with xCoord and yCoord BMU of samples on the som"""
    #Cast dataframe as array
    data_arr = np.asarray(data)
    
    #find winning node per item
    win_list = []
    for d in data_arr:
        win_list.append(som.winner(d))
    
    #Sort the lists together for plotting
    x = list(enumerate(win_list))
    x.sort(key=lambda tup:tup[1])
    indices, pos = zip(*x)
    
    #map that sorting to the datapoints
    idx = np.asarray(indices)

    winners_list = []
    for cnt, xx in enumerate(data_arr):
        #print(cnt)
        w = som.winner(xx)
        winners_list.append(w)
    
    #add coords to df
    
    coord_df = pd.DataFrame(winners_list, columns = ["xCoord", "yCoord"])
 
    ##this is a problem
    data_coord = data.join(coord_df, how = "right", on = None) 
    data_coord_sorted = data_coord.sort_values(["xCoord","yCoord"])
    
    return data_coord_sorted


def saveSom(som, sizeX, sizeY, max_iter, *fname):
    """saves trained SOM as pickle"""

    if fname:
        fname = "%s/%sx%s_map_hyperopt_iter%s.p" % (fname[0], sizeX, sizeY, max_iter)
    else:
        fname = "./Search/%sx%s_map_hyperopt_iter%s.p" % (sizeX, sizeY, max_iter)
    with open(fname, "wb") as outfile:
        pickle.dump(som, outfile)
        print("file saved")


def loadSom(fpath):
    """loads saved som from pickle"""
    with open(fpath, "rb") as infile:
        som = pickle.load(infile)
    return som


def getSomVectors(som, scaler, save, *fname):
    vecs = som.get_weights()
    sizeX = vecs.shape[0]
    sizeY = vecs.shape[1]
    
    vecs_flat = vecs.reshape(-1, vecs.shape[-1])

    vecs_scaled = applyInverseScaling(vecs_flat, scaler)
    vecs_scaled = vecs_scaled.round(2)

    if save:
        if fname:
            fname = "%s/%sx%s_vectors_som.csv" % (fname[0], sizeX, sizeY)
            np.savetxt(fname, vecs_scaled,fmt='%.2f', delimiter=",")
            print("file saved as: ", fname)
        np.savetxt("SOM_%sx%s.csv" % (sizeX, sizeY), vecs_scaled,fmt='%.2f', delimiter=",")


    vecs_scaled = vecs_scaled.reshape(vecs.shape)    
    return vecs_scaled, sizeX, sizeY 


def searchFor3Ints(vecs):

    vecs_df = pd.DataFrame(np.reshape(vecs, (vecs.shape[0]*vecs.shape[1],  vecs.shape[2])), columns=columns)
    #filter for a maximum of 2 interactions
    vecs_df['interactions'] = vecs_df.gt(0).sum(axis=1)
    vecs_df['interactions'] = vecs_df["interactions"] - 1
    over = vecs_df[vecs_df["interactions"] > 2]

    for i,row in over.iterrows():
        if row.BF < 0.03:
            row.BF = 0
        if row.WF < 0.03:
            row.WF = 0
        if row.SG < 0.03:
            row.SG = 0
        if row.CT < 0.03:
            row.CT = 0


    return over



#_________________________
#Plotting helper functions
#_________________________

def pcaPlot_df(df, coeff, scale, categ, labels = None):
    """plots PCA transformed data in 2d """
    df["xs"] = df[df.columns[0]]
    df["ys"] = df[df.columns[1]]
        
    n = coeff.shape[0]
    print(n)
    
    if scale == True:
        scalex = 1.0/(df["xs"].max() - df["xs"].min())
        scaley = 1.0/(df["ys"].max() - df["ys"].min())

        df["xs"] = scalex *  df["xs"] 
        df["ys"] = scaley * df["ys"] 
    
    g = sns.scatterplot(x='xs',y='ys', data=df, hue = df[categ])
    g.legend(title = "PCA", loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, frameon = False)

    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, df.columns[i], color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')




def pcaPlot3d_df(df, categ):
    """plots a df in 3d """
    import plotly.express as px
    from plotly.offline import iplot
    fig = px.scatter_3d(df, x= df.columns[0], y= df.columns[1] , z=df.columns[2],
              color= categ)
    fig.update_traces(marker=dict(size=2))
    return iplot(fig)



def plotUmatrix(som, sizeX, sizeY, max_iter, sigma, save, *fname):
    """Plots the UMATRIX of the som, saving optional"""
    plt.figure(figsize=(9, 9))
    plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
    plt.colorbar()
    plt.title('UM %s x %s Umatrix iter%s sigma%s' % (sizeX, sizeY, max_iter, round(sigma, 2)))
    if save:
        if fname:
            plt.savefig("%s/%s x %s Umatrix iter%s sigma%s.jpeg" % (fname[0], sizeX, sizeY, max_iter, round(sigma, 2)))
        else:
            plt.savefig("./Search/%s x %s Umatrix iter%s sigma%s.jpeg" % (sizeX, sizeY, max_iter, round(sigma, 2)))
    
    return som.distance_map().T
    

def plotParams(som, data, sizeX, sizeY, max_iter, sigma, scaler, save, *fname):
    """Plots the distribution of each parameter seperately on the map, saving optional"""

    W = som.get_weights()
    #print(np.shape(W))
    #THE weights need to be invrse scaled. 
    W_flat = W.reshape(-1, W.shape[-1])
    W_unscaled = applyInverseScaling(W_flat, scaler)
    W_unscaled = np.reshape(W_unscaled,W.shape)
    #print(W_unscaled)
    #print(np.shape(W_unscaled))

    plt.figure(figsize=(40, 8))
    feature_names= list(data.columns)
    for i, f in enumerate(feature_names):
        plt.subplot(1, len(feature_names) +1, i+1)
        plt.title(f)
        plt.pcolor(W_unscaled[:,:,i].T, cmap='coolwarm')
        plt.xticks(np.arange(sizeX+1))
        plt.yticks(np.arange(sizeY+1))
        plt.colorbar()
    plt.tight_layout()
    plt.suptitle('Parameter distribution %s x %s iter%s sigma%s' % (sizeX, sizeY, max_iter, round(sigma, 2)))
    if save:
        if fname:
            plt.savefig("%s/%s x %s Parameters iter%s sigma%s.jpeg" % (fname[0], sizeX, sizeY, max_iter, round(sigma, 2)))
        else:
            plt.savefig("./Search/%s x %s Parameters iter%s sigma%s.jpeg" % (sizeX, sizeY, max_iter, round(sigma, 2)))
    

def plotTrainingStats(df, sizeX, sizeY, max_iter, sigma,save, ts_factor = 0.99):
    """plots the training stats dataframe with smoothing, saving optional
    df : training dataframe returned from som.trainngStats
    ts_factor: float between 0 and 1 """
    fig, axes = plt.subplots(len(df.columns), figsize = (10,15))
    for i,w in enumerate(df.columns):
        smooth = (df.ewm(alpha=(1 - ts_factor)).mean())
        sns.lineplot(data = df.reset_index(), x = 'index', y = df[w], ax = axes[i], alpha = 0.4)
        sns.lineplot(data = df.reset_index(), x = 'index', y = smooth[w], ax = axes[i])
        axes[i].grid()
    
    plt.suptitle('TrainingStats %s x %s Umatrix iter%s sigma%s' % (sizeX, sizeY, max_iter, round(sigma, 2)))
    if save:
        plt.savefig("./Search/%s x %s Umatrix iter%s sigma%s.jpeg" % (sizeX, sizeY, max_iter, round(sigma, 2)))
    


def drawSOMnodesInPCA(data, sizeX, sizeY, som, pca_model, scale, max_iter, sigma, save):
    """Draws the SOM nodes in PCA space 2D"""

    #flatten map array
    weights = som.get_weights()
    weights = np.reshape(weights, ((sizeX * sizeY), data.shape[1]))    
    #pass through PCA model
    weighs_projected = pca_model.transform(weights)
    weighs_projected_df = pd.DataFrame(weighs_projected)
    
    #cast
    df = weighs_projected_df
    df["xs"] = df[df.columns[0]]
    df["ys"] = df[df.columns[1]]
        
    if scale == True:
        scalex = 1.0/(df["xs"].max() - df["xs"].min())
        scaley = 1.0/(df["ys"].max() - df["ys"].min())

        df["xs"] = scalex *  df["xs"] 
        df["ys"] = scaley * df["ys"] 
    
    #plot nodes
    fig, ax = plt.subplots(figsize=(10, 10))
    g = sns.scatterplot(x='xs',y='ys', data=df, ax=ax)

    
    #Partition list for lines one way
    listX = df["xs"].tolist()
    listY = df["ys"].tolist()
    
    new_x = [listX[i:i + (sizeY)] for i in range(0, len(listX), sizeY)]
    new_y = [listY[i:i + (sizeY)] for i in range(0, len(listX), sizeY)]

    for i in range(len(new_x)):
        plt.plot(new_x[i], new_y[i], color = "black")


    #transpose list for lines the other way
    list_x_t = np.array(new_x).T.tolist()
    list_y_t = np.array(new_y).T.tolist()
    
    for i in range(min(len(list_x_t), len(list_y_t))):
        plt.plot(list_x_t[i], list_y_t[i], color = "red")


    plt.title('SOM in PCA %s x %s iter%s sigma%s' % (sizeX, sizeY, max_iter, round(sigma, 2)))

    if save:    
        plt.savefig("./Search/%s x %s SOM in PCA iter%s sigma%s.jpg" % (sizeX, sizeY, max_iter, round(sigma,2)))
    
    return df


def drawSOMnodesInPCA_3d(data, sizeX, sizeY, som, pca_model, scale, max_iter, sigma, save):
    """Draws the SOM nodes in PCA space 3D
    data is a the standard scaled dataset
    sizeX and sizeY int size of the map
    som the trained PLSOM model
    pca_model the trained PCA model over standard scaled data
    max_iter is training iteratins
    sigma is som.thisSigma last one of the trained model
    save is a boolean"""

    #flatten map array
    weights = som.get_weights()
    weights = np.reshape(weights, ((sizeX * sizeY), data.shape[1]))    
    #pass through PCA model
    weighs_projected = pca_model.transform(weights)
    weighs_projected_df = pd.DataFrame(weighs_projected)
    weighs_projected_df["origin"] = "SOMnodes"

    #get PCA representation of dataset
    dataPCA = pca_model.transform(data)
    data_projected_df = pd.DataFrame(dataPCA)
    data_projected_df["origin"] = "Data"

    #concatenate
    df = pd.concat([weighs_projected_df, data_projected_df])

    if len(df.columns) <= 2:
        print("cannot visualize in 3d because it is only 2 components or lower")
        exit

    else:
        df= df.rename(columns={ df.columns[0]: "xs", df.columns[1]: "ys", df.columns[2]: "zs"})

    if scale == True:
        scalex = 1.0/(df["xs"].max() - df["xs"].min())
        scaley = 1.0/(df["ys"].max() - df["ys"].min())
        scaley = 1.0/(df["zs"].max() - df["zs"].min())

        df["xs"] = scalex *  df["xs"] 
        df["ys"] = scaley * df["ys"] 
        df["zs"] = scaley * df["zs"] 

    import plotly.express as px
    from plotly.offline import iplot
    fig = px.scatter_3d(df, x= df["xs"], y= df["ys"] , z=df["zs"], color= "origin")
    
    fig.update_traces(marker=dict(size=2))
    
    return iplot(fig)



def plotSOM_base(som, sizeX, sizeY, max_iter):
    """initializes SOM plot base """
    #define Matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(15, 15 * (sizeY/sizeX)))

    #add rectangle to plot
    ax.add_patch(Rectangle((0, 0), sizeX, sizeY,
             edgecolor = 'black',
             fill=False,
             lw=1))

    ax.set_xticks(np.arange(0, sizeX +1))
    ax.set_yticks(np.arange(0,sizeY+1))
    ax.set_title('model_ σ %s maxIter %s' % (round(som.thisSigma,2), max_iter))
    plt.close(fig)

    return fig, ax


def plotSOM_addData(som, data, fig, ax, color, size, alpha, marker, max_iter, sizeX, sizeY, save):
    """draws data on exisitng SOM base"""
    #get winners of the dataset
    data_coord = findWinners(data, som)
    data_coord.plot(x = "xCoord", y = "yCoord", ax = ax, kind = "scatter", alpha=alpha, c = color, s= size, marker = marker)

    if save:
        plt.savefig("./Search/%s x %s SOM iter%s sigma%s.jpg" % (sizeX, sizeY, max_iter, som.thisSigma))
    
    return data_coord


def markersFromIdx(df):
    idxL = list(df.index.values)
    markerL = []
    for i in idxL:
        markerL.append("$%s$" % i)
    
    return markerL
    

def plotSOM_addDataWithIdx(som, data, fig, ax, color, size, alpha, max_iter, sizeX, sizeY, save):
    """draws data on exisitng SOM base, and annotates based on idx of experiment data"""
    #get winners of the dataset
    data_coord = findWinners(data, som)
    mList = markersFromIdx(data_coord)

    for i,row in data_coord.iterrows():
        ax.plot(row["xCoord"], row["yCoord"], marker = mList[i], markersize = size, color = color)
        #data_coord.plot(x = "xCoord", y = "yCoord", ax = ax, kind = "scatter", alpha=alpha, c = color, s= size, marker = mList)

    # #add markers
    # for i, txt in enumerate(markersFromIdx(data_coord)):
    #     ax.annotate(txt, (data_coord["xCoord"].iloc[i], data_coord["yCoord"].iloc[i]))

    if save:
        plt.savefig("./Search/%s x %s SOM iter%s sigma%s.jpg" % (sizeX, sizeY, max_iter, som.thisSigma))
    
    return data_coord






def plotHitmap(som, data, sizeX, sizeY, save, *fname):
    """plots how mnay times a neuron in the map is activated"""
    plt.figure(figsize=(7, 7))
    data = np.asarray(data)
    frequencies = som.activation_response(data)
    maxHits = np.amax(frequencies)
    plt.pcolor(frequencies.T, cmap='Blues') 
    plt.colorbar(ticks = list(range(0,int(maxHits))))

    if save:
        if fname:
            plt.savefig("%s/%s x %s HitMap.jpeg" % (fname[0], sizeX, sizeY))
        else:
            plt.savefig("./Search/%s x %s HitMap iter%s sigma%s.jpeg" % (sizeX, sizeY))
    

    #plt.show()


##########################################################
#GRAPH 
########################################################


#i don't think this is what i am looking for, i need a graph of the hitdata itself
def buildGraph(som):
    # Create a graph from the MiniSOM map
    
    W = som.get_weights()
    graph = nx.Graph()

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            node = (i, j) 
            graph.add_node(node)
            if i > 0:
                graph.add_edge(node, (i-1, j))
            if j > 0:
                graph.add_edge(node, (i, j-1))
            if i < W.shape[0]-1:
                graph.add_edge(node, (i+1, j))
            if j < W.shape[1]-1:
                graph.add_edge(node, (i, j+1))

    # Calculate degree of connectivity for each node
    degrees = dict(graph.degree())

    # Set edge colors based on the degree of connectivity
    edge_colors = [degrees[edge[0]] + degrees[edge[1]] for edge in graph.edges()]

    # Visualize the graph
    plt.figure(figsize=(20, 20))
    # Pass edge_colors and edge_cmap arguments to the nx.draw() function

    nx.draw(graph, node_size=50, with_labels=True, edge_color=edge_colors, cmap = "coolwarm", width = 1.5)
    return graph



def graphFromDict(dict, *colors):
    """takes in a dictionary of the form {node1: [neigh1, neigh2, neigh3], node2: [neigh1, neigh2, neigh3]}
    and returns a graph with the nodes and edges"""
    G = nx.Graph()
    for key, value in dict.items():
        G.add_node(key)
        for i in value:
            G.add_edge(key, i)
    nx.draw(G, node_size=50, with_labels=True,  width = 1.5)
    return G


def getNeighborsGrah(data_noMat, data_ff_coord, data_rad, som_rad, som):
    """calculates the neighbor adjacency matrix based on a minimum distance radius
    both in hiD space and lowD space and returns two graphs with the distance encoded as color
    Blue> immeadiate neighbord, Red> 2nd order neighbor
    
    data_noMat: data without the material column
    data_ff_coord: data with the x and y coordinates of the winning neurons
    data_rad: radius in hiD space obtained from GetDatasetDistanceForGraph functon
    som_rad: radius in lowD space which is the square root of 2 because grid has fixed size"""
    from sklearn.neighbors import radius_neighbors_graph

    #Runn knn on original data, make sure to scale the radius because of wierd float rounds
    nbrs_data = radius_neighbors_graph(data_noMat, data_rad*1.01, metric = "euclidean",
                                        mode='distance', include_self=False)
    
    #make adjacency array and build graph for original data
    G_data = nx.from_numpy_matrix(nbrs_data.toarray())
    G_data.edges(data=True)   #adds weight as edge attribute
    print(G_data.edges(data=True))



    #Runn knn on SOM data
    dataSOM = np.asarray(data_ff_coord[["xCoord", "yCoord"]])
    #print(dataSOM)
    #FIND THE RIGHT DISTANCE HERE
    nbrs_som = radius_neighbors_graph(dataSOM, som_rad, metric = "euclidean",
                                        mode='distance', include_self=False)
    #print(nbrs_som.toarray())
    
    #make adjacency array and build graph for original data
    G_som = nx.from_numpy_matrix(nbrs_som.toarray())
    G_som.edges(data=True)   #adds weight as edge attribute
    print(G_som.edges(data=True))

    return G_data, G_som


def drawGraphs(G_data, G_som, k, save, *fname):
    """takes in two graphs and draws them side by side"""
    # create figure with two subplots
    fig, axes = plt.subplots(ncols=2, figsize=(20, 10))   
    pos_data = nx.spring_layout(G_data, k)
    nx.draw(G_data, with_labels=True, pos = pos_data,
             edge_color= [G_data[u][v]['weight'] for u,v in G_data.edges()],
                 edge_cmap=plt.cm.bwr, ax = axes[0])
    
    pos_som = nx.spring_layout(G_som, k)
    nx.draw(G_som, with_labels=True, pos = pos_som,
             edge_color= [G_som[u][v]['weight'] for u,v in G_som.edges()],
                    edge_cmap=plt.cm.bwr, ax = axes[1])
    axes[0].set_title("Neighbors in high dim space")
    axes[1].set_title("Neighbors in low dim space")
    if save:
        if fname:
            plt.savefig("%s/NeighborsGraph.jpeg" % (fname[0]))
        else:
            plt.savefig("./Search/NeighborsGraph.jpeg")

    return fig

def compareGraphs(G_data, G_som, k, save , *fname):    

    # Find edges that are common to both graphs
    common_edges = list(set(G_data.edges()).intersection(set(G_som.edges())))

    # Find edges that are only in G_som
    unique_edges_som = list(set(G_som.edges()).difference(set(G_data.edges())))

    # Find edges that are only in G_data
    unique_edges_data = list(set(G_data.edges()).difference(set(G_som.edges())))

    # Create a new graph with the combined edges and colors
    G_combined = nx.Graph()
    G_combined.add_edges_from([(u, v, {'color': 'green'}) for u, v in common_edges])
    G_combined.add_edges_from([(u, v, {'color': 'blue'}) for u, v in unique_edges_som])
    G_combined.add_edges_from([(u, v, {'color': 'red'}) for u, v in unique_edges_data])

    # Draw the combined graph with colored edges
    plt.figure(figsize=(10, 10))         
    # add legend
    blue_patch = plt.Line2D([], [], color='green', label='Common Neighbors')
    green_patch = plt.Line2D([], [], color='blue', label='LowD Neighbors')
    red_patch = plt.Line2D([], [], color='red', label='HighD Neighbors')
    lg = plt.legend(handles=[blue_patch, green_patch, red_patch], loc='upper right',
               bbox_to_anchor=(1.2, 1))
    plt.title("comparison of connectivity between HD and LD space")

    #Draw graph
    edge_colors = nx.get_edge_attributes(G_combined, 'color').values()
    pos_combi = nx.spring_layout(G_combined, k)
    nx.draw(G_combined, with_labels=True, edge_color=edge_colors, pos = pos_combi)
    


    DrawHistogramFromCombiGraph(common_edges, unique_edges_som, unique_edges_data, save, *fname)

    if save:
        if fname:
            plt.savefig("%s/NeighborsGraphComparison.jpeg" % (fname[0]),
                         bbox_extra_artists=(lg, ), bbox_inches='tight')
        else:
            plt.savefig("./Search/NeighborsGraphComparison.jpeg")

    



    return G_combined, common_edges, unique_edges_som, unique_edges_data



def graphWalk(data_noMat, data_ff_coord):
    from sklearn.neighbors import kneighbors_graph
    # Compute the k-nearest neighbors graph for each dataset
    k = 9  # number of neighbors
    G_noMat = nx.Graph(kneighbors_graph(data_noMat, k, mode='distance'))

    data_som = np.asarray(data_ff_coord[["xCoord", "yCoord"]])
    G_som = nx.Graph(kneighbors_graph(data_som, k, mode='distance'))

    #drawGraphs(G_noMat, G_som, 0.5, save = False)


    # Compute the graph distances between pairs of nodes in each graph
    D_noMat = dict(nx.all_pairs_shortest_path_length(G_noMat))
    D_som = dict(nx.all_pairs_shortest_path_length(G_som))


    # Create a dictionary of distances for each pair of nodes with the same name
    for node in list(G_noMat.nodes())[0:2]:
        D1 = D_noMat[node]
        D1 = {k: v for k, v in D1.items() if v < 2}

        D2 = D_som[node] 
        D2 = {k: v for k, v in D1.items() if v < 2}

        print(D1)
        print(D2)



def graphWalkfromGrah(G_data, G_som):
    # Compute the graph distances between pairs of nodes in each graph
    D_data = dict(nx.all_pairs_shortest_path_length(G_data))
    D_som = dict(nx.all_pairs_shortest_path_length(G_som))


    # Create a dictionary of distances for each pair of nodes with the same name
    for node in list(G_data.nodes())[0:2]:
        D1 = D_data[node]
        #D1 = {k: v for k, v in D1.items() if v < 2}

        D2 = D_som[node] 
        #D2 = {k: v for k, v in D1.items() if v < 2}

        print(D1)
        print(D2)








def DrawHistogramFromCombiGraph(common_edges, unique_edges_som, unique_edges_data, save, *fname):
    """plots the edge count stat of the combiGraph"""
    #plot the histogram of the edge count
    fig, ax = plt.subplots()
    ax.bar([1,2],
             [len(common_edges), len(unique_edges_data)], 
             color = ['green', 'red'])
    ax.set_xlabel("Type")
    ax.set_ylabel("Count")
    ax.set_xticks([1,2],["Common", "HighD"])
    ax.set_xlim(0, 4)
    ax.set_title("Edge count in the combined graph")
    if save:
        if fname:
            plt.savefig("%s/NeighborsGraphComparisonChart.jpeg" % (fname[0]))
        else:
            plt.savefig("./Search/NeighborsGraphComparisonChart.jpeg")

    plt.show()



def getDatasetDistanceForGraph(data):
    """takes in a dataset and returns the radius value for the graph"""
    from scipy.spatial import distance

    #calculate the distance matrix
    dist_data = distance.pdist(data, metric='euclidean')
    dist_data = np.sort(dist_data)
    return dist_data[0]





#______________________________
#Error metrics helper functions
#______________________________



def getQuantError(som, data):
    """returns quantization error of the trained map for a dataset"""
    data_arr = np.asarray(data)
    q_e= som.quantization_error(data_arr)

    return q_e

def getTopoError(som, data):
    """returns topographic error of the trained map for a dataset"""
    data_arr = np.asarray(data)
    t_e= som.topographic_error(data_arr)

    return t_e

def getPPC(som, data, sizeX, sizeY, plot = True):
    """returns difference in mean and variance between the ff dataset and the som vectors 
    as tuple(mean_diff,std_diff)
    both should tend to 0 if the distribution has been reproduced  """

    #get map node weights
    weights = som.get_weights()
    
    weights = np.reshape(weights, ((sizeX * sizeY), len(data.columns)))
    weights_df = pd.DataFrame(weights, columns = list(data.columns.values)) 
    stats_SOM = pd.DataFrame(weights_df.describe())
    stats_dataFF = pd.DataFrame(data.describe())
    diff = stats_SOM.subtract(stats_dataFF)
    a = diff.iloc[1].to_numpy()
    b = diff.iloc[2].to_numpy()
    d_mstd = abs(np.concatenate((a, b)))

    if plot:
        fig, axes = plt.subplots(len(data.columns), figsize = (19,20))
        for i,w in enumerate(data.columns):
            sns.distplot(data[w], kde=True, ax=axes[i], color = "red", label = "data")
            sns.distplot(weights_df[w], kde=True, ax = axes[i], label = "map_nodes")
            axes[i].legend(loc="upper right")
        plt.savefig("./Search/%s x %s Distributions sigma%s.jpeg" % (sizeX, sizeY, round(som.thisSigma, 2)))
    


    return d_mstd


def getMapNodeStats(som,data, sizeX, sizeY):
    """returns metrics about the ff data distribution over the nodes
    spread_r is the average number of data points per map node divided by max number of data points per node
    deadNodes_r is the ratio of inactive nodes over total number of nodes in the map"""
    
    data_arr = np.asarray(data)
    wmap = som.win_map(data_arr)
    n_nodes = sizeX * sizeY
    gatherList = []
    for i in wmap.values():
        gatherList.append(len(i))   
    

    av_dpn = np.average(gatherList)
    max_dpn = np.max(gatherList)
    spread = abs((av_dpn/max_dpn) - 1)

    deadNodes = n_nodes - (len(wmap.keys()))
    deadNodes_r = round((n_nodes - (len(wmap.keys())) ) / n_nodes, 4)

    #print(["av_dpn", av_dpn, "max_dpn", max_dpn, "spread", spread,"deadNodes", deadNodes_r])


    return spread, deadNodes_r


def getMapMetrics(som, data, sizeX, sizeY):
    q_e = getQuantError(som, data)
    t_e = getTopoError(som, data)
    d_mstd = getPPC(som, data, sizeX, sizeY, plot = False)
    spread, deadNodes_r = getMapNodeStats(som, data, sizeX, sizeY)


    all_metrics = {"qe" : q_e,
                    "te": t_e,
                    "mean_diff": d_mstd[0],
                    "std_diff" : d_mstd[1],
                    "spread" : spread,
                    "deadNodes" : deadNodes_r}

    #all = np.asarray([q_e, t_e, d_mstd[0],d_mstd[1], spread,deadNodes_r])
    #print("combined cost", np.linalg.norm(all))
    print(all_metrics)
    return all_metrics

#_________________________
#Hyperopt helper functions
#_________________________



def defHyperoptSpace(param):
    """use for hyperopt search
    returns hyperopt search space dictionary 
    param is a list of tuples with ("name of param", lower bound, upper bound, quantization)"""
    space = {}
    for p in param:
        space[p[0]] = hp.quniform((p[0]), p[1],p[2],p[3])

    return space 


def plotHyperoptSpace(param):
    """Plots the distribution of search space parameters for check"""
    for p in param:
        print(p)
        values = []
        space = hp.quniform((p[0]), p[1],p[2],p[3])
        for x in range(1000):
            values.append(hyperopt.pyll.stochastic.sample(space))
        df = pd.DataFrame(values, columns=["var1"])
        df.hist(bins= np.arange(p[1],p[2],p[3]))
        plt.xticks(np.arange(p[1],p[2],p[3]), rotation = -60)
        #done
        

def setLargerMapSize(data, multiplier):
    """returns the map size for the hyperopt iteration
    data: DF of inputs
    multiplier: total number of nodes wrt data length
    returns sizeX and sizeY"""
    from math import sqrt

    matrixLen = data.shape[0]
    print(["matrix len", matrixLen])
    newlen = round(matrixLen * multiplier)

    prelim = sqrt(newlen)
    if round(prelim)**2 < newlen:
        sizeX = round(prelim) +1
        sizeY = round(prelim) 
    else:
        sizeX = round(prelim)
        sizeY = sizeX
    #print("difference between datalength and map nodes is: ", sizeX*sizeY - newlen , "nodes")
    return sizeX, sizeY


def setSigmaUpper(sizeX, sizeY, multiplier2 = 1):
    sigma = max(sizeX, sizeY) * multiplier2
    return sigma
    

def getLoss(metrics, loss, mode):
    """use for hyperopt search 
    returns the combined loss off the chosen som metrics
    metrics is the loss dictionary from """
    metrics_chosen = {k: v for k, v in metrics.items() if k in loss}
    
    if mode == "sum":
        metrics_chosen["loss"] = sum(metrics_chosen.values())
    elif mode == "norm":
        metrics_chosen["loss"] = np.linalg.norm(np.asarray(list(metrics_chosen.values())))

    print(metrics_chosen)
    return metrics_chosen


def unpack(x):
    """helper function for unpack_trialStats()"""
    if x:
        return x[0]    
    return np.nan    


def unpackTrialStats(trials, losslist):
    """unpacks and sorts the stats and metrics of the bayesian optimization search
    takes trials a dictionary produced by hyperopt during the search
    loss: list of strings metrics to consider for loss - see getMapMetrics for names
    """
    
    trials_df = pd.DataFrame([pd.Series(t["misc"]["vals"]).apply(unpack) for t in trials])

    losslist.append("loss")
    for l in losslist:
        try:
            trials_df[l] = [t["result"][l] for t in trials]
        except:
            pass
    
    trials_df["trial_number"] = trials_df.index
    trials_df = trials_df.round(4)
    trials_sorted = trials_df.sort_values(by = "loss")

    return trials_sorted


def getBestMap(data, m_1, m_2, max_iter_som, s):
    #build and train map
    sizeX, sizeY = setLargerMapSize(data, m_1)
    sigma = round(setSigmaUpper(sizeX,sizeY, m_2))
    print(["multiplier_1", m_1, "sizeX", sizeX,"m_2", m_2, "sgimaU", sigma])

    som = trainPLSOM(data, sigma, max_iter_som, sizeX, sizeY, s, False)
    
    return som, sizeX, sizeY




def RunHyperopt_PLSOM(data, param, max_iter_som, s, max_iter_BO, losslist, mode, save ):
    """perform bayesian optimizaiton to tune map size according to target loss
    data: standardized dataset
    name: list, name of search space variable
    bounds: list of tupes, bounds of search space variable
    max_iter_som training of SOM map
    s: seed of som training
    max_iter_BO: number of iterations for bayesian optimization
    loss: list of strings metrics to consider for loss - see getMapMetrics for names
    mode: string either sum or norm, used to define the combined loss
    returns trials_df with all trials and best trial """
    #Since PSLOM's only hyperparameter is the neighborhood range, we vary the mapsize

    #define search space
    space = defHyperoptSpace(param)
    
    #define objective function
    def objective(space):
        #read design space var
        m_1 = space["multiplier_1"]
        m_2 = space["multiplier_2"]

        #build and train map
        global sizeX, sizeY
        sizeX, sizeY = setLargerMapSize(data, m_1)
        
        sigma = round(setSigmaUpper(sizeX, sizeY, m_2))
        print(["multiplier_1", m_1, "sizeX", sizeX,"m_2", m_2, "sgimaU", sigma])
        som = trainPLSOM(data, sigma, max_iter_som, sizeX, sizeY, s, False)

        #evaluate map metrics
        metrics = getMapMetrics(som, data, sizeX, sizeY)
        costVal = getLoss(metrics, losslist, mode)
        costVal["status"] = STATUS_OK
        return costVal 

    #Run bayesian optimization
    trials = Trials()
    best = fmin(fn = objective,
               space = space,
               algo = tpe.suggest,
               max_evals = max_iter_BO,
               trials = trials)
    print("best: {}".format(best))
    
    #Unpack results
    trials_df = unpackTrialStats(trials, losslist)
    
    #in the combi test search we are saving outside 
    if save:
        pass
        #trials_df.to_csv("./Search/%sx%s_map_hyperopt_iter%s.csv" % (sizeX, sizeY, max_iter_som))
    
    return trials_df, best


def drawDendogram(data, som, scaler, save, *fname):
    """calculates the eucledian distance matrix and 
    draws a dendogram of the map nodes in low dimensional space and high dimensional space
    returns the cophenetic correlation coefficient for both dendograms
    data: standardized dataset
    som: trained som
    save: boolean the plot as png"""

    #This uses the standard scaled data 
    #get distance matrix
    vecs = getSomVectors(som, scaler, False)[0]
    vecs = applyScaling(vecs.reshape(-1, vecs.shape[-1]), scaler, False)
    data = np.asarray(data)

    # print(pd.DataFrame(vecs).describe())
    # print(pd.DataFrame(data.reshape(-1, data.shape[-1])).describe())

    #calculate distance matrix with best linkage
    dist_som = pdist(vecs,
                        metric = "euclidean")
    dist_data = pdist(data.reshape(-1, data.shape[-1]),
                        metric = "euclidean")

    linkages = ["single", "complete", "average", "weighted", "centroid", "median", "ward"]
    keeper = dict({"linkage": [], "c_som": [], "c_data": []})
    for l in linkages:
        #calculate linkage
        Z_som = linkage(dist_som, method = l)
        Z_data = linkage(dist_data, method = l)

        # Calculate the cophenetic correlation coefficients
        c_som, coph_dists1 = cophenet(Z_som, pdist(vecs))
        c_data, coph_dists2 = cophenet(Z_data, pdist(data))

        keeper["linkage"].append(l)
        keeper["c_som"].append(c_som)
        keeper["c_data"].append(c_data)
    
    keeper_df = pd.DataFrame(keeper)
    keeper_df.sort_values(by = ["c_som", "c_data"], ascending = False, inplace = True) 
    print(keeper_df)

    best_linkage = keeper_df.iloc[0,0]
    #calculate linkage
    Z_som = linkage(dist_som, method = best_linkage)
    Z_data = linkage(dist_data, method = best_linkage)

    #plot dendogram
    fig, ax = plt.subplots(1,2, figsize = (20,40))
    dendrogram(Z_som, ax = ax[0], orientation="left", truncate_mode = "none", show_contracted= True)
    dendrogram(Z_data, ax = ax[1], orientation="right", truncate_mode = "none", show_contracted= True)

    ax[0].set_title('SOM dendogram \n cophoenetic correlation coefficient: %s' % (round(c_som, 3)))
    ax[1].set_title('Data dendogram \n cophoenetic correlation coefficient: %s' % (round(c_data, 3)))

    if save:
        if fname:
            plt.savefig("%s/Dendogram.jpeg" % (fname[0]))
        plt.savefig("./Search/dendogram.png")   
    plt.show()

    # Calculate the cophenetic correlation coefficients
    c_som, coph_dists1 = cophenet(Z_som, pdist(vecs))
    c_data, coph_dists2 = cophenet(Z_data, pdist(data))

    # Print the correlation coefficients
    print(best_linkage)
    print("Cophenetic correlation coefficient for som dendrogram:", c_som)
    print("Cophenetic correlation coefficient for data dendrogram:", c_data)

    return c_som, c_data



#_________________________
#Making DOE  matrix 
# #_________________________



def makeLevels(dicty):
    """takes dictionary with the following format
    key "matrix col name
    values as list [start, end, step]
    returns oriented dictionary"""

    df = pd.DataFrame()

    for key in dicty:
        start, end, step = dicty[key]
        values = list(range(start, end + step, step))
        df[key] = values
    
    d = df.to_dict(orient = "list")

    return d

def filterInteractions(df, maxInt, plot):
    """ takes a full factorial DOE dataframe and applies experiment filters
    returns dataset df"""

    #filter if the percentages sum are larger than 1
    df["sum"] = df.sum(axis=1)
    df = df[df["sum"] == 100]
    df = df.drop(columns = "sum")
    
    #filter for a maximum of 2 interactions
    df['interactions'] = df.gt(0).sum(axis=1)
    df['interactions'] = df["interactions"] - 1
    if plot:
        df['interactions'].hist(bins = [0,1,2,3,4])
    df = df[df["interactions"] <= maxInt]
    df = df.drop(columns = "interactions")

    df = df.reset_index()

    return df




def makeFFdesign(dicty, maxInt,plot,  save, *fname):
    "returns filtered FF dataset df from levels dictionary"

    from doepy import build

    d = makeLevels(dicty)
    ff = build.full_fact(d)
    ff = filterInteractions(ff, maxInt, plot)
    ff = ff/100

    if save:
        ff.to_csv("%s.csv" % fname)

    return ff


def makeCCdesign(dicty, maxInt, plot, save, *fname):
    "returns filtered CC dataset df from levels dictionary"

    from doepy import build

    d = makeLevels(dicty)
    cc = build.central_composite(d)
    cc = filterInteractions(cc, maxInt, plot)
    cc = cc/100

    if save:
        cc.to_csv("%s.csv" % fname)

    return cc


def generateDataset(dicty, maxInt, plot,  fname):
    ff = makeFFdesign(dicty, maxInt, plot, True, "%s_ff" % fname)
    #print(ff)
    cc = makeCCdesign(dicty, maxInt, plot,True, "%s_cc" % fname)
    #print(cc)

    return ff, cc




def plotOptimization_2d(losslist, trials_df, top, save, max_iter):
    """Plots the results of the bayesian search"""

    
    #get top5
    topbest = trials_df.iloc[:top]
    topbest = topbest.reset_index(drop = True)

    for m in losslist:
        # plotly express does not support contour plots so we will use `graph_objects` instead. `go.Contour
        # automatically interpolates "z" values for our loss.
        fig = go.Figure(
            data=go.Contour(
                z=trials_df[m],
                x=trials_df["sig"],
                y=trials_df["learning_rate"],
                contours=dict(
                    size = 0.01, 
                    showlabels=True,  # show labels on contours
                    labelfont=dict(size=12, color="white",),  # label font properties
                ),
                colorbar=dict(title=m, titleside="right",),
                hovertemplate="metric: %{z}<br>sig: %{x}<br>n_lr: %{y}<extra></extra>",
            )
        )


        fig.add_trace(
        go.Scatter(
            x=topbest["sig"],
            y=topbest["learning_rate"],
            marker = dict(color = "green"),
            text = topbest[m],
            hovertemplate = "metric: %{text}<br>sig: %{x}<br>n_lr: %{y}<extra></extra>",
            showlegend=False)
        )



        fig.update_layout(
            xaxis_title="sigma",
            yaxis_title="learning_rate",
            title={
                "text": "%s over hyperparameter optimization" % m,
                "xanchor": "center",
                "yanchor": "top",
                "x": 0.5,
            },
        )

        fig.show()
        
        if save:
            fig.write_image("./Search/%s x %s map %s over hyperparameter optimization iter%s.jpeg" % (sizeX, sizeY, m, max_iter))
    







# #OLD VERSION
# def getMetrics(som, data_arr, sizeX, sizeY, sigma, learning_rate, max_iter):
#     #get error
#         q_e= som.quantization_error(data_arr)
#         t_e = som.topographic_error(data_arr)
#         #get avg data per node
#         wmap = som.win_map(data_arr)
#         n_nodes = sizeX * sizeY

#         gatherList = []
#         for i in wmap.values():
#             gatherList.append(len(i))
#         av_dpn = np.average(gatherList)
#         av_dpn_r = round((av_dpn / n_nodes),4) 
#         #DECIDE IF PERCENTAGE OR RATIO
#         #av_dpn_r = round((av_dpn / n_nodes) *100,2)  
#         deadNodes = n_nodes - (len(wmap.keys()))
#         deadNodes_r = round((n_nodes - (len(wmap.keys())) ) / n_nodes, 4)

#         #make combined loss 
#         metrics = np.asarray([q_e, t_e, av_dpn_r, deadNodes_r])
#         val = np.linalg.norm(metrics)
        
#         df_row = [sigma, learning_rate, max_iter, sizeX, sizeY, q_e, t_e, av_dpn_r, av_dpn, deadNodes_r, deadNodes, val]
#         metrics_df = pd.DataFrame([df_row], columns = ["sigma", "lr", "max_iter", "sizeX", "sizeY", "qe", "te", "Crowd Ratio", "crowd", "Dn ratio", "DN", "loss"])
#         metrics_df.to_csv('./Search/%s x %s model_%s_%s iter%s.csv' % (sizeX, sizeY, sigma, learning_rate, max_iter))
        
#         return df_row, val



def autopct(pct): # only show the label when it's > 10%
    return ("%1.0f%%" % pct) if pct > 1 else ''


def drawReceipe(sizeX, sizeY, data_ff_sorted, data_test_plot):
    for x in range(sizeX):
        for y in range(sizeY):
            subDF1 = data_ff_sorted.loc[(data_ff_sorted["xCoord"] == x) & (data_ff_sorted["yCoord"] == y)]
            subDF1.insert(len(subDF1.columns),"origin","ff")
            subDF2 = data_test_plot.loc[(data_test_plot["xCoord"] == x) & (data_test_plot["yCoord"] == y)]
            subDF2.insert(len(subDF2.columns),"origin","jmp")
            subDF2 = subDF2.drop(columns = ["Youngs", "Shrink", "Weight"])

            merged = pd.DataFrame()
            merged = merged.append(subDF1)
            merged = merged.append(subDF2)

            #print(x,y)
            #print(merged)
            rowNum = merged.shape[0]
            titles = merged["origin"].tolist()
            dic = {"ff" : "red", "jmp":"pink"}
            cols = merged["origin"].map(dic).tolist()
            #print(cols)
            merged = merged.drop(columns = ["xCoord", "yCoord", "origin"])

            if rowNum > 1:
                ax_unsorted = merged.T.plot.pie(subplots = True, ylabel = " ", layout = [1, rowNum], figsize=(10,10), legend = False, labeldistance = 1.1, autopct=autopct)
                for i in range(len(ax_unsorted[0])):
                    ax_unsorted[0][i].set_title(titles[i])
                    ax_unsorted[0][i].set_facecolor((1.0, 0.47, 0.42))
                fig = plt.gcf()
                fig.suptitle("Node (%s,%s)" % (x,y), y = 0.7)
    
    
    
def combineCSVfiles(dir_path):
    # list to store files
    res = []
    # Iterate directory
    for file in os.listdir(dir_path):
        # check only text files
        if file.endswith('.csv'):
            res.append(file)
    print(res)
    
    
    allStats = pd.DataFrame(columns = ["sigma", "lr","max_iter", "qe", "te", "Crowd Ratio", "crowd", "Dn ratio", "DN", "combined loss"])
    for i in range(len(res)):
        path = dir_path + res[i]
        df = pd.read_csv(path)
        df = df[["sigma", "lr", "max_iter", "qe", "te", "Crowd Ratio", "crowd", "Dn ratio", "DN", "combined loss"]]
        #print(df)
        df = df.iloc[[0]].reset_index(drop = True)
        allStats = pd.concat([allStats, df])

    print(allStats)  
    return allStats





