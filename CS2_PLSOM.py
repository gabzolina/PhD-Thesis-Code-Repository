from numpy import (array, unravel_index, nditer, linalg, random, subtract, max,
                   power, exp, zeros, ones, arange, outer, meshgrid, dot,
                   logical_and, mean, cov, argsort, linspace, transpose,
                   einsum, prod, nan, sqrt, sum, hstack, diff, argmin, multiply,
                   nanmean, nansum, tile, array_equal, apply_along_axis, shape)
from numpy.linalg import norm
from collections import defaultdict, Counter
from warnings import warn
from sys import stdout
from time import time
from datetime import timedelta
import pickle
import os
from pandas import DataFrame

# for unit tests
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from numpy.testing import assert_array_equal
import unittest
import math

"""
    Minimalistic implementation of the Self Organizing Maps (SOM).
"""


def _build_iteration_indexes(data_len, num_iterations,
                             verbose=False, random_generator=None,
                             use_epochs=False):
    """Returns an iterable with the indexes of the samples
    to pick at each iteration of the training.

    If random_generator is not None, it must be an instance
    of numpy.random.RandomState and it will be used
    to randomize the order of the samples."""
    if use_epochs:
        iterations_per_epoch = arange(data_len)
        if random_generator:
            random_generator.shuffle(iterations_per_epoch) 
        iterations = tile(iterations_per_epoch, num_iterations)
    else:
        iterations = arange(num_iterations) % data_len
        if random_generator:
            random_generator.shuffle(iterations)
    if verbose:
        return _wrap_index__in_verbose(iterations)
    else:
        return iterations


def _wrap_index__in_verbose(iterations):
    """Yields the values in iterations printing the status on the stdout."""
    m = len(iterations)
    digits = len(str(m))
    progress = '\r [ {s:{d}} / {m} ] {s:3.0f}% - ? it/s'
    progress = progress.format(m=m, d=digits, s=0)
    stdout.write(progress)
    beginning = time()
    stdout.write(progress)
    for i, it in enumerate(iterations):
        yield it
        sec_left = ((m-i+1) * (time() - beginning)) / (i+1)
        time_left = str(timedelta(seconds=sec_left))[:7]
        progress = '\r [ {i:{d}} / {m} ]'.format(i=i+1, d=digits, m=m)
        progress += ' {p:3.0f}%'.format(p=100*(i+1)/m)
        progress += ' - {time_left} left '.format(time_left=time_left)
        stdout.write(progress)


def fast_norm(x):
    """Returns norm-2 of a 1-D numpy array.
    """
    return sqrt(dot(x, x.T))

def asymptotic_decay(learning_rate, t, max_iter):
    """Decay function of the learning process.
    Parameters
    ----------
    learning_rate : float
        current learning rate.

    t : int
        current iteration.

    max_iter : int
        maximum number of iterations for the training.
    """
    return learning_rate / (1+t/(max_iter/2))


#################################################GABRIELLA EDITS NEW CLASS
class MiniSom_PLSOM(object):
    def __init__(self, x, y, input_len, sigma,
                 neighborhood_function='gaussian', topology='rectangular',
                 activation_distance='euclidean', random_seed=None):
        """Initializes a Self Organizing Maps.
        Parameters
        ----------
        x : int
            x dimension of the SOM.

        y : int
            y dimension of the SOM.

        input_len : int
            Number of the elements of the vectors in input.

        sigma : neighborhood range as defined by Berglund (he calls it beta, but we keep Minisom naming)
                There is currently no known analytical method of finding the single parameter to the PLSOM2 algorithm, β. 
                Some heuristic methods do, however, exist: 
                • The optimal value of β is dependent on the size of the network. 
                A value of β = M, where M is the largest dimension in output space, provides a possible starting point for the search. 
                • If the map does not become ordered or over-fits, increase the value of β and restart. 
                • If the map over-reacts to some inputs, decrease the value of β and restart.

        There is no learning rate
        The decay function is given relative to the normalized fitting error/dataset diameter

        neighborhood_function : string, optional (default='gaussian')
            Function that weights the neighborhood of a position in the map.
            Possible values: 'gaussian', 'mexican_hat', 'bubble', 'triangle'

        topology : string, optional (default='rectangular')
            Topology of the map.
            Possible values: 'rectangular', 'hexagonal'

        activation_distance : string, callable optional (default='euclidean')
            Distance used to activate the map.
            Possible values: 'euclidean', 'cosine', 'manhattan', 'chebyshev'

            Example of callable that can be passed:

            def euclidean(x, w):
                return linalg.norm(subtract(x, w), axis=-1)

        random_seed : int, optional (default=None)
            Random seed to use.
        """

        self._random_generator = random.RandomState(random_seed)

        #self._learning_rate = learning_rate
        
        self._sigma = sigma
        self._input_len = input_len
        # random initialization
        self._weights = self._random_generator.rand(x, y, input_len)*2-1
        self._weights /= linalg.norm(self._weights, axis=-1, keepdims=True)

        self._activation_map = zeros((x, y))
        self._neigx = arange(x)
        self._neigy = arange(y)  # used to evaluate the neighborhood function

        if topology not in ['hexagonal', 'rectangular']:
            msg = '%s not supported only hexagonal and rectangular available'
            raise ValueError(msg % topology)
        self.topology = topology
        self._xx, self._yy = meshgrid(self._neigx, self._neigy)
        self._xx = self._xx.astype(float)
        self._yy = self._yy.astype(float)
        if topology == 'hexagonal':
            self._xx[::-2] -= 0.5
            if neighborhood_function in ['triangle']:
                warn('triangle neighborhood function does not ' +
                     'take in account hexagonal topology')

        neig_functions = {'gaussian': self._gaussian,
                          'mexican_hat': self._mexican_hat,
                          'bubble': self._bubble,
                          'triangle': self._triangle}

        if neighborhood_function not in neig_functions:
            msg = '%s not supported. Functions available: %s'
            raise ValueError(msg % (neighborhood_function,
                                    ', '.join(neig_functions.keys())))

        if neighborhood_function in ['triangle',
                                     'bubble'] and (divmod(sigma, 1)[1] != 0
                                                    or sigma < 1):
            warn('sigma should be an integer >=1 when triangle or bubble' +
                 'are used as neighborhood function')

        self.neighborhood = neig_functions[neighborhood_function]

        distance_functions = {'euclidean': self._euclidean_distance,
                              'cosine': self._cosine_distance,
                              'manhattan': self._manhattan_distance,
                              'chebyshev': self._chebyshev_distance}

        if isinstance(activation_distance, str):
            if activation_distance not in distance_functions:
                msg = '%s not supported. Distances available: %s'
                raise ValueError(msg % (activation_distance,
                                        ', '.join(distance_functions.keys())))

            self._activation_distance = distance_functions[activation_distance]
        elif callable(activation_distance):
            self._activation_distance = activation_distance

    def get_weights(self):
        """Returns the weights of the neural network."""
        return self._weights

    def get_euclidean_coordinates(self):
        """Returns the position of the neurons on an euclidean
        plane that reflects the chosen topology in two meshgrids xx and yy.
        Neuron with map coordinates (1, 4) has coordinate (xx[1, 4], yy[1, 4])
        in the euclidean plane.

        Only useful if the topology chosen is not rectangular.
        """
        return self._xx.T, self._yy.T

    def convert_map_to_euclidean(self, xy):
        """Converts map coordinates into euclidean coordinates
        that reflects the chosen topology.

        Only useful if the topology chosen is not rectangular.
        """
        return self._xx.T[xy], self._yy.T[xy]

    def _activate(self, x):
        """Updates matrix activation_map, in this matrix
           the element i,j is the response of the neuron i,j to x."""
        self._activation_map = self._activation_distance(x, self._weights)

    def activate(self, x):
        """Returns the activation map to x."""
        self._activate(x)
        return self._activation_map

    def _gaussian(self, c, sigma):
        """Returns a Gaussian centered in c."""
        d = 2*sigma*sigma
        ax = exp(-power(self._xx-self._xx.T[c], 2)/d)
        ay = exp(-power(self._yy-self._yy.T[c], 2)/d)
        return (ax * ay).T  # the external product gives a matrix

    def _mexican_hat(self, c, sigma):
        """Mexican hat centered in c."""
        p = power(self._xx-self._xx.T[c], 2) + power(self._yy-self._yy.T[c], 2)
        d = 2*sigma*sigma
        return (exp(-p/d)*(1-2/d*p)).T

    def _bubble(self, c, sigma):
        """Constant function centered in c with spread sigma.
        sigma should be an odd value.
        """
        ax = logical_and(self._neigx > c[0]-sigma,
                         self._neigx < c[0]+sigma)
        ay = logical_and(self._neigy > c[1]-sigma,
                         self._neigy < c[1]+sigma)
        return outer(ax, ay)*1.

    def _triangle(self, c, sigma):
        """Triangular function centered in c with spread sigma."""
        triangle_x = (-abs(c[0] - self._neigx)) + sigma
        triangle_y = (-abs(c[1] - self._neigy)) + sigma
        triangle_x[triangle_x < 0] = 0.
        triangle_y[triangle_y < 0] = 0.
        return outer(triangle_x, triangle_y)

    def _cosine_distance(self, x, w):
        num = (w * x).sum(axis=2)
        denum = multiply(linalg.norm(w, axis=2), linalg.norm(x))
        return 1 - num / (denum+1e-8)

    def _euclidean_distance(self, x, w):
        return linalg.norm(subtract(x, w), axis=-1)

    def _manhattan_distance(self, x, w):
        return linalg.norm(subtract(x, w), ord=1, axis=-1)

    def _chebyshev_distance(self, x, w):
        return max(subtract(x, w), axis=-1)

    def _check_iteration_number(self, num_iteration):
        if num_iteration < 1:
            raise ValueError('num_iteration must be > 1')

    def _check_input_len(self, data):
        """Checks that the data in input is of the correct shape."""
        data_len = len(data[0])
        if self._input_len != data_len:
            msg = 'Received %d features, expected %d.' % (data_len,
                                                          self._input_len)
            raise ValueError(msg)

    def winner(self, x):
        """Computes the coordinates of the winning neuron for the sample x."""
        self._activate(x)
        return unravel_index(self._activation_map.argmin(),
                             self._activation_map.shape)

    ################################################# GR EDITS
    #PLSOM specific helper functions    
    def euclidean_distance_lmb(self, arr1, arr2):
        """Just a lambda function for the pt to pt distance in getDatasetDiameter(data)."""
        return sqrt(sum((arr1 - arr2)**2, axis=-1))

    def getDatasetDiameter(self, data):
        """Finds the larest distance between any two points in the dataset array"""        

        n = data.shape[0]
        pairwise_distances = zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                pairwise_distances[i, j] = linalg.norm(data[i]-data[j])
                pairwise_distances[j, i] = pairwise_distances[i, j]
        diameter = max(pairwise_distances)

        return diameter

    def getError(self, x):
        """Computes distance in output space between Winning Node Vector (BMU) and datapoint""" 
        #this doesnt work because quantization needs the whole dataset 
        win_coord = self.winner(x)
        win_qvect = self._weights[win_coord]
        err = self.euclidean_distance_lmb(win_qvect, x)
        return err
    
    def getEpsilon(self, err, datasetDiam):
        """returns epsilon(t) which is the smallest error divided by the set dataset diameter"""
        epsilon = err/datasetDiam
        epsilon = min(epsilon , 1)
        return epsilon

    def getNeiSize(self, epsilon, sigma):
        """Returns the neighborhood scaling 
        sigma here is the neighborhood range"""
        neiSize = sigma * math.log(1 + epsilon * (math.e -1) )
        
        return neiSize

    #########################################################################
    
    def quantization(self, data):
        """Assigns a code book (weights vector of the winning neuron)
        to each sample in data."""
        self._check_input_len(data)
        winners_coords = argmin(self._distance_from_weights(data), axis=1)
        return self._weights[unravel_index(winners_coords,
                                           self._weights.shape[:2])]


    def random_weights_init(self, data):
        """Initializes the weights of the SOM
        picking random samples from data."""
        self._check_input_len(data)
        it = nditer(self._activation_map, flags=['multi_index'])
        while not it.finished:
            rand_i = self._random_generator.randint(len(data))
            self._weights[it.multi_index] = data[rand_i]
            it.iternext()

    def pca_weights_init(self, data):
        """Initializes the weights to span the first two principal components.

        This initialization doesn't depend on random processes and
        makes the training process converge faster.

        It is strongly reccomended to normalize the data before initializing
        the weights and use the same normalization for the training data.
        """
        if self._input_len == 1:
            msg = 'The data needs at least 2 features for pca initialization'
            raise ValueError(msg)
        self._check_input_len(data)
        if len(self._neigx) == 1 or len(self._neigy) == 1:
            msg = 'PCA initialization inappropriate:' + \
                  'One of the dimensions of the map is 1.'
            warn(msg)
        pc_length, pc = linalg.eig(cov(transpose(data)))
        pc_order = argsort(-pc_length)
        for i, c1 in enumerate(linspace(-1, 1, len(self._neigx))):
            for j, c2 in enumerate(linspace(-1, 1, len(self._neigy))):
                self._weights[i, j] = c1*pc[pc_order[0]] + c2*pc[pc_order[1]]


    ############################################################### GR EDITS
    #This modifies the map training: mainly due to change in decay function
    
    def getSigma(self, epsilon):
        """returns the current sigma"""
        thisSig = self.getNeiSize(epsilon, self._sigma)

        return thisSig
    
    
    def update(self, x, win, epsilon, thisSig):
        """Updates the weights of the neurons.
        Parameters
        ----------
        x : np.array
            Current pattern to learn.
        win : tuple
            Position of the winning neuron for x (array or tuple).
        diameter: dataset diameter constant across SOM training
        epsilon: iteration specific "normalized" error rate
        """
        g = self.neighborhood(win, thisSig)*epsilon
        self._weights += einsum('ij, ijk->ijk', g, x-self._weights)




    def train(self, data, num_iteration,
              random_order=False, verbose=True, use_epochs=False):
        """Trains the SOM.

        Parameters
        ----------
        data : np.array or list
            Data matrix.

        num_iteration : int
            If use_epochs is False, the weights will be
            updated num_iteration times. Otherwise they will be updated
            len(data)*num_iteration times.

        random_order : bool (default=False)
            If True, samples are picked in random order.
            Otherwise the samples are picked sequentially.

        verbose : bool (default=False)
            If True the status of the training will be
            printed each time the weights are updated.

        use_epochs : bool (default=False)
            If True the SOM will be trained for num_iteration epochs.
            In one epoch the weights are updated len(data) times and
            the learning rate is constat throughout a single epoch.
        """
        self._check_iteration_number(num_iteration)
        self._check_input_len(data)
        random_generator = None
        if random_order:
            random_generator = self._random_generator
        iterations = _build_iteration_indexes(len(data), num_iteration,
                                              verbose, random_generator,
                                              use_epochs)
        if use_epochs:
            def get_decay_rate(iteration_index, data_len):
                return int(iteration_index / data_len)
        else:
            def get_decay_rate(iteration_index, data_len):
                return int(iteration_index)
        

        #MODIFIED FOR PLSOM   GR EDITS
        self.datasetDiameter = self.getDatasetDiameter(data)
        #print("datasetDiameter", self.datasetDiameter)
        self.trainingStats = {key: [] for key in ["LastErr", "Epsilon", "Sigma"]}
        
        for t, iteration in enumerate(iterations):
            self.thisErr = self.getError(data[iteration])
            self.thisEpsilon = self.getEpsilon(self.thisErr, self.datasetDiameter)
            self.thisSigma = self.getSigma(self.thisEpsilon)

            #print("Error", round(self.thisErr, 3), "Epsilon", round(self.thisEpsilon, 3), "Sigma", round(self.thisSigma,3) , "diam", round(self.datasetDiameter,3))
            self.update(data[iteration], self.winner(data[iteration]),
                        self.thisEpsilon, self.thisSigma)

            #self.trainingStats.append((self.thisErr, self.thisEpsilon, self.thisSigma))    
            self.trainingStats["LastErr"].append(self.thisErr)
            self.trainingStats["Epsilon"].append(self.thisEpsilon)
            self.trainingStats["Sigma"].append(self.thisSigma)
            # self.trainingStats["QE"] = self.quantization_error(data)


        if verbose:
            print('\n quantization error:', self.quantization_error(data))

        






    def distance_map(self, scaling='sum'):
        """Returns the distance map of the weights.
        If scaling is 'sum' (default), each cell is the normalised sum of
        the distances between a neuron and its neighbours. Note that this
        method uses the euclidean distance.

        Parameters
        ----------
        scaling : string (default='sum')
            If set to 'mean', each cell will be the normalized
            by the average of the distances of the neighbours.
            If set to 'sum', the normalization is done
            by the sum of the distances.
        """

        if scaling not in ['sum', 'mean']:
            raise ValueError(f'scaling should be either "sum" or "mean" ('
                             f'"{scaling}" not valid)')

        um = nan * zeros((self._weights.shape[0],
                          self._weights.shape[1],
                          8))  # 2 spots more for hexagonal topology

        ii = [[0, -1, -1, -1, 0, 1, 1, 1]]*2
        jj = [[-1, -1, 0, 1, 1, 1, 0, -1]]*2

        if self.topology == 'hexagonal':
            ii = [[1, 1, 1, 0, -1, 0], [0, 1, 0, -1, -1, -1]]
            jj = [[1, 0, -1, -1, 0, 1], [1, 0, -1, -1, 0, 1]]

        for x in range(self._weights.shape[0]):
            for y in range(self._weights.shape[1]):
                w_2 = self._weights[x, y]
                e = y % 2 == 0   # only used on hexagonal topology
                for k, (i, j) in enumerate(zip(ii[e], jj[e])):
                    if (x+i >= 0 and x+i < self._weights.shape[0] and
                            y+j >= 0 and y+j < self._weights.shape[1]):
                        w_1 = self._weights[x+i, y+j]
                        um[x, y, k] = fast_norm(w_2-w_1)

        if scaling == 'mean':
            um = nanmean(um, axis=2)
        if scaling == 'sum':
            um = nansum(um, axis=2)

        return um/um.max()

    def activation_response(self, data):
        """
            Returns a matrix where the element i,j is the number of times
            that the neuron i,j have been winner.
        """
        self._check_input_len(data)
        a = zeros((self._weights.shape[0], self._weights.shape[1]))
        for x in data:
            a[self.winner(x)] += 1
        return a

    def _distance_from_weights(self, data):
        """Returns a matrix d where d[i,j] is the euclidean distance between
        data[i] and the j-th weight.
        """
        input_data = array(data)
        weights_flat = self._weights.reshape(-1, self._weights.shape[2])
        input_data_sq = power(input_data, 2).sum(axis=1, keepdims=True)
        weights_flat_sq = power(weights_flat, 2).sum(axis=1, keepdims=True)
        cross_term = dot(input_data, weights_flat.T)
        return sqrt(-2 * cross_term + input_data_sq + weights_flat_sq.T)
    

    #Function to edit out invalid nodes
    def modifyWeights(self, nodeIdxLst, weightArrayList):
        for nodeIdx in nodeIdxLst:
            weightarray = weightArrayList[nodeIdx]
            self._weights[nodeIdx] = weightarray

        return self
########################################################################




    def quantization_error(self, data):
        """Returns the quantization error computed as the average
        distance between each input sample and its best matching unit."""
        self._check_input_len(data)
        return norm(data-self.quantization(data), axis=1).mean()

    def topographic_error(self, data):
        """Returns the topographic error computed by finding
        the best-matching and second-best-matching neuron in the map
        for each input and then evaluating the positions.

        A sample for which these two nodes are not adjacent counts as
        an error. The topographic error is given by the
        the total number of errors divided by the total of samples.

        If the topographic error is 0, no error occurred.
        If 1, the topology was not preserved for any of the samples."""
        self._check_input_len(data)
        total_neurons = prod(self._activation_map.shape)
        if total_neurons == 1:
            warn('The topographic error is not defined for a 1-by-1 map.')
            return nan
        if self.topology == 'hexagonal':
            return self._topographic_error_hexagonal(data)
        else:
            return self._topographic_error_rectangular(data)

    def _topographic_error_hexagonal(self, data):
        """Return the topographic error for hexagonal grid"""
        b2mu_inds = argsort(self._distance_from_weights(data), axis=1)[:, :2]
        b2mu_coords = [[self._get_euclidean_coordinates_from_index(bmu[0]),
                        self._get_euclidean_coordinates_from_index(bmu[1])]
                       for bmu in b2mu_inds]
        b2mu_coords = array(b2mu_coords)
        b2mu_neighbors = [(bmu1 >= bmu2-1) & ((bmu1 <= bmu2+1))
                          for bmu1, bmu2 in b2mu_coords]
        b2mu_neighbors = [neighbors.prod() for neighbors in b2mu_neighbors]
        te = 1 - mean(b2mu_neighbors)
        return te

    def _topographic_error_rectangular(self, data):
        """Return the topographic error for rectangular grid"""
        t = 1.42
        # b2mu: best 2 matching units
        b2mu_inds = argsort(self._distance_from_weights(data), axis=1)[:, :2]
        b2my_xy = unravel_index(b2mu_inds, self._weights.shape[:2])
        b2mu_x, b2mu_y = b2my_xy[0], b2my_xy[1]
        dxdy = hstack([diff(b2mu_x), diff(b2mu_y)])
        distance = norm(dxdy, axis=1)
        return (distance > t).mean()

    def _get_euclidean_coordinates_from_index(self, index):
        """Returns the Euclidean coordinated of a neuron using its
        index as the input"""
        if index < 0:
            return (-1, -1)
        y = self._weights.shape[1]
        coords = self.convert_map_to_euclidean((index % y, int(index/y)))
        return coords

    def win_map(self, data, return_indices=False):
        """Returns a dictionary wm where wm[(i,j)] is a list with:
        - all the patterns that have been mapped to the position (i,j),
          if return_indices=False (default)
        - all indices of the elements that have been mapped to the
          position (i,j) if return_indices=True"""
        self._check_input_len(data)
        winmap = defaultdict(list)
        for i, x in enumerate(data):
            winmap[self.winner(x)].append(i if return_indices else x)
        return winmap

    def labels_map(self, data, labels):
        """Returns a dictionary wm where wm[(i,j)] is a dictionary
        that contains the number of samples from a given label
        that have been mapped in position i,j.

        Parameters
        ----------
        data : np.array or list
            Data matrix.

        label : np.array or list
            Labels for each sample in data.
        """
        self._check_input_len(data)
        if not len(data) == len(labels):
            raise ValueError('data and labels must have the same length.')
        winmap = defaultdict(list)
        for x, l in zip(data, labels):
            winmap[self.winner(x)].append(l)
        for position in winmap:
            winmap[position] = Counter(winmap[position])
        return winmap


