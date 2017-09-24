import numpy as np
from scipy import spatial
from collections import defaultdict

def normalization(x):
        y = x/np.sqrt(np.dot(x,x.T))
        return y

class somClass(object):
    """
    Implementation of Self Organizing Maps(SOM)
    Inputs :input_dims -> number of features in input data
            num_iterations -> number of iterations
            init_lr -> initial learning rate
            init_r -> optional initial radius
            grid_dims -> dimensions of SOM grid
            distance_metric -> choice of euclidean (0) or cosine (1) distance metric
            seed -> random seed
            verbose -> Boolean to display training progress
            result -> retrieve final weights for the nodes
            
     """
    
    def __init__(self, input_dims, num_iterations = 1000, init_lr = 0.5, grid_dims = (10,10), distance_metric = 0, \
                 seed = None, verbose = False, result = False, init_r = None):
        
        self.num_iterations = num_iterations
        self.init_lr = init_lr
        self.r, self.c = grid_dims
        self.distance_metric = distance_metric
             
        if init_r is None:
            self.init_radius = max(self.r, self.c)/2
        else:
            self.init_radius = init_r
            
        self.tc = self.num_iterations/np.log(self.init_radius)
        self._shrink_function_radius = lambda x, i : x * np.exp(-i/self.tc)
        self._shrink_function_lr = lambda x, i : x * np.exp(-i/self.num_iterations)

        self.random_generator = np.random.RandomState(seed)
        
        self.d = input_dims # number of dimensions in a sample
        self.init_weight = self.random_generator.rand(self.r, self.c, self.d) #3D tensor
        for ii in range(self.r): # making weight vectors unit norm
            for jj in range(self.c):
                vect = self.init_weight[ii,jj,:]
                vect = normalization(vect)
                self.init_weight[ii,jj,:] = vect
                
                
        self.w = self.init_weight  
        self.bmu_map = defaultdict(list)
        self.verbose = verbose
        self.result = result
        self.all_radius = [self.init_radius]
        self.all_lr = [self.init_lr]
        
        
    def _euclid_distance(self,v1,v2):
        """
        Euclidean distance function. Square root is not computed for faster execution
        """
        return np.sum((v1 - v2) ** 2) # no need to compute square root. this makes it faster
    
    def _cosine_distance(self,v1,v2):
        """
        Cosine distance function
        """
        return spatial.distance.cosine(v1, v2)
   
    def weight_initialize_withdata(self, X):
        """
        Optional method to initialize weights by randomly choosing from the input dataset
        Inputs : X -> input dataset (NxD) N samples, D dimensions
        """
        
        self.m = X.shape[0] # number of samples
        for ii in range(self.r): # making weight vectors unit norm
            for jj in range(self.c):
                vect = X[self.random_generator.randint(0, self.m), :]
                vect = normalization(vect)
                self.init_weight[ii,jj,:] = vect
                
        self.w = self.init_weight
    
    def _find_bmu(self,v):
        """
        Function to find the best matching unit. Not to be called explicitly.
        Inputs : v -> the randomly chosen input sample
        Outputs: bmu -> weight vector of the best matching unit
                 bmu_pos -> coordinates in the grid  of the bmu
        """
        bmu_pos = np.array([0,0])
        min_dist = np.inf
        for ii in range(self.r):
            for jj in range(self.c):
                weight_vect = self.w[ii,jj,:].reshape(self.d, 1)
                
                if self.distance_metric == 0: # Euclidean distance
                    dist = self._euclid_distance(v,weight_vect)
                elif self.distance_metric == 1: # cosine distance
                    dist = self._cosine_distance(v,weight_vect)
                    
                if dist < min_dist:
                    min_dist = dist
                    bmu_pos = np.array([ii,jj])
                    
        bmu = self.w[bmu_pos[0], bmu_pos[1], :].reshape(self.d, 1)
        return bmu, bmu_pos
    
   
    def _weight_update(self, sample, bmu_pos, radius, lr):
        """
        Function to update weights during training. Not to be called explicitly.
        Inputs : sample -> Input sample
                 bmu_pos -> coordinates of the bmu
                 radius -> radius of the neighborhood at current iteration
                 lr -> learning rate at current iteration
        """
       
        for ii in range(self.r):
            for jj in range(self.c):
                weight_vect = self.w[ii,jj,:].reshape(self.d, 1)
                geom_dist = np.sum((bmu_pos - np.array([ii,jj])) ** 2)
                if geom_dist <= radius**2:
                    neighborhood_func = np.exp(-geom_dist/(2*(radius**2)))
                    weight_vect_new = weight_vect + neighborhood_func*lr*(sample-weight_vect)
                    weight_vect_new = weight_vect_new.reshape(1,self.d)
                    weight_vect_new = normalization(weight_vect_new)
                    self.w[ii,jj,:] = weight_vect_new
                        
           
    def train(self, X):
        """
        Function to train the SOM.
        Inputs : X -> Input dataset (NxD) N samples, D dimensions
        Output : weights_final -> 3D tensor with D dimensional weights for each SOM grid position 
        
        """
        self.m = X.shape[0]
        for iteration in range(self.num_iterations):
            index = self.random_generator.randint(0, self.m)
            sample = X[index,:].reshape(self.d, 1)
            bmu, bmu_pos = self._find_bmu(sample)
            self.bmu_map[tuple(bmu_pos)].append(index) # indices of data samples and the node in the grid it is mapped to
            
            radius = self._shrink_function_radius(self.init_radius, iteration) # updating the radius of the neighborhood
            lr = self._shrink_function_lr(self.init_lr, iteration) # updating the learning rate
            
            self._weight_update(sample, bmu_pos, radius, lr)
            
            if self.verbose:
                print('Iteration %d completed' %iteration)
             
            self.all_radius.append(radius)
            self.all_lr.append(lr)
        
        if self.result:
            weights_final = self.w
            return weights_final
            
    def bmu_mappings(self):
        """
        Function to retrive the mappings of input samples to grid nodes
        Output : A dictionary with the grid coordinate as the key and the row indices of the input samples
        mapped to that node in a list as the value
        """
        if not any(self.bmu_map):
            assert any(self.bmu_map), 'SOM needs to be trained first. Call the train method.'
        else:
            return self.bmu_map
    
    def neighborhood_distance_map(self):
        """
        Function to get normalized distance of each node with respect to its own neighborhood. 
        Output : A matrix same size as the grid with (i,j) element containing the normalized distance of its own weight
        vector and those of its neighbors. Higher the normalized distance, more is the difference of the node from its
        neigbors, higher is the chance of the input samples mapped to that node to be outliers.
        
        """
        max_norm_distances = np.zeros((self.r, self.c))
        coord = np.nditer(max_norm_distances, flags = ['multi_index'])
        while not coord.finished:
            weight_center = self.w[coord.multi_index[0], coord.multi_index[1], :].reshape(self.d,1)
            for ii in range(coord.multi_index[0]-1, coord.multi_index[0]+2):
                for jj in range(coord.multi_index[1]-1, coord.multi_index[1]+2):
                    if (ii >= 0) and (ii < self.r) and (jj >= 0) and (jj < self.c):
                        weight_neighbor = self.w[ii, jj, :].reshape(self.d,1) 
                        max_norm_distances[coord.multi_index] += np.sqrt(np.sum((weight_center - weight_neighbor) ** 2))
            coord.iternext()
            
        max_norm_distances /= max_norm_distances.max() # normalizing with respect to the largest distance in the grid
        return max_norm_distances