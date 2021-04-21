'''
File used to analyze charge stability diagrams in order to produce capacitance matrices, virtual gating, lever arms, etc.
For a good review of the Hough transform, check https://alyssaq.github.io/2014/understanding-hough-transform/.
Hough transfrom code based off of https://github.com/alyssaq/hough_transform.
'''

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter
from sklearn import cluster
from sklearn.neighbors import NearestCentroid

class CSDAnalysis:
    '''
    Initialize the charge stability diagram analysis class which analyzes charge stability diagrams to extract parameters. 
    '''
    def __init__(self, csd, capacitances=None, blur_sigma=None):
        '''
         
        Parameters
        ----------
        csd : CSD object from csd_cim

        Keyword Arguments
        -----------------
        capacitances: List used to convert from occupation to current for analysis. If no capacitances are supplied, a colormap will be
                      created but this colormap will be arbitrary and thus not suitable for further analysis (default None)
        blur_sigma: If not None, do a Gaussian blur on the data to simulate thermal broadening of transition lines with standard deviation of
                    the Gaussian kernel as blur_sigma. A larger number means more smeared out (default None)

        Returns
        -------
        None

        '''
        self.capacitances = capacitances
        self.csd = copy.copy(csd) # to avoid overwriting original csd object

        # Create a new empty DataFrame where we will put numbers instead of tuples corresponding to occupations
        self.csd.csd = pd.DataFrame(0, index=self.csd.v_1_values, columns=self.csd.v_2_values, dtype=np.float32)

        # If no capacitances are provided, create a color map using the hash of the occupation
        if self.capacitances is None:
            single_occupations = [np.zeros(self.csd.n_sites) for i in range(self.csd.n_sites)]
            for i in range(self.csd.n_sites):
                single_occupations[i][i] = 1
            hashes = np.array([hash(tuple(single_occupations[i])) for i in range(self.csd.n_sites)])
            for i in self.csd.v_2_values:
                for j in self.csd.v_1_values:
                    self.csd.csd[i][j] = np.sum(hashes * self.csd.occupation[i][j][0])
        else:
            if len(self.capacitances) != self.csd.n_sites:
                raise Warning("Number of dot to charge sensor capacitances does not match the number of dots")
            # cast capacitances to numpy array to use numpy functions
            self.capacitances = np.array(self.capacitances)
            # going over v_2 in outer loop due to how numpy indexes 2d arrays
            for i in self.csd.v_2_values:
                for j in self.csd.v_1_values:
                    self.csd.csd[i][j] = np.sum(self.capacitances * self.csd.occupation[i][j][0])

            # Create derivative of charge stability diagram
            df_der_row = self.csd.csd.diff(axis=0).fillna(0) # Replace Nans (where derivative is not defined) with 0s
            df_der_col = self.csd.csd.diff(axis=1).fillna(0)
            csd_der = np.sqrt(df_der_row**2 + df_der_col**2) # to be sensitive to changes in both the x and y direction
            self.csd.csd_der = csd_der

        if blur_sigma is not None:
            if self.capacitances is None:
                raise Warning("Blurring of data cannot occur when no capaciatnce are provided. Data will not be changed")
            else:
                self.csd.csd = pd.DataFrame(gaussian_filter(self.csd.csd, blur_sigma), columns=self.csd.v_1_values, index=self.csd.v_2_values)

    def generate_bitmap(self, threshold, threshold_type='percentile', plotting=False):
        '''
        Transforms the charge stability diagram into a bitmap. Threshold determines whether bit is considered 'on' or 'off'

        Parameters
        ----------
        threshold: threshold which determines whether bit is considered 'on' or 'off'
        threshold: number which specifies the threshold. Behaves differently depending on threshold_type

        Keyword Arguments
        -----------------
        threshold_type: String flag for which type of thresholding to do (default 'percentile')
            - 'percentile': will set all elements in the array above the set percentile to 1 and all those below to 0, ignoring NaN values 
                    e.g with threshold=99, only elements above the 99th percentile will be set to 1
            - 'absolute': will set all elements in the array above the specified value to 1 and all those below to 0 
                    e.g with threshold=20, only elements whos is value greater or equal 20 will be set to 1
        plotting: flag which determines whether or not to plot the resulting thresholded Hough accumulator (default False)

        Returns
        -------
        None

        '''
        if threshold_type.lower() == 'percentile':
            # Converts percentile type threshold into absolute type to use same for loop
            threshold = np.nanpercentile(self.csd.csd_der, threshold)

        elif threshold_type.lower() == 'absolute':
            # Do nothing to the threshold, but avoid raising an error
            pass

        else:
            raise ValueError('Unrecognized threshold type: ' + str(threshold_type))
        
        # Make bitmap
        self.csd_bitmap = self.csd.csd_der.mask(abs(self.csd.csd_der) > threshold, other=1).mask(abs(self.csd.csd_der) <= threshold, other=0)
        if plotting is True:
            self.__plot_heatmap(self.csd_bitmap, self.csd.v_1_values, self.csd.v_2_values, r'V$_1$', r'V$_2$')

    def plot(self, derivative=False):
        '''
        Wrapper which plots the charge stability diagram, and the derivative of the charge stability diagram if it is meaningful

        Parameters
        ----------
        None

        Returns
        -------
        None

        '''
        # Plot colobar and indicate "current" if that result is requested
        if derivative is True:
            cbar_flag = True
            cbar_kws={'label': 'Current (arb.)'}
        else:
            cbar_flag = False
            cbar_kws = dict()
        # Plot the chagre stability diagram
        self.__plot_heatmap(self.csd.csd, None, None, r'V$_1$', r'V$_2$', cbar=cbar_flag, cbar_kws=cbar_kws)

        # Plot the "derivative" of the charge stability diagram, if desired
        if derivative is True:
            self.__plot_heatmap(self.csd.csd_der, None, None, r'V$_1$', r'V$_2$', cbar=cbar_flag, cbar_kws=cbar_kws)

    def hough_transform(self, num_thetas=180, theta_min=0, theta_max=90, plotting=False):
        '''
        Performs the Hough transform on the charge stability diagram bitmap stored in the object

        Parameters
        ----------
        None

        Keyword Arguments
        -----------------
        num_thetas: number of angle points to sweep over (default 180)
        theta_min: smallest angle (in degrees) to start the sweep from, which should not be smaller than -90 (default -90)
        theta_max: largest angle (in degrees) to sweep to, which should not be greater than 90 (default 90)
        plotting: flag which determines whether or not to plot the resulting Hough accumulator (default False)

        Returns
        -------
        Accumulator: 2D array with counts of each theta and 
        thetas: values of theta for which accumulator swept over
        rhos: values of distance for which accumulator swept over

        '''
        # Get charge stability diagram bitmap
        img = self.csd_bitmap
        
        # Rho and Theta ranges
        thetas = np.deg2rad(np.linspace(theta_min, theta_max, num=num_thetas))
        width = img.columns[-1]
        height = img.index[-1]
        diag_len = np.sqrt(width ** 2 + height ** 2)

        index_height, index_width = img.shape
        index_diag_len = int(round(np.sqrt(index_width ** 2 + index_height ** 2)))
        rhos = np.linspace(-diag_len, diag_len, num=2*index_diag_len)

        # Cache some resuable values
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)  

        # Hough accumulator array of theta vs rho
        accumulator = np.zeros((2 * index_diag_len, num_thetas), dtype=np.uint64)
        y_idxs, x_idxs = np.nonzero(img.to_numpy())

        # Vote in the hough accumulator
        for i in range(len(x_idxs)):
            x_index = x_idxs[i]
            y_index = y_idxs[i]

            x = img.columns[x_index]
            y = img.index[y_index]

            for t_idx in range(num_thetas):
                rho = x * cos_t[t_idx] + y * sin_t[t_idx]
                # Find index of the nearest value in rhos to rho, and add vote there
                rho_index = (np.abs(rhos - rho)).argmin()
                accumulator[rho_index, t_idx] += 1

        # Store values in object for later use
        self.accumulator = accumulator
        self.thetas = thetas
        self.rhos = rhos

        if plotting is True:
            # Round data to avoid ridiculously long tick markers
            rhos = np.round(self.rhos, 6)
            thetas = np.round(self.thetas, 6)
            # Call heatmap plotting function
            self.__plot_heatmap(self.accumulator, thetas, rhos, r'$\theta$ (rad)', r'$\rho$ (V)')

        return accumulator, thetas, rhos

    def threshold_hough_accumulator(self, threshold, threshold_type='percentile', plotting=False):
        '''
        Transforms the Hough transform accumulator stored in the objects into a binary accumulator using a threshold.
        Threshold determines whether accumulator value is set to 1 or 0. The threshold flag determines how the thrsehold is interpretted

        Parameters
        ----------
        threshold: number which specifies the threshold. Behaves differently depending on threshold_type

        Keyword Arguments
        -----------------
        threshold_type: String flag for which type of thresholding to do (default 'percentile')
            - 'percentile': will set all elements in the array above the set percentile to 1 and all those below to 0, ignoring NaN values  
                    e.g with threshold=99, only elements above the 99th percentile will be set to 1
            - 'absolute': will set all elements in the array above the specified value to 1 and all those below to 0 
                    e.g with threshold=20, only elements whos is value greater or equal 20 will be set to 1
        plotting: flag which determines whether or not to plot the resulting thresholded Hough accumulator (default False)

        Returns
        -------
        accumulator_threshold: 2D array with counts 

        '''
        if threshold_type.lower() == 'percentile':
            # Converts percentile type threshold into absolute type to use same for loop
            threshold = np.nanpercentile(self.accumulator, threshold)

        elif threshold_type.lower() == 'absolute':
            # Do nothing to the threshold, but avoid raising an error
            pass

        else:
            raise ValueError('Unrecognized threshold type: ' + str(threshold_type))

        # Go through all the elements in the accumulator, setting all the elements above the threshold to 1
        accumulator_threshold = np.zeros(self.accumulator.shape)
        for index, value in np.ndenumerate(self.accumulator):
            if value >= threshold:
                accumulator_threshold[index] = 1
        self.accumulator_threshold = accumulator_threshold

        if plotting is True:
            # Round data to avoid ridiculously long tick markers
            rhos = np.round(self.rhos, 6)
            thetas = np.round(self.thetas, 6)
            # Call heatmap plotting function
            self.__plot_heatmap(accumulator_threshold, thetas, rhos, r'$\theta$ (rad)', r'$\rho$ (V)')

        return accumulator_threshold

    def hough_cluster(self, eps, min_samples, plotting=False):
        '''
        Clusters the points in the thresholded Hough transform accumulator.

        Parameters
        ----------
        eps: maximum distance for points to be considered in the local neighbourhood of each other
        min_samples: minimum number of samples within the local neighbourhood in order for the point to be considered a core part of the cluster

        Keyword Arguments
        -----------------
        potting: boolean flag which sets whether or not to plot the results of clustering (default False)
        
        Returns
        -------
        centroids: numpy array of pairs [theta, rhos] corresponding to valid charge transition line

        '''
        # Get the accumulator threshold parameters and reorder as a list of pairs of indices instead of a 2D array
        a = np.array(self.accumulator_threshold.nonzero())
        points_index = []
        for i in range(len(a[0])):
            points_index.append([a[1][i], a[0][i]])

        # Function that does the clusterting. Save the labels for later
        self.db = cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(points_index)
        labels = self.db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        # Convert from indices to pairs of the for [theta, rho] and convert into a numpy array
        points = []
        for pair in points_index:
            points.append([self.thetas[pair[0]], self.rhos[pair[1]]])
        points = np.array(points)

        # Plotting subroutine which plots all the point in different clusters in different colors
        if plotting is True:
            # take the transpose of points for ease of plotting 
            temp_points = points.transpose()
            # Create uniques colors for each cluster
            unique_labels = set(labels)
            colors = [plt.cm.viridis(each) for each in np.linspace(0, 1, len(unique_labels))]
            # Loop over each group and corresponding color
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Black used for noise.
                    col = [0, 0, 0, 1]

                points_to_plot_x = []
                points_to_plot_y = []
                for i in range(len(labels)):
                    if k == labels[i]:
                        points_to_plot_x.append(temp_points[0][i])
                        points_to_plot_y.append(temp_points[1][i])

                sns.scatterplot(x=points_to_plot_x, y=points_to_plot_y, color=tuple(col), s=100)

            plt.title('Estimated number of clusters: %d' % n_clusters_)
            plt.ylabel(r'$\rho$ (V)')
            plt.xlabel(r'$\theta$ (rad)')
            plt.show()

        # get centroids of each cluster and save for later
        clf = NearestCentroid()
        clf.fit(points, labels)
        centroids = clf.centroids_

        # Remove the noise from the clustering if it is present (since it's label is -1 and all others are positive, it will alwasy be the first in the list of centroids)
        if -1 in labels:
            centroids = np.delete(centroids, 0, 0)

        # TODO determine better way to determine invalid charge transitions and exclude them
        # drop centroids where theta < 0 (which correcpond to positive slope)
        valid_centroids = np.array([i for i in centroids if i[0]>0])
        self.centroids = valid_centroids

        return valid_centroids

    def __plot_heatmap(self, data, x_values, y_values, x_label, y_label, cbar=True, cbar_kws=dict()):
        '''
        Private function which formats and plots Seaborn heatmaps.

        Parameters
        ----------
        data: Numpy array or Pandas Dataframe which contains the data to plot
        x_values: List of x values for the heatmap. If "data" is a numpy array, can be None
        y_values: List of y values for the heatmap. If "data" is a numpy array, can be None
        x_label: Label to add to x axis if plot
        y_label: Label to add to y axis of plot

        Keyword Arguments
        -----------------
        cbar: Whether or not to display a colobar for the heatmap (Default False)
        cbar_kws: Colorbar keyword arguments to pass to plot (Defaults to empty dictionary)
        
        Returns
        -------
        None

        '''
        # Casts to a dataframe if data is not already for ease of plotting
        if type(data) != type(pd.DataFrame()):
            data = pd.DataFrame(data, index=y_values, columns=x_values)
        s = sns.heatmap(data, cbar=cbar, xticklabels=int(self.csd.num/5), yticklabels=int(self.csd.num/5), cbar_kws=cbar_kws)
        # Flip y axis so y_values increasing from bottom to top 
        s.axes.invert_yaxis()
        s.axes.set_xlabel(x_label)
        s.axes.set_ylabel(y_label)
        plt.show()

    def plot_csd_with_lines(self):
        '''
        Function which plots the CSD with the fitted lines over top of it

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        # Save number of points in charge stability diagram for later use
        num = self.csd.csd.shape[0]

        # Create the heatmap figure
        f, ax = plt.subplots(1,1)
        sns.heatmap(self.csd.csd, cbar=False, xticklabels=int(num/5), yticklabels=int(num/5))
        ax.axes.invert_yaxis()

        # Create second axis with same x and y axis as the heatmap
        ax2 = ax.twinx().twiny()

        # For each centroid, convert from polar coordiantes to slope/intercept for and plot on second axis
        x = np.linspace(self.csd.v_g1_min, self.csd.v_g1_max)
        for centroid in self.centroids:
            theta = centroid[0]
            rho = centroid[1]
            m = -np.cos(theta)/np.sin(theta)
            b = rho/np.sin(theta)
            y = m * x + b
            sns.lineplot(x=x, y=y, ax=ax2)

        # format the secodn axis and show the plot
        ax2.set_xlim([self.csd.v_g1_min,self.csd.v_g1_max])
        ax2.set_ylim([self.csd.v_g2_min,self.csd.v_g2_max])
        ax2.get_yaxis().set_ticks([])
        ax2.get_xaxis().set_ticks([])
        ax.set(xlabel=r'V$_1$', ylabel=r'V$_2$')
        plt.show()

    def find_tripletpoints(self):
        '''
        Finds the location of triple points in a charge stability diagram.
        This function is NOT general and only works in the case of 4 main transition lines with a middle transition that is missing.

        Parameters
        ----------
        None

        Returns
        -------
        triple_points: list of tuples with correspond to coordinates (x,y) of the triple point
        '''
        m_list = []
        b_list = []

        for centroid in self.centroids:
            theta = centroid[0]
            rho = centroid[1]
            m_list.append(-np.cos(theta)/np.sin(theta))
            b_list.append(rho/np.sin(theta))

        # Sort m, then use the same sorting index so each (m,b) pair is kept and store for later
        m_array = np.array(m_list)
        b_array = np.array(b_list)
        m_sort = m_array.argsort()
        self.m_array = m_array[m_sort[::]]
        self.b_array = b_array[m_sort[::]]
        self.line_params = np.column_stack((self.m_array, self.b_array))

        # Reorder the line parameters if the values don't match up with number scheme
        if self.line_params[0,1] > self.line_params[1,1]:
            self.line_params[[0,1]] = self.line_params[[1,0]]
        if self.line_params[2,1] > self.line_params[3,1]:
            self.line_params[[2,3]] = self.line_params[[3,2]]


        candidate_points = []
        for i in range(len(self.m_array)):
            # Make sure you aren't looping over the same pair twice
            for j in range(i):
                # Extract values and compute expected intersection point
                m1_temp = self.m_array[i]
                m2_temp = self.m_array[j]
                b1_temp = self.b_array[i]
                b2_temp = self.b_array[j]
                x_temp = (b2_temp-b1_temp)/(m1_temp-m2_temp)
                y_temp = m1_temp * x_temp + b1_temp
                # Discard if expected interection point lies outside the CSD
                if (x_temp < self.csd.v_g1_min) or (x_temp > self.csd.v_g1_max) or (y_temp < self.csd.v_g2_min) or (y_temp > self.csd.v_g2_max):
                    continue
                candidate_points.append([x_temp, y_temp])

        # Convert to numpy arrays for ease of manipulation
        candidate_points = np.array(candidate_points)

        # Remove max and min x elements from points, which removes the invalid triple points
        candidate_points = np.delete(candidate_points, np.argmin(candidate_points, axis=0)[0], axis=0)
        triple_points = np.delete(candidate_points, np.argmax(candidate_points, axis=0)[0], axis=0)
        # Sort so the triple point with the smallest x comes first
        triple_points = triple_points[triple_points[:,0].argsort()]

        self.triple_points = triple_points

        return triple_points

    def plot_triple_points(self):
        '''
        Plots charge stability diagram with fitted lines (terminated at the correct triple points)
        This function is NOT general and only works in the case of 4 main transition lines with a middle transition that is missing.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        # Extract the coordinates for the two triple points
        x_electron = self.triple_points[0][0]
        y_electron = self.triple_points[0][1]
        
        x_hole = self.triple_points[1][0]
        y_hole = self.triple_points[1][1]

        # Create the heatmap figure
        f, ax = plt.subplots(1,1)
        num = self.csd.csd.shape[0]
        sns.heatmap(self.csd.csd, cbar=False, xticklabels=int(num/5), yticklabels=int(num/5))
        ax.axes.invert_yaxis()

        # Create second axis with same x and y axis as the heatmap
        ax2 = ax.twinx().twiny()

        # Create the x ranges for the various lines to plot on
        x_1 = [x_electron, self.csd.v_g1_max]
        x_2 = [self.csd.v_g1_min, x_hole]
        x_3 = [self.csd.v_g1_min, x_electron]
        x_4 = [x_hole, self.csd.v_g1_max]
        x_5 = [x_electron, x_hole]
        x_ranges = np.array([x_1, x_2, x_3, x_4, x_5])
        
        # Add the last line for the transition betweent the two points
        m_5 = (y_hole - y_electron)/(x_hole - x_electron)
        b_5 = y_electron - m_5 * x_electron
        line_5 = np.array([m_5, b_5])
        self.line_params = np.vstack((self.line_params, line_5))

        for x_range, line in zip(x_ranges, self.line_params):
            y = line[0] * x_range + line[1]
            sns.lineplot(x=x_range, y=y, ax=ax2)

        # Plot the two triple points
        sns.scatterplot(x=[x_hole, x_electron], y=[y_hole, y_electron], ax=ax2)

        # format the secodn axis and show the plot
        ax2.set_xlim([self.csd.v_g1_min,self.csd.v_g1_max])
        ax2.set_ylim([self.csd.v_g2_min,self.csd.v_g2_max])
        ax2.get_yaxis().set_ticks([])
        ax2.get_xaxis().set_ticks([])
        ax.set(xlabel=r'V$_1$', ylabel=r'V$_2$')
        plt.show()
