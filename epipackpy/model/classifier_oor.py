from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from scipy.stats import chi2

class global_oor_query():

    '''
    global classification and oor detection
    '''

    def __init__(self, train_data, query_data, label):
        
        self.train_data = train_data
        self.query_data = query_data

        ## init label
        self.le = LabelEncoder()
        self.label = self.le.fit_transform(label)

        print("- Classifier prepared")

    def _gaussian_init(self):

        init_mu = []
        init_cov_inv = []
        
        ## parameter estimation of class-condition gaussian 
        for i in np.unique(self.label):
            class_i_data = self.train_data[np.where(self.label == i)]
            class_i_mean = np.mean(class_i_data, axis=0, keepdims=True)
            class_i_norm = class_i_data - class_i_mean
            class_i_cov = np.matmul(class_i_norm.T, class_i_norm)/(class_i_norm.shape[0]-1)

            init_mu.append(class_i_mean)
            init_cov_inv.append(np.linalg.inv(class_i_cov))

        dist_mat_ref = np.zeros((self.train_data.shape[0], len(np.unique(self.label))))

        ## in-class distance distribution
        #for i in range(len(init_mu)):
        #    ref_norm = self.train_data - init_mu[i]
        #    ref_dist_i = np.matmul(np.matmul(ref_norm, init_cov_inv[i]), ref_norm.T)
        #    dist_mat_ref[:,i] = np.sqrt(np.diag(ref_dist_i))

        return init_mu, init_cov_inv #, dist_mat_ref

    def _cal_dist_query(self, mu_set, cov_set):
        
        dist_mat_query = np.zeros((self.query_data.shape[0], len(np.unique(self.label))))
        for i in range(len(mu_set)):
            query_norm = self.query_data - mu_set[i]
            query_dist_i = np.matmul(np.matmul(query_norm, cov_set[i]), query_norm.T)
            dist_mat_query[:,i] = np.diag(query_dist_i)

        return dist_mat_query
        
    def annotate(self,k = 15, confidence_threshold = 1e-4):
        
        ###### primary classification

        ## init knn classifier

        print("- Running kNN classifier with k =",k,"...")
        neigh = KNeighborsClassifier(n_neighbors=k)  
        neigh.fit(self.train_data, self.label)
        querylabel = neigh.predict(self.query_data)
        print("- kNN complete")

        ###### confidence score calculation

        ## init gaussian assumption
        print("- Gaussian distance calculating...")
        init_mu, init_cov_inv = self._gaussian_init()
        dist_query_mat = self._cal_dist_query(init_mu, init_cov_inv)

        print("- Gaussian distance complete")

        ## distance of each class distribution

        #class_distribution_mu = []

        #for i in range(len(np.unique(self.label))):
        #    mu_c = np.mean(dist_mat_ref[np.where(self.label==i)][:,i])
        #    class_distribution_mu.append(mu_c)
        
        ## oor recogniztion

        print("- Start detecting global out-of-reference cells...")
        
        reject_cell_list = []
        prob_score = []

        df = self.query_data.shape[1]

        for i in range(self.query_data.shape[0]):
            raw_label = querylabel[i]
            dist_i_c = dist_query_mat[i][raw_label]
            #dist_c_in = class_distribution_mu[raw_label]
            # min gaussian dist
            min_dist_label = np.argmin(dist_query_mat[i])
            dist_i_c_gau = dist_query_mat[i][min_dist_label]
            # prob in poisson
            #q = 1-poisson.cdf(dist_i_c, dist_c_in)
            #if q<confidence_threshold:
            #    if adjust_by_dist:
            #        if raw_label != min_dist_label:
            #            q_updated = 1-poisson.cdf(dist_i_c_gau, dist_c_in)
            #            if q_updated > adjust_threshold:
            #                querylabel[i] = min_dist_label
            #            else:
            #                reject_cell_list.append(i)
            #        else:
            #            reject_cell_list.append(i)
            #    else:
            #        reject_cell_list.append(i)
            q = 1-chi2.cdf(dist_i_c, df)

            if q<confidence_threshold:
                reject_cell_list.append(i)
            prob_score.append(q)
        
        ## return final label
        final_cell_type = self.le.inverse_transform(querylabel)
        final_cell_type[reject_cell_list] = 'Unknown'

        print("- Annotation complete")

        return final_cell_type, prob_score
