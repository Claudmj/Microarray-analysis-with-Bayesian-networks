# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 08:21:38 2020

@author: claud
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import itertools as it
import numpy as np

from scipy.special import multigammaln
from numpy.linalg import slogdet
from numpy import logaddexp
from math import pi
from math import log
from math import exp
from math import expm1
from math import lgamma
import pandas as pd

from ast import literal_eval

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report

LOG2PI = log(2*pi)
LOG2 = log(2)


def bhc(data, data_model, crp_alpha=1.0):
    """
    Bayesian hierarchical clustering CRP mixture model.
    Notes
    -----
    The Dirichlet process version of BHC suffers from terrible numerical
    errors when there are too many data points. 60 is about the limit. One
    could use arbitrary-precision numbers if one were so inclined.
    Parameters
    ----------
    data : numpy.ndarray (n, d)
        Array of data where each row is a data point and each column is a
        dimension.
    data_model : CollapsibleDistribution
        Provides the approprite ``log_marginal_likelihood`` function for the
        data.
    crp_alpha : float (0, Inf)
        CRP concentration parameter.
    Returns
    -------
    assignments : list(list(int))
        list of assignment vectors. assignments[i] is the assignment of data to
        i+1 clusters.
    lml : float
        log marginal likelihood estimate.
    """
    # initialize the tree
    print("bhc starts")
    nodes = dict((i, Node(np.array([x]), data_model, crp_alpha))
                 for i, x in enumerate(data))
    print("initialised")
    n_nodes = len(nodes)
    assignment = [i for i in range(n_nodes)]
    assignments = [list(assignment)]
    rks = [0]

    while n_nodes > 1:
        max_rk = float('-Inf')
        merged_node = None
        cnt = 0
        # for each pair of clusters (nodes), compute the merger score.
        for left_idx, right_idx in it.combinations(nodes.keys(), 2):
            cnt = cnt + 1
            print(cnt, n_nodes)
            tmp_node = Node.as_merge(nodes[left_idx], nodes[right_idx])

            logp_left = nodes[left_idx].logp
            logp_right = nodes[right_idx].logp
            logp_comb = tmp_node.logp

            log_pi = tmp_node.log_pi

            numer = log_pi + logp_comb

            neg_pi = log(-expm1(log_pi))
            denom = logaddexp(numer, neg_pi+logp_left+logp_right)

            log_rk = numer-denom

            if log_rk > max_rk:
                max_rk = log_rk
                merged_node = tmp_node
                merged_right = right_idx
                merged_left = left_idx

        rks.append(exp(max_rk))

        # Merge the highest-scoring pair
        del nodes[merged_right]
        nodes[merged_left] = merged_node

        for i, k in enumerate(assignment):
            if k == merged_right:
                assignment[i] = merged_left
        assignments.append(list(assignment))

        n_nodes -= 1

    # The denominator of log_rk is at the final merge is an estimate of the
    # marginal likelihood of the data under DPMM
    lml = denom
    return assignments, lml


class Node(object):
    """ A node in the hierarchical clustering.
    Attributes
    ----------
    nk : int
        Number of data points assigned to the node
    data : numpy.ndarrary (n, d)
        The data assigned to the Node. Each row is a datum.
    crp_alpha : float
        CRP concentration parameter
    log_dk : float
        Some kind of number for computing probabilities
    log_pi : float
        For to compute merge probability
    """

    def __init__(self, data, data_model, crp_alpha=1.0, log_dk=None,
                 log_pi=0.0):
        """
        Parameters
        ----------
        data : numpy.ndarray
            Array of data_model-appropriate data
        data_model : idsteach.CollapsibleDistribution
            For to calculate marginal likelihoods
        crp_alpha : float (0, Inf)
            CRP concentration parameter
        log_dk : float
            Cached probability variable. Do not define if the node is a leaf.
        log_pi : float
            Cached probability variable. Do not define if the node is a leaf.
        """
        self.data_model = data_model
        self.data = data
        self.nk = data.shape[0]
        self.crp_alpha = crp_alpha
        self.log_pi = log_pi

        if log_dk is None:
            self.log_dk = log(crp_alpha)
        else:
            self.log_dk = log_dk

        self.logp = self.data_model.log_marginal_likelihood(self.data)

    @classmethod
    def as_merge(cls, node_left, node_right):
        """ Create a node from two other nodes
        Parameters
        ----------
        node_left : Node
            the Node on the left
        node_right : Node
            The Node on the right
        """
        crp_alpha = node_left.crp_alpha
        data_model = node_left.data_model
        data = np.vstack((node_left.data, node_right.data))

        nk = data.shape[0]
        log_dk = logaddexp(log(crp_alpha) + lgamma(nk),
                           node_left.log_dk + node_right.log_dk)
        log_pi = log(crp_alpha) + lgamma(nk) - log_dk
        print(crp_alpha, lgamma(nk), log_dk )

        if log_pi == 0:
            raise RuntimeError('Precision error')

        return cls(data, data_model, crp_alpha, log_dk, log_pi)


class CollapsibleDistribution(object):
    """ Abstract base class for a family of conjugate distributions. """

    def log_marginal_likelihood(self, X):
        """ Log of the marginal likelihood, P(X|prior).  """
        pass


class NormalInverseWishart(CollapsibleDistribution):
    """
    Multivariate Normal likelihood with multivariate Normal prior on mean and
    Inverse-Wishart prior on the covariance matrix.
    All math taken from Kevin Murphy's 2007 technical report, 'Conjugate
    Bayesian analysis of the Gaussian distribution'.
    """

    def __init__(self, **prior_hyperparameters):
        self.nu_0 = prior_hyperparameters['nu_0']
        self.mu_0 = prior_hyperparameters['mu_0']
        self.kappa_0 = prior_hyperparameters['kappa_0']
        self.lambda_0 = prior_hyperparameters['lambda_0']

        self.d = float(len(self.mu_0))

        self.log_z = self.calc_log_z(self.mu_0, self.lambda_0, self.kappa_0,
                                     self.nu_0)

    @staticmethod
    def update_parameters(X, _mu, _lambda, _kappa, _nu, _d):
        n = X.shape[0]
        xbar = np.mean(X, axis=0)
        kappa_n = _kappa + n
        nu_n = _nu + n
        mu_n = (_kappa*_mu + n*xbar)/kappa_n

        S = np.zeros(_lambda.shape) if n == 1 else (n-1)*np.cov(X.T)
        dt = (xbar-_mu)[np.newaxis]

        back = np.dot(dt.T, dt)
        lambda_n = _lambda + S + (_kappa*n/kappa_n)*back

        assert(mu_n.shape[0] == _mu.shape[0])
        assert(lambda_n.shape[0] == _lambda.shape[0])
        assert(lambda_n.shape[1] == _lambda.shape[1])

        return mu_n, lambda_n, kappa_n, nu_n

    @staticmethod
    def calc_log_z(_mu, _lambda, _kappa, _nu):
        d = len(_mu)
        sign, detr = slogdet(_lambda)
        log_z = LOG2*(_nu*d/2.0) + (d/2.0)*log(2*pi/_kappa) +\
            multigammaln(_nu/2, d) - (_nu/2.0)*detr

        return log_z

    def log_marginal_likelihood(self, X):
        n = X.shape[0]
        params_n = self.update_parameters(X, self.mu_0, self.lambda_0,
                                          self.kappa_0, self.nu_0, self.d)
        log_z_n = self.calc_log_z(*params_n)

        return log_z_n - self.log_z - LOG2PI*(n*self.d/2)


if __name__ == '__main__':

    hypers = {
        'mu_0': np.zeros(10),
        'nu_0': 12.0,
        'kappa_0': 1.0,
        'lambda_0': np.eye(10)
    }
    data_model = NormalInverseWishart(**hypers)

    #import data into a dataframe
    filename='markov_blanket_features.csv'
    df=pd.read_csv(filename) #no header



    #assign columns 1 to end to an array - data
    data = []
    #in the same loop, create the labels
    labels = []
    for i in np.arange(0,len(df)):
        data.append(df.iloc[i,1:-1].tolist()) #literal_eval prevents a list as been written as a string
        if df.iloc[i,0] == "ALL":
            labels.append(0)
        else:
            labels.append(1)

    asgn, _ = bhc(data, data_model, 40)
    z = np.array(asgn[-2], dtype=float)
    change = []
    for i in range(len(z)):
        if z[i] == 0:
          change.append(0)   
        else:
            change.append(1)
    #change z to 0 and 1 values
    
    z = change

    
    #now you can calculate performance metrics with sklearn, such as confusion matrix
    c = confusion_matrix(labels, z)
    print(c)
    accuracy = accuracy_score(labels, z)
    accuracy2 = (c[0,0]+c[1,1])/71
    print('The accuracy is:' ,accuracy) 
    precision = c[1,1]/(c[1,1]+c[0,1])
    precision2 = precision_score(labels,z)
    recall = c[1,1]/(c[1,1]+c[1,0])
    recall2 = recall_score(labels,z)
    print('The precision is: ',precision)
    print('The recall is: ',recall)
    f1 = f1_score(labels,z)
    f12 = 2 * (precision * recall) / (precision + recall)
    print('The F1 score is:' ,f1)