import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.metrics import classification_report
from tqdm import tqdm

"""
tree:
    node                              | leaf   
    (feature, (fuzzy_set, tree_sub)+) | {Ck: pk}
"""

def I(mu, y):
    """
    I(D) = -\sum_{k=1}^n (p_k log_2 p_k)
    """
    I_D = 0.0
    D = np.sum(mu)
    for C_k in np.unique(y):
        D_Ck = np.sum(mu[y==C_k])
        I_D -= (D_Ck/D)*np.log2(D_Ck/D) if D_Ck != 0 else 0
    return I_D
        
def E(D_fuzz, y, mu):
    """
    E(A_i, D) = -\sum_{j=1}^m (p_{ij} I(D_{F_{ij}}))
    """
    Ds = []
    I_Ds = []
    for fuzzy_set_mu in D_fuzz.values():
        Ds.append(np.sum(fuzzy_set_mu))
        I_Ds.append(I(fuzzy_set_mu, y))
    E_D = np.dot(Ds, I_Ds)/np.sum(Ds)
    return E_D

def gain(D_fuzz, y, mu):
    """
    G(A_i, D) = I(D) - E(A_i, D)
    """
    I_D = I(mu, y)
    E_D = E(D_fuzz, y, mu)
    return I_D - E_D

def calculate_target_weights(target_names, target_name_ratio):
    weights = np.ones(len(target_names))/len(target_names)
    inv_ratios = 1./np.array([target_name_ratio[t] for t in target_names])
    return np.multiply(weights, inv_ratios)

def check_condition_1(mu, y, theta_r, target_weights):
    D = 0.0
    D_Cks = []
    for Ck in np.unique(y):
        D_Ck = np.sum(mu[y == Ck])*target_weights[Ck]
        D_Cks.append(D_Ck)
        D += D_Ck
    return np.any(np.array(D_Cks)/D >= theta_r)
        
def check_condition_2(mu, theta_n):
    return np.sum(mu) < theta_n

def check_condition_3(feature_names):
    return len(feature_names) == 0 

def node(X, y, mu, feature_names, targets_names, features_fuzzy_sets, deep=0, debug=False, 
         max_deep=None, theta_r=0.8, theta_n=1, threshold=0.005, target_weights=None):
    if target_weights is None:
        target_weights = np.ones(len(targets_names))
    # Check if is leaf
    ## max deep
    ## Check conditions
    if ((max_deep is not None) and (deep > max_deep))\
            or check_condition_1(mu, y, theta_r, target_weights) \
            or check_condition_2(mu, theta_n) \
            or check_condition_3(feature_names):
        D = 0.0
        D_Cks = []
        for i, Ck in enumerate(targets_names):
            D_Ck = np.sum(mu[y == i])*target_weights[i]
            D += D_Ck
            D_Cks.append(D_Ck)
        leaf = {i: D_Ck/D for i, D_Ck in enumerate(D_Cks)}
        return leaf
    
    # Else Branching
    # Save max gain feature info
    max_gain = -1
    max_gain_feature = None
    max_gain_feature_fuzzy = None
    # fuzzify for each feature (A_i)
    for i, feature in enumerate(feature_names):
        # feature column
        A_i = X[:, i]       
        # save membership values for each fuzzy set
        D_fuzz = {}
        fuzzy_sets_mus = features_fuzzy_sets[feature] # fuzzy set membership function for this feature
        for fuzzy_set_m, mu_m in fuzzy_sets_mus.items():
            D_fuzz[fuzzy_set_m] = np.multiply(mu,mu_m(A_i)) # fuzzify!
            #D_fuzz[fuzzy_set_m] = np.multiply(mu, np.array([mu_m(v) for v in A_i]))
        # calcule gain for this feature
        feature_gain = gain(D_fuzz, y, mu)
        if debug: print("\t"*deep, feature, feature_gain)
        # save the feature with higher gain
        if feature_gain > max_gain:
            max_gain = feature_gain
            max_gain_feature = feature
            max_gain_feature_fuzzy = dict(D_fuzz)
        del D_fuzz
    # Branching
    branchs = []
    if debug: print("\t"*(deep+1), "Branch feature:", max_gain_feature)
    for fuzzy_set, fuzzy_set_mu in max_gain_feature_fuzzy.items():
        if debug: print("\t"*(deep+1), "Branch feature set:", fuzzy_set)
        feature_selector = (feature_names != max_gain_feature)
        new_feature_names = feature_names[feature_selector]
        X_members = X[fuzzy_set_mu > threshold][:,feature_selector]
        y_members = y[fuzzy_set_mu > threshold]
        new_mus = fuzzy_set_mu[fuzzy_set_mu > threshold]
        if len(y_members) > 0:
            branchs.append(
                ( 
                    fuzzy_set, 
                    node( X_members, y_members, new_mus, 
                        new_feature_names, targets_names, features_fuzzy_sets, 
                        deep+1, debug, max_deep, theta_r, theta_n, threshold, target_weights)
                )
            )
    return (max_gain_feature, branchs)
    
def build_fuzzy_tree(X, y, feature_names, target_names, features_fuzzy_sets, 
                     mu_init=None, deep=0, debug=False,
                     max_deep=None, theta_r=0.8, theta_n=1, threshold=0.005, target_name_ratio=None):
    mu = np.ones_like(y, dtype=float) if mu_init is None else mu_init
    feature_names = np.array(feature_names)
    target_weights = np.ones(len(target_names)) if target_name_ratio is None else calculate_target_weights(target_names, target_name_ratio)
    tree = node(X, y, mu, feature_names, target_names, features_fuzzy_sets, 
                deep, debug, max_deep, theta_r, theta_n, threshold, target_weights)
    return tree

def predict_(tree, Xi, features_fuzzy_sets, feature_names, target_names):
    if isinstance(tree, dict):
        return np.array([tree[Ck_i] for Ck_i, _ in enumerate(target_names)])
    elif isinstance(tree, tuple):
        feature, children = tree
        avg_predictions = np.zeros(len(target_names), dtype=float)
        feature_Xi = Xi[feature_names.index(feature)]
        for fuzzy_set, sub_tree in children:
            mu = features_fuzzy_sets[feature][fuzzy_set](feature_Xi)
            if mu > 1e-4:
                avg_predictions += mu*predict_(sub_tree, Xi, features_fuzzy_sets, feature_names, target_names)
        return avg_predictions

def predict(tree, X, features_fuzzy_sets, feature_names, target_names, tqdm_=None):
    if (X.shape) == 1:
        X = [X]
    predictions = []
    pbar = tqdm(X) if tqdm_ is None else tqdm_(X)
    for Xi in pbar:
        predictions_i = predict_(tree, Xi, features_fuzzy_sets, feature_names, target_names)
        predictions.append(np.argmax(predictions_i))
    return np.array(predictions)

def predict_proba(tree, X, features_fuzzy_sets, feature_names, target_names, tqdm_=None):
    if (X.shape) == 1:
        X = [X]
    predictions = []
    pbar = tqdm(X) if tqdm_ is None else tqdm_(X)
    for Xi in pbar:
        predictions_i = predict_(tree, Xi, features_fuzzy_sets, feature_names, target_names)
        predictions.append(predictions_i/np.sum(predictions_i))
    return np.array(predictions)

def tree_to_rules(tree, target_names, root=[]):
    if isinstance(tree, dict):
        max_ = (None, -1)
        for k,v in tree.items():
            if v > max_[1]: max_ = (k,v)
        conditions = " AND ".join(root)
        result = f'IS "{target_names[max_[0]]}"'
        return [f"IF ({conditions}), THEN {result}"]
    elif isinstance(tree, tuple):
        feature, children = tree
        rules = []
        for fuzzy_set, sub_tree in children:
            rules_ = tree_to_rules(sub_tree, target_names, root+[f'("{feature}" IS "{fuzzy_set}")'])
            rules.extend(rules_)
        return rules

def show_tree(tree, target_names, deep=0, fuzzy_set=None):
    if isinstance(tree, dict):
        print("\t"*deep, f"[{fuzzy_set}]->", {target_names[k]: np.round(v,2) for k,v in tree.items()})
    elif isinstance(tree, tuple):
        feature, children = tree
        print("\t"*deep, f"[{fuzzy_set}]->" if fuzzy_set is not None else ""  , f"{feature}:" )
        for fuzzy_set, sub_tree in children:
            show_tree(sub_tree, target_names, deep+1, fuzzy_set)