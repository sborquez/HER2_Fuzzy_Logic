import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def triangular(a, b, c):
    def mu(z):
        return np.piecewise(z, 
            [((a-b) <= z) & (z < a), (a <= z) & (z <= (a+c)), ((a-b) > z) | (z > (a+c))], 
            [lambda z: 1 - (a-z)/b,  lambda z:  1 - (z-a)/c,  lambda z: 0])
    return mu

def sigma(a, b):
    def mu(z):
        return np.piecewise(z, 
        [((a-b) <= z) & (z <= a), z > a,       z < (a-b)], 
        [lambda z: 1 - (a-z)/b,   lambda z: 1, lambda z: 0])
    return mu

def sigma_inv(a, b):
    def mu(z):
        return np.piecewise(z, 
            [(a <= z) & (z <= (a+b)), z < a,       z > (a+b)  ], 
            [lambda z: 1 - (z-a)/b,   lambda z: 1, lambda z: 0])
    return mu

def trapezoidal(a,b,c,d):
    def mu(z):
        return np.piecewise(z, 
        [((a-c) <= z) & (z < a), (a<= z) & (z < b), (b <= z) & (z <= (b+d)), (z < (a-c)) | (z > (b+d))], 
        [lambda z: 1 - (a-z)/c,  lambda z: 1,       lambda z:1 - (z - b)/d , lambda z: 0])
    return mu

def show_membership(features_fuzzy_sets, ranges=None):
    min_, max_ = (0,10) if ranges is None else ranges
    for feature, fuzzy_sets in features_fuzzy_sets.items():
        z = np.linspace(min_, max_, 500)
        plt.title(feature)
        plt.xlabel("z")
        plt.ylabel("$\mu(z)$")
        for fuzzy_set, mu in fuzzy_sets.items():
            plt.plot(z, mu(z), label=fuzzy_set)
        plt.legend()
        plt.show()

def show_fuzzy_set(feature, fuzzy_sets, ranges=None):
    min_, max_ = (0,10) if ranges is None else ranges
    z = np.linspace(min_, max_, 500)
    plt.title(feature)
    plt.xlabel("z")
    plt.ylabel("$\mu(z)$")
    for fuzzy_set, mu in fuzzy_sets.items():
        plt.plot(z, mu(z), label=fuzzy_set)
    plt.legend()
    plt.show()


def build_3_triangular(names, z1, z2, z3):
    if len(names) != 3:
        raise ValueError
    return {
        names[0]: sigma_inv(a=z1, b=z2-z1),
        names[1]: triangular(a=z2, b=z2-z1, c=z3-z2) ,
        names[2]: sigma(a=z3, b=z3-z2)
    }

def build_3_trapezoidal(names, z1, z2, z3, z4):
    if len(names) != 3:
        raise ValueError
    return {
        names[0]: sigma_inv(a=z1, b=z2-z1),
        names[1]: trapezoidal(a=z2, b=z3, c=z2-z1, d=z4-z3) ,
        names[2]: sigma(a=z4, b=z4-z3)
    }

def build_2_sigma(names, z1, z2):
    if len(names) != 2:
        raise ValueError
    return {
        names[0]: sigma_inv(a=z1, b=z2-z1),
        names[1]: sigma(a=z2, b=z2-z1)
    }

    