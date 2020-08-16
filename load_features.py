import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_features(csv_filepath, drop_na=True):
    features = pd.read_csv(csv_filepath)
    if drop_na: features = features.dropna()
    return features

def merge_features(list_of_features):
    new_features_df = pd.concatente(list_of_features, ignore_index=True)
    return new_features_df

def split_features_target(features):
    target_col = "Class"
    non_features_cols = ["image", "row_i", "col_j", 'B', 'G', 'R']
    feature_names = set(features.columns) - set(non_features_cols) - set([target_col])
    feature_names = list(feature_names)

    X = features[feature_cols].to_numpy()
    y = features[target_col].to_numpy()
    
    return (feature_names, target_col), (X, y)


def to_image(content, points, size, multichannel=False):
    if multichannel:
        image = np.zeros((size[0], size[1], 3))
        image[points[:,0], points[:,1], :] = content/255.
    else:
        image = np.zeros(size)
        image[points[:,0], points[:,1]] = content
    return image

def imshow(image, title, multichannel=False):
    plt.figure(figsize=(6,6))
    plt.title(title)
    if multichannel: 
        plt.imshow(image)
    else:    
        im = plt.imshow(image, cmap="gray")
        plt.colorbar(im)
    plt.grid(False)
    plt.xticks([]); plt.yticks([]);
    plt.show()

def show_image_and_features(image_name, features, features_names):
    image_features = features[features["image"] == image_name]
    points = image_features[["row_i", "col_j"]].to_numpy()
    size = points.max(axis=0) + 1
    
    image_ = to_image(image_features[["R", "G", "B"]].to_numpy(), points, size, True)
    imshow(image_, image_name, True)

    image_ = to_image(image_features["Class"].to_numpy(), points, size)
    imshow(image_, f"{image_name} - Mask" )

    for f in features_names:
        image_ = to_image(image_features[f].to_numpy(), points, size)
        imshow(image_, f"{image_name} - {f}" )