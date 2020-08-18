import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_features(csv_filepath, drop_na=True, selected_features=None):
    features = pd.read_csv(csv_filepath)
    if drop_na: features = features.dropna()
    if selected_features is not None:
        if not isinstance(selected_features, list): 
            selected_features = [selected_features]
        target_col = ["Class"]
        non_features_cols = ["image", "row_i", "col_j", 'B', 'G', 'R']
        drop_features = set(features.columns) - set(non_features_cols) - set(target_col) - set(selected_features)
        features.drop(columns=drop_features, inplace=True)
    return features

def merge_features(list_of_features):
    new_features_df = pd.concat(list_of_features, ignore_index=True)
    return new_features_df

def split_features_target(features, only_data=False):
    target_col = "Class"
    non_features_cols = ["image", "row_i", "col_j", 'B', 'G', 'R']
    feature_names = set(features.columns) - set(non_features_cols) - set([target_col])
    feature_names = list(feature_names)

    X = features[feature_names].to_numpy()
    y = features[target_col].to_numpy()
    if only_data:
        return  (X, y)
    else:
        return (feature_names, target_col), (X, y)


def to_image(content, points, size, multichannel=False):
    if multichannel:
        image = np.zeros((size[0], size[1], 3))
        image[points[:,0], points[:,1], :] = content/255.
    else:
        image = np.zeros(size)
        image[points[:,0], points[:,1]] = content
    return image

def imshow(image, title, multichannel=False, vmin=None, vmax=None, ax=None):
    if ax is None:
        plt.figure(figsize=(6,6))
        ax = plt.gca()
        plot=True
    else:
        plot=False
    ax.set_title(title)
    if multichannel: 
        ax.imshow(image, vmin=vmin, vmax=vmax)
    else:    
        im = ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
        if plot: plt.colorbar(im)
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]);
    if plot: plt.show()

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

def show_images_and_masks(image_name, features, predicted):
    image_features = features[features["image"] == image_name]
    points = image_features[["row_i", "col_j"]].to_numpy()
    size = points.max(axis=0) + 1
    
    image_ = to_image(image_features[["R", "G", "B"]].to_numpy(), points, size, True)
    imshow(image_, image_name, True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    image_ = to_image(image_features["Class"].to_numpy(), points, size)
    imshow(image_, f"{image_name} - Mask", vmin=0, vmax=1, ax=ax1 )
    image_ = to_image(predicted, points, size)
    imshow(image_, f"{image_name} - Predicted", vmin=0, vmax=1, ax=ax2 )
    plt.show()