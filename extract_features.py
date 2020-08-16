import cv2
from skimage.color import rgb2hed
from skimage.feature import local_binary_pattern, greycomatrix
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import csv
from os.path import exists, basename

features_ranges = {
    # Color
    "mean_rawblue":     (0, 255),
    "mean_rawgreen":    (0, 255),
    "mean_rawbred":     (0, 255),
    "mean_exblue":      (-510, 510),
    "mean_exgreen":     (-510, 510),
    "mean_exred":       (-510, 510),
    "mean_intentsity":  (0, 255),
    "mean_hue":         (0, 255),
    "mean_saturation":  (0, 255),
    "mean_value":       (0, 255),
    "mean_dab":         (0, 255),
    "mean_eosin":       (0, 255),
    "mean_hematoxylin": (0, 255),
    # Texture
    "lpb_ror":          (0, 255),
}

def images_and_features_to_table(image_name, image, mask, color, filters, textures, csv_filepath, overwrite=False, quiet=True):
    all_features = {**color, **filters, **textures}
    rows, cols, _ = image.shape
    feature_keys = list(all_features.keys())
    table_columns = ["image", "row_i", "col_j", "B", "G", "R", "Class"] + feature_keys
    table_rows = []
    if not quiet: pbar = tqdm(total=(rows*cols))
    for row_i in range(rows):
        for col_j in range(cols):
            table_row = [image_name, row_i, col_j, 
                         image[row_i, col_j, 0], image[row_i, col_j, 1], image[row_i, col_j, 2],
                         mask[row_i, col_j]]
            for feature_k in feature_keys:
                table_row.append(all_features[feature_k][row_i, col_j])
            table_rows.append(table_row)
            if not quiet: pbar.update(1)
    if not quiet: pbar.close()

    if not quiet: print("Saving in", csv_filepath)
    if exists(csv_filepath):
        if overwrite: 
            print("File", csv_filepath, "exists!")
            print("Warning! overwriting")
            mode="w" 
        else: 
            print("Appending")
            mode="a"
    else:
        mode="w"

    with open(csv_filepath, mode=mode, newline='') as csvfile:
        tablewriter = csv.writer(csvfile, delimiter=',')
        if mode == "w": 
            csvfile.write(",".join(table_columns) + "\n")
        tablewriter.writerows(table_rows)
    return csv_filepath

def show_features(color, filters, textures, image_name, to_folder=None):
    all_features = {**color, **filters, **textures}


def filter_features(image):
    """
    Extract color features from image.
    List of Filter Features (2)
    ======================
    sobel_magnitud: magnitud of sobelx (dx) and sobely (dy)
    Laplacian: apply laplacian 
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (5,5))
    sobelx = cv2.Sobel(gray, cv2.CV_64F,1, 0,ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F,0, 1,ksize=3)
    magnitud = (255*(np.sqrt(sobelx**2 + sobely**2) / (255*np.sqrt(20)))).astype(int)
    laplacian = (255 * cv2.Laplacian(gray,cv2.CV_64F) / (255*8)).astype(int)
    return {
        "sobel_magnitud" : magnitud,
        "laplacian": laplacian
    }


def texture_features(image, region_size=5, quiet=True):
    """
    Extract color features from image.
    List of Texture Features (9)
    ======================
        lpb_ror: Local Binary Patter extension with roation invariant.
        GLCN features:
            mean_vertical:  np.zeros((rows, cols), dtype=float),
            mean_horizontal: np.zeros((rows, cols), dtype=float),
            homogeneity_horizontal: np.zeros((rows, cols), dtype=float),
            homogeneity_vertical: np.zeros((rows, cols), dtype=float),
            energy_horizontal : np.zeros((rows, cols), dtype=float),
            energy_vertical:  np.zeros((rows, cols), dtype=float),
            correlation_vertical: np.zeros((rows, cols), dtype=float),
            correlation_horizontal: np.zeros((rows, cols), dtype=float)
    """
    # Color  convertions
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)//4
    # Local Binary Pattern with rotation invariant
    lpb_ror = local_binary_pattern(image_gray, P=8, R=1.0, method="ror")
    # Grey Level Co-occurrence Matrix (GLCM)
    pad = (region_size - 1)//2
    rows, cols = image_gray.shape
    image_gray_pad = cv2.copyMakeBorder(image_gray, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    glcm_features = {
        "mean_vertical":  np.zeros((rows, cols), dtype=float),
        "mean_horizontal": np.zeros((rows, cols), dtype=float),
        "homogeneity_horizontal": np.zeros((rows, cols), dtype=float),
        "homogeneity_vertical": np.zeros((rows, cols), dtype=float),
        "energy_horizontal" : np.zeros((rows, cols), dtype=float),
        "energy_vertical":  np.zeros((rows, cols), dtype=float),
        "correlation_vertical": np.zeros((rows, cols), dtype=float),
        "correlation_horizontal": np.zeros((rows, cols), dtype=float)
    }
    ## levels
    ii, jj = np.meshgrid(np.arange(256//4), np.arange(256//4))
    ii = ii.reshape(256//4,256//4,1)
    jj = jj.reshape(256//4,256//4,1)
    if not quiet: pbar = tqdm(total=rows*cols)
    for row_i in range(rows):
        for col_j in range(cols):
            # Calcule GLCM for window 
            row_start = row_i
            row_end = row_i + 2*(pad)
            col_start = col_j
            col_end = col_j + 2*(pad)
            window = image_gray_pad[row_start:row_end + 1, col_start:col_end + 1]
            glc_matrix = greycomatrix(window, [1], [0, np.pi/2], levels=64, symmetric=True, normed=True)[:,:,0,:]
            # Mean - Descriptive Statistics
            mu_ij = np.multiply(ii, glc_matrix).sum(axis=(0,1))
            mu_h_ij, mu_v_ij = mu_ij
            # Correlation - Descriptive Statistics
            s2_ij = np.multiply(glc_matrix,(ii - mu_ij)**2).sum(axis=(0,1))
            corr_v_ij, corr_h_ij = np.multiply(glc_matrix,(((ii - mu_ij)*(jj-mu_ij))/s2_ij)).sum(axis=(0,1))
            # Homogeneity - Contrast Measure
            homogeneity_h_ij, homogeneity_v_ij = np.multiply(glc_matrix, 1/(1+(ii - jj)**2)).sum(axis=(0,1))
            # Energy - orderliness Measure
            energy_h_ij, energy_v_ij =  np.sqrt((glc_matrix**2).sum(axis=(0,1)))
            # Update features
            glcm_features["mean_vertical"][row_i, col_j] = mu_h_ij
            glcm_features["mean_horizontal"][row_i, col_j] = mu_v_ij
            glcm_features["correlation_vertical"][row_i, col_j] = corr_v_ij
            glcm_features["correlation_horizontal"][row_i, col_j] = corr_h_ij
            glcm_features["homogeneity_horizontal"][row_i, col_j] = homogeneity_h_ij
            glcm_features["homogeneity_vertical"][row_i, col_j] = homogeneity_v_ij
            glcm_features["energy_horizontal"][row_i, col_j] = energy_h_ij
            glcm_features["energy_vertical"][row_i, col_j] = energy_v_ij
            if not quiet: pbar.update(1)
    if not quiet: pbar.close()
    return {
        "lpb_ror": lpb_ror,
        **glcm_features
    }

def color_features(image, region_size=3):
    """
    Extract color features from image.
    region = (region_size x region_size)
    
    List of Color Features (13)
    ======================
        mean_rawblue: the average over the region of the B value.
        mean_rawgreen: the average over the region of the G value.
        mean_rawred: the average over the region of the R value.
        mean_exblue: measure the excess blue: (2B - (G + R))
        mean_exgreen: measure the excess green: (2G - (R + B))
        mean_exred: measure the excess red: (2R - (G + B))
        mean_intensiy: Intensity of the region.
        mean_saturation: Saturation of the region.
        mean_hue: Hue of the region.
        mean_value: Value of the region.
        mean_hematoxylin: hematoxylin of the region.
        mean_eosin: eosin of the region.
        mean_dab: dab of the region.
    """
    region = (region_size, region_size)
    # Blur image
    image_blur = cv2.blur(image, region)
    image_data = np.array(image_blur, dtype=int)  # support values out of the range [0, 255]          
    # Color  convertions
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_hsv_blur = cv2.blur(image_hsv, region)
    # HED color deconvolution
    hematoxylin_min = -0.6949930050571282436067122034728527069091796875
    hematoxylin_max = -0.2467002013182242603495097910126787610352039337158203125
    eosin_min = -0.0934433874521349017161497840788797475397586822509765625
    eosin_max = 0.36841286044231302820861628788406960666179656982421875
    dab_min = -0.54258864651267213474739037337712943553924560546875
    dab_max = -0.14370220292798296934932977819698862731456756591796875
    hed_image = rgb2hed(image_data[:,:,::-1]/255.)
    """
    Color Features
    """
    # Mean Blur
    mean_rawblue = image_data[:, :, 0]
    # Mean Green
    mean_rawgreen = image_data[:, :, 1]
    # Mean Red
    mean_rawbred = image_data[:, :, 2]
    # Mean exblue
    mean_exblue = 2*image_data[:, :, 0] - (image_data[:, :, 1] + image_data[:, :, 2])
    # Mean exgreen
    mean_exgreen = 2*image_data[:, :, 1] - (image_data[:, :, 0] + image_data[:, :, 2])
    # Mean exred
    mean_exred = 2*image_data[:, :, 2] - (image_data[:, :, 1] + image_data[:, :, 0])
    # Mean Intensity
    mean_intentsity = cv2.blur(image_gray, region)
    # Mean Hue
    mean_hue = image_hsv_blur[:,:,0]
    # Mean Saturation
    mean_saturation = image_hsv_blur[:,:,1]
    # Mean Value
    mean_value = image_hsv_blur[:,:,2]
    # Mean hematoxylin
    mean_hematoxylin = 255 * ((hed_image[:,:,0] - hematoxylin_min)/ (hematoxylin_max - hematoxylin_min))
    mean_hematoxylin = mean_hematoxylin.astype(int)
    # Mean eosin
    mean_eosin = 255 * ((hed_image[:,:,1] - eosin_min)/(eosin_max - eosin_min))
    mean_eosin = mean_eosin.astype(int)
    # mean DAB
    mean_dab = 255 * ((hed_image[:,:,2] - dab_min)/(dab_max - dab_min))
    mean_dab = mean_dab.astype(int)
    return {
        "mean_rawblue":     mean_rawblue,
        "mean_rawgreen":    mean_rawgreen,
        "mean_rawbred":     mean_rawbred,
        "mean_exblue":      mean_exblue,
        "mean_exgreen":     mean_exgreen,
        "mean_exred":       mean_exred,
        "mean_intentsity":  mean_intentsity,
        "mean_hue":         mean_hue,
        "mean_saturation":  mean_saturation,
        "mean_value":       mean_value,
        "mean_dab":         mean_dab,
        "mean_eosin":       mean_eosin,
        "mean_hematoxylin": mean_hematoxylin
    }

def extract_features(image_path, mask_path, csv_filepath, overwrite_csv=False, display_images=False, quiet=True):
    """
    Extract features from a images.
    """

    # Open Images
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Read files
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) // 255

    # Check for invalid inputs
    if image is None:
        raise ValueError(f"Could not open or find the image.", image_path)
    if mask is None:
        raise ValueError(f"Could not open or find the image.", mask_path)

    # Extract Color Features
    if not quiet: print("Color Features...", end=" ")
    extracted_color_features = color_features(image)
    if not quiet: print("Done!")
    if not quiet: print("Texture Features...")
    extracted_texture_features = texture_features(image, quiet=quiet)
    #extracted_texture_features = {}; print("skiped")
    if not quiet: print("Done!")
    if not quiet: print("Filter Features...", end=" ")
    extracted_filter_features = filter_features(image)
    if not quiet: print("Done!")

    # Save results
    image_name = basename(image_path)
    csv_file = images_and_features_to_table(image_name, image, mask, 
                                 extracted_color_features, extracted_filter_features, extracted_texture_features, 
                                 csv_filepath, overwrite_csv, quiet)

    return csv_file


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Extract features from a images")
    ap.add_argument("-i", "--image", required=True, help="Path to tiff image.", type=str)
    ap.add_argument("-m", "--mask", required=True, help="Path to tiff mask image.", type=str)
    ap.add_argument("-o", "--csv", help=" Path to csv file.", type=str, default="./features.csv")
    ap.add_argument('--overwrite', dest='overwrite', action='store_true')
    ap.add_argument('--display', dest='display', action='store_true')
    ap.add_argument('--show_progress', dest='not_quiet', action='store_true')
    args = vars(ap.parse_args())

    # Parse arguments
    image_path = args["image"]
    mask_path = args["mask"]
    csv_filepath = args["csv"]
    overwrite_csv = args["overwrite"]
    display_images = args["display"]
    quiet = not args["not_quiet"]
    # Run
    extract_features(image_path, mask_path, csv_filepath, overwrite_csv, display_images, quiet)

