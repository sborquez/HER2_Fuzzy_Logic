#!/bin/bash

images="/mnt/e/datasets/medical/her2/Raquel/big_images"
masks="/mnt/e/datasets/medical/her2/Raquel/big_images_mask"

for image in  $(ls $images);
do
    mask=${image/.tif/"_mask.tif"}
    csv=${image/.tif/".csv"}
    echo "$images/$image $masks/$mask $csv"
    python extract_features.py -i "$images/$image" -m "$masks/$mask" -o "./data/$csv" --show_progress
done
