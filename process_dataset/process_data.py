"""
    process_data.py
    - Convert data information in COCO format. 

    @author Alesson Scapinello
	@author Bernardo Henz
    @author Daniel Souza
    @author Felipe Kitamura
    @author Igor Santos
    @author José Venson
"""

###################################
# Misc
###################################
import os
import re
import argparse
import datetime
import fnmatch
import pandas
import json
import random

###################################
# Dicom Images
###################################
import pydicom
import numpy as np

# Info, licenses, categories based on COCO format (see http://cocodataset.org/#home)
INFO = {
    "description": "RSNA-Pneumonia-Detection-Challenge Dataset",
    "url": "https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "DASA-FIDI-IARA",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 0,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 0,
        'name': 'Lung Opacity',
    },
    {
        'id': 1,
        'name': 'No Lung Opacity / Not Normal',
    },
    {
        'id': 2,
        'name': 'Normal',
    },
]

def create_annotation_info(id, image_id, category_id, bbox):
    """ Create an annotation to the image id
    """
    return {
        "id": id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "iscrowd" : 0
    }

def create_image_info(image_id, image_size, file_name, license_id=1):
    """ Create an image information with coco format
    """
    return {
        "id": image_id,
        "width": image_size[0],
        "height": image_size[1],
        "file_name": file_name,
        "license": license_id,
    }

def filter_for_dcm(root, files):
    """ Filter an directory for dicom images
    """
    file_types = ['*.dcm']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    
    return files

def convert_to_coco(train_csv_path, class_info_csv_path, train_path,output_train_json_name, output_validation_json_name, output_test_json_name, split=[70,20,10]):
    """ Convert kaggle dataset to coco format
    """

    train_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
    val_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
    test_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
    # Split dataset into train, validation and test
    if split is None:
        split = [70,20,10]

    image_id = 0
    annotation_id = 0

    if not (os.path.exists(train_csv_path) and os.path.exists(class_info_csv_path)):
        raise ValueError("Error openning csv files!")
    
    # Open csv and group by patientId 
    df_class_info = pandas.read_csv(class_info_csv_path)
    df_train = pandas.read_csv(train_csv_path)
    df_train = df_train.groupby('patientId')

    train_images = 0
    val_images = 0
    test_images = 0

    for patient_id, rows in df_train:
        file_name = '{}.dcm'.format(patient_id)
        image_path = os.path.join(train_path, file_name)
        image = pydicom.dcmread(image_path).pixel_array
        
        # Create current image info 
        image_info = create_image_info(image_id,image.shape, file_name)
        rand_value = random.random()
        
        # Add current info into train, val or test dataset
        if rand_value < split[0]/100:
            train_images+=1
            train_output["images"].append(image_info)
        elif rand_value < split[0]/100+split[1]/100:
            val_images+=1
            val_output["images"].append(image_info)
        else:
            test_images+=1
            test_output["images"].append(image_info)
        
        # Find category according to detailed csv provided by rsna dataset
        class_info_row = df_class_info[df_class_info.patientId==patient_id]
        category_id = list(filter(lambda category:category["name"] == class_info_row["class"][class_info_row.index[0]],CATEGORIES))[0]["id"] 

        #Each image can have 0 or n annotations
        for _, annotation_info in rows.iterrows():  
            # Create current image annotation             
            annotation_info = create_annotation_info(annotation_id,image_id,category_id,
                                    [annotation_info.x, annotation_info.y, annotation_info.width, annotation_info.height])
            annotation_id += 1

            # Add current annotation into train, val or test dataset
            if rand_value < split[0]/100:
                train_output["annotations"].append(annotation_info)
            elif rand_value < split[0]/100+split[1]/100:
                val_output["annotations"].append(annotation_info)
            else:
                test_output["annotations"].append(annotation_info)                        
        image_id += 1

    print("Json created. train: {}, val: {}, test: {}".format(train_images,val_images,test_images))
    # Save json to train
    with open(output_train_json_name, 'w') as output_json_file:
        json.dump(train_output, output_json_file)
    output_json_file.close()
    with open(output_validation_json_name, 'w') as output_json_file:
        json.dump(val_output, output_json_file)
    output_json_file.close()
    with open(output_test_json_name, 'w') as output_json_file:
        json.dump(test_output, output_json_file)
    output_json_file.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    #required arguments
    parser.add_argument('input_csv_train', help='path to stage_1_train_labels.csv file')
    parser.add_argument('class_info_csv_path', help='path to class_info_csv_path.csv file')
    parser.add_argument('input_train_images_path', help='path to rsna-challenge-2018 dataset')
    parser.add_argument('output_train_json_name', help='Name to json with train dataset')
    parser.add_argument('output_validation_json_name', help='Name to json with validation dataset')
    parser.add_argument('output_test_json_name', help='Name to json with test dataset')
    parser.add_argument('--split', help='Values to split dataset into train, val and test. (Default [70 20 10])', nargs='+', type=int)

    # parse arguments and store them into variables
    args = parser.parse_args()
    train_csv_path = args.input_csv_train
    train_path = args.input_train_images_path
    class_info_csv_path = args.class_info_csv_path
    output_train_json_name = args.output_train_json_name
    output_validation_json_name = args.output_validation_json_name
    output_test_json_name = args.output_test_json_name
    split = args.split
    
    convert_to_coco(train_csv_path, class_info_csv_path, train_path,output_train_json_name, output_validation_json_name, output_test_json_name, split)
