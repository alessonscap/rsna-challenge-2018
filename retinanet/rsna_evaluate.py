#!/usr/bin/env python

"""
    rsna_evaluate.py
    - Script to evaluate model and print mAP based on GT information. Also, create file with detections to submit. 

    @author Alesson Scapinello
	@author Bernardo Henz
    @author Daniel Souza
    @author Felipe Kitamura
    @author Igor Santos
    @author José Venson
"""

import argparse
import os
import sys
import json
import keras
import tensorflow as tf
import numpy as np
import pandas

from keras_retinanet.keras_retinanet import models
from keras_retinanet.keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.keras_retinanet.preprocessing.pascal_voc import PascalVocGenerator
from keras_retinanet.keras_retinanet.utils.keras_version import check_keras_version

from rsna_generator import RsnaGenerator
from rsna_eval import evaluate
from keras import backend as K

def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_generator(rsna_test_json,rsna_path,hist_eq=False):
    """ Create generators for test.
    """
    test_generator = RsnaGenerator(
        rsna_test_json,
        rsna_path,
        image_only_transformations=None,
        hist_eq=hist_eq,
    )
    return test_generator


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')

    parser.add_argument('rsna_path', help='Path to dataset directory (ie. /tmp/COCO).')
    parser.add_argument('rsna_test_json', help='Path to test json.')
    parser.add_argument('model',             help='Path to RetinaNet model.')
    parser.add_argument('--kaggle_output_file', help='File to generate kaggle submission file.')

    parser.add_argument('--convert-model',   help='Convert the model to an inference model (ie. the input is a training model).', action='store_true')
    parser.add_argument('--backbone',        help='The backbone of the model.', default='resnet50')
    parser.add_argument('--gpu',             help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--score-threshold', help='Threshold on score to filter detections with (defaults to 0.05).', default=0.05, type=float)
    parser.add_argument('--iou-threshold',   help='IoU Threshold to count for a positive detection (defaults to 0.5).', default=0.5, type=float)
    parser.add_argument('--nms_threshold',   help='Non maximum suppression threshold',type=float, default=0.5)
    parser.add_argument('--max-detections',  help='Max Detections per image (defaults to 100).', default=100, type=int)
    parser.add_argument('--save-path',       help='Path for saving images with detections (doesn\'t work for COCO).')
    parser.add_argument('--image-min-side',  help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side',  help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    parser.add_argument('--anchor_boxes', help='List of anchor boxes', type=str, default='0.5,1,2')
    parser.add_argument('--anchor_scales', help='List of anchor scales', type=str, default='1, 1.25992105, 1.58740105')
    parser.add_argument('--hist_eq', help='Perform histogram equalization', action='store_true')

    parser.add_argument('--classifier_csv', help='Path to classifier output file.')
    parser.add_argument('--classifier_threshold', help='Threshold on classifier to filter detections (when classifier result is No LungOpacity)')
    parser.add_argument('--classifier_threshold_negative', help='Threshold on classifier to filter detections (when classifier result is No LungOpacity).')


    return parser.parse_args(args)

def rsna_evaluate(model_path, backbone, anchor_boxes,score_threshold, nms_threshold, rsna_path, rsna_test_json, anchor_scales):
    """ Evaluate an json using retinanet model, print mAP based on GT and generate kaggle submission file.
    """

    save_path = None
    from tensorflow import Graph, Session
    graph1 = Graph()
    with graph1.as_default():
        session1 = Session()
        with session1.as_default():
            model2 = models.load_model(model_path, backbone_name=backbone, convert=True, nms_threshold=nms_threshold, anchors_ratios=anchor_boxes,
                                    anchors_scales=anchor_scales)
            # create the generator
            generator = create_generator(rsna_test_json, rsna_path)

            map = evaluate(
               generator,
               model2,
               iou_threshold=0.5,
               score_threshold=score_threshold,
               max_detections=100,
               generate_kaggle_output='teste.csv'
            )
            del model2
            import gc
            gc.collect()
        with open ('output_map.txt', 'a') as output_map:
            output_map.write('{} : {} \n'.format(model_path,map))

def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    args.anchor_boxes = [float(item) for item in args.anchor_boxes.split(',')]
    args.anchor_scales = [float(item) for item in args.anchor_scales.split(',')]
    print('Using anchors: {}'.format(args.anchor_boxes))
    print('Using scales: {}'.format(args.anchor_scales))
    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # create the generator
    generator = create_generator(args.rsna_test_json, args.rsna_path,args.hist_eq)

    # load the model
    print('Loading model, this may take a second...')
    model = models.load_model(args.model, backbone_name=args.backbone, convert=args.convert_model,nms_threshold=args.nms_threshold,anchors_ratios=args.anchor_boxes,anchors_scales=args.anchor_scales)

    # print(model.summary())

    # start evaluation
    final_mAP = evaluate(
        generator,
        model,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
        max_detections=args.max_detections,
        save_path=args.save_path,
        generate_kaggle_output=args.kaggle_output_file
    )

    print('Map:', final_mAP)
    print('-'*80)

if __name__ == '__main__':
    main()
