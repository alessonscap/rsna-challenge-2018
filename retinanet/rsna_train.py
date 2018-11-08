#!/usr/bin/env python

"""
    rsna_train.py
    - Train models using keras and tensorflow backend.

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
import warnings

import keras
import keras.preprocessing.image
import tensorflow as tf

# Change these to absolute imports if you copy this script outside the keras_retinanet package.

from keras_retinanet.keras_retinanet  import layers  # noqa: F401
from keras_retinanet.keras_retinanet  import losses
from keras_retinanet.keras_retinanet  import models
from keras_retinanet.keras_retinanet.callbacks import RedirectModel
from keras_retinanet.keras_retinanet.callbacks.eval import Evaluate
from keras_retinanet.keras_retinanet.models.retinanet import retinanet_bbox
from keras_retinanet.keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.keras_retinanet.preprocessing.kitti import KittiGenerator
from keras_retinanet.keras_retinanet.preprocessing.open_images import OpenImagesGenerator
from keras_retinanet.keras_retinanet.preprocessing.pascal_voc import PascalVocGenerator
from keras_retinanet.keras_retinanet.utils.anchors import make_shapes_callback
from keras_retinanet.keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.keras_retinanet.utils.model import freeze as freeze_model
from keras_retinanet.keras_retinanet.utils.model import freeze_first_N_layers
from keras_retinanet.keras_retinanet.utils.transform import random_transform_generator
from keras_retinanet.keras_retinanet.models.retinanet import AnchorParameters
from keras.callbacks import TensorBoard


from rsna_generator import RsnaGenerator, ImageOnlyTransformations

from keras import backend as K

class TrainValTensorBoard(TensorBoard):
    """ Tensorboard callback for train and validation.
    """
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


def set_keras_backend(backend):
    """ Configure tensorflow as backend
    """
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        importlib.reload(K)
        assert K.backend() == backend
    if backend == "tensorflow":
        K.get_session().close()
        cfg = K.tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        K.set_session(K.tf.Session(config=cfg))
        K.clear_session()



def makedirs(path):
    """ Try to create the directory,
        pass if the directory exists already, fails otherwise.
    """
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def model_with_weights(model, weights, skip_mismatch):
    """ Load weights for model.

    Args
        model         : The model to load weights for.
        weights       : The weights to load.
        skip_mismatch : If True, skips layers whose shape of weights doesn't match with the model.
    """
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


def create_models(backbone_retinanet, num_classes, weights, multi_gpu=0, freeze_backbone=False,anchors_ratios=None,anchors_scales=None,noise_aug_std=None,dropout_rate=None):
    """ Creates three models (model, training_model, prediction_model).

    Args
        backbone_retinanet : A function to call to create a retinanet model with a given backbone.
        num_classes        : The number of classes to train.
        weights            : The weights to load into the model.
        multi_gpu          : The number of GPUs to use for training.
        freeze_backbone    : If True, disables learning for the backbone.

    Returns
        model            : The base model. This is also the model that is saved in snapshots.
        training_model   : The training model. If multi_gpu=0, this is identical to model.
        prediction_model : The model wrapped with utility functions to perform object detection (applies regression values and performs NMS).
    """
    modifier = freeze_model if freeze_backbone else None
    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    anchor_params = AnchorParameters.default
    anchor_params.ratios = anchors_ratios
    anchor_params.scales = anchors_scales
    if multi_gpu > 1:
        from keras.utils import multi_gpu_model
        with tf.device('/cpu:0'):
            model = model_with_weights(backbone_retinanet(num_classes, modifier=modifier,num_anchors=anchor_params.num_anchors(),noise_aug_std=noise_aug_std,dropout_rate=dropout_rate), weights=weights, skip_mismatch=True)
        training_model = multi_gpu_model(model, gpus=multi_gpu)
    else:
        model          = model_with_weights(backbone_retinanet(num_classes, modifier=modifier,num_anchors=anchor_params.num_anchors(),noise_aug_std=noise_aug_std,dropout_rate=dropout_rate), weights=weights, skip_mismatch=True)
        training_model = model

    # make prediction model
    prediction_model = retinanet_bbox(model=model,anchors_ratios=anchors_ratios,anchors_scales=anchors_scales)

    # compile model
    training_model.compile(
        loss={
            'regression'    : losses.smooth_l1(),
            'classification': losses.focal()
        },
        optimizer=keras.optimizers.adam(lr=1e-4, clipnorm=0.001)
    )

    return model, training_model, prediction_model


def create_callbacks(model, training_model, prediction_model, validation_generator, args):
    """ Creates the callbacks to use during training.

    Args
        model: The base model.
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.

    Returns:
        A list of callbacks used for training.
    """
    callbacks = []

    tensorboard_callback = None

    if args.tensorboard_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir                = args.tensorboard_dir,
            histogram_freq         = 0,
            batch_size             = args.batch_size,
            write_graph            = True,
            write_grads            = False,
            write_images           = False,
            embeddings_freq        = 0,
            embeddings_layer_names = None,
            embeddings_metadata    = None
        )
        # callbacks.append(tensorboard_callback)
        callbacks.append(TrainValTensorBoard(write_graph=False, log_dir=args.tensorboard_dir))

    # save the model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        makedirs(args.snapshot_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                '{backbone}_{dataset_type}_{{epoch:02d}}.h5'.format(backbone=args.backbone, dataset_type=args.dataset_type)
            ),
            verbose=1
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)

    # Append Reduce LR on plateau callback (see keras documentation)
    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor  = 'loss',
        factor   = 0.1,
        patience = 2,
        verbose  = 1,
        mode     = 'auto',
        epsilon  = 0.0001,
        cooldown = 0,
        min_lr   = 0
    ))

    return callbacks


def create_generators(args, preprocess_image):
    """ Create generators for training and validation.

    Args
        args             : parseargs object containing configuration for generators.
        preprocess_image : Function that preprocesses an image for the network.
    """
    common_args = {
        'batch_size'       : args.batch_size,
        'image_min_side'   : args.image_min_side,
        'image_max_side'   : args.image_max_side,
        'preprocess_image' : preprocess_image,
    }

    # create random transform generator for augmenting training data
    if args.random_transform:
        transform_generator = random_transform_generator(
            min_rotation=-0.1,
            max_rotation=0.1,
            min_translation=(-0.1, -0.1),
            max_translation=(0.1, 0.1),
            min_shear=-0.1,
            max_shear=0.1,
            min_scaling=(0.9, 0.9),
            max_scaling=(1.1, 1.1),
            flip_x_chance=0.5,
            flip_y_chance=0.0,
        )
    else:
        transform_generator = random_transform_generator(flip_x_chance=0.0)
    
    # Add proper data augmentation
    if args.image_only_transformations:
        image_only_transformations = ImageOnlyTransformations(noise_std = 0.02, 
                                                              contrast_level= 0.3,
                                                              brightness_level = 0.1)
    else:
        image_only_transformations = None

    #Create train and validation generator
    train_generator = RsnaGenerator(
        args.rsna_train_json,
        args.rsna_path,
        transform_generator=transform_generator,
        image_only_transformations = image_only_transformations,
        bbox_aug_std = args.bbox_aug_std,
        anchors_ratios = args.anchor_boxes,
        anchors_scales = args.anchor_scales,
        dicom_load_mode=args.dicom_load_mode,
        hist_eq=args.hist_eq,
        **common_args
    )
    
    train_generator.transform_parameters.fill_mode = 'constant'
    
    validation_generator = RsnaGenerator(
        args.rsna_val_json,
        args.rsna_path,
        image_only_transformations=None,
        anchors_ratios = args.anchor_boxes,
        anchors_scales = args.anchor_scales,
        dicom_load_mode=args.dicom_load_mode,
        hist_eq=args.hist_eq,
        **common_args
    )

    return train_generator, validation_generator


def check_args(parsed_args):
    """ Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.

    Args
        parsed_args: parser.parse_args()

    Returns
        parsed_args
    """

    if parsed_args.multi_gpu > 1 and parsed_args.batch_size < parsed_args.multi_gpu:
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size,
                                                                                             parsed_args.multi_gpu))
    if parsed_args.multi_gpu > 1 and parsed_args.snapshot:
        raise ValueError(
            "Multi GPU training ({}) and resuming from snapshots ({}) is not supported.".format(parsed_args.multi_gpu,
                                                                                                parsed_args.snapshot))

    if parsed_args.multi_gpu > 1 and not parsed_args.multi_gpu_force:
        raise ValueError("Multi-GPU support is experimental, use at own risk! Run with --multi-gpu-force if you wish to continue.")

    if 'resnet' not in parsed_args.backbone:
        warnings.warn('Using experimental backbone {}. Only resnet50 has been properly tested.'.format(parsed_args.backbone))

    return parsed_args


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    rsna_parser = subparsers.add_parser('rsna')
    rsna_parser.add_argument('rsna_path', help='Path to dataset directory (ie. /tmp/COCO).')
    rsna_parser.add_argument('rsna_train_json', help='Path to training json.')
    rsna_parser.add_argument('rsna_val_json', help='Path to validation json.')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot',          help='Resume training from a snapshot.')
    group.add_argument('--imagenet-weights',  help='Initialize the model with pretrained imagenet weights. This is the default behaviour.', action='store_const', const=True, default=True)
    group.add_argument('--weights',           help='Initialize the model with weights from a file.')
    group.add_argument('--no-weights',        help='Don\'t initialize the model with any weights.', dest='imagenet_weights', action='store_const', const=False)

    parser.add_argument('--backbone',        help='Backbone model used by retinanet.', default='resnet50', type=str)
    parser.add_argument('--batch-size',      help='Size of the batches.', default=1, type=int)
    parser.add_argument('--gpu',             help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--multi-gpu',       help='Number of GPUs to use for parallel processing.', type=int, default=0)
    parser.add_argument('--multi-gpu-force', help='Extra flag needed to enable (experimental) multi-gpu support.', action='store_true')
    parser.add_argument('--epochs',          help='Number of epochs to train.', type=int, default=50)
    parser.add_argument('--steps',           help='Number of steps per epoch.', type=int, default=10000)
    parser.add_argument('--val_steps',           help='Number of steps per epoch.', type=int, default=400)
    parser.add_argument('--snapshot-path',   help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default='./snapshots')
    parser.add_argument('--tensorboard_dir', help='Log directory for Tensorboard output', default='./logs')
    parser.add_argument('--no-snapshots',    help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--no-evaluation',   help='Disable per epoch evaluation.', dest='evaluation', action='store_false')
    parser.add_argument('--freeze-backbone', help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument('--data-aug', help='Enables random-transforms and image-only-transforms.', action='store_true')
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--image_only_transformations', help='Randomly perform image-only transformations.', action='store_true')
    parser.add_argument('--noise_aug_std', help='Defines de std of the random noise added during training. If noise_aug_std=None, no noise is added.', type=float,default=None)
    parser.add_argument('--bbox_aug_std', help='Defines the std of the bounding box augs (none aug with not set).', type=float,default=None)
    parser.add_argument('--dropout_rate', help='Defines the dropout rate.', type=float,default=None)

    parser.add_argument('--image-min-side', help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side', help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    parser.add_argument('--dicom_load_mode', help='Decide to load only image (image) or sex and view position as well (image_sex_view).', type=str, default='image')
    parser.add_argument('--hist_eq', help='Perform histogram equalization', action='store_true')
    
    parser.add_argument('--anchor_boxes', help='List of anchor boxes', type=str, default='0.5,1,2')
    parser.add_argument('--anchor_scales', help='List of anchor scales', type=str, default='1, 1.25992105, 1.58740105')
    parser.add_argument('--score_threshold', help='Threshold on score to filter detections with (defaults to 0.05).', default=0.2, type=float)
    parser.add_argument('--nms_threshold',   help='Non maximum suppression threshold',type=float, default=0.1)

    return check_args(parser.parse_args(args))

def main(args=None):
    set_keras_backend("tensorflow")  
    # parse arguments
    if args is None:
        args = sys.argv[1:]

    #Parse anchor boxes/scales
    args = parse_args(args)
    args.anchor_boxes = [float(item) for item in args.anchor_boxes.split(',')]
    args.anchor_scales = [float(item) for item in args.anchor_scales.split(',')]
    print('Using anchors: {}'.format(args.anchor_boxes))
    
    if (args.data_aug):
        args.random_transform=True
        args.image_only_transformations=True

    # create object that stores backbone information
    backbone = models.backbone(args.backbone,args.noise_aug_std)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # create the generators
    train_generator, validation_generator = create_generators(args, backbone.preprocess_image)
    
    # create the model
    if args.snapshot is not None:
        print('Loading model, this may take a second...')
        model            = models.load_model(args.snapshot, backbone_name=args.backbone, anchors_ratios=args.anchor_boxes,anchors_scales=args.anchor_scales)
        training_model   = model
        prediction_model = retinanet_bbox(model=model, anchors_ratios=args.anchor_boxes,anchors_scales=args.anchor_scales)
    else:
        weights = args.weights
        # default to imagenet if nothing else is specified
        if weights is None and args.imagenet_weights:
            weights = backbone.download_imagenet()

        print('Creating model, this may take a second...')
        model, training_model, prediction_model = create_models(
            backbone_retinanet=backbone.retinanet,
            num_classes=1,
            weights=weights,
            multi_gpu=args.multi_gpu,
            freeze_backbone=args.freeze_backbone,
            anchors_ratios=args.anchor_boxes,
            anchors_scales=args.anchor_scales,
            noise_aug_std=args.noise_aug_std,
            dropout_rate=args.dropout_rate
        )

    # print model summary
    print(model.summary())

    # this lets the generator compute backbone layer shapes using the actual backbone model
    if 'vgg' in args.backbone or 'densenet' in args.backbone:
        train_generator.compute_shapes = make_shapes_callback(model)
        if validation_generator:
            validation_generator.compute_shapes = train_generator.compute_shapes

    # create the callbacks
    callbacks = create_callbacks(
        model,
        training_model,
        prediction_model,
        validation_generator, 
        args,
    )

    # start training
    training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=validation_generator,
        validation_steps=args.val_steps
    )

    
if __name__ == '__main__':
    main()
