"""
    rsna_eval.py
    - Auxiliar scripts to evaluation process.

    @author Alesson Scapinello
	@author Bernardo Henz
    @author Daniel Souza
    @author Felipe Kitamura
    @author Igor Santos
    @author José Venson
"""

from __future__ import print_function

from keras_retinanet.keras_retinanet.utils.anchors import compute_overlap
from keras_retinanet.keras_retinanet.utils.visualization import draw_detections, draw_annotations

import keras
import numpy as np
import os
import pandas
import cv2

class Evaluate(keras.callbacks.Callback):
    """ Evaluation callback for arbitrary datasets.
    """

    def __init__(self, generator, iou_threshold=0.5, score_threshold=0.05, max_detections=100, save_path=None, tensorboard=None, verbose=1):
        """ Evaluate a given dataset using a given model at the end of every epoch during training.

        # Arguments
            generator       : The generator that represents the dataset to evaluate.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
            score_threshold : The score confidence threshold to use for detections.
            max_detections  : The maximum number of detections to use per image.
            save_path       : The path to save images with visualized detections to.
            tensorboard     : Instance of keras.callbacks.TensorBoard used to log the mAP value.
            verbose         : Set the verbosity level, by default this is set to 1.
        """
        self.generator       = generator
        self.iou_threshold   = iou_threshold
        self.score_threshold = score_threshold
        self.max_detections  = max_detections
        self.save_path       = save_path
        self.tensorboard     = tensorboard
        self.verbose         = verbose

        super(Evaluate, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # run evaluation
        average_precisions = evaluate(
            self.generator,
            self.model,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
            save_path=self.save_path
        )

        # compute per class average precision
        present_classes = 0
        precision = 0
        for label, (average_precision, num_annotations ) in average_precisions.items():
            if self.verbose == 1:
                print('{:.0f} instances of class'.format(num_annotations),
                      self.generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
            if num_annotations > 0:
                present_classes += 1
                precision       += average_precision
        self.mean_ap = precision / present_classes

        if self.tensorboard is not None and self.tensorboard.writer is not None:
            import tensorflow as tf
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = self.mean_ap
            summary_value.tag = "mAP"
            self.tensorboard.writer.add_summary(summary, epoch)

        logs['mAP'] = self.mean_ap

        if self.verbose == 1:
            print('mAP: {:.4f}'.format(self.mean_ap))



def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(generator, model, score_threshold=0.5, max_detections=100, save_path=None, generate_kaggle_output=None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
    kaggle_results = []
    # Predict on all images 
    for i in range(generator.size()):
        raw_image    = generator.load_image(i)

        if generator.hist_eq:
            from rsna_generator import histogram_equalize
            raw_image = histogram_equalize(raw_image[:,:,0])*255
            raw_image = np.stack((raw_image,)*3, -1)
            
        image_name   = generator.get_image_name(i)
        image        = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)
        
        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))
        
        # run network
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]
        
        # correct boxes for image scale
        boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes      = boxes[0, indices[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = labels[0, indices[scores_sort]]

        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)
        
        # Save processed images
        if save_path is not None:
            draw_annotations(raw_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
            draw_detections(raw_image, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name, score_threshold=score_threshold)

            cv2.imwrite(os.path.join(save_path, '{}.png'.format(image_name)), raw_image)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]

        # change to (x, y, w, h) (Kaggle)
        boxes[:, :, 2] -= boxes[:, :, 0]
        boxes[:, :, 3] -= boxes[:, :, 1]
        rsna_boxes      = boxes[0, indices[scores_sort], :]

        prediction_string = ''
        selection = np.where(image_scores > score_threshold)[0]
        
        #Generate list with predictions to save with kaggle csv format  
        predictions=[]
        bbox_score = 1.0
        for select_i in selection:
            predictions.append(bbox_score)
            predictions.extend(list(rsna_boxes[select_i,:]))
        kaggle_result = {
            'patientId'         : image_name,
            'PredictionString'  : ' '.join(map(str, list(predictions))) 
        }
        kaggle_results.append(kaggle_result)
        if generate_kaggle_output:
            data_csv = pandas.DataFrame(kaggle_results)
            data_csv.to_csv(generate_kaggle_output, encoding='utf-8',
                                index=False, columns=["patientId", "PredictionString"])

        print('{}/{}'.format(i + 1, generator.size()), end='\r')

    return all_detections

def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, generator.size()]

    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        print('{}/{}'.format(i + 1, generator.size()), end='\r')

    return all_annotations

# helper function to calculate IoU
def _iou(box1, box2):
    x11, y11, w1, h1 = box1
    x21, y21, w2, h2 = box2
    assert w1 * h1 > 0
    assert w2 * h2 > 0
    x12, y12 = x11 + w1, y11 + h1
    x22, y22 = x21 + w2, y21 + h2

    area1, area2 = w1 * h1, w2 * h2
    xi1, yi1, xi2, yi2 = max([x11, x21]), max([y11, y21]), min([x12, x22]), min([y12, y22])
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    else:
        intersect = (xi2-xi1) * (yi2-yi1)
        union = area1 + area2 - intersect
        return intersect / union
    
def _map_iou(boxes_true, boxes_pred, scores, thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]):
    """
    Mean average precision at differnet intersection over union (IoU) threshold
    
    input:
        boxes_true: Mx4 numpy array of ground true bounding boxes of one image. 
                    bbox format: (x1, y1, w, h)
        boxes_pred: Nx4 numpy array of predicted bounding boxes of one image. 
                    bbox format: (x1, y1, w, h)
        scores:     length N numpy array of scores associated with predicted bboxes
        thresholds: IoU shresholds to evaluate mean average precision on
    output: 
        map: mean average precision of the image
    """
    
    # According to the introduction, images with no ground truth bboxes will not be 
    # included in the map score unless there is a false positive detection (?)
        
    # return None if both are empty, don't count the image in final evaluation (?)
    if len(boxes_true) == 0 and len(boxes_pred) == 0:
        return None
    
    # assert boxes_true.shape[1] == 4 or boxes_pred.shape[1] == 4, "boxes should be 2D arrays with shape[1]=4"
    if len(boxes_pred):
        assert len(scores) == len(boxes_pred), "boxes_pred and scores should be same length"
        # sort boxes_pred by scores in decreasing order
        boxes_pred = boxes_pred[np.argsort(scores)[::-1], :]
    
    map_total = 0
    
    # loop over thresholds
    for t in thresholds:
        matched_bt = set()
        tp, fn = 0, 0
        for i, bt in enumerate(boxes_true):
            matched = False
            for j, bp in enumerate(boxes_pred):
                miou = _iou(bt, bp[:4])
                if miou >= t and not matched and j not in matched_bt:
                    matched = True
                    tp += 1 # bt is matched for the first time, count as TP
                    matched_bt.add(j)
            if not matched:
                fn += 1 # bt has no match, count as FN
                
        fp = len(boxes_pred) - len(matched_bt) # FP is the bp that not matched to any bt
        m = tp / (tp + fn + fp)
        map_total += m
    
    return map_total / len(thresholds)

def evaluate(
    generator,
    model,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    generate_kaggle_output=None,
    save_path=None,
    hist_eq=False
):
    """ Evaluate a given dataset using a given model.

    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections     = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections, 
                                        save_path=save_path,generate_kaggle_output=generate_kaggle_output)
    all_annotations    = _get_annotations(generator)

    # Manually calculate MAP
    label = 0
    num_annotations = 0.0
    our_map = []
     
    for i in range(generator.size()):
        detections           = np.array(all_detections[i][label])
        annotations          = np.array(all_annotations[i][label])
        num_annotations     += annotations.shape[0]

        scores               = np.zeros((0,))
        cur_detections =  []
        cur_annotations =  []
        
        # Format annotations to get iou
        for i in range(len(annotations)):
            t = [annotations[i][0], annotations[i][1], annotations[i][2]-annotations[i][0], annotations[i][3]-annotations[i][1]]
            cur_annotations.append(t)
        # Format detections to get iou
        for i in range(len(detections)):
            scores = np.append(scores, detections[i][4])
            t = [detections[i][0], detections[i][1], detections[i][2]-detections[i][0], detections[i][3]-detections[i][1]]
            cur_detections.append(t)
        
        cur_detections = np.array(cur_detections)
        cur_annotations = np.array(cur_annotations)
        
        # Get mAP to the current image
        current_map = _map_iou(cur_annotations, cur_detections,scores)
        
        # Current_map can be [None, 0 or an float]. None means that both annotations and detections are empty
        if current_map is not None: 
            our_map.append(current_map)

    return np.mean(our_map)
