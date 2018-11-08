from keras_retinanet.keras_retinanet.preprocessing.generator import Generator
import keras
import os
import numpy as np

import pydicom

from pycocotools.coco import COCO

from skimage import exposure
def histogram_equalize(img):
    img_cdf, bin_centers = exposure.cumulative_distribution(img)
    return np.interp(img, bin_centers, img_cdf)
    
class ImageOnlyTransformations():
    def __init__(self, noise_std, contrast_level, brightness_level):
        self.noise_std = noise_std
        self.contrast_level = contrast_level
        self.brightness_level = brightness_level
    
    def performTransformations(self,image):
        #outimage = np.clip(image + np.random.normal(scale= np.random.uniform(0.0001,self.noise_std),size=image.shape),0,1)  #too slow, moving to keras_model
        outimage = np.clip( ((image-0.5)*np.random.uniform(1-self.contrast_level,1+self.contrast_level))+0.5 ,0,1)
        outimage = np.clip( outimage + np.random.uniform(-self.brightness_level,self.brightness_level),0,1 )
        return outimage


class RsnaGenerator(Generator):
    """ 
    Generate data from the RSNA dataset.
    """

    def __init__(self, rsna_json_path, data_dir,image_only_transformations,dicom_load_mode='image',hist_eq=False, **kwargs):
        """ Initialize a RSNA data generator.

        Args
            data_dir: Path to where the RSNA dataset is stored.
        """
        self.data_dir  = data_dir
        self.coco      = COCO(rsna_json_path)
        self.image_ids = self.coco.getImgIds()
        self.image_only_transformations = image_only_transformations
        self.dicom_load_mode = dicom_load_mode
        self.hist_eq = hist_eq

        self.load_classes()
        super(RsnaGenerator, self).__init__(**kwargs)

    def load_classes(self):
        """ Loads the class to label mapping (and inverse) for COCO.
        """
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes             = {}
        self.coco_labels         = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key


    def size(self):
        """ Size of the COCO dataset.
        """
        return len(self.image_ids)

    def num_classes(self):
        """ Number of classes in the dataset. For COCO this is 80.
        """
        return len(self.classes)

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        if label == -1:
            label = 2
        return self.labels[label]

    def coco_label_to_label(self, coco_label):
        """ Map COCO label to the label as used in the network.
        COCO has some gaps in the order of labels. The highest label is 90, but there are 80 classes.
        """
        return self.coco_labels_inverse[coco_label]

    def coco_label_to_name(self, coco_label):
        """ Map COCO label to name.
        """
        return self.label_to_name(self.coco_label_to_label(coco_label))

    def label_to_coco_label(self, label):
        """ Map label as used by the network to labels as used by COCO.
        """
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path       = os.path.join(self.data_dir, image_info['file_name'])
        return self.read_image_dicom(path,mode=self.dicom_load_mode)

    def get_image_name(self, image_index):
        """ Get the file name at the image_index.
        """
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        return image_info['file_name'][:-4]  #remove .dcm from file_name

    def get_image_category(self, image_index):
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        if len(annotations_ids) == 0:
            return 
        coco_annotations = self.coco.loadAnns(annotations_ids)
        return coco_annotations[0]['category_id']

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        # get ground truth annotations
        # print('image_index: ',image_index)
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations     = np.zeros((0, 5))
        # print('annotations before: ',annotations)
        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):
            #print('idx: ',idx)
            #print('a: ',a)
            annotation        = np.zeros((1, 5))
            if (self.coco_label_to_label(a['category_id']) is not 0):
                annotation[0,4]  = -1
                annotation[0,2:4] = 0.1
                annotations       = np.append(annotations, annotation, axis=0)
                continue
                #return annotations
            # some annotations have basically no width / height, skip them
            if not a['bbox']:   
                #annotation[0, 4]  = self.coco_label_to_label(a['category_id'])
                #annotations       = np.append(annotations, annotation, axis=0)
                continue
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue
            
            annotation[0, :4] = a['bbox']
            annotation[0, 4]  = self.coco_label_to_label(a['category_id'])
            annotations       = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]
        
        #print('annotations: ',annotations)
        return annotations

    def read_image_dicom(self,path,mode='image'):
        """ Read an image in dicom format.
        Args
            path: Path to the image.
            mode: image|image_sex_view
        """
        dicom_img = pydicom.dcmread(path)
        image = dicom_img.pixel_array
        image = np.stack((image,)*3, -1) #convert grayscale to rgb
        if mode=='image_sex_view':
            if dicom_img.PatientSex == 'F':
                image[:,:,1] = 0
            elif dicom_img.PatientSex == 'M':
                image[:,:,1] = 1
            else:
                raise Exception('Invalid Sex on dicom {}.'.format(path))
            if dicom_img.ViewPosition == 'AP':
                image[:,:,2] = 0
            elif dicom_img.ViewPosition == 'PA':
                image[:,:,2] = 1
            else:
                raise Exception('Invalid View Position on dicom {}. View position is: {}'.format(path,dicom_img.ViewPosition))
        return image[:, :].copy()

    def preprocess_imageRSNA_notusing(self, x, mode='rsna'):
        """ Preprocess an image by subtracting the ImageNet mean.

        Args
            x: np.array of shape (None, None, 3) or (3, None, None).
            mode: One of "caffe" or "tf".
                - caffe: will zero-center each color channel with
                    respect to the ImageNet dataset, without scaling.
                - tf: will scale pixels between -1 and 1, sample-wise.

        Returns
            The input with the ImageNet mean subtracted.
        """
        # mostly identical to "https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py"
        # except for converting RGB -> BGR since we assume BGR already
        x = x.astype(keras.backend.floatx())
        if mode == 'tf':
            x /= 127.5
            x -= 1.
        elif mode == 'caffe':
            x[..., 0] -= 103.939
            x[..., 1] -= 116.779
            x[..., 2] -= 123.68
        elif mode == 'rsna':
            x /= 255
            print('RSNA [0,1]')
        return x

    def preprocess_group_entry(self, image, annotations):
        """ Preprocess image and its annotations.
        """
        # preprocess the image
        if self.image_only_transformations:
            image = self.image_only_transformations.performTransformations(image.astype(keras.backend.floatx())/255.0)*255
        

            # # randomly transform image and annotations
        image, annotations = self.random_transform_group_entry(image, annotations)
        
        if (self.dicom_load_mode=='image_sex_view'):
            img_backup = np.copy(image[:,:,1:])

        if(self.hist_eq):
            image = histogram_equalize(image[:,:,0]) * 255 #grayscale image
            image = np.stack((image,)*3, -1)

        if (self.dicom_load_mode=='image_sex_view'):
            image[:,:,1:] = img_backup
        
        image = self.preprocess_image(image)

        # resize image
        image, image_scale = self.resize_image(image)

        # apply resizing to annotations too
        annotations[:, :4] *= image_scale

        return image, annotations

    def preprocess_group(self, image_group, annotations_group):
        """ Preprocess each image and its annotations in its group.
        """
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # preprocess a single group entry
            image, annotations = self.preprocess_group_entry(image, annotations)

            # copy processed data back to group
            image_group[index]       = image
            annotations_group[index] = annotations

        return image_group, annotations_group



    def compute_input_output(self, group):
        """ Compute inputs and target outputs for the network.
        """
        # load images and annotations
        image_group       = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # perform preprocessing steps
        image_group, annotations_group = self.preprocess_group(image_group, annotations_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group)
        mask = np.zeros_like(inputs)
        for image_index, image in enumerate(image_group):
            for annot in annotations_group[image_index]:
                if annot[-1]>=0:
                    x1,x2,y1,y2 = int(annot[0]),int(annot[2]),int(annot[1]),int(annot[3])
                    mask[image_index,y1:y2,x1:x2]=1
        return inputs,mask
        # compute network targets
        targets = self.compute_targets(image_group, annotations_group)
        targets[1] = targets[1][:,:,(0,3)]

        return inputs, targets