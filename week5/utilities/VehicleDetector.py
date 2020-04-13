import os
import numpy as np
import tensorflow as tf
import cv2

from week5.utilities import LabelMap
from week5.utilities.BoundingBox import draw_box_label


class VehicleDetector:
    # Method: Constructor
    def __init__(self, kitti=False, min_conf=0.7, info=False):
        """
        :param kitti: If True, use the Kitti model
        :param min_conf: Minimum acceptable confidence level
        :param info: If True, display all visualisations
        """
        # Change to current working directory
        os.chdir(os.getcwd())

        self.bounding_boxes = []
        self.min_conf = min_conf
        self.info = info
        self.kitti = kitti
        # path_to_model = 'D:/Master/m6/fin/mcv-m6-2020-team3/models/research/object_detection/data/frozen_model.pb'
        path_to_model = 'D:/Master/m6/fin/mcv-m6-2020-team3/models/research/object_detection/data/frozen_inference_graph_faster_rcnn_inception_v2.pb'
        #path_to_model = 'D:/Master/m6/fin/mcv-m6-2020-team3/models/research/object_detection/data/frozen_inference_graph_ssd_mobilenet_v2.pb'



        if self.kitti:
            path_to_label_map = 'D:/Master/m6/fin/mcv-m6-2020-team3/models/research/object_detection/data/kitti_label_map.pbtxt' #CAR=1
            num_classes = 2
        else:
            path_to_label_map = 'D:/Master/m6/fin/mcv-m6-2020-team3/models/research/object_detection/data/mscoco_label_map.pbtxt' #CAR=3, BUS=6 AND TRUNCK=8
            num_classes = 7

        # Define TensorFlow graph
        self.detection_graph = tf.Graph()

        # Load model and initialize the TensorFlow graph
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            # Setup session and image tensor
            self.session = tf.Session(graph=self.detection_graph)
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected
            self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represents the classification confidence for each of the objects
            self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        # Load label map and convert into categories
        loaded_label_map = LabelMap.load_label_map(path_to_label_map)
        categories = LabelMap.convert_label_map_to_categories(label_map=loaded_label_map,
                                                              max_num_classes=num_classes,
                                                              use_display_name=True)

        # Assign an index to each category
        self.category_index = LabelMap.create_category_index(categories)

    # Method: Used to convert image into numpy array
    @staticmethod
    def load_image_into_numpy_array(image):
        """
        :param image: Image
        :return: Image as NumPy array
        """
        (im_width, im_height) = image.size

        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    # Method: Used to convert normalized coordinates to pixel coordinates
    @staticmethod
    def normalized_to_pixel_coordinates(box, dims):
        """
        :param box: Box with normalized coordinates
        :param dims: Image dimensions
        :return: Box with pixel coordinates
        """
        height, width = dims[0], dims[1]
        pixel_coords = [int(box[0] * height), int(box[1] * width), int(box[2] * height), int(box[3] * width)]

        return np.array(pixel_coords)

    # Method: Used to detect the locations of the vehicles in the image
    def get_bounding_box_locations(self, image, frame,pklList, out, x):
        """
        :param image: Image
        :return: Bounding box locations surrounding detected vehicles
        """
        with self.detection_graph.as_default():
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_expanded = np.expand_dims(image, axis=0)

            # Actual detection
            (boxes, scores, classes, num_detections) = self.session.run([self.boxes,
                                                                         self.scores,
                                                                         self.classes,
                                                                         self.num_detections],
                                                                        feed_dict={self.image_tensor: image_expanded})

            # Remove 1D entries from the shape of the array
            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes)
            scores = np.squeeze(scores)

            # Convert to a list
            classes_list = classes.tolist()

            # Find cars detected in the image
            if self.kitti:
                index_vector = [i for i, id in enumerate(classes_list) if ((id == 1) and (scores[i] > self.min_conf))]
            else:
                index_vector = [i for i, id in enumerate(classes_list) if ((id == 3) and (scores[i] > self.min_conf))]

            if len(index_vector) > 0:
                temp_boxes = []
                temp_score = []

                for index in index_vector:
                    # Get image dimensions
                    dims = image.shape[0:2]

                    # Convert normalized coordinates to pixel coordinates
                    box = self.normalized_to_pixel_coordinates(boxes[index], dims)

                    # Calculate height, width and ratio to filter out boxes that not the right shape or size
                    box_h = box[2] - box[0]
                    box_w = box[3] - box[1]
                    ratio = box_h / box_w

                    # Filter out boxes that are not the right shape or size
                    # if ratio < 0.8 and box_h > 20 and box_w > 20:
                    # if box_h > 20 and box_w > 20:
                    #if ratio < 0.8:
                    if True:
                        temp_boxes.append(box)
                        #temp_boxes.append(scores[index])
                        detection = {}
                        detection['frame'] = frame + 1
                        detection['left'] = box[1]
                        detection['top'] = box[0]
                        detection['width'] = box_w
                        detection['height'] = box_h
                        detection['confidence'] = scores[index]
                        pklList.append(detection)
                        print(detection)

                        image2 = draw_box_label(image, box)
                        cv2.imwrite(out  + '/frame_' + '%05d.jpg' % x, image2)

                        print('[INFO]: Vehicle Detected at {} with {:.2f}% confidence'.format(box,scores[index]*100.0 ))

                if self.info:
                    cv2.imshow("detections", image2)
                    cv2.waitKey(300)
                self.bounding_boxes = temp_boxes
                # self.actualscore  = temp_score

        return self.bounding_boxes
