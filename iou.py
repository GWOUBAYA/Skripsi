import cv2
import os
import numpy as np

def get_face_detector(
                      ):
    modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "models/deploy.prototxt"
    model = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return model

def find_faces(img, model):

    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    res = model.forward()
    faces = []
    for i in range(res.shape[2]):
        confidence = res[0, 0, i, 2]
        if confidence > 0.5:
            box = res[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            faces.append([x, y, x1, y1])
    return faces


def elliptical_to_bbox(ellipse):
    major_axis_radius, minor_axis_radius, angle, center_x, center_y , _ = ellipse
    
    # Convert angle to radians
    angle_rad = np.radians(angle)
    
    # Calculate the coordinates of the four corners of the rotated rectangle
    x1 = center_x + major_axis_radius * np.cos(angle_rad)
    y1 = center_y + major_axis_radius * np.sin(angle_rad)
    x2 = center_x - major_axis_radius * np.cos(angle_rad)
    y2 = center_y - major_axis_radius * np.sin(angle_rad)
    x3 = center_x + minor_axis_radius * np.cos(np.pi/2 + angle_rad)
    y3 = center_y + minor_axis_radius * np.sin(np.pi/2 + angle_rad)
    x4 = center_x - minor_axis_radius * np.cos(np.pi/2 + angle_rad)
    y4 = center_y - minor_axis_radius * np.sin(np.pi/2 + angle_rad)
    
    # Calculate the bounding box coordinates
    bbox_x = min(x1, x2, x3, x4)
    bbox_y = min(y1, y2, y3, y4)
    bbox_width = max(x1, x2, x3, x4) - bbox_x
    bbox_height = max(y1, y2, y3, y4) - bbox_y
    
    return [bbox_x, bbox_y, bbox_width, bbox_height]


# Path to the FDDB dataset images
images_dir = r'originalPics'

# Path to the FDDB dataset annotation files
annotations_dir = r'FDDB-folds'

def load_annotations(annotation_path):
    annotations = []
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            image_path = lines[i].strip()
            i+=1
            num_faces = int(lines[i].strip())
            faces = []
            for _ in range(num_faces):
                i += 1  # Move to the next line for face annotation
                face_annotation = lines[i].strip().split()
                faces.append(list(map(float, face_annotation)))
            annotations.append((image_path, faces))
            i += 1  # Move to the next image annotation after processing faces
    return annotations

def calculate_iou(pred, ground_truth):
    # Convert to rectangular bounding boxes
    ground_truth = elliptical_to_bbox(ground_truth)

    # Convert coordinates and dimensions to integers
    pred = list(map(int, pred))
    ground_truth = list(map(int, ground_truth))
    
    # Coordinates of the area of intersection.
    ix1 = np.maximum(ground_truth[0], pred[0])
    iy1 = np.maximum(ground_truth[1], pred[1])
    ix2 = np.minimum(ground_truth[2], pred[2])
    iy2 = np.minimum(ground_truth[3], pred[3])
     
    # Intersection height and width.
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0))
     
    area_of_intersection = i_height * i_width
     
    # Ground Truth dimensions.
    gt_height = ground_truth[3] - ground_truth[1] + 1
    gt_width = ground_truth[2] - ground_truth[0] + 1
     
    # Prediction dimensions.
    pd_height = pred[3] - pred[1] + 1
    pd_width = pred[2] - pred[0] + 1
     
    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
     
    iou = area_of_intersection / area_of_union
     
    return iou



def perform_iou_testing(annotations, detection_results):
    total_annotations = len(annotations)
    true_positives = 0
    false_positives = 0
    
    for image_path, image_annotations in annotations:
        try:
            image_path = os.path.join('originalPics', image_path + '.jpg')
        except FileNotFoundError:
            print(f"Image not found: {image_path}")
            continue  # Skip to the next image
        
        if image_path in detection_results:
            detected_bboxes = detection_results[image_path]
            for bbox_detected in detected_bboxes:
                for annotation_bbox in image_annotations:
                    iou = calculate_iou(bbox_detected, annotation_bbox)
                    if iou <= 0.5:
                        true_positives += 1
                        break
                else:
                    false_positives += 1
    
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / total_annotations
    
    return precision, recall, iou



# Load file paths from FDDB-fold-01.txt
def load_file_paths(file_paths_path):
    file_paths = []
    with open(file_paths_path, 'r') as f:
        for line in f:
            file_paths.append(line.strip())
    return file_paths

# Load FDDB file paths
file_paths = load_file_paths('FDDB-folds\FDDB-fold-01.txt')

# Initialize face detection model
face_detection_model = get_face_detector()

# Dictionary to store detection results
detection_results = {}



# Perform face detection using the model on FDDB images
for file_path in file_paths:
    image_path = os.path.join('originalPics', file_path + '.jpg')
    print(f"Loading image: {image_path}")
    
    # Debugging: Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        continue  # Skip to the next image
    
    detected_faces = find_faces(image, face_detection_model)
    detection_results[image_path] = detected_faces




# Perform IOU testing
annotations = load_annotations(os.path.join(annotations_dir, 'FDDB-fold-01-ellipseList.txt'))
precision, recall, iou = perform_iou_testing(annotations, detection_results)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")



