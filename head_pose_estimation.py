import cv2

import numpy as np
import math
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks
import keras
from skimage.transform import resize
from time import time
# from keras import backend as K
# import os
# import csv

def get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val):
    """Return the 3D points present as 2D for making annotation box"""
    point_3d = []
    dist_coeffs = np.zeros((4,1))
    rear_size = val[0]
    rear_depth = val[1]
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))
    
    front_size = val[2]
    front_depth = val[3]
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype="float").reshape(-1, 3)
    
    # Map to 2d img points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    return point_2d
def predict(X_new):
    X_reshaped = X_new.reshape(-1, 1, 2)

    # Resize X to (num_samples, 224, 224, 3)
    X_resized = np.zeros((X_reshaped.shape[0], 224, 224, 3))
    for i in range(X_reshaped.shape[0]):
        X_resized[i] = resize(X_reshaped[i], (224, 224, 3), mode='constant')

    # Normalize X_resized to the range [0, 1]
    X_normalized = X_resized / X_resized.max()

    result = modelku(X_normalized) 
    threshold = 0.5
    result = (result > threshold) 
    if result:
        return "cheat"
    else:
        return "non-cheat"
    
# =============================================================================
# def draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix,
#                         rear_size=300, rear_depth=0, front_size=500, front_depth=400,
#                         color=(255, 255, 0), line_width=2):
#     rear_size = 1
#     rear_depth = 0
#     front_size = img.shape[1]
#     front_depth = front_size*2
#     val = [rear_size, rear_depth, front_size, front_depth]
#     point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
#     # # Draw all the lines
#     cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
#     cv2.line(img, tuple(point_2d[1]), tuple(
#         point_2d[6]), color, line_width, cv2.LINE_AA)
#     cv2.line(img, tuple(point_2d[2]), tuple(
#         point_2d[7]), color, line_width, cv2.LINE_AA)
#     cv2.line(img, tuple(point_2d[3]), tuple(
#         point_2d[8]), color, line_width, cv2.LINE_AA)
# =============================================================================
    
    
def head_pose_points(img, rotation_vector, translation_vector, camera_matrix):
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size*2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    y = (point_2d[5] + point_2d[8])//2
    x = point_2d[2]
    
    return (x, y)
 
face_model = get_face_detector()
landmark_model = get_landmark_model()
cap = cv2.VideoCapture(0)
ret, img = cap.read()
previous = time()
delta = 0
size = img.shape
font = cv2.FONT_HERSHEY_SIMPLEX 
# 3D model points.
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])

# Camera internals
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
coor1 = []
coor2=[]
coorType=[]
current = time()
coordinates = []
time_hold = 2
def save_image(image):
    try:
        image_path = "static\im"+"_"+"_image.jpg"
        cv2.imwrite(image_path, image)
        print("Image saved successfully")
        return str("static\im"+"_"+"_image.jpg")
    except Exception as e:
        print("Error saving image:", str(e))
        return None
while True:

    ret, img = cap.read()
    if ret == True:
        

        faces = find_faces(img, face_model)
        for face in faces:
            marks = detect_marks(img, landmark_model, face)
            # mark_detector.draw_marks(img, marks, color=(0, 255, 0))
            image_points = np.array([
                                    marks[30],     # Nose tip
                                    marks[8],     # Chin
                                    marks[36],     # Left eye left corner
                                    marks[45],     # Right eye right corne
                                    marks[48],     # Left Mouth corner
                                    marks[54]      # Right mouth corner
                                ], dtype="double")
            dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
            
            
            # Project a 3D point (0, 0, 1000.0) onto the image plane.
            # We use this to draw a line sticking out of the nose
            
            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            
            for p in image_points:
                cv2.circle(img, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
            
            
            p1 = ( int(image_points[0][0]), int(image_points[0][1]))
            p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            x1, x2 = head_pose_points(img, rotation_vector, translation_vector, camera_matrix)

            cv2.line(img, p1, p2, (0, 255, 255), 2)
            cv2.line(img, tuple(x1), tuple(x2), (255, 255, 0), 2)
            current = time() 
            delta += current - previous
            previous = current
            print(delta)
     
            # Check if 3 (or some other value) seconds passed
     
            # for (x, y) in marks:
            #     cv2.circle(img, (x, y), 4, (255, 255, 0), -1)
            # cv2.putText(img, str(p1), p1, font, 1, (0, 255, 255), 1)
            try:
                m = (p2[1] - p1[1])/(p2[0] - p1[0])
                ang1 = int(math.degrees(math.atan(m)))
            except:
                ang1 = 90
                
            try:
                m = (x2[1] - x1[1])/(x2[0] - x1[0])
                ang2 = int(math.degrees(math.atan(-1/m)))
            except:
                ang2 = 90
                result = []
                # print('div by zero error')
            modelku = keras.models.load_model('s160419048.h5',compile=False)
            if delta > time_hold:
                X_new = np.array([[ang1,ang2]])
                result = predict(X_new)
                row=[] 
                row.append(ang1)
                row.append(ang2)
                row.append(result)
                coordinates.append(row)
                delta=0
# =============================================================================
#             if ang1 >= 48:
#                 if delta > time_hold:
#                     X_new = np.array([[ang1,ang2]])
#                     result = predict(X_new)
#                     row=[] 
#                     row.append(ang1)
#                     row.append(ang2)
#                     row.append(result)
#                     coordinates.append(row)
#                     delta=0
#             elif ang1 <= -48:
#            
#                 if delta > time_hold:
#                     X_new = np.array([[ang1,ang2]])
#                     result = predict(X_new)
#                     row=[] 
#                     row.append(ang1)
#                     row.append(ang2)
#                     row.append(result)
#                     coordinates.append(row)
#                     delta=0
#             if ang2 >= 48:
#                 
#                 if delta > time_hold:
#                     X_new = np.array([[ang1,ang2]])
#                     result = predict(X_new)
#                     row=[] 
#                     row.append(ang1)
#                     row.append(ang2)
#                     row.append(result)
#                     coordinates.append(row)
#                     delta=0
#                     
#             elif ang2 <= -48:
#                 if delta > time_hold:
#                     X_new = np.array([[ang1,ang2]])
#                     result = predict(X_new)
#                     row=[] 
#                     row.append(ang1)
#                     row.append(ang2)
#                     row.append(result)
#                     coordinates.append(row)
#                     delta=0
# =============================================================================
            # cv2.putText(img, str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
            # cv2.putText(img, str(ang2), tuple(x1), font, 2, (255, 255, 128), 3)
        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
# =============================================================================
#             filename = os.path.join(os.getcwd(), "result.csv")
#             coordinates = []
#             for i in range(len(coor1)):
#                 row = []
#                 row.append(coor1[i])
#                 row.append(coor2[i])
#                 row.append(coorType[i])
#                 coordinates.append(row)
# 
#             with open(filename, 'w', newline='') as csvfile:
#                 csvwriter = csv.writer(csvfile)
#                 csvwriter.writerows(coordinates)
# =============================================================================
            print(coordinates)
            break
# =============================================================================
#         elif cv2.waitKey(1) & 0xFF == ord('s'):
#                 # Save values if 's' key is pressed
#                 value1 = ang1
#                 value2 = ang2
#                 coor1.append(ang1)
#                 coor2.append(ang2)
#                 coorType.append("cheat")
#         elif cv2.waitKey(1) & 0xFF == ord('a'):
#                 # Save values if 's' key is pressed
#                 value1 = ang1
#                 value2 = ang2
#                 coor1.append(ang1)
#                 coor2.append(ang2)
#                 coorType.append("non-cheat")
# =============================================================================
         
    else:
        break
cv2.destroyAllWindows()
cap.release()