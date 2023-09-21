
import datetime
import os
from flask import Flask, render_template, request, Response
import sys
import cv2
import numpy as np
import math
from face_detector import find_faces
from face_landmarks import get_landmark_model, detect_marks
import keras
from skimage.transform import resize
from time import time
import mysql.connector
import base64
from flask_socketio import SocketIO, emit
import asyncio
from PIL import Image
import io
app = Flask(__name__,template_folder='templates')
app.config['CACHE_TYPE'] = 'null'
modelku = keras.models.load_model('s160419048.h5',compile=False)
if __name__=="__main__":
    app.run(debug=True)
app.secret_key = 'your_secret_key'
sys.setrecursionlimit(3000)  # Set a higher value than the default
app.config['CACHE_TYPE'] = 'null'

id= 0
name =''
socketio = SocketIO(app)
loop = asyncio.get_event_loop()



def base64_to_image(base64_string):
    idx = base64_string.find('base64,')
    base64_string  = base64_string[idx+7:]

    sbuf = io.BytesIO()

    sbuf.write(base64.b64decode(base64_string, ' /'))
    pimg = Image.open(sbuf)


    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

 # Replace with a strong secret key
@app.route('/')
def home():
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        connection = mysql.connector.connect(
                      host='localhost',
                        user='root',
                        database='skripsi'
                    )
        cursor = connection.cursor()
        
        # Query the database for the given username and password
        query = "SELECT * FROM users WHERE email = %s AND password = %s"
        cursor.execute(query, (username, password))
        user = cursor.fetchone()
        # Close the database connection
        if user:
            name = user[1]
            id= user[0]
            if name == 'admin':
                cursor = connection.cursor()
                query = "SELECT user_id, name, timestamp, status, image,ang1,ang2 FROM detects WHERE status = 1 ORDER BY ang2 ASC;"
                cursor.execute(query)
                records = cursor.fetchall()
                connection.commit()
                cursor.close()
                connection.close()
                processed_records = []
                for record in records:
                    user_id = record[0]
                    user_name=record[1]
                    timestamp = record[2]
                    status = record[3]
                    image_path = record[4]
                    angle1= record[5]
                    angle2= record[6]
                    image_path_str = image_path # Convert bytearray to string
                    image_filename = os.path.basename(image_path_str) 
                    
                    processed_record = {
                        'user_id': user_id,
                        'name':user_name,
                        'timestamp': timestamp,
                        'status': status,
                        'ang1':angle1,
                        'ang2':angle2,
                        'image_path': "/static/"+image_filename,
                    }
                    
                    processed_records.append(processed_record)
                return render_template('admin.html', records=processed_records)
            connection.commit()
            cursor.close()
            connection.close()
            return render_template('access.html')
        
        else:
            connection.commit()
            cursor.close()
            connection.close()
            return render_template('login.html', error='Invalid credentials')
       
    return render_template('login.html', error='')
    

@socketio.on("image")
@app.route('/gen_frame')
def gen_frame(image):
    img = base64_to_image(image)
    size = img.shape
    landmark_model = get_landmark_model()
    model_points = np.array([
                                (0.0, 0.0, 0.0),            
                                (0.0, -330.0, -65.0),       
                                (-225.0, 170.0, -135.0),     
                                (225.0, 170.0, -135.0),     
                                (-150.0, -150.0, -125.0),    
                                (150.0, -150.0, -125.0)      
                            ])
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )
        
        #print(1)
    faces = find_faces(img)
    #print(2)
    for face in faces:
        marks = detect_marks(img, landmark_model, face)
        image_points = np.array([
                                marks[30],     
                                marks[8],     
                                marks[36],     
                                marks[45],    
                                marks[48],     
                                marks[54]
                            ], dtype="double")
        dist_coeffs = np.zeros((4,1)) 
        #print(3)
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)

        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        p1 = ( int(image_points[0][0]), int(image_points[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        x1, x2 = head_pose_points(img, rotation_vector, translation_vector, camera_matrix)
        # print(4)
    
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


        result = 0

        X_new = np.array([[ang1,ang2]])
        result = predict(X_new)
        print(result)
        if(result >= 0.65):
            try:
                connection2 = mysql.connector.connect(
                    host='localhost',
                    user='root',
                    database='skripsi'
                )
                cursor2 = connection2.cursor()

                timestamp = datetime.datetime.now()    
                path=save_image(img,id,timestamp)
            
                query = "INSERT INTO `detects` (`user_id`, `name`,`test_id`, `timestamp`, `status`, `image`,`ang1`,`ang2`)  VALUES (%s, %s, %s, %s, %s, %s,%s,%s)"
# Provide values for all six placeholders
                values = (id, name, 1, timestamp, 1, path,ang1,ang2)
                cursor2.execute(query, values)

                connection2.commit()
                cursor2.close()
                connection2.close()


                
            except Exception as e:
                print("Error saving image:", str(e))

                connection2.commit()
                cursor2.close()
                connection2.close()

        img = None
        ang1 = None
        ang2 = None
        result = None
        processed_image_data = img # Replace this with actual processing logic
        socketio.emit('processed_image', processed_image_data, namespace='/image')

def save_image(image, id, timestamp):
    id_str = str(id)
    timestamp_str = str(timestamp)
    id_str = id_str.replace(" ", "_")
    timestamp_str = timestamp_str.replace(":", "_")
    try:
        image_path = "static\im"+id_str+"_"+timestamp_str+"_image.jpg"
        cv2.imwrite(image_path, image)
        print("Image saved successfully")
        return str("static\im"+id_str+"_"+timestamp_str+"_image.jpg")
    except Exception as e:
        print("Error saving image:", str(e))
        return None

def predict(X_new):
    result = modelku.predict(X_new) 
    return result[0][0]
        
def get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val):
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
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    return point_2d

def head_pose_points(img, rotation_vector, translation_vector, camera_matrix):
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size*2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    x2 = (point_2d[5] + point_2d[8])//2
    x1 = point_2d[2]
    return (x1, x2)
 
if __name__ == "__main__":
    socketio.run(app, debug=True, port=5000, host='0.0.0.0')