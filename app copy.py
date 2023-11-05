import numpy as np
import face_recognition
import cv2
from flask import Flask, request
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})


UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

detected_faces_per_frame = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_and_encode_images(image_filenames):
    encodings = []
    for filename in image_filenames:
        image = cv2.imread(filename)
        print("image", image)
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encodings.append(face_recognition.face_encodings(image_rgb)[0])
    return encodings

# Set up the VideoWriter object
def blur_matching_faces(image, net, picture_encodings, threshold=0.5, match_tolerance=0.5):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    windows = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            try:
                face_rgb = cv2.cvtColor(image[startY:endY, startX:endX], cv2.COLOR_BGR2RGB)
            except:
                continue
            face_encodings = face_recognition.face_encodings(face_rgb)

            if len(face_encodings) > 0:
                matches = face_recognition.compare_faces(picture_encodings, face_encodings[0], match_tolerance)
                if any(matches):
                    windows.append((startX, startY, endX, endY))
                    face = image[startY:endY, startX:endX]
                    blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
                    image[startY:endY, startX:endX] = blurred_face
                    detected_faces_per_frame.append(len(windows))
    return image, windows

@app.route('/BlurFaces', methods=['POST'])
def Blur_Faces():
   
    if 'video' not in request.files or 'photos' not in request.files:
        return "No files received. Please upload a video and photos.", 400

    video_file = request.files['video']

    photo_files = request.files.getlist('photos')

    # Check if the uploaded files have allowed extensions
    if not allowed_file(video_file.filename) or not all(allowed_file(f.filename) for f in photo_files):
        return "Invalid file type. Allowed extensions are jpg, jpeg, png, and mp4.", 400
    
    # Save the uploaded video and photos
    video_filename = secure_filename(video_file.filename)
    video_file.save(os.path.join(app.config['UPLOAD_FOLDER'], video_filename))

    photo_filenames = []
    for photo in photo_files:
        photo_filename = secure_filename(photo.filename)
        photo.save(os.path.join(app.config['UPLOAD_FOLDER'], photo_filename))
        photo_filenames.append(os.path.join(app.config['UPLOAD_FOLDER'], photo_filename))

    # Load the DNN model for face detection
    model_file = "res10_300x300_ssd_iter_140000.caffemodel"
    config_file = "deploy.prototxt.txt"
    net = cv2.dnn.readNetFromCaffe(config_file, model_file)

    # Load the video and reference pictures
    print('Video file name : ', video_filename)
    video = cv2.VideoCapture(os.path.join(app.config['UPLOAD_FOLDER'], video_filename))
    picture_encodings = load_and_encode_images(photo_filenames)

    # Get the video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # Set up the VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") # mp4v for mp4
    output = cv2.VideoWriter("output_video.mp4", fourcc, fps, (width, height)) # (width, height) for mp4

    print('ENTERING WHILE LOOP')

    try:
        i = 0
        while i < 1000:
            ret, frame = video.read()
            #print(frame, ret)
            if not ret:
                break
            if i % 10 == 0:
                blurred_frame= blur_matching_faces(frame, net, picture_encodings)
           
            output.write(blurred_frame)
            # yield the frame as a response
            # frameee = cv2.imencode('.jpg', blurred_frame)[1].tobytes()
            # yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frameee + b'\r\n')
            i += 1
            print('i : ', i)
           
            # If the user presses the `q` key, stop the loop
            if cv2.waitKey(1) == ord("q"):
              break
    finally:
        # Release the video capture object and the VideoWriter object
        video.release()
        output.release()
        print('EXITED WHILE LOOP')

        plt.plot(detected_faces_per_frame)
        plt.xlabel("Frame Number")
        plt.ylabel("Number of Detected Faces")
        plt.title("Accuracy of Face Detection")
        plt.show()

        # Close all windows
        cv2.destroyAllWindows()
        return "Faces have been blurred successfully.", 200 

if __name__ == '__main__':
    app.run(debug=True)

