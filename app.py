from flask import Flask, render_template, request, Response
import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
import tensorflow as tf


new_model = tf.keras.models.load_model('saved_model/my_model')
detector = MTCNN()

def gen_frames():  
	cap = cv2.VideoCapture(0)


	while True:
		__, frame = cap.read()
		result = detector.detect_faces(frame)
		if result != []:
			for person in result:
				bounding_box = person['box']
				keypoints = person['keypoints']
				cv2.rectangle(frame,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (0,155,255),
                          2)
				cv2.circle(frame,(keypoints['left_eye']), 2, (0,155,255), 2)
				cv2.circle(frame,(keypoints['right_eye']), 2, (0,155,255), 2)
				cv2.circle(frame,(keypoints['nose']), 2, (0,155,255), 2)
				cv2.circle(frame,(keypoints['mouth_left']), 2, (0,155,255), 2)
				cv2.circle(frame,(keypoints['mouth_right']), 2, (0,155,255), 2)
				data=[]
				img_size = 128
				# replace any negative value with zero
				boxes = [face['box'] for face in result if face['confidence']>0.60]
				print(boxes)
				x,y,width,height = [0 if value < 0 else value for value in boxes[0]]
				# read image with green channel
				img_array = cv2.imread('opencv.png', 1)
				# crop image with bounding box
				img_cropped = img_array[y:y+height,x:x+width]
				# resize cropped image
				img = cv2.resize(img_cropped,(img_size,img_size))
				data.append(img)
				X = np.array(data)/255
				predict = new_model.predict(X)
				output = ['face_with_mask' if i > 0.8 else 'face_with_no_mask' for i in predict]
				cv2.putText(frame, output[0], (x, y-
					10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
		ret, buffer = cv2.imencode('.jpg', frame)
		frame = buffer.tobytes()
		if cv2.waitKey(1) &0xFF == ord('q'):
			break

		yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def contact():

    if "Start Streaming" in request.form:
        print('Started Streaming')
        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return render_template('index.html')

if __name__ == '__main__':
    app.run()