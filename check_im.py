import numpy as np
import cv2
import tensorflow as tf
import psycopg2
import math
import operator
tf.logging.set_verbosity(tf.logging.ERROR)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras.models import load_model
cap = cv2.VideoCapture(0)
model = load_model('landmark_model.h5')
conn = psycopg2.connect(database="project", user = "postgres", password = "root", host = "127.0.0.1", port = "5432")
print("Opened database successfully")
def cosine_similarity(v1,v2):
	"compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
	sumxx, sumxy, sumyy = 0, 0, 0
	for i in range(len(v1)):
		x = v1[i]; y = v2[i]
		sumxx += x*x
		sumyy += y*y
		sumxy += x*y
	return sumxy/math.sqrt(sumxx*sumyy)
cur = conn.cursor()
rol =[16010107,16010108,16010112,16010104,16010114, 16010116,16010223,16010103,16010101,16010111,16010124, 16010121,16010105,16010106,16010122,16010125,16010118,16010117, 16010120,16010113,16010119,16010102,16010115]
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while(True):
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		fr=np.zeros((1,96,96,1))
		cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_gray = cv2.resize(roi_gray,(96,96))
		fr[0,:,:,0]=roi_gray[:,:]/255.0
		cv2.imwrite("test.png",roi_gray)
		pre = model.predict(fr)[0]
		dist = []
		i = 0
		pts = []
		while i < len(pre):
			pts.append([pre[i],pre[i+1]])
			i+=2
		for item in pts:
			j = 0
			while j <len(pts):
				dist.append(math.hypot(item[0]-pts[j][0], item[1]-pts[j][1]))
				j+=1
	
		sim = {}
		for i in range(0,len(rol)):
			cur.execute("SELECT * FROM persons where roll ="+str(rol[i]))
			data = cur.fetchone()[0:16]
			q = []
			for x in data:
				q.append(x)
			#print q
			sim[rol[i]]=cosine_similarity(q,dist)
			#print "Operation done successfully";
		sorted_x = sorted(sim.items(), key=operator.itemgetter(1))
		print sorted_x[18:23]
    # Display the resulting frame
	cv2.imshow('frame',gray)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
conn.commit()
conn.close()
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
