import numpy as np
import cv2
import os
import tensorflow as tf
import pandas as pd
import math
tf.logging.set_verbosity(tf.logging.ERROR)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras.models import load_model
filename = os.listdir("list")
model = load_model('landmark_model.h5')
for f in filename:
	print f
	img = cv2.imread("list/"+f,0)
	fr=np.zeros((1,96,96,1))
	gray = cv2.resize(img,(96,96))
	fr[0,:,:,0]=gray[:,:]/255.0
	pre = model.predict(fr)[0]
	for i in range(0,len(pre),2):
		cv2.circle(fr[0],(int(pre[i]*96),int(pre[i+1]*96)), 4, (0,0,255), -1)
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
	dist.append(f.split(".")[0])
	dis = [dist]
	my_df = pd.DataFrame(dis)
	my_df.to_csv('data.csv', mode='a',index=False, header=False)
	cv2.imshow('img',fr[0])
	cv2.waitKey(0)
cv2.destroyAllWindows()
