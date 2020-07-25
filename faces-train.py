import os
import cv2
import numpy
import hashlib


class Recognizer:
	def __init__(self):
		self.inputs = []
		self.labels = []
		self.recognizer = cv2.face.LBPHFaceRecognizer_create()
		self.face_cascade = cv2.CascadeClassifier(
			cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
		)
		self.image_dir = os.path.join(os.getcwd(), 'images')

	def uid(self, string):
		return int(str(int(hashlib.md5(string.encode('utf-8')).hexdigest(),
					16))[0:8])

	def make_model(self, ):
		for name in os.listdir(self.image_dir):
			print(name)
			path_ = os.path.join(self.image_dir, name)
			if not os.path.isdir(path_):
				continue
			for photo in os.listdir(path_):
				image = cv2.imread(os.path.join(path_, photo), cv2.IMREAD_COLOR)
				gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				face_rects = self.face_cascade.detectMultiScale(gray, 1.5, 5)

				for (x, y, w, h) in face_rects:
					face = gray[y: y + h, x: x + w]

				self.inputs.append(cv2.resize(face, (550, 550)))
				self.labels.append(self.uid(name))
		self.recognizer.train(self.inputs, numpy.array(self.labels))
		self.recognizer.save("recognizers/LBPH.yml")


if __name__ == '__main__':
	train = Recognizer()
	train.make_model()
