from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", default= 'dataset',
	help="coloque o caminho das faces + imagens")
ap.add_argument("-e", "--encodings", default= 'nutri',
	help="caminho do db das faces codificadas")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
	help="escolha um modelo de detecção: `hog` or `cnn`")
args = vars(ap.parse_args())

# grab the paths to the input images in our dataset
print("[INFO] Contando faces...")
imagePaths = list(paths.list_images(args["dataset"]))
# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []


for (i, imagePath) in enumerate(imagePaths):

	print("[INFO] Processando imagens: {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])

	encodings = face_recognition.face_encodings(rgb, boxes)

	for encoding in encodings:
		knownEncodings.append(encoding)
		knownNames.append(name)

print("[INFO] codificando rostos...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()











