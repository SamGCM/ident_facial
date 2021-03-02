import face_recognition
import argparse
import pickle
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", default='nutri_faces',
	help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("[INFO] recognizing faces...")
boxes = face_recognition.face_locations(rgb,
	model=args["detection_method"])
encodings = face_recognition.face_encodings(rgb, boxes)

names = []

for encoding in encodings:

	matches = face_recognition.compare_faces(data["encodings"],
		encoding)
	name = "Unknown"


if True in matches:

    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
    counts = {'Jaine': 30, 'Lais': 20, 'Thayz': 20 }

    for i in matchedIdxs:
        name = data["names"][i]
        counts[name] = counts.get(name, 0) + 1

    name = max(counts, key=counts.get)


names.append(name)



# loop over the recognized faces
for ((top, right, bottom, left), name) in zip(boxes, names):
	# draw the predicted face name on the image
	cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
	y = top - 15 if top - 15 > 15 else top + 15
	cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
		0.75, (0, 255, 0), 2)
# show the output image
cv2.imshow("Quadro", image)
cv2.waitKey(0)











