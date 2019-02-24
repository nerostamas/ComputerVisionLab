import cv2
from mtcnn.mtcnn import MTCNN
import os

detector = MTCNN()

faceID = 0

print("Start")
for file in os.listdir("FaceDB"):
    if file.endswith("png"):
        print(file)
        image = cv2.imread("FaceDB/" + file)
        results = detector.detect_faces(image)

        for result in results:
            bounding_box = result['box']
            print(bounding_box)
            img_crop = image[bounding_box[1]:bounding_box[1] + bounding_box[3],
                       bounding_box[0]:bounding_box[0] + bounding_box[2]]
            cv2.imwrite("Result/face_" + faceID.__str__() + ".jpg", img_crop)
            faceID += 1
