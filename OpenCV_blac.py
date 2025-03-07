import cv2
from cvzone.SelfiSegmentationModule import SelfiSegmentation

# connecting the internal camera (first camera index will be 0, it is the default)
cap = cv2.VideoCapture(0)

# extracting the camera capture size
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

# loading and resizing the background image
# background_image = cv2.resize(cv2.imread("rio.jpg"), (width, height))
background_image = cv2.imread("bianco.jpg")

if background_image is None:
    print("Wrong path:")
else:
    background_image = cv2.resize(background_image, (width, height))

# creating segmentation instance for taking the foreground (the person).
segmentor = SelfiSegmentation()

# iterating the camera captures
while True:
    # Reading the captured images from the camera
    ret, frame = cap.read()

    # segmenting the image
    segmentated_img = segmentor.removeBG(frame, background_image, 0.9)

    # concatenating the images horizontally - makes the two frames next to eachother
    concatenated_img = cv2.hconcat([frame, segmentated_img])

    # cv2.imshow("Camera Capture", concatenated_img)
    cv2.imshow("Camera Live", concatenated_img)

    # ending condition
    if cv2.waitKey(1) == ord("q"):
        break

# relasing the sources
cap.release()
cv2.destroyAllWindows()
