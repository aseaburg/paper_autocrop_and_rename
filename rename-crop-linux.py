import os 
import shutil
import cv2
from inference import get_model
import supervision as sv

#use this if you want to input at runtime
#src = input(str("enter the source folder: "))
#dest = input(str("enter the destination folder: "))
#prefix = input(str("enter the prefix for each renamed file: "))

#use this if you'd rather just manually do it
src = "11th"
dest = "junior"
prefix = "junior"

# copy all files in src folder to destination folder
destination = shutil.copytree(src,dest)
# define path to destination folder
destination_path = os.path.abspath(destination)

count = 1
for file in os.listdir(destination):
    # define the path of old image and new image
    old = destination_path + "/" + file
    new = destination_path + "/" + prefix + '_' + str(count) + ".jpg"
    # rename based on new parameters
    os.rename(old, new)
    count += 1

# load trained model from roboflow for paper detection
model = get_model(model_id="paper-detection-cringe/3", api_key="NYx88sSrQuVutibtPcoK")

for file in os.listdir(destination):
    # get full path
    file = destination_path + "/" + file
    # open the image from file
    image = cv2.imread(file)
    # gets the frame and runs the model on it
    results = model.infer(image)[0]
    # gets coordinates of detection
    detections = sv.Detections.from_inference(results)
    try:
        # crop the image using the coordinates 
        cropped_image = sv.crop_image(image=image, xyxy=detections.xyxy[0])
    except:
        # prints message if crop fails
        print(file, "may need manual cropping")
    # overwrite the image file with the new image
    cv2.imwrite(file, cropped_image)



