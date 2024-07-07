import os 
import shutil
import cv2
from inference import get_model
import supervision as sv

# source file, destination folder, and prefix for each renamed image
src = "12"
dest = "senior"
prefix = "senior"

# copy the entire directory from src to dest
destination = shutil.copytree(src,dest)
# get the complete path of the destination folder
destination_path = os.path.abspath(destination)

count = 1
for file in sorted(os.listdir(destination)):
    # original file path
    old = destination_path + '/' + file
    # new file path
    new = destination_path + '/' + prefix + '_' + str(count) + '.jpg'
    # rename 
    os.rename(old,new)
    #iterate
    count+=1

# load trained model from roboflow for paper detection
model = get_model(model_id="paper-detection-cringe/3", api_key="NYx88sSrQuVutibtPcoK")

for file in os.listdir(destination):
    # get full path
    file = destination_path + "/" + file
    try:
        # open the image from file
        image = cv2.imread(file)
        # gets the frame and runs the model on it
        results = model.infer(image)[0]
        # gets coordinates of detection
        detections = sv.Detections.from_inference(results)
        # crop the image using the coordinates 
        cropped_image = sv.crop_image(image=image, xyxy=detections.xyxy[0])
        # overwrite the image file with the new image
        cv2.imwrite(file, cropped_image)
        # status
        print(file, "cropped")
    except:
        # prints message if crop fails
        print(file, "may need manual cropping")
    
        
    



