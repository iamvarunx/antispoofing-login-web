import os
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

# Ignore warnings
warnings.filterwarnings('ignore')

SAMPLE_IMAGE_PATH = "./images/sample/"


# Function to check if the image's width and height are in the correct aspect ratio (3:4)
def check_image(image):
    height, width, channel = image.shape
    if width / height != 3 / 4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True


def test(image, model_dir, device_id):
    model_test = AntiSpoofPredict(device_id)  # Initialize the anti-spoofing model
    image_cropper = CropImage()  # Initialize the image cropper

    # Resize image to match the expected aspect ratio
    image = cv2.resize(image, (int(image.shape[0] * 3 / 4), image.shape[0]))
    result = check_image(image)
    if not result:
        return

    image_bbox = model_test.get_bbox(image)  # Get the bounding box for the image
    prediction = np.zeros((1, 3))  # Initialize prediction
    test_speed = 0  # Initialize speed tracker

    # Sum the prediction from all models in the model directory
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False

        # Crop the image based on the parameters
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time() - start  # Track the time taken for prediction

    # Get the final result of the prediction
    label = np.argmax(prediction)
    value = prediction[0][label] / 2
    return label


if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="Directory for the model library used to test"
    )
    parser.add_argument(
        "--image_name",
        type=str,
        default="image_F1.jpg",
        help="Image used to test"
    )
    args = parser.parse_args()

    # Call the test function with the provided arguments
    test(args.image_name, args.model_dir, args.device_id)
