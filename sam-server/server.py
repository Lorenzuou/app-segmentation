

from flask import Flask, request, jsonify
from flask_cors import CORS  # import the flask_cors module
from PIL import Image
import io
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import cv2
import base64
import supervision as sv
import os 
from pathlib import Path

SESSIONS_FOLDER = 'sessions'


def rmdir(directory):
    directory = Path(directory)
    for item in directory.iterdir():
        if item.is_dir():
            rmdir(item)
        else:
            item.unlink()
    directory.rmdir()


os.makedirs(SESSIONS_FOLDER, exist_ok=True)

# remove any folder from the sessions folder that does not have any image inside the folder masked
for folder in os.listdir(SESSIONS_FOLDER):
    folder_path = os.path.join(SESSIONS_FOLDER, folder)
    masked_folder_path = os.path.join(folder_path, 'masked')
    if len(os.listdir(masked_folder_path)) == 0:
        rmdir(folder_path)

app = Flask(__name__)
CORS(app)  # enable CORS on the app


CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
MODEL_TYPE = "vit_h"

model = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
model.to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(model,
                                           crop_n_layers=1,
                                           crop_n_points_downscale_factor=2
                                           )

                                           
predictor = SamPredictor(model)

current_image = None
current_mask = None


def get_largest_area(result_dict):
    sorted_result = sorted(result_dict, key=(lambda x: x['area']),
                           reverse=True)
    return sorted_result[0]


def apply_mask(image, mask, color=None):
    print(image.shape, mask.shape)
    # Convert the mask to a 3 channel image
    if color is None:
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    else:
        mask_rgb = np.zeros_like(image)
        mask_rgb[mask > 0] = color

    # Overlay the mask and image
    # overlay_image = cv2.addWeighted(image, 0.7, mask_rgb, 0.3, 0)

    return mask_rgb


# def generate_image_of_mask(output_mask, shape):
#     image_rgb = np.zeros((shape[0], shape[1], 3), dtype='uint8')

#     for i in range(len(output_mask)):
#         mask = output_mask[i]['segmentation']
#         mask = np.where(mask, 255, 0).astype('uint8')
#         color = np.random.randint(0, 255, 3)
#         image = apply_mask(image_rgb, mask, color=color)
#         image_rgb = cv2.addWeighted(image_rgb, 1, image, 1, 0)

#     return image


def get_current_image():
    return current_image


def set_current_image(image):
    current_image = image



def generate_image(image):
    # Generate segmentation mask
    output_mask = mask_generator.generate(image)
    # get second largest area
    largest_area = get_largest_area(output_mask)
    mask = largest_area['segmentation']

    return mask


def set_image(image):
    predictor.set_image(image)


def generate_images_with_box(image, box):
    box = np.array(box)

    # exit()
    output_mask, scores, logits = predictor.predict(
                            box=box,
                            multimask_output=True,
                        )
    
    mask_input = output_mask[np.argmax(scores), :, :]  # Choose the model's best mask

    # return generate_image_of_mask(output_mask, image.shape)

    image_rgb = np.zeros((image.shape[0], image.shape[1], 3), dtype='uint8')

    color = (255, 255, 255)
    mask = np.where(mask_input, 255, 0).astype('uint8')
    image = apply_mask(image_rgb, mask, color=color)

    # for i in range(len(output_mask)):
    #     mask = output_mask[i]
    #     mask = np.where(mask, 255, 0).astype('uint8')
    
    #     color = (255, 255, 255)
    #     image = apply_mask(image_rgb, mask, color=color)

        #invert black and white
    # image = cv2.bitwise_not(image)
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return image

def generate_image_with_prompt(image, input_labels, input_points):
    # input_labels = input_labels.split(',') if input_labels else []
    # input_points = input_points.split(',') if input_points else []
    input_points = np.array(input_points)
    # input_labels = np.array(input_labels, dtype=np.float32)
    print(input_points)
    print(input_labels)
    # exit()
    
    output_mask, scores, logits = predictor.predict(
                            point_coords=input_points,
                            point_labels=input_labels,
                            multimask_output=True,
                        )
    
    mask_input = output_mask[np.argmax(scores), :, :]  # Choose the model's best mask


    # return generate_image_of_mask(output_mask, image.shape)

    image_rgb = np.zeros((image.shape[0], image.shape[1], 3), dtype='uint8')

    color = (255, 255, 255)
    mask = np.where(mask_input, 255, 0).astype('uint8')
    image = apply_mask(image_rgb, mask, color=color)

    # for i in range(len(output_mask)):
    #     mask = output_mask[i]
    #     mask = np.where(mask, 255, 0).astype('uint8')
    
    #     color = (255, 255, 255)
    #     image = apply_mask(image_rgb, mask, color=color)

        #invert black and white
    # image = cv2.bitwise_not(image)
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return image


@app.route('/save', methods=['POST'])
def save():
    data = request.json
    if 'sessionIdentifier' not in data:
        return jsonify({'error': 'No sessionIdentifier in request'}), 400
    sessionIdentifier = data['sessionIdentifier']
    sessionFolder = os.path.join(SESSIONS_FOLDER, sessionIdentifier)

    originalFolder = os.path.join(sessionFolder, 'original')
    maskedFolder = os.path.join(sessionFolder, 'masked')

    os.makedirs(originalFolder, exist_ok=True)
    os.makedirs(maskedFolder, exist_ok=True)

    original_image = data['originalImage']
    masked_image = data['maskedImage']
    file_name = data['fileName']

    original_image = Image.open(io.BytesIO(base64.b64decode(original_image)))
    original_cv2 = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)

    masked_image = Image.open(io.BytesIO(base64.b64decode(masked_image)))
    masked_cv2 = cv2.cvtColor(np.array(masked_image), cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(originalFolder, file_name), original_cv2)
    cv2.imwrite(os.path.join(maskedFolder, file_name), masked_cv2)


    return jsonify({'message': 'Saved successfully'})



@app.route('/predict/box', methods=['POST'])
def predict_box():
    data = request.json
    if 'file' not in data:
        return jsonify({'error': 'No file in request'}), 400
    if 'box' not in data:
        return jsonify({'error': 'No box in request'}), 400
    
    sessionIdentifier = data['sessionIdentifier']

    # check if a folder with the sessionIdentifier exists 
    sessionFolder = os.path.join(SESSIONS_FOLDER, sessionIdentifier)
    os.makedirs(sessionFolder, exist_ok=True)

    #create a original and masked folder
    originalFolder = os.path.join(sessionFolder, 'original')
    maskedFolder = os.path.join(sessionFolder, 'masked')
    os.makedirs(originalFolder, exist_ok=True)
    os.makedirs(maskedFolder, exist_ok=True)

    image = Image.open(io.BytesIO(base64.b64decode(data['file'])))
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    image_name = data['fileName']

    cv2.imwrite(os.path.join(originalFolder, image_name), image_cv2)

    box = data['box']
    if (data['again'] is False or predictor.is_image_set is False):
        set_image(image_cv2)
    image_masked = generate_images_with_box(image_cv2, box)

    cv2.imwrite(os.path.join(maskedFolder, image_name), image_masked)

    # turn black to white and white to black 
    # image_masked = cv2.bitwise_not(image_masked)

    image_masked = Image.fromarray(image_masked)

    #send image as response to the client in json format
    image = io.BytesIO()
    image_masked.save(image, format='PNG')
    image.seek(0)
    image_bytes = image.getvalue()
    base64_encoded_result = base64.b64encode(image_bytes).decode()
    return jsonify({'image': base64_encoded_result})

                            

@app.route('/predict/prompt', methods=['POST'])
def predict_prompt():
    data = request.json
    if 'file' not in data:
        return jsonify({'error': 'No file in request'}), 400
    if 'input_labels' not in data:
        return jsonify({'error': 'No input_labels in request'}), 400
    if 'input_points' not in data:
        return jsonify({'error': 'No input_points in request'}), 400

    image = Image.open(io.BytesIO(base64.b64decode(data['file'])))
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    #save image to disk
    cv2.imwrite('image.jpg', image_cv2)
    input_labels = data['input_labels']
    input_points = data['input_points']

    
    if (data['again'] is False or predictor.is_image_set is False):
        set_image(image_cv2)

    image_masked = generate_image_with_prompt(image_cv2, input_labels, input_points)


    # turn black to white and white to black 
    # image_masked = cv2.bitwise_not(image_masked)
    #save masked image to disk
    cv2.imwrite('image_masked.jpg', image_masked)

    image_masked = Image.fromarray(image_masked)

    #send image as response to the client in json format
    image = io.BytesIO()
    image_masked.save(image, format='PNG')
    image.seek(0)
    image_bytes = image.getvalue()
    base64_encoded_result = base64.b64encode(image_bytes).decode()
    return jsonify({'image': base64_encoded_result})




@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file'}), 400

    file = request.files['file'].read()  # get the file from the request
    image = Image.open(io.BytesIO(file))  # open the image
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    #save image to disk
    cv2.imwrite('image.jpg', image_cv2)

    image_masked = generate_image(image_cv2)
    #save masked image to disk
    cv2.imwrite('image_masked.jpg', image_masked)
    
    image_masked = Image.fromarray(image_masked)

    #send image as response to the client in json format
    image = io.BytesIO()
    image_masked.save(image, format='PNG')
    image.seek(0)
    image_bytes = image.getvalue()
    base64_encoded_result = base64.b64encode(image_bytes).decode()  # encode as base64
    return jsonify({'image': base64_encoded_result})
@app.route('/')
def hello():
    return 'Hello, World!'



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)