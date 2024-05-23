

from flask import Flask, request, jsonify
from flask_cors import CORS  # import the flask_cors module
from PIL import Image
import io
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import cv2
import base64


app = Flask(__name__)
CORS(app)  # enable CORS on the app


# CHECKPOINT_PATH = "../../../sam_vit_h_4b8939.pth"
CHECKPOINT_PATH = "./sam_vit_b_01ec64.pth"

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
MODEL_TYPE = "vit_b"

model = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
model.to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(model)
predictor = SamPredictor(model)

current_image = None


def get_second_largest_area(result_dict):
    sorted_result = sorted(result_dict, key=(lambda x: x['area']),
                           reverse=True)
    return sorted_result[0]


def apply_mask(image, mask, color=None):
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
    largest_area = get_second_largest_area(output_mask)
    mask = largest_area['segmentation']

    return mask


def set_image(image):
    predictor.set_image(image)


def generate_image_with_prompt(image, input_labels, input_points):
    # input_labels = input_labels.split(',') if input_labels else []
    # input_points = input_points.split(',') if input_points else []
    input_points = np.array(input_points, dtype=np.float32)
    input_labels = np.array(input_labels, dtype=np.float32)
    # exit()
    output_mask, socres, logits = predictor.predict(
                            point_coords=input_points,
                            point_labels=input_labels,
                            multimask_output=False,
                        )
    # return generate_image_of_mask(output_mask, image.shape)

    image_rgb = np.zeros((image.shape[0], image.shape[1], 3), dtype='uint8')

    for i in range(len(output_mask)):
        mask = output_mask[i]
        mask = np.where(mask, 255, 0).astype('uint8')
    
        color = (255, 255, 255)
        image = apply_mask(image_rgb, mask, color=color)

        #invert black and white
    # image = cv2.bitwise_not(image)
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


    return image

                            

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
    input_labels = data['input_labels']
    input_points = data['input_points']
    
    if (data['again'] is False or current_image is None):
        set_image(image_cv2)

    image_masked = generate_image_with_prompt(image_cv2, input_labels, input_points)


    # turn black to white and white to black 
    image_masked = cv2.bitwise_not(image_masked)

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

    image_masked = generate_image(image_cv2)
    
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