
# 1. Imports 
import os
from flask import Flask
from flask import request as call_request
from transformers import DetrFeatureExtractor, DetrForObjectDetection
import torch
from PIL import Image

# Creates Flask serving engine
app = Flask(__name__)

model = None

@app.before_first_request
def init():
    """
    Load model else crash, deployment will not start
    """
    global model
    global extractor
    extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-101")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101")
    return None

@app.route("/v2/greet", methods=["GET"])
def status():
    global model
    global extractor
    if model is None:
        return "Flask Code: Model was not loaded."
    else:
        return "Model is loaded."

# You may customize the endpoint, but must have the prefix `/v<number>`
@app.route("/v2/predict", methods=["POST"])
def detectTrash():
    """
    Perform an inference on the model created in initialize

    Returns:
        String value price.
    """
    global model
    global extractor
    
    query = dict(call_request.json)
    img_path = query['imgUrl']
    
    image = Image.open(img_path)

    inputs = extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    target_sizes = torch.tensor([image.size[::-1]])
    results = extractor.post_process(outputs, target_sizes=target_sizes)[0]
    obj_dict = {}
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        # only keep detections with score > 0.8
        if score > 0.8:
            item = model.config.id2label[label.item()]
            if item not in obj_dict:
                obj_dict[item] = 1
            else: 
                obj_dict[item] += 1
    # Response
    return obj_dict

if __name__ == "__main__":
    print("Serving Initializing")
    init()
    print(f'{os.environ["greetingmessage"]}')
    print("Serving Started")
    app.run(host="0.0.0.0", debug=True, port=9001)