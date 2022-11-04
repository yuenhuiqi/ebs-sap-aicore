MODEL_PATH = '/app/model/model.h5'

# Imports
from transformers import DetrFeatureExtractor, DetrForObjectDetection
import torch
from PIL import Image

# HuggingFace Transformer
extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-101")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101")
img = "data/trash.jpeg"


def detectTrash(img_path, model, extractor): 
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

  return obj_dict

# to test out 
detectTrash(img, model, extractor)



# Save model
torch.save(model, MODEL_PATH)