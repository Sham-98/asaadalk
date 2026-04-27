import torch
#################################### For Image ####################################
from PIL import Image
import pandas as pd
import glob
import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay


# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)

# Load images
full_image_paths = glob.glob("/projappl/project_2001382/asaadalk/Correct_SAM_Classification_Images/*.jpg")
#image_paths = full_image_paths[:100]

thresholds = 0.9
output_csv = "sam_has_animal.csv"

# Write results incrementally to CSV instead of accumulating in memory
first_write = True
for image_path in full_image_paths:
    image = Image.open(image_path)
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt="animal")
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    # Close image to free memory immediately
    image.close()

    detection = len(scores)
    has_animal = "true" if detection > 0 else "false"

    image_name = os.path.basename(image_path)
    print("image_path:", image_name)
    print("Has animal:", has_animal)
    print("Detection:", detection)
    print("Localization:", boxes)
    # Don't print/store masks — they are large arrays not needed for evaluation

    # Write one row at a time to avoid accumulating data in RAM
    row = pd.DataFrame([{
        "image_path": image_name,
        "Detection": detection,
        "Has animal": has_animal,
        "Localization": str(boxes)  # store as string summary, not raw tensor
    }])
    row.to_csv(output_csv, mode='w' if first_write else 'a', header=first_write, index=False)
    first_write = False

    # Explicitly free large objects
    del masks, boxes, scores, inference_state, output


print("sam_has_animal.csv written.")
print("working2!")










