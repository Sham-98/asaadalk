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
full_image_paths = glob.glob("/scratch/project_2001382/data/shared/zebra/images/*.jpg")
#image_paths = full_image_paths[:1000]

thresholds = 0.9
output_csv = "SAM_results.csv"

# Write results incrementally to CSV instead of accumulating in memory
count_empty_images = 0
first_write = True
for image_path in full_image_paths:
    try:
        image = Image.open(image_path)
    except Exception:
        print("Invalid image with 0.00 B:", image_path)
        count_empty_images += 1
        print(count_empty_images)
        continue

    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt="animal")
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

    # Close image to free memory immediately
    image.close()

    detection = len(scores)
    has_animal = "True" if detection > 0 else "False"

    image_name = os.path.basename(image_path)
    print("image_path:", image_name)
    print("has_animal_SAM:", has_animal)
    print("Detection:", detection)
    print("Localization:", boxes)
    # Don't print/store masks — they are large arrays not needed for evaluation

    # Write one row at a time to avoid accumulating data in RAM
    row = pd.DataFrame([{
        "image_path": image_name,
        "Detection": detection,
        "has_animal_SAM": has_animal,
        "Localization": str(boxes)  # store as string summary, not raw tensor
    }])
    row.to_csv(output_csv, mode='w' if first_write else 'a', header=first_write, index=False)
    first_write = False

    # Explicitly free large objects
    del masks, boxes, scores, inference_state, output

print("SAM_results.csv written.")

# Load the JSON ground truth data
with open('/scratch/project_2001382/data/shared/zebra/annotations/GZCD_gt.json', 'r') as f:
    data = json.load(f)
print("working1!")

df_gt = pd.DataFrame(data['images'])
#print("Ground truth table:", df_gt)

# Keep only needed columns
df_gt_cul = df_gt[['image_path', 'has_animal']]

df_sam_pred = pd.read_csv(output_csv, usecols=['image_path', 'Detection', 'has_animal_SAM'])
#print("SAM results table:", df_sam_pred)
#print(df_sam_pred.dtypes)
#print("Ground truth needed columns:", df_gt_cul)
#print(df_gt_cul.dtypes)

print("working2!")

columns_merge = pd.merge(df_sam_pred, df_gt_cul, how="inner", on="image_path")
pd.set_option('display.max_columns', None)

print("\nComparison table", columns_merge)
print("working3!")

columns_merge[['image_path', 'Detection', 'has_animal', 'has_animal_SAM']]
print("\nComparison table", columns_merge)

print("working4!")

# Convert to NumPy arrays for evaluation
actual = columns_merge['has_animal'].to_numpy()       # ground truth
predicted = columns_merge['has_animal_SAM'].to_numpy()

# Confusion matrix, accuracy, and full classification report
cm = confusion_matrix(actual, predicted)
print("Confusion matrix:", cm)

plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
 xticklabels=['Predicted Positive', 'Predicted Negative'],
 yticklabels=['Actual Positive', 'Actual Negative'])
plt.xlabel('SAM Label')
plt.ylabel('Ground Truth Label')
plt.title('Confusion matrix')
plt.savefig('cm.jpg')
plt.show()

accuracy = accuracy_score(actual, predicted)
print("Accuracy:", accuracy)
print(classification_report(actual, predicted))

print("working5!")






