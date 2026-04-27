import torch
#################################### For Image ####################################
from PIL import Image
import pickle
import pandas as pd 
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)

# Load an image
image_path = "/scratch/project_2001382/data/shared/zebra/images/2ddf70b1-813f-dccb-757a-9ad0eb20478c.jpg"
image = Image.open(image_path)
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="animal")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

print("Detection:", len(scores))
print("Localization:", boxes)
print("Segmentation:", masks)
#print("Has real animal:", output)

#cope the following outputs into the list[]
#SAM_results = []
#SAM_results.append({"image path": image_path, "Detection": len(scores), "Localization": boxes, "Segmentation": masks})

#print("This is the before writing")
#print(SAM_results)
#create pkl file (serialization)
#with open ('SAM_results_pickle_1', 'ab') as f:
    #write the dictionary, not the list [0]
#    pickle.dump(SAM_results[0], f)

#check the results of the file (read)for one object only
#with open ('SAM_results_pickle_1', 'rb') as f:
#    loaded_SAM_results = pickle.load(f)
#print(loaded_SAM_results)  

#flatten()


#loaded_SAM_results = []
#with (open("SAM_results_pickle_1", "rb")) as f:
#    while True:
    #    try:
   #         loaded_SAM_results.append(pickle.load(f))
  #      except EOFError:
 #           break
#print(loaded_SAM_results)  
print("This is the After writing")

#read unpickled data
#unpickle_df = pd.DataFrame(loaded_SAM_results)
#print("\nDataFrame:")
#print(unpickle_df)
#print(unpickle_df.columns)
#print("Hiiii")
      