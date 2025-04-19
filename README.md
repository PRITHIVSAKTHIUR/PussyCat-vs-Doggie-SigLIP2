
![sdfsdfc.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/nOQzJQNWRvWDskXlG2IJ4.png)

# **PussyCat-vs-Doggie-SigLIP2**  

> **PussyCat-vs-Doggie-SigLIP2** is an image classification vision-language encoder model fine-tuned from **google/siglip2-base-patch16-224** for a single-label classification task. It is designed to classify images as either a cat or a dog using the **SiglipForImageClassification** architecture.  

The model categorizes images into two classes:  
- **Class 0:** "Pussy Cat"  
- **Class 1:** "Doggie"  

```py
Classification Report:
              precision    recall  f1-score   support

   Pussy Cat     0.9194    0.8745    0.8964     12500
      Doggie     0.8803    0.9234    0.9013     12500

    accuracy                         0.8989     25000
   macro avg     0.8999    0.8989    0.8989     25000
weighted avg     0.8999    0.8989    0.8989     25000
```

![download.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/WyUXhTD5UYzG_uBS3tI_l.png)

# **Run with Transformers🤗**  

```python
!pip install -q transformers torch pillow gradio
```

```python
import gradio as gr
from transformers import AutoImageProcessor
from transformers import SiglipForImageClassification
from transformers.image_utils import load_image
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/PussyCat-vs-Doggie-SigLIP2"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def animal_classification(image):
    """Predicts whether the image contains a cat or a dog."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    labels = {
        "0": "Pussy Cat", 
        "1": "Doggie"
    }
    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    
    return predictions

# Create Gradio interface
iface = gr.Interface(
    fn=animal_classification,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="Cat vs Dog Classification",
    description="Upload an image to classify whether it contains a cat or a dog."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
```  

# **Intended Use:**  

The **PussyCat-vs-Doggie-SigLIP2** model is designed to classify images as either a cat or a dog. Potential use cases include:  

- **Pet Identification:** Helping users distinguish between cats and dogs.  
- **Automated Pet Sorting:** Useful for shelters and pet adoption platforms.  
- **Educational Purposes:** Assisting in teaching image classification concepts.  
- **Surveillance & Security:** Identifying animals in security footage.
