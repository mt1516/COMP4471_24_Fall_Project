import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import time
import json

mapping = json.load(open("data/ImageNet1k/image_net_mapping.json"))
labels = list(mapping.keys())

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
image = Image.open(requests.get(url, stream=True).raw)

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        # {"type": "text", "text": "Determine the main body in this picture. Your response should only consist of the name of the object, such as 'frog', 'planet', 'girl', etc. Reply with your top five guesses.\nFor example: given the image, you are supposed to respond ***['train','plane','truck','car','bird']***, ranked by train being your most confident classification, plane being the second most confident, to bird being the least confident option."}
        {"type": "text", "text": "A photo of a {label}."}
    ]}
]
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to(model.device)

# start = time.time()
# output = model.generate(**inputs, max_new_tokens=500)
# print("Time:", time.time()-start)
# print(processor.decode(output[0]))


# Get the image and prompt embeddings
with torch.no_grad():
    image_embedding = model(**inputs).hidden_states


# Get the text embeddings for the labels
labels_inputs = processor(
    labels,
    return_tensors="pt",
).to(model.device)
with torch.no_grad():
    label_embeddings = model(**labels_inputs).hidden_states

# Calculate the similarity between the image and the labels
similarity = torch.einsum("be,le->bl", image_embedding, label_embeddings)
topk = torch.topk(similarity, 5).indices[0]
print([labels[i] for i in topk])
