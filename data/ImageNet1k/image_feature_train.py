import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import argparse
import torch
# from torchvision import transforms
from datasets import load_dataset
from transformers import AutoProcessor, MllamaVisionModel
from tqdm import tqdm
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("split", "-s", "--split", type=str, default=None, nargs="?", help="Dataset split to use: train, validation, test")
parser.add_argument("skip", "-s", "--skip", type=int, default=None, nargs="?", help="Skip first n samples from the dataset")
parser.add_argument("take", "-t", "--take", type=int, default=None, nargs="?", help="First n samples to take from the dataset (after skipping)")
args = parser.parse_args()

def arg_check(args):
    if args.split not in ["train", "validation", "test"] or args.split is not None:
        raise ValueError("Invalid split argument. Must be one of: train, validation, test")
    if args.skip and args.skip < 0 or args.skip is not None:
        raise ValueError("Invalid skip argument. Must be a positive integer")
    if args.take and args.take < 0 or args.take is not None:
        raise ValueError("Invalid take argument. Must be a positive integer")
    return args

args = arg_check(args)
if args.split is None:
    args.split = "train"
dataset = load_dataset("ILSVRC/imagenet-1k", split=args.split, streaming=True)
if args.skip:
    dataset = dataset.skip(args.skip)
if args.take:
    dataset = dataset.take(args.take)

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model_path = "../../model/"

processor = AutoProcessor.from_pretrained(model_path)
model = MllamaVisionModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# def custom_collate_fn(data):
#     # data = Image.open(data[0]["image"]).convert("RGB") if isinstance(#something here) else #something here
#     return data
# #something here about data transformation

def infer(processor, model, data):
    with torch.no_grad():
        inputs = processor(
            images=data["image"],
            return_tensors="pt",
        ).to(model.device)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(**inputs).last_hidden_state[0]
        outputs = outputs.mean(dim=1).mean(dim=1)
        return outputs
        
with torch.inference_mode():
    tensor = torch.tensor([]).to(model.device)
    for i, data in enumerate(tqdm(dataset)):
        try:
            # inputs = processor(
            #     images=data["image"],
            #     return_tensors="pt",
            # ).to(model.device)
            # inputs = {k: v.to(model.device) for k, v in inputs.items()}
            # outputs = model(**inputs).last_hidden_state[0]
            # outputs = outputs.mean(dim=1).mean(dim=1)
            outputs = infer(processor, model, data)
            tensor = torch.cat((tensor, outputs), 0)
            if not i % 1000:
                torch.save(tensor, f"train_features_cp_{i//1000}.pt")
                print(f"checkpoint {i//1000} saved")
                tensor = torch.tensor([]).to(model.device)
        except Exception as e:
            with open("train_features_error.txt", "a") as f:
                f.write(f"{i}: {e}\n")