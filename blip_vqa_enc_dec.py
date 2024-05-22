# !pip install -q git+https://github.com/huggingface/peft.git transformers bitsandbytes datasets

from datasets import load_dataset
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

dataset = load_dataset("Graphcore/vqa", split="validation")

from PIL import Image
import requests
import torch
from transformers import BlipProcessor, Blip2ForConditionalGeneration

# model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", device_map="auto", torch_dtype=torch.float16)
# # processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

from transformers import AutoProcessor, Blip2ForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", device_map="auto", torch_dtype=torch.float16)

from peft import LoraConfig, get_peft_model

# Let's define the LoraConfig
# config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     lora_dropout=0.05,
#     bias="none",
#     target_modules=["q_proj", "k_proj"]
# )

# model = get_peft_model(model, config)
# model.print_trainable_parameters()

# max_memory = get_balanced_memory(
#     model,
#     max_memory=None,
#     no_split_module_classes=["Blip2ForConditionalGeneration","IntermediateHead"],
#     dtype='float16',
#     low_zero=False,
# )

# device_map = infer_auto_device_map(
#     model,
#     max_memory=max_memory,
#     no_split_module_classes=["Blip2ForConditionalGeneration","IntermediateHead"],
#     dtype='float16'
# )

# model = dispatch_model(model, device_map=device_map)



import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

idxs = []
for i in range(len(dataset)):
    if len(dataset[i]["label"]["ids"])==0:
      idxs.append(i)
      
print(idxs)

dataset = dataset.filter(lambda example, idx: idx not in idxs, with_indices=True)

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
from PIL import Image
import numpy as np
from datasets import load_dataset
from torch.optim import AdamW
import tqdm
import os

# Load the VQAv2 dataset with images
# dataset = load_dataset("vqa_v2", split="train")
max_length = 10

# Define the tokenizer
# processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

class VQADataset(Dataset):
    def __init__(self, dataset, processor, max_length):
        self.dataset = dataset
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        question = item["question"]
        answer = item["label"]["ids"][np.argmax(np.array(item["label"]["weights"]))]
        image = Image.open(item["image_id"])
        # Tokenize question and answer
        encoding = self.processor(image, question, padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt")
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        # question = self.processor.tokenizer(
        #     question, padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt"
    
        encoding["text"] = item["label"]["ids"][np.argmax(np.array(item["label"]["weights"]))]
        # encoding["question"] = question
        return encoding
    
def collate_fn(batch):
    # pad the input_ids and attention_mask
    processed_batch = {}
    for key in batch[0].keys():
        if key != "text":
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            text_inputs = processor.tokenizer(
                [example["text"] for example in batch], max_length=10, padding="max_length", return_tensors="pt"
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
    return processed_batch


# Create an instance of the custom dataset
vqa_dataset = VQADataset(dataset, processor, max_length = 10)

# Define the DataLoader
batch_size = 16
train_dataloader = DataLoader(vqa_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer

class IntermediateHead(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_heads, dropout):
        super(IntermediateHead, self).__init__()
        self.transformer_decoder_layer = TransformerDecoderLayer(
            d_model=input_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout
        )
        self.transformer_decoder = TransformerDecoder(
            decoder_layer=self.transformer_decoder_layer,
            num_layers=num_layers
        )
        self.classifier = nn.Linear(input_size, output_size)

    def forward(self, x):
        # Assuming x has shape (seq_len, batch_size, input_size)
        x = x.to(self.transformer_decoder_layer.self_attn.out_proj.weight.dtype)
        x = x.transpose(0, 1)
        
        memory = torch.zeros_like(x)
        
        # Transformer decoder layer forward pass
        transformer_output = self.transformer_decoder(x, memory)
        transformer_output = transformer_output.transpose(0,1)
        # print(transformer_output.shape)
        # Apply classifier to the transformer output
        output = self.classifier(transformer_output.contiguous().view(-1, transformer_output.size(-1)))  # Take the output of the last layer
        # print(output.shape)
        # output = output.unsqueeze(0)
        
        return output.view(transformer_output.size(0), -1, output.size(-1))

# Example usage:
input_size =2048 # Example input size
hidden_size = 2048  # Example hidden size
output_size = 32128  # Example output size
num_layers = 2  # Number of decoder layers
num_heads = 8  # Number of attention heads
dropout = 0.1  # Dropout probability

# Initialize IntermediateHead module


# # Test forward pass with dummy input
# dummy_input = torch.randn(10, 32, input_size)  # Example input with sequence length 10 and batch size 32
# output = intermediate_head(dummy_input)
# print("Output shape:", output.shape)  # Example output shape


    
# vocab_size = 50272
# input_size = 2560
# hidden_size = 6072
num_epochs = 5
# Create intermediate head modules
layers_for_exit = [3, 5, 7, 9, 12, 15, 18, 21, 23]
# intermediate_heads = nn.ModuleList([IntermediateHead(input_size,hidden_size, vocab_size) for _ in range(len(layers_for_exit))])
intermediate_heads = nn.ModuleList([IntermediateHead(input_size, hidden_size, output_size, num_layers, num_heads, dropout) for _ in range(len(layers_for_exit))])


print("The length of dataset is", len(dataset))

# define the optimizer
# optimizer = AdamW(intermediate_heads.parameters(), lr=1e-5)
from transformers import get_linear_schedule_with_warmup
current_step = 0
save_steps = 3000
optimizer = AdamW(intermediate_heads.parameters(), lr=1e-4)
n_train_steps = num_epochs * len(train_dataloader)
print("The number of training steps are:", n_train_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=n_train_steps)

kl_div_loss = torch.nn.KLDivLoss(reduction='batchmean')

intermediate_heads.train()
# model.train()
for epoch in range(num_epochs):
    # set the model to training model
    # initialize the training loss
    train_loss = 0
    for idx, batch in enumerate((train_dataloader)):
        print("Samples Processed:", idx)
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device, torch.float16)
        labels=input_ids
      # forward pass
        outputs = model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        labels=input_ids, output_hidden_states= True)
        # print(outputs.logits.shape)
        # print(outputs.language_model_outputs.keys())
        int_loss_train = 0
        for exit in range(len(layers_for_exit)):
            intermediate_head = intermediate_heads[exit]
            intermediate_head = intermediate_head.to(device)
            # print(outputs.language_model_outputs.hidden_states[layers_for_exit[exit]].shape)
            intermediate_logits = intermediate_head(outputs.language_model_outputs.decoder_hidden_states[layers_for_exit[exit]])
            # print(intermediate_logits.shape)
            # intermediate_logits = intermediate_logits[:, :128, :]
            if current_step > 0:
              generated_caption = processor.batch_decode(intermediate_logits.argmax(dim=-1), skip_special_tokens=True)[0]
            #   print(f"The generated caption for exit {layers_for_exit[exit]} is {generated_caption}")
              with open("generated_vqa_enc_dec.txt", "a") as f:
                f.write(f"Exit: {layers_for_exit[exit]}, Caption: {generated_caption}\n")

            new_shape = (-1, intermediate_logits.size(-1))

            # Reshape the tensor using the calculated shape
            reshaped_logits = intermediate_logits.reshape(new_shape)
            intermediate_loss = (F.cross_entropy(reshaped_logits, labels.view(-1), ignore_index=-100))
            # kd_loss = kl_div_loss(outputs.logits, intermediate_logits)
            int_loss_train+=layers_for_exit[exit]*intermediate_loss#+outputs.loss#+kd_loss#+0.5*outputs.loss
            # Backpropagate the intermediate loss and accumulate gradients
          #   intermediate_loss.backward()
        # get the loss
        # print(outputs.decoder_hidden_states[layers_for_exit[0]])
        int_loss_train = int_loss_train / len(layers_for_exit)
        # backward pass
        int_loss_train.backward()
        # save_steps+=1
        # update the weights
        optimizer.step()
        scheduler.step()
        # zero the gradients
        optimizer.zero_grad()
        # log the loss
        loss_v = int_loss_train.item()
        train_loss += loss_v
        # increment the step
        current_step += 1
        # log the training loss
        # summary_writer.add_scalar("train_loss", loss_v, global_step=current_step)
        
        if current_step%save_steps==0:
          print(f"Epoch: {epoch}, Step: {current_step}, Train Loss: {train_loss / save_steps:.4f} " )  
          intermediate_head_weights_dir = f"./multi_heads_vqa_enc_dec/checkpoint/intermediate_head_weights/-{current_step}"
          os.makedirs(intermediate_head_weights_dir, exist_ok=True)

          # Save the weights of each intermediate head
          for layer_idx, intermediate_head in enumerate(intermediate_heads):
              head_path = os.path.join(intermediate_head_weights_dir, f"head_layer_{layers_for_exit[layer_idx]}.pt")
              torch.save(intermediate_head.state_dict(), head_path)

