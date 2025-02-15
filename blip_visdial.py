from datasets import load_dataset
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

dataset = load_dataset('HuggingFaceM4/VisDial', split="train")

from PIL import Image
import requests
import torch
from transformers import BlipProcessor, Blip2ForConditionalGeneration

# model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", device_map="auto", torch_dtype=torch.float16)
# # processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
device = 'cuda:0'

from transformers import AutoProcessor, Blip2ForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("ybelkada/blip2-opt-2.7b-fp16-sharded", output_hidden_states = True, torch_dtype=torch.float16)

from peft import LoraConfig, get_peft_model

# Let's define the LoraConfig
config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj"]
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

max_memory = get_balanced_memory(
    model,
    max_memory=None,
    no_split_module_classes=["Blip2ForConditionalGeneration","IntermediateHead"],
    dtype='float16',
    low_zero=False,
)

device_map = infer_auto_device_map(
    model,
    max_memory=max_memory,
    no_split_module_classes=["Blip2ForConditionalGeneration","IntermediateHead"],
    dtype='float16'
)

model = dispatch_model(model, device_map=device_map)

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
from PIL import Image
from datasets import load_dataset
import random

max_length = 32

# Custom Dataset class for VisDial
class VisDialDataset(Dataset):
    def __init__(self, dataset, processor, max_length):
        self.dataset = dataset
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        dialog = item["dialog"]
        image = item["image"]
        # true_labels = [turn[1] for turn in dialog]  # Extract true labels from dialog

        # Construct dialog history
        # dialog_history = []
        # for i in range(idx):
        #     dialog_history.append(dialog[i][0])  # Add question from previous turn
        #     dialog_history.append(dialog[i][1])  # Add answer from previous turn
        
        
        rnd = random.randint(0, len(dialog)-2)
            
        context = [(dialog[i][0], dialog[i][1]) for i in range(rnd)]
        question = dialog[rnd+1][0]
        template = "Question: {} Answer: {}."

        prompt = " ".join([template.format(context[i][0], context[i][1]) for i in range(len(context))]) + " Question: " + question + " Answer:"
        # print(prompt)

        # Tokenize and encode dialog history
        encoding = self.processor(image, prompt, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        # encoding["text_1"] = " Question: " + question + " Answer:"
        encoding["text"] = dialog[rnd+1][1]

        return encoding

# Function to collate batch items
def collate_fn(batch):
    # Pad the input_ids and attention_mask
    processed_batch = {}
    for key in batch[0].keys():
        if key != "text": #and key != "text_1":
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            text_inputs = processor.tokenizer(
                [example["text"] for example in batch], max_length=128, padding="max_length", return_tensors="pt"
            )
            # text_inputs_new = processor.tokenizer(
            #     [example["text_1"] for example in batch], max_length=128, padding="max_length", return_tensors="pt"
            # )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
            # processed_batch["input_ids_new"] = text_inputs_new["input_ids"]
    return processed_batch

# Load the VisDial dataset
# dataset = load_dataset("visdial")

# Define the processor
# processor = AutoProcessor.from_pretrained("salesforce/blip2-opt-2.7b")

# Create an instance of the custom dataset
train_dataset = VisDialDataset(dataset, processor, max_length)

# Define the DataLoader
batch_size = 8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer

import torch
# from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
from PIL import Image
import numpy as np
from datasets import load_dataset
from torch.optim import AdamW
import tqdm
import os

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


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)



# usage:
input_size =2560 # Example input size
hidden_size = 5072  # Example hidden size
output_size = 50272  # Example output size
num_layers = 2  # Number of decoder layers
num_heads = 8  # Number of attention heads
dropout = 0.1  # Dropout probability

# Initialize IntermediateHead module


# # Test forward pass with dummy input
# dummy_input = torch.randn(10, 32, input_size)  # Example input with sequence length 10 and batch size 32
# output = intermediate_head(dummy_input)
# print("Output shape:", output.shape)  # Example output shape


    
vocab_size = 50272
# input_size = 2560
# hidden_size = 6072
num_epochs = 15
# Create intermediate head modules
layers_for_exit = [3, 5, 18, 21, 27]
# intermediate_heads = nn.ModuleList([IntermediateHead(input_size,hidden_size, vocab_size) for _ in range(len(layers_for_exit))])
intermediate_heads = nn.ModuleList([IntermediateHead(input_size, hidden_size, output_size, num_layers, num_heads, dropout) for _ in range(len(layers_for_exit))])




# define the optimizer
# optimizer = AdamW(intermediate_heads.parameters(), lr=1e-5)
from transformers import get_linear_schedule_with_warmup
current_step = 0
save_steps = 5000
optimizer = AdamW(intermediate_heads.parameters(), lr=1e-4)
n_train_steps = num_epochs * len(train_dataloader)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=n_train_steps)

kl_div_loss = torch.nn.KLDivLoss(reduction='batchmean')




intermediate_heads = nn.ModuleList([IntermediateHead(input_size, hidden_size, output_size, num_layers, num_heads, dropout) for _ in layers_for_exit])
discriminator = Discriminator(input_size, hidden_size)

opt_int = AdamW(intermediate_heads.parameters(), lr=1e-4)
opt_disc = AdamW(discriminator.parameters(), lr=1e-4)
criterion = nn.BCELoss()

for epoch in range(num_epochs):
    for batch in train_dataloader:
        outputs = model(**batch, output_hidden_states=True)
        real_features = outputs.hidden_states[-1].detach()
        fake_features = [head(outputs.hidden_states[l]) for head, l in zip(intermediate_heads, layers_for_exit)]
        
        # Discriminator Training
        opt_disc.zero_grad()
        disc_loss = criterion(discriminator(real_features), torch.ones_like(real_features[:, :1])) + \
                    criterion(torch.cat([discriminator(f) for f in fake_features]), torch.zeros(len(fake_features), 1))
        disc_loss.backward()
        opt_disc.step()
        
        # Generator (Intermediate Heads) Training
        opt_int.zero_grad()
        gen_loss = sum(criterion(discriminator(head(outputs.hidden_states[l])), torch.ones_like(real_features[:, :1]))
                       for head, l in zip(intermediate_heads, layers_for_exit))
        gen_loss.backward()
        opt_int.step()
        
    print(f"Epoch {epoch}: Disc Loss {disc_loss.item()}, Gen Loss {gen_loss.item()}")


# intermediate_heads.train()
# # model.train()
# for epoch in range(num_epochs):
#     # set the model to training model
#     # initialize the training loss
#     train_loss = 0
#     for idx, batch in enumerate((train_dataloader)):
#         print("Samples Processed", idx)
#         input_ids = batch.pop("input_ids").to(device)
#         # input_ids_new = batch.pop("input_ids_new").to(device)
#         pixel_values = batch.pop("pixel_values").to(device, torch.float16)
#         labels=input_ids
#       # forward pass
#         outputs = model(input_ids=input_ids,
#                         pixel_values=pixel_values,
#                         labels=input_ids, output_hidden_states= True)
#         # print(outputs.language_model_outputs.keys())
#         int_loss_train = 0
#         for exit in range(len(layers_for_exit)):
#             intermediate_head = intermediate_heads[exit]
#             intermediate_head = intermediate_head.to(device)
#             # print(outputs.language_model_outputs.hidden_states[layers_for_exit[exit]].shape)
#             intermediate_logits = intermediate_head(outputs.language_model_outputs.hidden_states[layers_for_exit[exit]])
#             # print(intermediate_logits.shape)
#             intermediate_logits = intermediate_logits[:, :128, :]
#             # pooling_layer = torch.nn.AvgPool1d(kernel_size=2, stride=2)
#             # Pad the sequence to ensure output size is 48
#             # padded_hidden_states = F.pad(intermediate_logits, (0, 0, 0, 1))  # Padding the last dimension by 1

#             # Apply pooling to reduce the sequence length
#             # pooled_hidden_states = pooling_layer(padded_hidden_states.transpose(1, 2)).transpose(1, 2)
#             # intermediate_logits = pooling_layer(intermediate_logits.transpose(1, 2)).transpose(1, 2)
#             # print(intermediate_logits.shape)
#             # predictions.extend(intermediate_logits.argmax(dim=-1).tolist())
#             # print(intermediate_logits.argmax(dim=-1))
#             if current_step > 0:
#               generated_caption = processor.batch_decode(intermediate_logits.argmax(dim=-1), skip_special_tokens=True)[0]
#               true_caption = processor.batch_decode(labels, skip_special_tokens=True)[0]
#               with open("generated_captions_visdial.txt", "a") as f:
#                   f.write(f"Exit: {layers_for_exit[exit]}, Caption: {generated_caption}\n")
#                   f.write(f"The true caption is: {true_caption}\n")
                  
#             new_shape = (-1, intermediate_logits.size(-1))

#             # Reshape the tensor using the calculated shape
#             reshaped_logits = intermediate_logits.reshape(new_shape)
#             intermediate_loss = (F.cross_entropy(reshaped_logits, labels.view(-1), ignore_index=-100))
#             # intermediate_loss = (F.cross_entropy(intermediate_logits.view(-1, intermediate_logits.size(-1)), labels.view(-1), ignore_index=-100))
#             # kd_loss = kl_div_loss(outputs.logits, intermediate_logits)
#             int_loss_train+=intermediate_loss#+outputs.loss#+kd_loss#+0.5*outputs.loss
#             # Backpropagate the intermediate loss and accumulate gradients
#           #   intermediate_loss.backward()
#         # get the loss
#         # print(outputs.decoder_hidden_states[layers_for_exit[0]])
#         int_loss_train = int_loss_train / len(layers_for_exit)
#         # backward pass
#         int_loss_train.backward()
#         # save_steps+=1
#         # update the weights
#         optimizer.step()
#         scheduler.step()
#         # zero the gradients
#         optimizer.zero_grad()
#         # log the loss
#         loss_v = int_loss_train.item()
#         train_loss += loss_v
#         # increment the step
#         current_step += 1
#         # log the training loss
#         # summary_writer.add_scalar("train_loss", loss_v, global_step=current_step)
        
#         if current_step%save_steps==0:
#           print(f"Epoch: {epoch}, Step: {current_step}, Train Loss: {train_loss / save_steps:.4f} " )  
#           intermediate_head_weights_dir = f"./multi_heads_visdial/checkpoint/intermediate_head_weights/-{current_step}"
#           os.makedirs(intermediate_head_weights_dir, exist_ok=True)

#           # Save the weights of each intermediate head
#           for layer_idx, intermediate_head in enumerate(intermediate_heads):
#               head_path = os.path.join(intermediate_head_weights_dir, f"head_layer_{layers_for_exit[layer_idx]}.pt")
#               torch.save(intermediate_head.state_dict(), head_path)

