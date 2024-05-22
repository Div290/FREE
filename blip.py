import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import os
from datasets import load_dataset
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory



# with open('train_ds_coco.pkl', 'rb') as f1:
#     train_ds = pickle.load(f1)
    
with open('/home/iitb/divya/val_ds_coco.pkl', 'rb') as f:
    dataset = pickle.load(f)
    
# with open('test_ds_coco.pkl', 'rb') as f2:
#     test_ds = pickle.load(f2)


with open('/home/iitb/divya/coco_synthetic.pkl', 'rb') as f:
    captions = pickle.load(f)


# dataset = load_dataset("coco", split= 'train')


device = "cuda" if torch.cuda.is_available() else "cpu"
print("The device being used is", device)


from torch.utils.data import Dataset, DataLoader
class ImageCaptioningDataset(Dataset, captions):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = item["sentences"]["raw"]
        return encoding

def collate_fn(batch):
    # pad the input_ids and attention_mask
    processed_batch = {}
    for key in batch[0].keys():
        if key != "text":
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            text_inputs = processor.tokenizer(
                [example["text"] for example in batch], max_length=128, padding="max_length", return_tensors="pt"
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
    return processed_batch



# processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
# model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", device_map="auto")

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("ybelkada/blip2-opt-2.7b-fp16-sharded", output_hidden_states = True, torch_dtype=torch.float16)


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
    no_split_module_classes=["Blip2ForConditionalGeneration", "IntermediateHead"],
    dtype='float16',
    low_zero=False,
)

device_map = infer_auto_device_map(
    model,
    max_memory=max_memory,
    no_split_module_classes=["Blip2ForConditionalGeneration", "IntermediateHead"],
    dtype='float16'
)

model = dispatch_model(model, device_map=device_map)

train_dataset = ImageCaptioningDataset(dataset, processor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16, collate_fn=collate_fn)



# class IntermediateHead(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(IntermediateHead, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)
#         self.tanh = nn.Tanh()
        
#     def forward(self, x):
#         x = x.to(self.fc1.weight.dtype)
#         x = self.fc1(x)
#         x = self.tanh(x)
#         x = self.fc2(x)
#         return x

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
input_size = 2560  # Example input size
hidden_size = 2560  # Example hidden size
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
layers_for_exit = [2, 3, 5, 20]
# intermediate_heads = nn.ModuleList([IntermediateHead(input_size,hidden_size, vocab_size) for _ in range(len(layers_for_exit))])
intermediate_heads = nn.ModuleList([IntermediateHead(input_size, hidden_size, output_size, num_layers, num_heads, dropout) for _ in range(len(layers_for_exit))])




# define the optimizer
optimizer = AdamW(intermediate_heads.parameters(), lr=1e-5)
from transformers import get_linear_schedule_with_warmup
current_step = 0
save_steps = 3000
# optimizer = AdamW(intermediate_heads.parameters(), lr=1e-4)
n_train_steps = num_epochs * len(train_dataloader)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=n_train_steps)

kl_div_loss = torch.nn.KLDivLoss(reduction='batchmean')

intermediate_heads.train()
# model.train()
for epoch in range(num_epochs):
    # set the model to training model
    # initialize the training loss
    train_loss = 0
    for idx, batch in enumerate(tqdm(train_dataloader)):
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device, torch.float16)
        labels=input_ids
      # forward pass
        outputs = model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        labels=input_ids, output_hidden_states= True)
        int_loss_train = 0
        for exit in range(len(layers_for_exit)):
            intermediate_head = intermediate_heads[exit]
            intermediate_head = intermediate_head.to(device)
            # print(outputs.language_model_outputs.hidden_states[layers_for_exit[exit]].shape)
            intermediate_logits = intermediate_head(outputs.language_model_outputs.hidden_states[layers_for_exit[exit]])
            # print(intermediate_logits.shape)
            intermediate_logits = intermediate_logits[:, :128, :]
            # pooling_layer = torch.nn.AvgPool1d(kernel_size=2, stride=2)
            # Pad the sequence to ensure output size is 48
            # padded_hidden_states = F.pad(intermediate_logits, (0, 0, 0, 1))  # Padding the last dimension by 1

            # Apply pooling to reduce the sequence length
            # pooled_hidden_states = pooling_layer(padded_hidden_states.transpose(1, 2)).transpose(1, 2)
            # intermediate_logits = pooling_layer(intermediate_logits.transpose(1, 2)).transpose(1, 2)
            # print(intermediate_logits.shape)
            # predictions.extend(intermediate_logits.argmax(dim=-1).tolist())
            # print(intermediate_logits.argmax(dim=-1))
            if current_step > 0:
              generated_caption = processor.batch_decode(intermediate_logits.argmax(dim=-1), skip_special_tokens=True)[0]
            #   print(f"The generated caption for exit {layers_for_exit[exit]} is {generated_caption}")
              # Write the generated caption to a text file
              with open("generated_captions_decod_fina_unfrozen.txt", "a") as f:
                  f.write(f"Exit: {layers_for_exit[exit]}, Caption: {generated_caption}\n")
            # Calculate cosine similarity loss
            # Calculate cosine similarity loss
            # batch_size, seq_len, hidden_size = intermediate_logits.shape
            # Calculate cosine similarity manually
            # cosine_similarity = F.cosine_similarity(intermediate_logits, outputs.logits, dim=-1)
            # cosine_loss = 1 - cosine_similarity.mean()
            # print( labels.view(-1).shape)
            new_shape = (-1, intermediate_logits.size(-1))

            # Reshape the tensor using the calculated shape
            reshaped_logits = intermediate_logits.reshape(new_shape)
            # print(reshaped_logits.shape)

            # Calculate intermediate loss
            intermediate_loss = F.cross_entropy(reshaped_logits, labels.view(-1), ignore_index=-100)
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
          intermediate_head_weights_dir = f"./multi_heads/checkpoint/intermediate_head_weights/-{current_step}"
          os.makedirs(intermediate_head_weights_dir, exist_ok=True)

          # Save the weights of each intermediate head
          for layer_idx, intermediate_head in enumerate(intermediate_heads):
              head_path = os.path.join(intermediate_head_weights_dir, f"head_layer_{layers_for_exit[layer_idx]}.pt")
              torch.save(intermediate_head.state_dict(), head_path)


# import torch
# from tqdm import tqdm
# import torch.nn.functional as F

# from torch.optim import AdamW
# import os

# from torch.utils.tensorboard import SummaryWriter

# summary_writer = SummaryWriter(log_dir="./image-captioning/tensorboard")

# # define the optimizer
# # optimizer = AdamW(intermediate_heads.parameters(), lr=1e-5)
# from transformers import get_linear_schedule_with_warmup
# current_step = 0
# save_steps = 10000
# # optimizer = AdamW(intermediate_heads.parameters(), lr=1e-4)
# n_train_steps = num_epochs * len(train_dataloader)
# # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=n_train_steps)

# kl_div_loss = torch.nn.KLDivLoss(reduction='batchmean')


# # summary_writer = SummaryWriter(log_dir="./image-captioning/tensorboard")

# # Define the optimizer
# optimizer = AdamW(intermediate_heads.parameters(), lr=1e-5)

# # Define the learning rate scheduler
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1, num_training_steps=n_train_steps)

# # Define the KLDivLoss
# kl_div_loss = torch.nn.KLDivLoss(reduction='batchmean')

# # Autoregressive training loop
# for epoch in range(num_epochs):
#     # Set the model to training mode
    
#     # Initialize the training loss
#     train_loss = 0
    
#     for idx, batch in enumerate(tqdm(train_dataloader)):
#         input_ids = batch.pop("input_ids").to(device)
#         pixel_values = batch.pop("pixel_values").to(device, torch.float16)
#         labels = input_ids.clone()  # Clone the input_ids to use as labels
#         # print("\n",idx,"\n")
#         int_loss_train = 0
#         label_1 = []
#         for i in labels[0]:
#             if i>1:
#                 label_1.append(i)
#             else:
#                 pass
#         # print(labels.shape)
#         # print("\n Next token starts here\n")
#         # print(len(label_1))
#         outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels, output_hidden_states=True)
#         # Autoregressive generation loop
#         for t in range(len(label_1)+3):  # Loop over each token position (except the last one)
#             # Forward pass
#             # print(outputs.logits.shape)
#             # outputs = output

#             # Process intermediate heads
#             for exit in range(len(layers_for_exit)):
#                 intermediate_head = intermediate_heads[exit].to(device)
#                 intermediate_logits = intermediate_head(outputs.language_model_outputs.hidden_states[layers_for_exit[exit]])
#                 intermediate_logits = intermediate_logits
#                 # print(intermediate_logits.shape)
#                 # print(outputs.language_model_outputs.hidden_states[25].shape)
#                 # print(outputs.language_model_outputs.hidden_states[1])

#                 # Calculate intermediate loss
#                 # autoregressive_labels = labels.clone()
#                 # Update labels for autoregressive generation
#                 # Update labels for autoregressive generation
#                 autoregressive_labels = labels.clone()
#                   # Mask out future tokens beyond the current position
#                 autoregressive_labels[:, t+1:] = -100
#                 # print(autoregressive_labels)

#                 # Compute the logits and reshape them
#                 logits_shape = intermediate_logits.shape
#                 intermediate_logits_flat = intermediate_logits.view(-1, logits_shape[-1])
#                 # print(intermediate_logits)

#                 # Flatten the autoregressive labels tensor
#                 autoregressive_labels_flat = autoregressive_labels.view(-1)
#                 # print(autoregressive_labels_flat.shape)
#                 generated_caption = processor.batch_decode(intermediate_logits.argmax(dim=-1), skip_special_tokens=True)[0]
#                 # print(f"The generated caption for exit {layers_for_exit[exit]} is {generated_caption}")

#                 # Compute the cross-entropy loss
#                 intermediate_loss = 128*F.cross_entropy(intermediate_logits_flat[t], autoregressive_labels_flat[t], ignore_index=-100)

#                 # Accumulate the loss
#                 int_loss_train += intermediate_loss
#         int_loss_train+=outputs.loss
#         # Backpropagate the intermediate loss
#         int_loss_train.backward()
#         with open("generated_captions_auto_dec_unfrozen.txt", "a") as f:
#                   f.write(f"index is :{idx}, Exit: {layers_for_exit[exit]}, Caption: {generated_caption}\n")
        
#         # Update the weights
#         optimizer.step()
#         scheduler.step()
#         optimizer.zero_grad()
        
#         # Log the loss
#         loss_v = intermediate_loss.item()
#         train_loss += loss_v
#         # summary_writer.add_scalar("train_loss", loss_v, global_step=current_step)
                
#         # Log the epoch training loss
#         print(f"Epoch: {epoch}, Train Loss: {train_loss / len(train_dataloader):.4f}")
    
#     # Save intermediate head weights
#     if epoch % save_steps == 0:
#         intermediate_head_weights_dir = f"./multi_heads/checkpoint/intermediate_head_weights/epoch-{epoch}"
#         os.makedirs(intermediate_head_weights_dir, exist_ok=True)
        
#         for layer_idx, intermediate_head in enumerate(intermediate_heads):
#             head_path = os.path.join(intermediate_head_weights_dir, f"head_layer_{layers_for_exit[layer_idx]}.pt")
#             torch.save(intermediate_head.state_dict(), head_path)