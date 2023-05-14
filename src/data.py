import transformers, datasets, torch

def get_data_loaders(args): 
   
    dataset = datasets.load_dataset(args.data.dataset_name,
        split=args.data.load_split,
        cache_dir=args.model.cache_dir,
        num_proc=args.data.num_proc)




    # return tokenized_dataset, loader, val_loader

# Split into train/val 
split = dataset.train_test_split(test_size=0.2)
train_set, val_set = split["train"], split["test"]

# Package into DatasetDict 
dataset_dict = DatasetDict({
    'train': train_set,
    'val': val_set 
})

# Get tokenizer 
model_checkpoint = "bigscience/bloomz-560m"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize_function(examples, cmp_len=20, seq_len=100, overlap=0, min_seq=5):
    ...

tokenized_dataset = dataset_dict.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"], batch_size=100)

# Create data loaders 
loader = torch.utils.data.DataLoader(tokenized_dataset['train'], batch_size=100, shuffle=True)
val_loader = torch.utils.data.DataLoader(tokenized_dataset['val'], batch_size=100, shuffle=True)

def get_data_loaders(args): 
    return tokenized_dataset, loader, val_loader

