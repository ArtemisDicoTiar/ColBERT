import torch
from torch.utils.data import DataLoader
from transformers import GPTNeoForCausalLM, AutoTokenizer, LineByLineDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B')
model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B')

dataset = LineByLineDataset(...)
data_loader = DataLoader(dataset=dataset,
                         num_workers=args.num_workers,
                         batch_size=args.batch_size,
                         drop_last=False)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                max_lr=args.lr,
                                                epochs=args.num_epochs,
                                                steps_per_epoch=len(data_loader),
                                                anneal_strategy='linear')

for step, batch in enumerate(data_loader):
    input_ids, attention_masks, labels = batch

    input_ids = input_ids.to(device)
    attention_masks = attention_masks.to(device)
    labels = labels.to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_val)
    optimizer.step()
    scheduler.step()