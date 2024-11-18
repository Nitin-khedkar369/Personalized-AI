# # Fine-Tuning Pretrained Models:
# from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
#
# # Load a pretrained model and tokenizer
# model_name = "llama3.1:8b"
# model = AutoModelForCausalLM.from_pretrained(model_name)
# print(model)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# print(tokenizer)

#
# # Tokenize the dataset
# def tokenize_function(examples):
#     return tokenizer(examples['text'], padding='max_length', truncation=True)
#
# # Set up training arguments
# training_args = TrainingArguments(
#     output_dir="./results",
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=4,
#     num_train_epochs=3,
#     weight_decay=0.01,
# )
#
# # Initialize the Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     tokenizer=tokenizer,
# )
#
# # Train the model
# trainer.train()
#
#
# from transformers import get_linear_schedule_with_warmup
# import torch
#
#
#
# # Example scheduler
# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
# scheduler = get_linear_schedule_with_warmup(
#     optimizer, num_warmup_steps=0, num_training_steps=total_steps
# )
#
# gradient_accumulation_steps = 4
#
#
# from torch.cuda.amp import autocast, GradScaler
#
# scaler = GradScaler()
#
# for batch in train_dataloader:
#     with autocast():
#         outputs = model(**batch)
#         loss = outputs.loss
#     scaler.scale(loss).backward()
#     scaler.step(optimizer)
#     scaler.update()
#     optimizer.zero_grad()
