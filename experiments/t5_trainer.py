from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    AutoTokenizer,
    logging,
    GenerationConfig
)
import argparse
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os
import numpy as np
from accelerate import Accelerator
import random
import torch

# If logging bothers, you can disable it
logging.set_verbosity_info()
logging.disable_progress_bar()

def set_deterministic_training(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def split_tokens(sentence):
    return sentence.split(' ')

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

def compute_metrics_bleu(pred):
    references = pred.label_ids
    generated_texts = pred.predictions
    
    bleu_scores = []
    for reference, generated in zip(references, generated_texts):
        reference = np.where(reference != -100, reference, tokenizer.pad_token_id)
        reference_text = tokenizer.decode(reference, skip_special_tokens=False)
        refs_tokenized = [split_tokens(reference_text.strip())]

        generated = np.where(generated != -100, generated, tokenizer.pad_token_id)
        generated_text = tokenizer.decode(generated, skip_special_tokens=False)
        cands_tokenized = split_tokens(generated_text.strip())
        
        bleu_score = sentence_bleu(refs_tokenized, cands_tokenized,smoothing_function=SmoothingFunction().method1)
        bleu_scores.append(bleu_score)

    return {
        'bleu': sum(bleu_scores) / len(bleu_scores)
    }

def g2t_preprocess_function(examples):
    inputs = ['Graph to Text:' + doc for doc in examples["triplet"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True,padding='max_length')

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["text"], max_length=512, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def define_args():
    parser = argparse.ArgumentParser(description="T5 Trainer")
    #Base model and dataset
    parser.add_argument("--model_name", type=str, default="t5-large")
    parser.add_argument("--dataset_train", type=str, default="")
    parser.add_argument("--dataset_valid", type=str, default="")
    #New model and output
    parser.add_argument("--new_model", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    #training arguments
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--fp16", type=bool, default=False)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    #save and logging steps
    parser.add_argument("--save_steps", type=int, default=4000)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--device_map", type=str, default="")
    parser.add_argument("--evaluation_strategy", type=str, default="steps")
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--deepspeed",type=str, default="")
    parser.add_argument("--local_rank",type=int,default="")
    parser.add_argument("--seed",type=int,default=42)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = define_args()
    #model name
    model_name = args.model_name
    new_model = args.new_model
    output_dir = args.output_dir
    #Dataset
    dataset_train = args.dataset_train
    dataset_valid = args.dataset_valid  
    #Training arguments
    num_train_epochs = args.num_train_epochs
    fp16 = args.fp16
    per_device_train_batch_size = args.per_device_train_batch_size
    per_device_eval_batch_size = args.per_device_eval_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    gradient_checkpointing = args.gradient_checkpointing
    max_grad_norm = args.max_grad_norm
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    lr_scheduler_type = args.lr_scheduler_type
    # kwargs
    save_steps = args.save_steps
    logging_steps = args.logging_steps
    device_map = args.device_map
    evaluation_strategy = args.evaluation_strategy
    save_total_limit = args.save_total_limit
    deepspeed=args.deepspeed
    seed=args.seed

    set_deterministic_training(seed)
    # Load model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map={"": Accelerator().local_process_index}
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True, 
        TOKENIZERS_PARALLELISM=True,
        )
    tokenizer.padding_side = "right"

    # Load datasets
    train_dataset = load_dataset('json', data_files=dataset_train, split="train")
    dev_dataset = load_dataset('json', data_files=dataset_valid, split="train")
    tokenized_train_dataset = train_dataset.map(g2t_preprocess_function, batched=True)
    tokenized_dev_dataset = dev_dataset.map(g2t_preprocess_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Set generation parameters
    generation_config=GenerationConfig(
        num_beams=5,
        use_cache=True,
        length_penalty=1.0,
        temperature=1.0,
        decoder_start_token_id=0,
        eos_token_id=1,
        pad_token_id=0
    )
    # Set training parameters
    training_arguments = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy=evaluation_strategy,
        save_strategy=evaluation_strategy,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        num_train_epochs=num_train_epochs,
        lr_scheduler_type=lr_scheduler_type,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=save_steps,
        save_total_limit=save_total_limit,
        run_name=new_model,
        fp16=fp16,
        bf16=True,
        report_to="wandb",
        deepspeed=deepspeed,
        load_best_model_at_end=True,
        warmup_steps=2000,
        generation_config=generation_config
    )
    
    # Set supervised fine-tuning parameters
    trainer = Seq2SeqTrainer(
        model=model,
        data_collator = data_collator,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_dev_dataset, 
        tokenizer=tokenizer,
        args=training_arguments,
        compute_metrics=compute_metrics_bleu,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    trainer.train()
    trainer.model.save_pretrained(new_model)
