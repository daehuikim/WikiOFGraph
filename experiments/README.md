# Evaluation
We provide training, inference and evaluation codes used in the paper.


## Trainning

installation 
```
pip install transformers, deepspeed
```

Our training process utilizes DeepSpeed to optimize performance. You can train the model using the following DeepSpeed command:

(Modify arguments according to your needs)

```
deepspeed --num_gpus=NUM_GPU t5_trainer.py \
    --dataset_train YOUR_TRAIN_DATA_PATH \
    --dataset_valid YOUR_DEV_DATA_PATH \
    --new_model "YOUR_NEW_MODEL" \
    --output_dir ./results-YOUR_NEW_MODEL\
    --num_train_epochs NUM_EPOCH \
    --per_device_train_batch_size TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size EVAL_BATCH_SIZE \
    --gradient_accumulation_steps GRAD_ACCUM \
    --learning_rate LR \
    --save_steps SAVE_STEPS \
    --logging_steps LOGGING_STEPS \
    --lr_scheduler_type SCHEDULER_TYPE \
    --model_name "t5-large" \
    --deepspeed deepspeed_config.json
```


## Inference

You can inference trainned model through python run.

(Modify arguments according to your needs)
```
python t5_inference.py \
    --model YOUR_MODEL \
    --test_file YOUR_TEST_FILE \
    --save_path YOUR_SAVE_PATH 
```

## Evaluation metrics

We used [GenerationEval](https://github.com/WebNLG/GenerationEval) for BLEU, Chr++, TER, METEOR, BLEURT.

We used [evaluate.rouge](https://huggingface.co/spaces/evaluate-metric/rouge) for ROUGE-L and bert-score from [official repository](https://github.com/Tiiiger/bert_score).

## Results

Every inferenced results are included in ```results``` directory.

```
data_questeval_test # Results of Data-QuestEval impact (Table5)
genwiki_test # Results of Genwiki test file (Table2)
wikiofgraph_test # Results of WikiOFGraph test file (Table3)
```
Source for evaluation and reproduce are provided in ```source``` directory.

```c#_n@g-genwiki.txt``` means the ratio of "curated" samples and "noise" samples are #:@.