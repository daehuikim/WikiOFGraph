from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    GenerationConfig
)
import json
import argparse

def define_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--test_file", type=str, default="")
    parser.add_argument("--save_path", type=str, default="")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = define_args()
    model = args.model
    device = args.device
    test_file = args.test_file
    save_path = args.save_path
    end_num = args.end_num
    start_num = args.start_num
        
    _model = T5ForConditionalGeneration.from_pretrained(
        model,
        local_files_only=True
    )
    _model.to(device)
    tokenizer_path = model+"/tokenizer"
    tokenizer = T5Tokenizer.from_pretrained(
        tokenizer_path, 
        trust_remote_code=True, 
        TOKENIZERS_PARALLELISM=True,
        local_files_only=True,
        skip_special_tokens=True
    )
    tokenizer.padding_side = "right"
    
    generation_config=GenerationConfig(
        num_beams=5,
        use_cache=True,
        length_penalty=1.0,
        temperature=1.0,
        decoder_start_token_id=0,
        eos_token_id=1,
        pad_token_id=0
    )
    
    result_lines=[]
    with open(test_file, "r", encoding="utf-8") as file:
        for l in file:
            line = json.loads(l)
            inputtext = line.get("triplet", "")
            inputtext = "Graph to Text:" + inputtext
            tokens = tokenizer(inputtext, return_tensors ='pt').input_ids.cuda("cuda:0")
            result = _model.generate(
                input_ids=tokens,
                max_length=512,
                generation_config=generation_config 
            )
            validated_list = result[0].tolist()
            temp_output = tokenizer.decode(validated_list,skip_special_tokens=True)
            result_lines.append("sample result is\n"+temp_output)
            
    with open(save_path, "w", encoding="utf-8") as f:
        for line in result_lines:
            f.write(line+"\n")
    print("file saved at",save_path)
