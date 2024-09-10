from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import argparse

class GraphExtractor:
    def __init__(self, model_name, quantization, dtype, trust_remote_code, tensor_parallel_size, temperature, top_p, max_tokens):
        self.model_name = model_name
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        self.llm = LLM(
            model=model_name, 
            quantization=quantization,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_prompts(self, text_data, num, instruction):
        prompts = [instruction.format(text=text_data[i]) for i in range(min(len(text_data), num))]
        return prompts

    def prepare_prompts_for_chat(self, prompts):
        prompts_dict = [[{"role": "user", "content": prompt}] for prompt in prompts]
        prompts_chat_applied = [self.tokenizer.apply_chat_template(prompt_dict, tokenize=False, add_generation_prompt=True) for prompt_dict in prompts_dict]
        return prompts_chat_applied

    def generate_triplets(self, text_data, num, instruction):
        prompts = self.generate_prompts(text_data, num, instruction)
        prompts_chat_applied = self.prepare_prompts_for_chat(prompts)
        outputs = self.llm.generate(prompts_chat_applied, self.sampling_params)
        generated = ["sample result is \n" + output.outputs[0].text for output in outputs]
        return generated

    def save_results(self, generated_texts, save_path):
        with open(save_path, 'w', encoding='utf-8') as f:
            for text in generated_texts:
                f.write(text + '\n')

def main():
    parser = argparse.ArgumentParser(description="Generate triplets from text using a pretrained model.")
    
    # Add arguments for file paths, number of samples, and model parameters
    parser.add_argument('--input_data_path', type=str, required=True, help='Path to the input text file.')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the generated triplets.')
    parser.add_argument('--intended_number', type=int, required=True, help='Number of triplets to generate.')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default="casperhansen/llama-3-70b-instruct-awq", help='Model name for the LLM.')
    parser.add_argument('--quantization', type=str, default="AWQ", help='Quantization type.')
    parser.add_argument('--dtype', type=str, default="auto", help='Data type for the model.')
    parser.add_argument('--trust_remote_code', type=str, default="True", help='Whether to trust remote code for the model.')
    parser.add_argument('--tensor_parallel_size', type=int, default=4, help='Tensor parallel size for the model.')
    
    # Sampling parameters
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature for sampling.')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling parameter.')
    parser.add_argument('--max_tokens', type=int, default=256, help='Maximum number of tokens to generate.')

    args = parser.parse_args()

    instruction="""Your task is to create a set of triplets that can represent all the entities that appear in the given text.
    Triplet is consist of three parts (<S>, <P>, <O>)
    <S> means subject, <P> means predicate(relation), <O> means object
    You can not simply copy the entities from the text, you need to create a set of triplets that can represent all the entities in the text.
    For example, you cannot just use "is" or "are in <P> part, you need to find a more specific predicate that can represent the relationship between the subject and the object.
    Complete the [TRIPLET] to represent [TEXT], as shown in the Examples.
    Please just complete [TRIPLET] without saying anything else.

    ### Example:
    [TEXT]: The Acharya Institute of Technology is located in Soldevanahalli, Acharya Dr. Sarvapalli Radhakrishnan Road, Hessarghatta Main Road, Bangalore – 560090, Karnataka, India. It is affiliated with the Visvesvaraya Technological University in Belgaum.
    [TRIPLETS]: (<S> Acharya Institute of Technology| <P> affiliation| <O> Visvesvaraya Technological University), (<S> Visvesvaraya Technological University| <P> city| <O> Belgaum), (<S> Acharya Institute of Technology| <P> state| <O> Karnataka), (<S> Acharya Institute of Technology| <P> country| <O> India), (<S> Acharya Institute of Technology| <P> campus| <O> In Soldevanahalli, Acharya Dr. Sarvapalli Radhakrishnan Road, Hessarghatta Main Road, Bangalore – 560090.)

    [TEXT]: Albert Jennings Fountain was born in Staten Island in New York City and died in Dona Ana County, New Mexico.
    [TRIPLETS]: (<S> Albert Jennings Fountain| <P> death Place| <O> Doña Ana County, New Mexico), (<S> Albert Jennings Fountain| <P> birth Place| <O> New York City), (<S> Albert Jennings Fountain| <P> birth Place| <O> Staten Island)

    [TEXT]: Abilene, Texas is in Jones County in the United States. Washington, D.C. is the capital of the U.S. with New York City being the largest city. English is their native language.
    [TRIPLETS]: (<S> United States| <P> capital| <O> Washington, D.C.), (<S> Abilene, Texas| <P> is Part Of| <O> Jones County, Texas), (<S> Jones County, Texas| <P> country| <O> United States), (<S> United States| <P> largest City| <O> New York City), (<S> United States| <P> language| <O> English language)

    ### Query:
    [TEXT]: {text}
    [TRIPLET]:"""

    # Read input text file
    with open(args.input_data_path, 'r') as f:
        text_data = f.readlines()

    # Create an instance of GraphExtractor with user-provided model parameters
    graph_extractor = GraphExtractor(
        model_name=args.model_name,
        quantization=args.quantization,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )

    generated_texts = graph_extractor.generate_triplets(text_data, args.intended_number, instruction)
    graph_extractor.save_results(generated_texts, args.save_path)

if __name__ == '__main__':
    main()
