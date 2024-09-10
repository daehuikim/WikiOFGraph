from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

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
        prompts_dict = [{"role": "user", "content": prompt} for prompt in prompts]
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
    input_data_path = "YOUR_INPUT_PATH"
    save_path = "YOUR_SAVE_PATH"
    intended_number = 100_000

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

    with open(input_data_path, 'r') as f:
        text_data = f.readlines()

    graph_extractor = GraphExtractor(
        model_name="casperhansen/llama-3-70b-instruct-awq", 
        quantization="AWQ",
        dtype="auto",
        trust_remote_code="True",
        tensor_parallel_size=4,
        temperature=0.5,
        top_p=0.9,
        max_tokens=256
    )

    generated_texts = graph_extractor.generate_triplets(text_data, intended_number, instruction)
    graph_extractor.save_results(generated_texts, save_path)

if __name__ == '__main__':
    main()
