from openai import OpenAI
import json

# Function for formatting input data into prompt
def transfer_style(triplets):
    items = triplets.split("), ")
    st=""
    for i,item in enumerate(items):
        written = str(i+1)+"."+item+")\n"
        if item == items[-1]:
            st += written[:-2]
        else:
            st += written 
    return st

def process():
    prompt="""Your job is to validate Triplet-Text pair data for Triplet-to-Text generation task.
    Based on the provided instructions and examples, write all errors in the Triplet-Text pairs present in the given query.

    ###Triplet Composition
    Format: (<S> subject| <P> predicate| <O> object)
    A triplet is composed of three elements: subject, predicate, and object. Each of these
    elements is delineated by specific symbols with capital letters such as <S>, <P>, and <O>.
    These symbols help to clearly identify the role of each element within the triplet.
    ⚫ <S> stands for the subject, which represents the entity that is performing an action or
    being described.
    ⚫ <P> stands for the predicate, which illustrates the relationship or action that connects
    the subject to the object.
    ⚫ <O> stands for the object, which is the entity that is receiving the action or being
    described in relation to the subject.

    ###Error Types
    A.Unused Triplet
    *Example*
    <Triplet set>
    1.(<S> Ian Hornak| <P> birthPlace| <O> Philadelphia)
    2.(<S> Ian Hornak| <P> nationality| <O> American)
    3.(<S> Ian Hornak| <P> birthPlace| <O> Pennsylvania)
    <Text>
    Ian Hornak was born in Philadelphia , Pennsylvania , to parents who immigrated from Slovakia .
    <Errors>
    [Unused Triplet]:2
    [Unguessable Text]:x

    *Extra explanations*
    Because <Text> does not include information of American nationality.
    Also, If there are duplicate triplets, one of them is considered an unused triplet.
    For example, in case of [1.(<S> Alice | <P> date of birth| <O> 1997) 2.(<S> Alice| <P> date of birth| <O> 12DEC97)] and “Alice was born in December 12th in 1997” are paired, the text can be guessed with only triplet 2. Therefore 1 can be determined as unused triplet.

    B.Unguessable Text
    *Example*
    <Triplet Set>
    1. (<S> Flora Drummond| <P> sex or gender| <O> female)
    <Text>
    Nicknamed 'The General' for Flora Drummond's habit of leading Women's Rights marches wearing a military style uniform 'with an officers cap and epaulettes' and riding on a large horse, Drummond was an organiser for the Women's Social and Political Union (WSPU) andwas imprisoned nine times for her activism in the Women's Suffrage movement.
    <Errors>
    [Unused Triplet]:x
    [Unguessable text]:Nicknamed 'The General' for, habit of leading, Rights marches wearing a military style uniform 'with an officers cap and epaulettes' and riding on a large horse, Drummond was an organiser for the Women's Social and Political Union (WSPU) and was imprisoned nine times for her activism in the, Suffrage movement.

    *Extra explanations*
    Because most part is unguessable from given triplet set.
    Also, if the information in the <Reference Text> is based on facts or common knowledge but cannot be inferred from the <Triplet Set>, it is considered [Unguessable text].
    From this descriptions, please find all errors of following query pair.

    Please just fill the format below. No need to write any extra explanations.
    ###Query
    <Triplet set>
    {triplet}
    <Text>
    {text}
    <Errors>
    [Unused Triplets]:
    [Unguessable Text]:
    """

    client = OpenAI(
        api_key="YOUR_API_KEY" # Replace with your API key, Note: Do not share your API key publicly
    )
    number=5
    input_path = "YOUR_SAVE_PATH"+str(number)+".jsonl"
    prompts=[]
    with open(input_path, 'r') as f:
        for line in f:
            input_data = json.loads(line)
            triplets = input_data["triplet"]
            text = input_data["text"]
            prompts.append(prompt.format(triplet=transfer_style(triplets), text=text))
    
    validation_results = []
    for prompt in prompts:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=1,
            max_tokens=700,
        )
        validation_results.append(response.choices[0].message.content)
    
    save_path = "YOUR_SAVE_PATH"+str(number)+"_result.txt"
    with open(save_path, 'w') as f:
        for result in validation_results:
            f.write("Generated Validation Summary:\n") # Labels for differentiation
            f.write(result)
            f.write("\n")

if __name__ == "__main__":
    process()
            