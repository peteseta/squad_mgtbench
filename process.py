from dotenv import load_dotenv
import csv
import json
import os
from tqdm import tqdm

import tiktoken
import openai
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# encoding
enc = tiktoken.get_encoding("cl100k_base")

# https://platform.openai.com/docs/models/model-endpoint-compatibility
GPT_35_MODEL_CODE = "gpt-3.5-turbo-0613"
GPT_4_MODEL_CODE = "gpt-4-0613"

total_spending = 0


def call_openai(question, model, system_prompt="You are a helpful assistant."):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message["content"].replace('\n', ' ')

def call_anthropic(question):
    anthropic = Anthropic()
    completion = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=500,
        prompt=f"{HUMAN_PROMPT} {question} {AI_PROMPT}",
    )
    return completion.completion.replace('\n', ' ')

def generate_prompt(question, context):
    prompt = (
        f"I will provide a passage and a question to you. The answer should be extracted from the context."
        f"You need to return me your answer. The passage is '{context}' and the question is '{question}'."
        f"Now, please answer the question."
    )
    return prompt

def cost_per_qna(prompt, gpt_35_response, gpt_4_response):
    global total_spending
    
    prompt_tokens = len(enc.encode(prompt))
    gpt_35_response_tokens = len(enc.encode(gpt_35_response))
    gpt_4_response_tokens = len(enc.encode(gpt_4_response))
    
    # GPT-4: $0.03 / 1K tokens input, $0.06 / 1K tokens output
    # GPT-3.5: $0.0015 / 1K tokens	$0.002 / 1K tokens
    # USD to THB conversion rate: 1 USD = 35 THB
    
    gpt_4_cost = (prompt_tokens * 0.03 + gpt_4_response_tokens * 0.06) / 1000
    gpt_35_cost = 2*((prompt_tokens * 0.0015 + gpt_35_response_tokens * 0.002) / 1000)
    total_cost = (gpt_4_cost + gpt_35_cost)*35
    
    total_spending += total_cost
    return total_cost, total_spending
    
def process(input_data):
    output_data = []
    
    for topic_index, entry in enumerate(tqdm(input_data, desc="Topics")):  # To loop over each topic
        for paragraph_index, item in enumerate(tqdm(entry["paragraphs"], desc=f"Paragraphs for {entry['title']}")):  # loop over each paragraph in a topic
            context = item["context"]
            
            # Filtering out 'impossible' questions.
            valid_questions = [q for q in item["qas"] if not q.get('is_impossible', False)]
            print(f"{len(valid_questions)} questions in this paragraph.")
            
            for qna_index, qna in enumerate(tqdm(valid_questions, desc="Questions")): # loop over each q&a in the paragraph
                id_ = qna["id"]
                question = qna["question"]
                answer_text = [ans["text"] for ans in qna["answers"]]
                answer_start = [ans["answer_start"] for ans in qna["answers"]]
                answers = {"text": answer_text, "answer_start": answer_start}
                
                llm_question = generate_prompt(question, context)
                
                # print(f"TOPIC {topic_index+1} OF {len(input_data)}")
                # print(f"PARAGRAPH {paragraph_index+1} OF {len(entry['paragraphs'])}")
                # print(f"QUESTION {qna_index+1} OF {len(valid_questions)}")
                tqdm.write(f"{question}")

                gpt35_answer = call_openai(llm_question, GPT_35_MODEL_CODE)
                tqdm.write(f"gpt-3.5: {gpt35_answer}")

                gpt35_prompted_answer = call_openai(
                    llm_question,
                    GPT_35_MODEL_CODE,
                    system_prompt="You are a genius student who excels at comprehending texts and answering questions correctly and concisely. You write with extremely high perplexity and burstiness: Perplexity is related to how predictible each subsequent word in the text is, while burstiness refers to variations between sentences. Humans tend to write with greater burstiness, for example, with some longer or complex sentences alongside shorter ones, while AI sentences tend to be more uniform. Keeping in mind these concepts, use fluent and natural human-like language, with variable sentence/paragraph lengths and occansional unconventional/complex sentence structures.",
                )
                tqdm.write(f"gpt-3.5-prompted: {gpt35_prompted_answer}")

                gpt4_answer = call_openai(llm_question, GPT_4_MODEL_CODE)
                tqdm.write(f"gpt-4: {gpt4_answer}")

                claude_answer = call_anthropic(llm_question)
                tqdm.write(f"claude: {claude_answer}")
                
                qna_cost, acc_cost = cost_per_qna(llm_question, gpt35_answer, gpt4_answer)
                tqdm.write(f"qna cost: {qna_cost}THB | total cost: {acc_cost}THB")
                tqdm.write("-------------------------------------------------------------------------")
                
                # Append this processed QnA to the output data
                output_data.append({
                    "id": id_,
                    "title": entry["title"],
                    "context": context,
                    "question": question,
                    "answers": answers,
                    "Question": llm_question,
                    "gpt35_answer": gpt35_answer,
                    "gpt35_prompted_answer": gpt35_prompted_answer,
                    "gpt4_answer": gpt4_answer,
                    "claude_answer": claude_answer
                })

                # Write this QnA to the CSV file
                with open("SQuAD_2.csv", 'a', newline='', encoding='utf-8') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow([id_, entry["title"], context, question, answers, llm_question, gpt35_answer, gpt35_prompted_answer, gpt4_answer, claude_answer])

# Assuming the JSON data from SQuAD2.0 is stored in a variable named 'data'
with open("train-v2.0.json", "r") as file:
    raw_data = json.load(file)
    data = raw_data['data']

write_header = False
if write_header:
    with open("SQuAD_2.csv", 'a', newline='', encoding='utf-8') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(["id", "title", "context", "question", "answers", "Question", "ChatGPT-0613_answer", "ChatGPT-0613-prompted_answer", "GPT4_answer", "Claude2_answer"])
    
process(data)