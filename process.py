from dotenv import load_dotenv
import csv
import json

from openai import ChatCompletion
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

load_dotenv()

# https://platform.openai.com/docs/models/model-endpoint-compatibility
GPT_35_MODEL_CODE = "gpt-3.5-turbo-0613"
GPT_4_MODEL_CODE = "gpt-4-0613"


def call_openai(question, context, model, system_prompt="You are a helpful assistant."):
    response = ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": generate_prompt(question, context)},
        ],
    )
    return response.choices[0].message["content"]


def call_anthropic(question, context):
    anthropic = Anthropic()
    completion = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=500,
        prompt=f"{HUMAN_PROMPT} {generate_prompt(question, context)} {AI_PROMPT}",
    )
    return completion.completion

def call_hf(question, context):
    # Implementation of the HuggingFace LLaMa2 API call here
    return "LLaMa2 answer here"

def generate_prompt(question, context):
    prompt = (
        f"I will provide a passage and a question to you. The answer should be extracted from the context."
        f"You need to return me your answer. The passage is '{context}' and the  question is '{question}'."
        f"Now, please answer the question."
    )
    return prompt

def rephrase(answer):
    return answer

def process(input_data):
    output_data = []
    for item in input_data:
        context = item["context"]
        for qna in item["questions"]:
            id_ = qna["id"]
            question = qna["question"]
            answer_text = [ans["text"] for ans in qna["answers"]]
            answer_start = [ans["answer_start"] for ans in qna["answers"]]

            answers = {"text": answer_text, "answer_start": answer_start}

            gpt35_answer = call_openai(question, context, GPT_35_MODEL_CODE)
            print(f"gpt-3.5: {gpt35_answer}")

            gpt35_prompted_answer = call_openai(
                question,
                context,
                GPT_35_MODEL_CODE,
                system_prompt="You are a helpful assistant. Please answer the question.",
            )
            print(f"gpt-3.5-prompted: {gpt35_prompted_answer}")

            gpt35_rephrased_answer = rephrase(
                call_openai(question, context, GPT_35_MODEL_CODE)
            )
            print(f"gpt-3.5-rephrased: {gpt35_rephrased_answer}")

            gpt4_answer = call_openai(question, context, GPT_4_MODEL_CODE)
            print(f"gpt-4: {gpt4_answer}")

            claude_answer = call_anthropic(question, context)
            print(f"claude: {claude_answer}")

            llama2_answer = call_hf(question, context)
            print(f"llama: {llama2_answer}")

            # Store the results in the desired format
            row = [
                id_,
                "SQuAD2.0",
                context,
                question,
                json.dumps(answers),
                gpt35_answer,
                gpt4_answer,
                claude_answer,
                llama2_answer,
                gpt35_prompted_answer,
                gpt35_rephrased_answer,
            ]
            output_data.append(row)

    # write to CSV
    with open("SQuAD_2.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "id",
                "title",
                "context",
                "question",
                "answers",
                "ChatGPT-0613_answer",
                "GPT4_answer",
                "Claude2_answer",
                "LLaMa2_answer",
                "ChatGPT-0613-prompted_answer",
                "ChatGPT-0603-rephrased_answer",
            ]
        )
        writer.writerows(output_data)


# Assuming the JSON data from SQuAD2.0 is stored in a variable named 'data'
data = json.loads("train-v2.0.json")
process(data)
