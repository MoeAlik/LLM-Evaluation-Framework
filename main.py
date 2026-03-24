import torch
import re
import csv

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# TODO: throw into config file
HUGGINGFACE_MODEL = "Qwen/Qwen2.5-Math-1.5B-Instruct"
TASKS_SELECTED = 200
CSV_FILE_OUTPUT_BATCH1 = "out_batch_1.csv"
CSV_FILE_OUTPUT_BATCH2 = "out_batch_2.csv"
CSV_FILE_OUTPUT_BATCH3 = "out_batch_3.csv"
CSV_FILE_OUTPUT_BATCH4 = "out_batch_4.csv"

tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    HUGGINGFACE_MODEL,
    dtype=torch.float16,
    device_map="auto"
)
dataset = load_dataset("cais/mmlu", "elementary_mathematics", split="test").select(range(TASKS_SELECTED))

headers = ["Input","answer1","answer2","answer3","answer4","Model Output"]
all_respones = []

incomplete_responses = []
multiple_responses = []
incorrect_responses = []

def convert_offset_to_letter(offset):
    return chr(ord('A') + offset)

def format_question(example):
    question = example["question"]
    choices = example["choices"]

    formatted_question = ""

    formatted_question += f"{question}<br>"
    for i, choice in enumerate(choices):
        formatted_question += f"{convert_offset_to_letter(i)}, {choice}<br>"

    return formatted_question

def validate_settings():
    assert torch.cuda.is_available()

def run_model():
    prompts = [format_question(ex) for ex in dataset]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side='left').to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    pattern = r"boxed\{(.)\}"

    for prompt, formatted_prompts, resp in zip(dataset, prompts, decoded):
        matches = re.findall(pattern, resp)
        offset = int(prompt["answer"]) # pyright: ignore[reportArgumentType, reportCallIssue]

        choices = list(prompt["choices"]) # pyright: ignore[reportArgumentType, reportCallIssue]
        question = str(prompt["question"]) # pyright: ignore[reportArgumentType, reportCallIssue]

        print(resp)

        resp = resp.replace('\r\n', '<br>')
        resp = resp.replace('\r', '<br>')
        resp = resp.replace('\n', '<br>')

        all_respones.append(
            [question, *choices, resp]
        )

        if not matches:
            incomplete_responses.append(resp)
        elif (len(set(matches)) != 1):
            multiple_responses.append(resp)
        elif matches[0] != convert_offset_to_letter(offset):
            incorrect_responses.append(resp)

def print_stats():
    num_incomplete = len(incomplete_responses)
    num_multiple = len(multiple_responses)
    num_incorrect = len(incorrect_responses)
    num_correct = TASKS_SELECTED - num_incorrect - num_incomplete - num_multiple

    print(f"Number of incomplete: {num_incomplete}")
    print(f"Number of multiple: {num_multiple}")
    print(f"Number of incorrect: {num_incorrect}")
    print(f"Number of correct: {num_correct}")

    print(f"Accurary is {num_correct / TASKS_SELECTED}")

def generate_csv():

    with open(CSV_FILE_OUTPUT_BATCH1, mode="w", newline="", encoding='utf-8') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        for row in [headers] + all_respones:
            writer.writerow(row)

    # with open(CSV_FILE_OUTPUT_BATCH2, mode="w", newline="", encoding='utf-8') as file:
    #     writer = csv.writer(file, quoting=csv.QUOTE_ALL)
    #     for row in [headers] + all_respones[50:100]:
    #         writer.writerow(row)

    # with open(CSV_FILE_OUTPUT_BATCH3, mode="w", newline="", encoding='utf-8') as file:
    #     writer = csv.writer(file, quoting=csv.QUOTE_ALL)
    #     for row in [headers] + all_respones[100:150]:
    #         writer.writerow(row)

    # with open(CSV_FILE_OUTPUT_BATCH4, mode="w", newline="", encoding='utf-8') as file:
    #     writer = csv.writer(file, quoting=csv.QUOTE_ALL)
    #     for row in [headers] + all_respones[150:]:
    #         writer.writerow(row)

def main():
    run_model()
    print_stats()
    generate_csv()

if __name__ == '__main__':
    main()
