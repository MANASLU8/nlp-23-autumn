import os

from ctransformers import AutoModelForCausalLM

from db import DB, EmbeddingFunction, OldEmbeddingFunction

llm = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-OpenOrca-GGUF",
                                           model_file="mistral-7b-openorca.Q4_K_S.gguf",
                                           model_type="mistral",
                                           gpu_layers=50,
                                           max_new_tokens=1000,
                                           context_length=4000)


def create_prompt_with_context(context, question):
    prompt = "Context: " + context + "\n\n"
    prompt += "Question: " + question + "\n\n"
    prompt += "Answer the question using only provided context. Your answer should be no longer than 50 words.\n\n"

    return prompt


def process_request(question, db):
    context = "\n".join(db.query(question, 5)["documents"][0])
    context2 = "\n===\n".join(db.query(question, 5)["documents"][0])

    llm_prompt = create_prompt_with_context(context, question)
    result_prompt = create_prompt_with_context(context2, question)
    answer = llm(llm_prompt)
    return f"{result_prompt}{answer}\n\n{'=' * 50}\n\n"


def batch_processing(questions, db):
    res_list = []
    for i, question in enumerate(questions):
        print(f"processing {i}th question from {len(questions)}")
        response = process_request(question, db)
        res_list.append(response)

    return res_list


def split_text_into_sentences_by_empty_line(file_path):
    with open(file_path, 'r') as file:
        text = file.read()

    sentences = text.split("\n")
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    return sentences


if __name__ == "__main__":
    dbs = {
        "lab5_old": DB("lab5_cosine", "/results/vec_db_old", OldEmbeddingFunction),
        "sem_split_proskurin": DB("sem_split_proskurin", "/results", EmbeddingFunction),
        "sem_split_bolshim": DB("sem_split_bolshim", "/results", EmbeddingFunction),
        "sem_split_bolshim_mangarakov": DB("sem_split_bolshim_mangarakov", "/results", EmbeddingFunction),
        "sem_split_proskurin_mangarakov": DB("sem_split_proskurin_mangarakov", "/results", EmbeddingFunction)
    }
    try:
        os.mkdir("/results/vec_db__bench_results2")
    except FileExistsError:
        pass

    file_path = 'questions.txt'
    sentences = split_text_into_sentences_by_empty_line(file_path)

    for dn_name, db in dbs.items():
        print("processing {}...".format(dn_name))
        result = batch_processing(sentences, db)

        with open(f"/results/vec_db__bench_results2/{dn_name}.txt", "w") as file:
            for sentence in result:
                print(sentence, file=file)
