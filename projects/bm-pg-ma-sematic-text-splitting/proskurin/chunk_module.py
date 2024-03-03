import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def split_to_chunks(min, max, text):
    marked = [0] * text.__len__()
    embeddings = model.encode(text, device='cpu',
                              # да я лох без gpu и что
                              normalize_embeddings=True)
    res_list = []
    for i in range(len(text)):
        res_list_temp = []
        if marked[i] == 1:
            continue
        marked[i] = 1
        res_list_temp.append(i)
        curr_text_size = len(text[i].split(" "))
        flag = True
        while flag:
            max_pos = -1
            curr_max = 0
            # 1 2 3 4 5 6
            # 1 0 0 1 0 0
            for j in range(i + 1, len(text)):
                if marked[j] == 1 or len(text[j].split(" ")) + curr_text_size > max:
                    continue
                if abs(cosine_similarity(embeddings[j], embeddings[i])) > curr_max:
                    max_pos = j
                    curr_max = abs(cosine_similarity(embeddings[j], embeddings[i]))

            if max_pos != -1:
                res_list_temp.append(max_pos)
                marked[max_pos] = 1
                curr_text_size = curr_text_size + len(text[max_pos].split(" "))
            else:
                if curr_text_size < min:
                    for result in range(len(res_list)):
                        marked[result] = 0
                else:
                    res_list.append(res_list_temp)

                curr_text_size = 0
                flag = False

    return res_list
