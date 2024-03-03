from scipy.spatial.distance import cosine
import numpy as np
from sentence_transformers import SentenceTransformer

from common.utils import split_to_sentences

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')


def split_to_chunks(single_sentences_list,
                    splitting_to_subchunks=False,
                    min_chunk_size=None,
                    max_chunk_size=None,
                    max_percentage_splitting=95,
                    min_percentage_splitting=80):
    if max_chunk_size is None and splitting_to_subchunks:
        raise ValueError('Unexpected args: max_chunk_size is None on splitting_to_subchunks is True')

    sentences = [{'sentence': x, 'index': i} for i, x in enumerate(single_sentences_list)]
    sentences = combine_sentences(sentences)

    embeddings = model.encode([x['combined_sentence'] for x in sentences],
                              device='cuda',
                              normalize_embeddings=True)

    for i, sentence in enumerate(sentences):
        sentence['combined_sentence_embedding'] = embeddings[i]

    distances, sentences = calculate_cosine_distances(sentences)

    breakpoint_percentile_threshold = max_percentage_splitting

    chunks = None
    if not splitting_to_subchunks:
        indices_above_thresh = get_threshold_indices(distances, breakpoint_percentile_threshold)
        chunks = merge_sentences_to_chunks(indices_above_thresh, sentences)

        # if min_chunk_size is not None:
        #     chunks = merge_iterative_smaller_chunks(chunks, min_chunk_size)

        if max_chunk_size is not None:
            large_chunks_idx = test_largest_chunks(chunks, max_chunk_size)
            if len(large_chunks_idx) > 0:
                large_chunks = [chunks[i] for i in large_chunks_idx]
                subchunk_list = []
                for i, chunk in enumerate(large_chunks):
                    large_chunk_sentences = split_to_sentences(chunk['combined_sentence'])
                    subchunks = split_to_chunks(large_chunk_sentences,
                                                splitting_to_subchunks=True,
                                                min_chunk_size=min_chunk_size,
                                                max_chunk_size=max_chunk_size,
                                                max_percentage_splitting=max_percentage_splitting,
                                                min_percentage_splitting=min_percentage_splitting)

                    subchunk_list += subchunks

                chunks = delete_by_idx_array(chunks, large_chunks_idx)
                chunks += subchunk_list
    else:
        splitting = True
        while splitting:
            indices_above_thresh = get_threshold_indices(distances, breakpoint_percentile_threshold)
            chunks = merge_sentences_to_chunks(indices_above_thresh, sentences)

            if len(test_largest_chunks(chunks, max_chunk_size)) > 0:
                breakpoint_percentile_threshold -= 1
                if breakpoint_percentile_threshold < min_percentage_splitting:
                    break
            else:
                splitting = False

        if min_chunk_size is not None:
            avg_distance = np.average(np.array(distances)) * 1.2
            chunks = merge_iterative_smaller_chunks(chunks,
                                                    min_chunk_size,
                                                    max_chunk_size,
                                                    avg_distance)

    return chunks


def merge_iterative_smaller_chunks(chunks, min_chunk_size, max_chunk_size=None, avg_distance=None):
    while test_smallest_chunks(chunks, min_chunk_size):
        new_chunks = combine_chunks(chunks, min_chunk_size, max_chunk_size, avg_distance)
        if len(new_chunks) == len(chunks):
            break
        chunks = new_chunks
    return chunks


def get_threshold_indices(distances, breakpoint_percentile_threshold):
    breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)
    indices_above_thresh = [i for i, x in enumerate(distances) if
                            x > breakpoint_distance_threshold]
    return indices_above_thresh


def merge_sentences_to_chunks(indices_above_thresh, sentences):
    start_index = 0
    chunks = []
    for index in indices_above_thresh:
        end_index = index

        group = sentences[start_index:end_index + 1]
        combined_text = ' '.join([d['sentence'] for d in group])
        chunks.append({"combined_sentence": combined_text,
                       "combined_sentence_embedding": [d["combined_sentence_embedding"] for d in group]})

        start_index = index + 1
    if start_index < len(sentences):
        combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
        chunks.append({"combined_sentence": combined_text,
                       "combined_sentence_embedding": [d["combined_sentence_embedding"] for d in
                                                       sentences[start_index:]]})
    return chunks


def combine_sentences(sentences, buffer_size=1):
    for i in range(len(sentences)):
        combined_sentence = ''

        for j in range(i - buffer_size, i):
            if j >= 0:
                combined_sentence += sentences[j]['sentence'] + ' '
        combined_sentence += sentences[i]['sentence']

        for j in range(i + 1, i + 1 + buffer_size):
            if j < len(sentences):
                combined_sentence += ' ' + sentences[j]['sentence']

        sentences[i]['combined_sentence'] = combined_sentence

    return sentences


def calculate_cosine_distances(sentences):
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]['combined_sentence_embedding']
        embedding_next = sentences[i + 1]['combined_sentence_embedding']
        distance = cosine(embedding_current, embedding_next)

        distances.append(distance)

        sentences[i]['distance_to_next'] = distance

    return distances, sentences


def combine_chunks(chunks, min_length, max_length=None, avg_distance=None):
    if (max_length is None) != (avg_distance is None):
        raise ValueError('Must provide either max_length or avg_distance')

    combined_chunks = []
    chunks = chunks.copy()
    idx_to_delete = set()

    for i, chunk in enumerate(chunks):
        chunk_text_size = len(chunk['combined_sentence'])
        if chunk_text_size < min_length:
            closest_distance = float('inf')
            closest_chunk_index = None

            for j, other_chunk in enumerate(chunks):
                if i != j:
                    embs = chunk['combined_sentence_embedding']
                    embs_other = other_chunk['combined_sentence_embedding']
                    chunk_embedding = np.mean(np.array(embs), axis=0)
                    other_chunk_embedding = np.mean(np.array(embs_other), axis=0)

                    distance = cosine(chunk_embedding, other_chunk_embedding)

                    other_chunk_text_size = len(other_chunk['combined_sentence'])
                    if distance < closest_distance \
                            and ((max_length is None and avg_distance is None)
                                 or (other_chunk_text_size + chunk_text_size <= max_length
                                     and distance <= avg_distance)):
                        closest_distance = distance
                        closest_chunk_index = j

            if closest_chunk_index is not None:
                combined_sentence = chunk['combined_sentence'] + chunks[closest_chunk_index]['combined_sentence']
                combined_embedding = chunk['combined_sentence_embedding'] + chunks[closest_chunk_index][
                    'combined_sentence_embedding']
                combined_chunk = {'combined_sentence': combined_sentence,
                                  'combined_sentence_embedding': combined_embedding}

                idx_to_delete.add(closest_chunk_index)
                idx_to_delete.add(i)

                combined_chunks.append(combined_chunk)

    chunks = delete_by_idx_array(chunks, idx_to_delete)
    return chunks + combined_chunks


def delete_by_idx_array(collection, idx_to_delete):
    collection = [e[1] for e in enumerate(collection) if e[0] not in idx_to_delete]
    return collection


def test_smallest_chunks(new_chunks, min_len):
    if new_chunks is None:
        return True

    flag = False
    for chunk in new_chunks:
        val = len(chunk['combined_sentence']) < min_len
        flag |= val
    return flag


def test_largest_chunks(new_chunks, max_len):
    if new_chunks is None:
        return list()

    large_chunks = list()
    for i, chunk in enumerate(new_chunks):
        if len(chunk['combined_sentence']) > max_len:
            large_chunks.append(i)

    return large_chunks
