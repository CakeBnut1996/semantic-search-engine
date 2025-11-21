from src.semantic_search_engine import retrieve_data,  _collection, _load_gemini_client
import json, os
import pandas as pd
import time

ROOT = r"C:\Users\mmh\Documents\data science kdf"

OUTPUT_JSON = os.path.join(ROOT, "eval", "chunk_training_examples.json")
gold_dataset = pd.read_json(OUTPUT_JSON)
eval_pos_records = []
eval_neg_records = []

for _, record in gold_dataset.iterrows():

    gold_ds = record["dataset_id"]
    gold_chunk = record["chunk_index"]
    gold_chunk_text = record["chunk_text"]
    # -----------------------------------------
    # (1) POSITIVE QUESTIONS
    # -----------------------------------------
    for qa in record["examples"]["positives"]:

        q = qa["question"]
        start = time.perf_counter()
        rankings = retrieve_data(q, sim_thre=-1)  # no cutoff for evaluation
        end = time.perf_counter()
        retrieval_time = end - start  # seconds

        # Extract retrieved dataset IDs in ranked order
        if rankings:
            retrieved_datasets = [r["dataset_id"] for r in rankings]   # list: ds_id, ordered
        else:
            retrieved_datasets = []

        # ---------- Compute Metrics ----------
        # Dataset rank
        if gold_ds in retrieved_datasets:
            ds_rank = retrieved_datasets.index(gold_ds) + 1
            ds_mrr = 1 / ds_rank
        else:
            ds_rank = None
            ds_mrr = 0

        ds_hit1 = (ds_rank == 1)
        ds_hit3 = (ds_rank is not None and ds_rank <= 3)
        ds_hit5 = (ds_rank is not None and ds_rank <= 5)

        # Flatten chunks: list of (sim, chunk_text)
        flat_chunks = []
        for r in rankings:
            for sim, chunk_text in r["top_chunks"]:
                flat_chunks.append(chunk_text)

        # Chunk rank
        if gold_chunk_text in flat_chunks:
            ch_rank = flat_chunks.index(gold_chunk_text) + 1
            ch_mrr = 1 / ch_rank
        else:
            ch_rank = None
            ch_mrr = 0

        ch_hit1 = (ch_rank == 1)
        ch_hit3 = (ch_rank is not None and ch_rank <= 3)
        ch_hit5 = (ch_rank is not None and ch_rank <= 5)

        # Store detailed evaluation record (optional)
        eval_pos_records.append({
            "dataset_id": gold_ds,
            "chunk_id": gold_chunk,
            "question_type": "positive",
            "question": q,

            # Dataset-level
            "ds_rank": ds_rank,
            "ds_mrr": ds_mrr,
            "ds_hit3": ds_hit3,

            # Chunk-level
            "chunk_rank": ch_rank,
            "chunk_mrr": ch_mrr,
            "chunk_hit3": ch_hit3,

            "retrieval_time_sec": retrieval_time

        })
    # -----------------------------------------
    # (2) HARD-NEGATIVE QUESTIONS
    # -----------------------------------------
    for hn in record["examples"]["hard_negatives"]:

        q = hn["question"]

        start = time.perf_counter()
        rankings = retrieve_data(q, sim_thre=-1)
        end = time.perf_counter()

        retrieval_time = end - start

        retrieved_datasets = [r["dataset_id"] for r in rankings] if rankings else []

        # NEGATIVE METRICS
        gold_in_results = (gold_ds in retrieved_datasets)

        # False positive: model retrieved the gold dataset when it should NOT
        false_positive = 1 if gold_in_results else 0

        # True negative: model successfully avoided retrieving gold dataset
        true_negative = 1 - false_positive

        # NegRecall@3 (aka hard-negative recall)
        neg_recall3 = 1 if (not gold_in_results or retrieved_datasets.index(gold_ds) >= 3) else 0

        # Negative MRR
        if gold_in_results:
            rank = retrieved_datasets.index(gold_ds) + 1
            neg_mrr = 1 / rank
        else:
            neg_mrr = 0

        eval_neg_records.append({
            "dataset_id": gold_ds,
            "chunk_id": gold_chunk,
            "question_type": "negative",
            "question": q,

            # Negative metrics
            "false_positive": false_positive,
            "true_negative": true_negative,
            "neg_recall3": neg_recall3,
            "neg_mrr": neg_mrr,

            # Speed
            "retrieval_time_sec": retrieval_time
        })

eval_pos_records = pd.DataFrame(eval_pos_records)
eval_neg_records = pd.DataFrame(eval_neg_records)
