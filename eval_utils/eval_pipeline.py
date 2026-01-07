import time
from retrieval_utils.retriever import retrieve_data
from eval_utils.metrics_retrieval import MetricCalculator


# --- PIPELINE FUNCTIONS ---

def run_retrieval_step_positive(
        qa_pair: dict,
        gold_ds_id: str,
        db_config: dict
) -> dict:
    """
    Step 1 (Positive): Retrieve data and calculate Statistical Metrics.
    """
    query = qa_pair["question"]
    gold_quote = qa_pair["context_quote"]

    # 1. Execute Retrieval
    start_t = time.perf_counter()

    # Using the functional retriever we built earlier
    retrieved_items = retrieve_data(
        query=query,
        db_path=db_config["db_path"],
        collection_name=db_config["collection_name"],
        model_name=db_config["embedding_model"],
        num_docs=10,
        chunks_per_doc=10
    )

    latency = time.perf_counter() - start_t

    # 2. Extract Context for Generation
    # We take the very first chunk as the "context" for the generator
    top_text = retrieved_items[0].chunk_text if retrieved_items else ""
    top_ds = retrieved_items[0].dataset_id if retrieved_items else "N/A"

    # 3. Calculate Retrieval Metrics (Math/Stats)
    gt_metrics = MetricCalculator.evaluate_positives(retrieved_items, gold_ds_id, gold_quote)

    return {
        "type": "positive",
        "query": query,
        "gold_ds": gold_ds_id,
        "retrieved_ds": top_ds,
        "retrieved_text_snippet": top_text,  # IMPORTANT: Passed to Step 2
        "latency": latency,
        **gt_metrics
    }


def run_generation_step_positive(
        query: str,
        context: str,
        student_generator,
        judge_model
) -> dict:
    """
    Step 2 (Positive): Generate answer and Judge Faithfulness.
    """
    # 1. Generate Answer
    gen_answer = student_generator.generate(query, context)

    # 2. AI Judge Evaluation
    faith_res = judge_model.evaluate_faithfulness(query, context, gen_answer)

    return {
        "gen_answer": gen_answer,
        "faithfulness": faith_res.score,
        "faithfulness_reason": faith_res.reasoning
    }


def run_retrieval_step_negative(
        hn_pair: dict,
        gold_ds_id: str,
        db_config: dict
) -> dict:
    """
    Step 1 (Negative): Retrieve data and check for Leakage.
    """
    query = hn_pair["question"]

    # 1. Execute Retrieval
    start_t = time.perf_counter()
    retrieved_items = retrieve_data(
        query=query,
        db_path=db_config["db_path"],
        collection_name=db_config["collection_name"],
        model_name=db_config["embedding_model"],
        num_docs=10,
        chunks_per_doc=10
    )
    latency = time.perf_counter() - start_t

    top_text = retrieved_items[0].chunk_text if retrieved_items else ""
    top_ds = retrieved_items[0].dataset_id if retrieved_items else "N/A"

    # 2. Retrieval Metrics (Negatives)
    neg_metrics = MetricCalculator.evaluate_negatives(retrieved_items)

    return {
        "type": "negative",
        "query": query,
        "gold_ds": gold_ds_id,
        "retrieved_ds": top_ds,
        "retrieved_text_snippet": top_text,
        "latency": latency,
        **neg_metrics
    }


def run_generation_step_negative(
        query: str,
        context: str,
        student_generator,
        judge_model
) -> dict:
    """
    Step 2 (Negative): Generate answer and Judge Abstention.
    """
    # 1. Generate Answer (Should be a refusal)
    gen_answer = student_generator.generate(query, context)

    # 2. AI Judge Evaluation
    abst_res = judge_model.evaluate_abstention(query, context, gen_answer)

    return {
        "gen_answer": gen_answer,
        "correct_refusal": abst_res.is_correct_refusal,
        "refusal_reason": abst_res.reasoning
    }