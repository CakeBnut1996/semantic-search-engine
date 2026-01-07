import importlib, yaml
import io_utils.pre_processor
import retrieval_utils.retriever
import generation_utils.generator
import generation_utils.schema
import eval_utils.eval_pipeline
import eval_utils.metrics_llm

# Force reload the modules
importlib.reload(io_utils.pre_processor)
importlib.reload(retrieval_utils.retriever)
importlib.reload(generation_utils.generator)
importlib.reload(generation_utils.schema)
importlib.reload(eval_utils.eval_pipeline)
importlib.reload(eval_utils.metrics_llm)

# Now import the specific functions/classes
from io_utils.pre_processor import run_ingestion
from retrieval_utils.retriever import retrieve_data, rank_datasets
from generation_utils.generator import StudentGenerator
from generation_utils.schema import KDFResponse, DatasetSummary
from eval_utils.eval_pipeline import run_retrieval_step_positive, run_generation_step_positive
from eval_utils.metrics_llm import UniversalJudge

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Resolve Active Selections
active_emb = cfg['embeddings'][cfg["retrieval"]['active_embedding']]
active_stu = cfg['llm'][cfg["generation"]['active_student']]
active_db = cfg['db'][cfg["retrieval"]['active_db']]
active_jd = cfg['llm'][cfg["generation"]['active_judge']]

# Config Variables
DB_PATH = cfg['data']['db_path']
DATA_DIR = cfg['data']['data_dir']
COLLECTION_NAME = active_db['collection']
EMBEDDING_MODEL = active_emb['model']

USER_QUERY = cfg['user_query']

student = StudentGenerator(
    provider=active_stu['provider'],
    model_name=active_stu['model']
)


judge = UniversalJudge(provider=active_jd['provider'], model_name=active_jd['model'])

run_ingestion(
    data_dir=DATA_DIR,
    db_path=DB_PATH,
    collection_name=COLLECTION_NAME,
    embedding_model_name=EMBEDDING_MODEL,
    chunk_size=cfg['data']['chunk_size'],
    chunk_overlap=cfg['data']['chunk_overlap']
)

retrieved_data = retrieve_data(
    query=USER_QUERY,
    db_path=DB_PATH,
    collection_name=COLLECTION_NAME,
    model_name=EMBEDDING_MODEL,
    num_docs=cfg['retrieval']['num_docs'],
    chunks_per_doc=cfg['retrieval']['chunks_per_doc']
)

ranked_data = rank_datasets(retrieved_data)

# top_dataset = ranked_data[0]
# best_context = top_dataset.top_chunks[0]['text']

context_str = str(ranked_data)

answer = student.generate(
    query=USER_QUERY,
    context=context_str,
    schema=KDFResponse
)


# eval
retrieval_res = run_retrieval_step_positive(
    qa_pair=qa_data,
    gold_ds_id="report_q3.pdf",
    db_config=DB_CONFIG
)

gen_res = run_generation_step_positive(
        query=retrieval_res["query"],
        context=retrieval_res["retrieved_text_snippet"],
        student_generator=student,
        judge_model=judge
    )


