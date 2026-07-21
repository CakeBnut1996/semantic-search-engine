import os
import yaml
import logging
from retrieval_utils.retriever_solr import SolrRetriever, SolrResult
from generation_utils.generator import StudentGenerator
from generation_utils.schema import Response

# Configure simple logging
logging.basicConfig(level=logging.WARNING)

def solr_retrieve_helper(
        query: str,
        sys_cfg: dict,
        num_docs: int
):
    """
    Helper for Solr-only retrieval logic.
    """
    solr = SolrRetriever(url=sys_cfg["SOLR_URL"])
    solr_results = solr.search(query, rows=num_docs)
    
    final_results = []
    for s_res in solr_results:
        # Map SolrResult to a format the generator expects (similar to RankedDataset)
        final_results.append({
            "dataset_id": s_res.id,
            "top_score": s_res.score,
            "source_url": s_res.url,
            "source_title": s_res.title,
            "top_chunks": [{"text": s_res.snippet or s_res.metadata.get('content', '')[:1000], "score": s_res.score}]
        })
    
    return final_results

def load_config():
    """Load system configuration from config.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_main():
    # 1. Load configuration
    cfg = load_config()
    
    active_stu = cfg['llm'][cfg["generation"]['active_student']]

    workspace_root = os.path.abspath(os.path.dirname(__file__))

    sys_cfg = {
        "SOLR_URL": cfg['retrieval'].get('solr_url', "http://solr:8983/solr/biokdf"),
        "NUM_DOCS": cfg['retrieval']['num_docs']
    }

    # 2. Initialize Student Generator
    print(f"--- Initializing {active_stu['model']} ---")
    student = StudentGenerator(
        provider=active_stu['provider'],
        model_name=active_stu['model']
    )

    print("SEED Solr Search (CLI Mode)")
    print("Type 'exit' to quit.")
    print("-" * 30)

    while True:
        try:
            query_text = input("\nEnter your search query: ").strip()
            
            if not query_text:
                continue
            if query_text.lower() in ['exit', 'quit']:
                break

            print("\n🔍 Searching Solr...")
            
            # A. Retrieval (Solr Only)
            ranked_data = solr_retrieve_helper(
                query=query_text,
                sys_cfg=sys_cfg,
                num_docs=sys_cfg["NUM_DOCS"]
            )

            if not ranked_data:
                print("No results found in Solr.")
                continue

            # B. Generation
            print("🧠 Generating Answer...")
            context_str = str(ranked_data)
            answer_object = student.generate(
                query=query_text,
                context=context_str,
                schema=Response
            )

            # C. Output Results
            print("\n" + "="*50)
            if hasattr(answer_object, 'answer'):
                print(f"AI SUMMARY:\n{answer_object.answer}")
            else:
                print(f"AI SUMMARY:\n{answer_object}")
            print("="*50)

            print("\nSUPPORTING EVIDENCE (SOLR):")
            for i, dataset in enumerate(ranked_data[:3], 1):
                print(f"\n[{i}] Source: {dataset['source_title'] or dataset['dataset_id']}")
                if dataset['source_url']:
                    print(f"    URL: {dataset['source_url']}")
                
                if dataset['top_chunks']:
                    print(f"    Snippet: \"{dataset['top_chunks'][0]['text'][:200]}...\"")
            
            print("\n" + "-"*30)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error occurred: {e}")

    print("\nShutting down. Goodbye!")

if __name__ == "__main__":
    run_main()
