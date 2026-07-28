import os
import logging
import argparse
from seed.retrieval.retriever_solr import SolrRetriever
from seed.generation.generator import StudentGenerator
from seed.generation.schema import Response
from seed.config import load_config

# Configure simple logging
logging.basicConfig(level=logging.WARNING)

def run_cli():
    # 1. Load configuration
    cfg = load_config("config.yaml") # Assumes config.yaml is in the execution root
    
    active_provider = cfg.llm_providers.get(cfg.active_student)
    if not active_provider:
        print(f"Error: Active student '{cfg.active_student}' not found.")
        return

    # 2. Initialize components
    print(f"--- Initializing {active_provider.model} via {active_provider.provider} ---")
    student = StudentGenerator(
        provider=active_provider.provider,
        model_name=active_provider.model,
        base_url=active_provider.base_url
    )

    solr_client = SolrRetriever(
        url=cfg.solr.url,
        query_fields=cfg.solr.query_fields,
        result_fields=cfg.solr.result_fields,
        sort_order=cfg.solr.sort_order,
        url_template=cfg.solr.url_template
    )

    print("SEED Search Explorer (CLI Mode)")
    print("Type 'exit' to quit.")
    print("-" * 30)

    while True:
        try:
            query_text = input("\nEnter your search query (or 'exit'): ").strip()
            
            if not query_text:
                continue
            if query_text.lower() in ['exit', 'quit']:
                break

            print("\n🔍 Searching Solr...")
            
            # Retrieval
            res = solr_client.search(query=query_text, rows=cfg.solr.num_docs)

            if not res:
                print("No results found in Solr.")
                continue

            print(f"Found {len(res)} documents.")
            for r in res:
                print(f" - [{r.id}] {r.title or 'No Title'} (Score: {r.score:.2f})")

            # Generation
            print("\n🧠 Generating Answer...")
            context_str = "\n".join([f"Source [{r.id}]: {r.snippet}" for r in res])
            answer_object = student.generate(
                query=query_text,
                context=context_str,
                schema=Response
            )

            # Output Results
            print("\n" + "="*50)
            if hasattr(answer_object, 'answer'):
                print(f"AI SUMMARY:\n{answer_object.answer}")
            else:
                print(f"AI SUMMARY:\n{str(answer_object)}")
            print("="*50 + "\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    run_cli()
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
