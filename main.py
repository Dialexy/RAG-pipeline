import argparse
from config import PipelineConfig
from src.pipeline import build_pipeline, query_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG pipeline")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("index", help="Ingest corpus, chunk, embed, and store")

    q = sub.add_parser("query", help="Ask a question against the index")
    q.add_argument("question", type=str)
    q.add_argument(
        "--model",
        type=str,
        default=None,
        help="This flag allows you to run your own model of choice",
    )

    args = parser.parse_args()
    cfg = PipelineConfig()

    if args.model is not None:
        cfg.generation.model = args.model

    if args.command == "index":
        build_pipeline(cfg)
    elif args.command == "query":
        result = query_pipeline(args.question, cfg)
        print(result["answer"])
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
