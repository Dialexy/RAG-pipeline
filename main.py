import argparse
from config import PipelineConfig
from src.pipeline import build_pipeline, query_pipeline
from src.logger import set_debug


def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--debug", action="store_true", help="Enable DEBUG logging")

    parser = argparse.ArgumentParser(description="RAG pipeline")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("index", parents=[common], help="Ingest corpus, chunk, embed, and store")

    q = sub.add_parser("query", parents=[common], help="Ask a question against the index")
    q.add_argument("question", type=str)
    q.add_argument(
        "--model",
        type=str,
        default=None,
        help="This flag allows you to run your own model of choice",
    )

    args = parser.parse_args()
    if getattr(args, "debug", False):
        set_debug()
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
