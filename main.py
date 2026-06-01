import argparse
import uvicorn
from config import PipelineConfig
from src.pipeline import build_pipeline, query_pipeline
from src.logger import set_debug


def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--debug", action="store_true", help="Enable DEBUG logging")

    parser = argparse.ArgumentParser(description="RAG pipeline")
    sub = parser.add_subparsers(dest="command")

    idx = sub.add_parser(
        "index", parents=[common], help="Ingest corpus, chunk, embed, and store"
    )
    idx.add_argument(
        "--force", action="store_true", help="Reindex even if the corpus is unchanged"
    )

    srv = sub.add_parser("serve", parents=[common], help="Start the FastAPI server")
    srv.add_argument("--host", type=str, default="0.0.0.0")
    srv.add_argument("--port", type=int, default=8000)

    q = sub.add_parser(
        "query", parents=[common], help="Ask a question against the index"
    )
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

    if getattr(args, "model", None) is not None:
        cfg.generation.model = args.model

    if args.command == "index":
        build_pipeline(cfg, force=getattr(args, "force", False))
    elif args.command == "query":
        result = query_pipeline(args.question, cfg)
        print(result["answer"])
    elif args.command == "serve":
        from api import app
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
