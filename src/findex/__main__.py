import importlib
import multiprocessing
import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("Використання: python -m findex <команда> [аргументи]")
        print("Доступні команди: index, search, embed, crawl, serve")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd in {"crawl", "serve", "embed"}:
        from findex.cli import app

        app()
        return
    # ``findex.search`` in the package is the function, not the module.
    sub_args = sys.argv[2:]

    if cmd == "index":
        index_mod = importlib.import_module("findex.index")
        sys.exit(index_mod.main(sub_args))
    elif cmd == "search":
        search_mod = importlib.import_module("findex.search")
        sys.exit(search_mod.main(sub_args))
    else:
        print(f"❌ Невідома команда: {cmd}")
        sys.exit(1)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
