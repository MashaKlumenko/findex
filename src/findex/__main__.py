import sys
import findex.index
import findex.search

def main() -> None:
    if len(sys.argv) < 2:
        print("Використання: python -m findex <команда> [аргументи]")
        print("Доступні команди: index, search")
        sys.exit(1)

    cmd = sys.argv[1]
    # Передаємо модулям чистий список аргументів без самої команди index/search
    sub_args = sys.argv[2:]

    if cmd == "index":
        sys.exit(findex.index.main(sub_args))
    elif cmd == "search":
        sys.exit(findex.search.main(sub_args))
    else:
        print(f"❌ Невідома команда: {cmd}")
        sys.exit(1)

if __name__ == "__main__":
    main()
