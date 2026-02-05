from dotenv import load_dotenv
load_dotenv()

from rag_assistant.rag import RAG

def main():
    rag = RAG()
    while True:
        q = input("\nQuestion (or 'exit'): ").strip()
        if q.lower() == "exit":
            break

        try:
            res = rag.ask(q)
        except RuntimeError as e:
            print(f"\n {e}")
            continue

        print("\n--- Answer ---")
        print(res.answer)

        print("\n--- Retrieved sources ---")
        for i, d in enumerate(res.docs, start=1):
            md = d.metadata or {}
            src = md.get("filename", md.get("source", "unknown"))
            page = md.get("page", "?")
            chunk_id = md.get("chunk_id", "")
            snippet = d.page_content[:120].replace("\n", " ")
            print(f"[{i}] {src} p{page} {chunk_id}: {snippet}...")


if __name__ == "__main__":
    main()















