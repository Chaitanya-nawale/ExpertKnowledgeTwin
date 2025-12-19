from langsmith import Client

def main():
    client = Client()
    prompt = client.pull_prompt("rlm/rag-prompt")  # no await here
    print(prompt)

if __name__ == "__main__":
    main()
