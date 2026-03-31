from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM

if __name__ == "__main__":
    load_dotenv()
    llm = HelloAgentsLLM()

    messages = [{"role": "user", "content": "你好！"}]
    for chunk in llm.think(messages):
        print(chunk, end="")