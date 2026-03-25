import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

class HelloAgentsLLM:
    def __init__(self,model:str=None,apiKey:str=None, baseUrl: str = None, timeout: int = None):
        self.model=model or os.getenv("LLM_MODEL_ID")
        self.apiKey=apiKey or os.getenv("LLM_API_KEY")
        self.baseUrl=baseUrl or os.getenv("LLM_BASE_URL")
        self.timeout=timeout or 30

        if not all([self.model,self.apiKey,self.baseUrl]):
            raise ValueError("请确保环境变量中包含LLM_MODEL_ID、LLM_API_KEY和LLM_BASE_URL。")
        self.client=OpenAI(api_key=self.apiKey,base_url=self.baseUrl, timeout=self.timeout)

    def think(self, message:List[Dict[str,str]], temperature: float = 0)->str:
        print("正在调用 LLM API...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=message,
                temperature=temperature,
                stream=True
            )
            print("大语言模型响应成功。")
            collected_content = []
            for chunk in response:
                content=chunk.choices[0].delta.cotent or ""
                print(content,end="",flush=True)
                collected_content.append(content)
            print()
            return "".join(collected_content)
        except Exception as e:
            print(f"调用LLM API时发生错误: {e}")
            return None
        
if __name__ == "__main__":
    try:
        llmClient=HelloAgentsLLM()
        exampleMessage=[
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "写一个快速排序算法"}
        ]

        print("正在生成代码...")
        responseText=llmClient.think(exampleMessage,0)
        if responseText:
            print("生成的代码如下:")
            print(responseText)
    except ValueError as e:
        print(f"发生错误: {e}")

