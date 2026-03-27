import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any
from serpapi import SerpApiClient
import re

class HelloAgentsLLM:
    def __init__(
        self,
        model: str = None,
        apiKey: str = None,
        baseUrl: str = None,
        timeout: int = None,
    ):
        load_dotenv()
        self.model = model or os.getenv("LLM_MODEL_ID")
        self.apiKey = apiKey or os.getenv("LLM_API_KEY")
        self.baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        self.timeout = timeout or 30

        if not all([self.model, self.apiKey, self.baseUrl]):
            raise ValueError(
                "请确保环境变量中包含LLM_MODEL_ID、LLM_API_KEY和LLM_BASE_URL。"
            )
        self.client = OpenAI(
            api_key=self.apiKey, base_url=self.baseUrl, timeout=self.timeout
        )

    def Think(self, message: List[Dict[str, str]], temperature: float = 0) -> str:
        print("正在调用 LLM API...")
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=message, temperature=temperature, stream=True
            )
            print("大语言模型响应成功。")
            collected_content = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()
            return "".join(collected_content)
        except Exception as e:
            print(f"调用LLM API时发生错误: {e}")
            return None


class ToolExcutor:
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def RegisterTool(self, name: str, description: str, func: callable):
        if name in self.tools:
            print(f"工具 '{name}' 已经注册，覆盖原有工具。")
        self.tools[name] = {"description": description, "func": func}
        print(f"工具 '{name}' 注册成功。")

    def GetTool(self, name: str) -> callable:
        return self.tools.get(name, {}).get("func")

    def GetAvailableTools(self) -> str:
        return "\n".join(
            [f"- {name}: {info['description']}" for name, info in self.tools.items()]
        )


def Search(query) -> str:
    print(f"搜索查询: {query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "SerpApi API key error"

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",
            "hl": "zh-CN",
        }

        client = SerpApiClient(params)
        results = client.get_dict()

        if "answer_box_list" in results:
            return "".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
        return f"对不起，没有找到关于 '{query}' 的信息。"

    except Exception as e:
        return f"搜索时发生错误: {e}"


# ReAct 提示词模板
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `{{tool_name}}[{{tool_input}}]`:调用一个可用工具。
- `Finish[最终答案]`:当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用 Finish[最终答案] 来输出最终答案。

现在，请开始解决以下问题:
Question: {question}
History: {history}
"""


class ReActAgent:
    def __init__(
        self, llm_client: HelloAgentsLLM, tool_excutor: ToolExcutor, max_steps: int = 5
    ):
        self.client = llm_client
        self.tool_excutor = tool_excutor
        self.max_steps = max_steps
        self.history = []

    def Run(self, question: str):
        self.history = []
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- Step {current_step} ---")

            tools_description = self.tool_excutor.GetAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_description, question=question, history=history_str
            )

            message = [{"role": "user", "content": prompt}]
            response_text = self.client.Think(message=message)

            if not response_text:
                print("LLM没有返回响应，结束执行。")
                break
            
            thought, action=self.ParseOutput(response_text)
            if thought:
                print(f"Thought: {thought}")
            if not action:
               print("没有解析到Action，结束执行。")
               break
            if action.startswith("Finish"):
                final_answer = re.match(r"Finish\[(.*)\]",action).group(1)
                print(f"最终答案: {final_answer}")
                return final_answer
            tool_name, tool_input = self.ParseAction(action)
            if not tool_name or not tool_input:
                print("Action格式错误，无法解析工具名称和输入。")
                continue
            print(f"执行工具: {tool_name}，输入: {tool_input}")
            tool_function=self.tool_excutor.GetTool(tool_name)
            if not tool_function:
                observition=f"工具 '{tool_name}' 未找到。"
            else:
                observition=tool_function(tool_input)
            print(f"Observision: {observition}")
            self.history.append(f"action:{action}")
            self.history.append(f"observition:{observition}")
        print("达到最大步骤限制，结束执行。")
        return
            

    def ParseOutput(self,text:str):
        thought_match=re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        action_match=re.search(r"Action:\s*(.*?)$", text, re.DOTALL)

        thought=thought_match.group(1).strip() if thought_match else ""
        action=action_match.group(1).strip() if action_match else ""
        return thought,action
    
    def ParseAction(slef,action_text:str):
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return None,None


if __name__ == "__main__":
    # 初始化工具执行器
    tool_excutor = ToolExcutor()
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_excutor.RegisterTool("Search", search_description, Search)

    # 初始化 LLM 客户端
    llm_client = HelloAgentsLLM()

    # 初始化并运行 ReAct Agent
    agent = ReActAgent(llm_client=llm_client, tool_excutor=tool_excutor, max_steps=5)
    
    question = "what is harness engineering"
    print(f"\n=== 开始处理问题：{question} ===\n")
    result = agent.Run(question)
    print(f"\n=== 最终结果：{result} ===")
