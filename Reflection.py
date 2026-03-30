from typing import List, Dict, Any, Optional
from HelloAgentsLLM import HelloAgentsLLM

INITIAL_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。请根据以下要求，编写一个Python函数。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。

要求: {task}

请直接输出代码，不要包含任何额外的解释。
"""

REFLECT_PROMPT_TEMPLATE = """
你是一位极其严格的代码评审专家和资深算法工程师，对代码的性能有极致的要求。
你的任务是审查以下Python代码，并专注于找出其在<strong>算法效率</strong>上的主要瓶颈。

# 原始任务:
{task}

# 待审查的代码:
```python
{code}
```

请分析该代码的时间复杂度，并思考是否存在一种<strong>算法上更优</strong>的解决方案来显著提升性能。
如果存在，请清晰地指出当前算法的不足，并提出具体的、可行的改进算法建议（例如，使用筛法替代试除法）。
如果代码在算法层面已经达到最优，才能回答“无需改进”。

请直接输出你的反馈，不要包含任何额外的解释。
"""


REFINE_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。你正在根据一位代码评审专家的反馈来优化你的代码。

# 原始任务:
{task}

# 你上一轮尝试的代码:
{last_code_attempt}
评审员的反馈：
{feedback}

请根据评审员的反馈，生成一个优化后的新版本代码。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。
请直接输出优化后的代码，不要包含任何额外的解释。
"""


class Memory:
    def __init__(self):
        self.records: List[Dict[str,Any]]=[]
    
    def AddRecord(self,record_type:str,content:Any):
        record={"type": record_type,"content":content}
        self.records.append(record)
        print(f"已添加记忆记录: {record}")

    def GetTrajectory(self)->str:
        trajectory_parts=[]
        for record in self.records:
            if record["type"]=="exection":
                trajectory_parts.append(f"--- 上一轮尝试 (代码) ---\n{record['content']}")
            elif record["type"]=="reflection":
                trajectory_parts.append(f"--- 评审员反馈 ---\n{record['content']}")
        return "\n\n".join(trajectory_parts)
    
    def GetLastExecution(self)->Optional[str]:
        for record in reversed(self.records):
            if record["type"]=="exection":
                return record["content"]
        return None
    
class ReflectionAgent:
    def __init__(self, llm_client, max_iterations=5):
        self.llm_client=llm_client
        self.memory=Memory()
        self.max_iterations=max_iterations

    def _get_llm_response(self, prompt:str)->str:
        message=[{"role":"user","content":prompt}]
        response=self.llm_client.Think(message=message) or ""
        return response
    
    def Run(self, task:str):
        print(f"\n--- 开始处理任务 ---\n任务: {task}")
        print("\n正在生成初始代码...")
        initial_prompt=INITIAL_PROMPT_TEMPLATE.format(task=task)
        initial_code=self._get_llm_response(initial_prompt)
        self.memory.AddRecord("exection",initial_code)

        for i in range(self.max_iterations):
            print(f"\n--- 第 {i+1}/{self.max_iterations} 轮迭代 ---")
            print("\n-> 正在进行反思...")

            last_code=self.memory.GetLastExecution() or ""
            reflect_prompt=REFLECT_PROMPT_TEMPLATE.format(task=task,code=last_code)
            feed_back=self._get_llm_response(reflect_prompt)
            self.memory.AddRecord("reflection",feed_back)

            if "无需改进" in feed_back:
                print("\n评审员认为当前代码无需改进，已达到最优。")
                break

            refine_prompt=REFINE_PROMPT_TEMPLATE.format(
                task=task,
                last_code_attempt=last_code,
                feedback=feed_back
            )

            refined_code=self._get_llm_response(refine_prompt)
            self.memory.AddRecord("exection",refined_code)

        final_code=self.memory.GetLastExecution()
        print(f"\n--- 迭代结束 ---\n最终代码:\n{final_code}")
        return final_code
    
if __name__ == "__main__":
    # 初始化 LLM 客户端
    llm_client = HelloAgentsLLM()

    # 初始化并运行 ReAct Agent
    agent = ReflectionAgent(llm_client=llm_client)
    
    task = "任务： 编写一个Python函数，找出1到n之间所有的素数 (prime numbers)。"
    print(f"\n=== 开始处理问题：{task} ===\n")
    result = agent.Run(task)