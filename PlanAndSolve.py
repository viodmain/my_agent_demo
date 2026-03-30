PLANNER_PROMPT_TEMPLATE = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划,```python与```作为前后缀是必要的:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决“当前步骤”，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对“当前步骤”的回答:
"""

from HelloAgentsLLM import HelloAgentsLLM
import ast

class Palnner:
    def __init__(self, llm_client):
        self.llm_client=llm_client

    def Plan(self,question) -> list[str]:
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)
        message=[{"role":"user","content":prompt}]
        print(f"正在生成计划，问题: {question}")

        response_text=self.llm_client.Think(message=message) or ""
        print(f"生成的计划: {response_text}")

        try:
            plan_str=response_text.split("```python")[1].split("```")[0].strip()
            plan=ast.literal_eval(plan_str)
            return plan if isinstance(plan,list) else []
        except Exception as e:
            print(f"解析计划时发生错误: {e}")
            return []

class Executor:
    def __init__(self,llm_client):
        self.llm_client=llm_client

    def Excute(self, question: str, plan: list[str])->str:
        history=""
        print(f"正在执行计划")

        for i, step in enumerate(plan):
            print(f"\n正在执行步骤 {i+1}/{len(plan)}: {step}")
            prompt=EXECUTOR_PROMPT_TEMPLATE.format(
                question=question,
                plan=plan,
                history=history if history else "无",
                current_step=step
            )

            messsage=[{"role":"user","content":prompt}]
            reponse_text=self.llm_client.Think(message=messsage) or ""
            history += f"步骤 {i+1}: {step}\n结果: {reponse_text}\n\n"
            print(f"步骤 {i+1} 的结果: {reponse_text}")
        
        final_answer=reponse_text
        return final_answer

class PlanAndSolver:
    def __init__(self, llm_client):
        self.planner=Palnner(llm_client)
        self.executor=Executor(llm_client)

    def Run(self, question: str):
        print(f"\n---开始处理问题---\n问题: {question}")

        plan=self.planner.Plan(question)

        if not plan:
            print("未能生成有效的计划，无法继续执行。")
            return "对不起，我无法为这个问题生成一个可行的计划。" 
        
        final_answer=self.executor.Excute(question,plan)
        print(f"\n---最终答案---\n{final_answer}")

if __name__ == "__main__":
    # 初始化 LLM 客户端
    llm_client = HelloAgentsLLM()

    # 初始化并运行 ReAct Agent
    agent = PlanAndSolver(llm_client=llm_client)
    
    question = "2026年第一个月销售额10W,后面每个月预计提示50%，全年销售额是多少？"
    print(f"\n=== 开始处理问题：{question} ===\n")
    result = agent.Run(question)