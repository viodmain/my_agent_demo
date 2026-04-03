MY_REACT_PROMPT = """你是一个具备推理和行动能力的AI助手。你可以通过思考分析问题，然后调用合适的工具来获取信息，最终给出准确的答案。

## 可用工具
{tools}

## 工作流程
请严格按照以下格式进行回应，每次只能执行一个步骤:

Thought: 分析当前问题，思考需要什么信息或采取什么行动。
Action: 选择一个行动，格式必须是以下之一:
- `{{tool_name}}[{{tool_input}}]` - 调用指定工具
- `Finish[最终答案]` - 当你有足够信息给出最终答案时

## 重要提醒
1. 每次回应必须包含Thought和Action两部分
2. 工具调用的格式必须严格遵循:工具名[参数]
3. 只有当你确信有足够信息回答问题时，才使用Finish
4. 如果工具返回的信息不够，继续使用其他工具或相同工具的不同参数

## 当前任务
**Question:** {question}

## 执行历史
{history}

现在开始你的推理和行动:
"""

import re
import os
from dotenv import load_dotenv
from typing import Optional, List,Tuple,Dict,Any
from serpapi import SerpApiClient
from hello_agents  import ReActAgent, HelloAgentsLLM,Config,Message,ToolRegistry

class MyReActAgent(ReActAgent):
    def __init__(
            self,
            name:str,
            llm:HelloAgentsLLM,
            tool_registry:Optional[ToolRegistry]=None,
            system_prompt:Optional[str]=None,
            config:Optional[Config]=None,
            max_steps:int=5,
            custom_prompt:Optional[str]=None
        ):
            super().__init__(name,llm,system_prompt,config)
            self.tool_registry=tool_registry
            self.max_steps=max_steps
            self.current_history:List[str]=[]
            self.prompt_template=custom_prompt if custom_prompt else MY_REACT_PROMPT
            print(f"✅ {name} 初始化完成，最大步数: {max_steps}")

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

    def run(self,input_text:str,**kwargs)->str:
        self.current_history=[]
        self.current_step=0;

        print(f"🤖 {self.name} 正在处理: {input_text}")
        while self.current_step < self.max_steps:
            self.current_step+=1
            print(f"🔄 步骤 {self.current_step}/{self.max_steps}")

            tool_desc = self.tool_registry.get_tools_description()
            history_str="\n".join(self.current_history)
            prompt=self.prompt_template.format(
                 tools=tool_desc,
                 question=input_text,
                 history=history_str)
            
            message=[{"role":"user","content":prompt}]
            response_text = self.llm.invoke(messages=message,**kwargs)

            thought, action = self.ParseOutput(response_text.content)
            if action and action.startswith("Finish"):
                _,final_answer = self.ParseAction(action)
                self.add_message(Message(input_text, "user"))
                self.add_message(Message(final_answer, "assistant"))
                return final_answer
            
            if action:
                tool_name,tool_input=self.ParseAction(action)
                observation=self.tool_registry.execute_tool(tool_name,tool_input)
                self.current_history.append(f"action:{action}")
                self.current_history.append(f"observation:{observation}")
        final_answer = "抱歉，我无法在限定步数内完成这个任务。"
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        return final_answer

if __name__ == "__main__":
    load_dotenv()
    llm=HelloAgentsLLM()
    tool_registry=ToolRegistry()
    from HelloAgentsLLM import Search
    tool_registry.register_function("search", "搜索互联网信息", Search)
    print("✅ 搜索工具注册成功")
    agent=MyReActAgent(
        name="测试 ReActAgent",
        llm=llm,
        tool_registry=tool_registry,
        max_steps=3
    )
    question="夏天去哪里玩"
    answer=agent.run(question)
    print(f"最终答案：{answer}")
