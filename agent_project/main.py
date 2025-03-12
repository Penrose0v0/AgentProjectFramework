import os
import yaml

from src.modules import Agent
from src.utils import output_folder, logger


def main(): 
    logger.info("程序启动")

    # Read config
    config_file = "config.yaml"
    with open(config_file, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    # Init modules
    logger.info("正在初始化模组")
    agent = Agent(config=config)
    logger.info("模组初始化成功")

    # Start main loop
    logger.info("开始进行对话...")
    while True: 
        user_input = input("User: ")
        if user_input == "q": 
            break
        agent_response = agent.chat(contents=user_input)
        print(f"Agent: {agent_response}")
    logger.info("对话结束")

    # Export memory
    logger.info("正在导出对话记录")
    export_file = os.path.join(output_folder, "memory.json")
    agent.llm.export_memory(export_file, include_system_prompt=True)
    logger.info("对话记录导出成功")

    logger.info("程序结束")


if __name__ == "__main__": 
    main()