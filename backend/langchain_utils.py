from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

from llm_wrapper import GroqLLM


def make_conv_chain():
    
    llm = GroqLLM()
    memory = ConversationBufferMemory(memory_key="chat_history")

    template = """You are a helpful assistant. Use the conversation history
and then answer the user's latest query.

Conversation History:
{chat_history}

User: {question}
Assistant:"""

    prompt = PromptTemplate(input_variables=["chat_history", "question"], template=template)
    return LLMChain(llm=llm, prompt=prompt, memory=memory)
