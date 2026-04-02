from langchain_core.prompts import ChatPromptTemplate


chat_template = ChatPromptTemplate([
  ('system', "You are a helpful {domain} expert"),
  ('human', "explain in simple terms, what is {topic}?")
    
])

prompt = chat_template.invoke({"domain": "math", "topic": "calculus"})

print(prompt)