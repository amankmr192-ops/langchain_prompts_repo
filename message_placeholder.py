from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder

#chat template
chat_template = ChatPromptTemplate([
  ('system', "You are a helpful customer support agent"),
  ('human', '{query}'),
  MessagesPlaceholder(variable_name="chat_history")
    
])
chat_history = []


#load chat history
with open("chat_history.txt") as f:
    f.readlines()
    
print(chat_history)   

#create prompt
prompt = chat_template.invoke({"query": "Where is my refund?", "chat_history": chat_history})

print(prompt)