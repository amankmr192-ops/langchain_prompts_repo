from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.7,
        max_new_tokens=512
    )
)

model = ChatHuggingFace(llm=llm)

while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    response = model.invoke(user_input)
    print("Chatbot:", response.content)