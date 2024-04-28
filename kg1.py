from langchain.prompts import PromptTemplate
from langchain import LLMChain
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

load_dotenv()


# example 1
# prompt = PromptTemplate(
#     input_variables=["product"],
#     template="What is a good name for a company that makes {product}?",
# )

# chain = LLMChain(llm=llm, prompt=prompt)
# print(chain.run("mentoring online for testers"))


# example 2
# llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
# conversation = ConversationChain(llm=llm, verbose=True, memory=ConversationBufferMemory())
# conversation.predict(input="Tell me about yourself.")

# conversation.predict(input="What can you do?")
# conversation.predict(input="How can you help me with data analysis?")

# print(conversation)


# example 3

# # instantiate the LLM and embeddings models
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# # create our documents
# texts = ["Napoleon Bonaparte was born in 15 August 1769", "Louis XIV was born in 5 September 1638"]
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.create_documents(texts)

# # create Deep Lake dataset
my_activeloop_org_id = "hyurii3"
my_activeloop_dataset_name = "langchain_course_from_zero_to_hero"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

# # add documents to our Deep Lake dataset
# db.add_documents(docs)

# example 3-2


# create new documents
texts = [
    "Lady Gaga was born in 28 March 1986",
    "Michael Jeffrey Jordan was born in 17 February 1963"
]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)

# add documents to our Deep Lake dataset
db.add_documents(docs)


# example 4

# retrieval_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())

# tools = [
#     Tool(name="Retrieval QA System", func=retrieval_qa.run, description="Useful for answering questions."),
# ]

# agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# response = agent.run("When was Napoleone born?")
# print(response)