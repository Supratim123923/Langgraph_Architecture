from langchain_openai import ChatOpenAI

class Rewriter:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

    def rewrite(self, query: str, previous_answer: str) -> str:
        prompt = (
            "The user asked the following query but the answer wasn't satisfactory:\n\n"
            f"Query: {query}\n"
            f"Answer: {previous_answer}\n\n"
            "Please rewrite or clarify the query to make it more precise and easier for a document retrieval system to understand."
        )

        response = self.llm.invoke(prompt)
        return response.content.strip()
