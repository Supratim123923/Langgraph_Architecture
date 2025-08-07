from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

class Grader:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def grade(self, docs: list[Document]) -> float:
        if not docs:
            return 0.0

        prompt = (
            "You're an evaluator for document relevance.\n"
            "Given the following documents, rate how well they seem to answer a user's question "
            "on a scale of 0 to 1 (0 = not helpful, 1 = perfectly helpful):\n\n"
        )
        context = "\n---\n".join(doc.page_content for doc in docs)
        prompt += context + "\n\nReturn ONLY the score."

        response = self.llm.invoke(prompt)
        try:
            score = float(response.content.strip())
            return max(0.0, min(score, 1.0))  # Clamp between 0 and 1
        except:
            return 0.0
