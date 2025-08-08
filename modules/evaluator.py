from langchain_openai import ChatOpenAI

class Evaluator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def is_answered(self, answer: str) -> bool:
        if not answer.strip():
            return False

        prompt = (
            "Is the following answer meaningful, complete, and helpful for the user?\n\n"
            f"Answer:\n{answer}\n\n"
            "Respond with only 'yes' or 'no'."
        )

        response = self.llm.invoke(prompt)
        result = response.content.strip().lower()
        if not result:
            return False
        # Ensure we only accept 'yes' or 'no'
        if result not in ["yes", "no"]:
            raise ValueError("Evaluator response must be 'yes' or 'no'.")
        # Return True if the answer is deemed satisfactory
        if result == "yes":
            return True
        
        return False
