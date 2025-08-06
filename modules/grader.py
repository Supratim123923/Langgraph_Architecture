class Grader:
    def grade(self, docs):
        joined = " ".join(docs).lower()
        return 0.8 if "bamboo" in joined and "steel" in joined else 0.5
