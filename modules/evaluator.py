class Evaluator:
    def is_answered(self, content):
        return "bamboo" in content.lower() and "steel" in content.lower()
