def load_rules():
    with open("prompts/guidelines.txt") as f:
        return f.read()