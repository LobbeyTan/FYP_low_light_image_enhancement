import os


class Option:

    def __init__(self, root_dir=os.getcwd(), phase="train") -> None:
        self.phase = phase
        self.root_dir = root_dir
