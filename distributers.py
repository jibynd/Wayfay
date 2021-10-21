class Generator:
    def __init__(self, distributer):
        self.dist = distributer
        self.params = {}
    def spit(self, size = 1):
        return self.dist(size = size, **self.params) # assuming params contains all required args
    
    def set(self, arg, val):
        self.params[arg] = val

