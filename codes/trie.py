class PrefixNode:
    def __init__(self):
        self.children = {}
        self.word = False
        
class PrefixTree:
    def __init__(self):
        self.root = PrefixNode()
        
    def insert(self, word):
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = PrefixNode()
            cur = cur.children[c]
        cur.word = True
        
    def search(self, word):
        cur = self.root
        for c in word:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return cur.word
    
    def startsWith(self, prefix):
        cur = self.root
        for c in prefix:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return True
    
prefixTree = PrefixTree()
prefixTree.insert("apple")

print(prefixTree.search("apple"))
print(prefixTree.startsWith("app"))
print(prefixTree.startsWith("api"))