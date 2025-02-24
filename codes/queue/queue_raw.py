class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None
        
class Queue:
    def __init__(self):
        self.head = self.tail = None
        
    def enqueue(self, val):
        new_node = ListNode(val)
        
        if self.tail:
            self.tail.next = new_node
            self.tail = self.tail.next
        # Queue is empty
        else:
            self.head = self.tail = new_node
            
    def dequeue(self):
        if not self.head:
            return None
        
        val = self.head.val
        self.head = self.head.next
        if not self.head:
            self.tail = None
            
        return val
    
    def print(self):
        cur = self.head
        while cur:
            print(cur.val, ' -> ', end="")
            cur = cur.next
        print()
        
q = Queue()
q.enqueue(1)
q.enqueue(3)
q.print()