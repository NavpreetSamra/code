import unittest


def brackets(b='(){}[]'):
    stack = Stack()
    d = {'(': ')', '{': '}', '[': ']'}
    for c in b:
        if c in d:
            stack.push(d[c])
        else:
            if stack.peek() == c:
                stack.pop()
            else:
                return False

    return stack.is_empty()

class Stack():
    """
    Generic Stack

    :param list lst: option to begin with values in stack
    """
    def __init__(self, lst=[]):
        self.lst = lst

    def push(self, item):
        """
        Push itme onto top of stack

        :param object item: item to push
        """
        self.lst.append(item)

    def pop(self):
        """
        Pop and return item from top of stack

        :return: top element of stack
        :rtype: object
        """
        return self.lst.pop()

    def peek(self):
        """
        Inspect and return top element of stack

        :return: top element of stack
        :rtype: object
        """
        if not self.is_empty():
            return self.lst[-1]

    def peek_n(self, n=1):
        """
        Inspect and return top n elements of stack

        NB: if n > len(stack) entire stack will be returned

        :param int(>0) n: number of elements to peek

        :return: top n elements of stack
        :rtype: list
        """
        if not self.is_empty():
            return self.lst[-1: -1-n: -1]

    def is_empty(self):
        """
        :return: is empty
        :rtype: bool
        """
        return not self.lst

    def __len__(self):
        return len(self.lst)

    def __repr__(self):
        return ", ".join([str(i) for i in self.lst[::-1]])


class QueueStacks():
    """
    Queue from two stacks

    :param lst queue: option to begin with values in queue
    """
    def __init__(self, lst=[]):
        self.stack1 = Stack(lst)
        self.stack2 = Stack()

    def enqueue(self, item):
        """
        Put item in queue

        :param object item: item to add
        """
        self.stack1.push(item)

    def dequeue(self):
        """
        Return first element of queue
        """
        if not self.stack2.is_empty():
            return self.stack2.pop()

        self.requeue()
        return self.stack2.pop()

    def peek(self):
        """
        Return first element of queue
        """
        if not self.stack2.is_empty():
            return self.stack2.peek()

        self.requeue()
        return self.stack2.peek()

    def requeue(self):
        while not self.stack1.is_empty():
            self.stack2.push(self.stack1.pop())
class TestBrackets(unittest.TestCase):

    def test_asym(self):
        self.assertTrue(brackets)

    def test_sym(self):
        self.assertTrue(brackets('([{}])'))

    def test_nested(self):
        self.assertTrue(brackets('([{}[]()])'))

    def test_broken(self):
        self.assertFalse(brackets('[({])'))

if __name__ == "__main__":
    unittest.main()
