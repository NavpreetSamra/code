# Complete the function below.
class Stack():
    """
    Generic Stack #This is my stack class available in my github

    :param list lst: option to begin with values in :class:`Stack`
    """
    def __init__(self, lst=None):
        if list:
            self.lst = lst
        else:
            self.lst = []

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
        
    def is_empty(self):
        """
        :return: is empty
        :rtype: bool
        """
        return not self.lst
    
def count(s):
    stack0 = Stack()
    stack1 = Stack()
    c = 0
    for i in s:
        if i == 1:
            if not stack0.is_empty():
                count +=1
                stack0.pop()
            else:
                stack1.push(i)
        else:
            if not stack1.is_empty():
                count +=1
                stack1.pop()
            else:
                stack0.push(i)        
    return count
    
