import unittest


def build_bst(slst, node=None):
    """
    Build sorted array into BSTree
    """
    n = len(slst)
    if not node:
        node = Node(slst.pop(n/2))
        root = node
    if n < 3:
        if n > 0:
            node.left = Node(slst[0])
        if n > 1:
            node.right = Node(slst[1])
    else:
        a, b = slst[:n/2], slst[n/2:]
        node.left = Node(a.pop(len(a)/2))
        node.right = Node(b.pop(len(b)/2))
        build_bst(a, node.left)
        build_bst(b, node.right)
    if 'root' in dir():
        return root


def preorder_traversal(node):
    print node.value
    if node.left:
        preorder_traversal(node.left)
    if node.right:
        preorder_traversal(node.right)


def inorder_traversal(node):
    if node.left:
        inorder_traversal(node.left)
    print node.value
    if node.right:
        inorder_traversal(node.right)


def postorder_traversal(node):
    if node.left:
        postorder_traversal(node.left)
    if node.right:
        postorder_traversal(node.right)
    print node.value


class Node():
    """
    Node Class 

    :param obj value: value of Node

    :attribute left: pointer to left Node
    :attribute right: pointed to right Node
    """
    def __init__(self, value):
        self.value = value
        self.right = None
        self.left = None

class TestX(unittest.TestCase):

    def test_x(self):
        pass

if __name__ == "__main__":
    unittest.main()
