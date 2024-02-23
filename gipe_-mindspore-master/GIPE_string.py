'''
Descripttion: The Gini-impurity preserving encryption method(get each node's cipher t.cipher by the corresponding node position, left path append '0' while righe path append '1').
version: v1
Author: anonymous
'''

import mindspore.numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import os


class TreeNode:
    '''Definition of Binary Search Tree Node'''
    def __init__(self, val, label, flag=True):
        """
        Parameters
        ----------
        val : list 
            Store plaintexts in the current node.
        left : TreeNode
            The left child node.
        right : TreeNode
            The right child node.
        label : int 
            Record the label in the current node(available when flag is true).
        min : float
            The minimum value of the child nodes of the current node.
        max : float
            The maximum value of the child nodes of the current node.
        flag : boolean
            Judge whether the current node type, True: the node can store different value with consistent lable, False: the node can store consistent value with different labels.
        """
        self.val = val
        self.left = None
        self.right = None
        self.label = label
        self.min = float("inf")
        self.max = float("-inf")
        self.flag = flag


class OperationTree:
    '''Insert plaintext to recursively build encrypted binary search tree'''
    def insert(self, root, val, label):
        """
        Parameters
        ----------
        root : TreeNode
            The node of the binary search tree.
        val : float
            The value of the insert plaintext.
        label : int 
            The label of the insert plaintext.
        """
        if root == None:
            root = TreeNode([val], label)
            root.min = val
            root.max = val
        else:
            # Based on the plaintext's value and label to determine its position node in the binary search tree.
            if (root.flag == True):
                if (label == root.label):
                    if (val <= max(root.val) and val >= min(root.val)):
                        root.val.append(val)
                    elif (val < min(root.val)):
                        if val < root.min:
                            root.min = val
                        if (root.left and val <= root.left.max):
                            root.left = self.insert(root.left, val, label)
                        else:
                            root.val.append(val)
                    elif (val > max(root.val)):
                        if val > root.max:
                            root.max = val
                        if (root.right and val >= root.right.min):
                            root.right = self.insert(root.right, val, label)
                        else:
                            root.val.append(val)

                else:  #label != root.label
                    if (val < min(root.val)):
                        if val < root.min:
                            root.min = val
                        root.left = self.insert(root.left, val, label)
                    elif (val > max(root.val)):
                        if val > root.max:
                            root.max = val
                        root.right = self.insert(root.right, val, label)
                    else:  #Split the current node
                        rootLeft = root.left
                        rootright = root.right

                        leftVal = [item for item in root.val if item < val]
                        rightVal = [item for item in root.val if item > val]
                        rootmin = root.min
                        rootmax = root.max
                        if (leftVal):
                            left = TreeNode(leftVal, root.label, True)
                            left.min = root.min
                            left.max = max(leftVal)
                        if (rightVal):
                            right = TreeNode(rightVal, root.label, True)
                            right.min = min(rightVal)
                            right.max = root.max

                        if (val in root.val):
                            rootValue = [
                                item for item in root.val if item == val
                            ]
                            rootValue.append(val)
                            root.val = rootValue
                            root.flag = False
                        else:
                            root.val = [val]
                        root.min = rootmin
                        root.max = rootmax

                        if (leftVal):
                            root.left = left
                            root.left.left = rootLeft
                        else:
                            root.left = rootLeft
                        if (rightVal):
                            root.right = right
                            root.right.right = rootright
                        else:
                            root.right = rootright

            else:
                if (val == root.val[0]):
                    root.val.append(val)
                elif (val < root.val[0]):
                    if val < root.min:
                        root.min = val
                    root.left = self.insert(root.left, val, label)
                else:
                    if val > root.max:
                        root.max = val
                    root.right = self.insert(root.right, val, label)
        return root


def levelOrder(root):
    '''
    levelOrder traversal the built binary search tree.
    '''
    res = []
    if root:
        queue = [root]
    else:
        print(res)
    while len(queue):
        n = len(queue)
        r = []
        for _ in range(n):
            node = queue.pop(0)
            r.append([node.val, node.flag])
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        res.append(r)
    for item in res:
        print(item)


def Encode(root, value, code='0'):
    '''
    Find the training plaintext dataset's  corresponding node position in the built binary search tree and return its ciphertext.
    '''
    if (value in root.val):
        return code
    if (value > max(root.val)):
        return Encode(root.right, value, code + '1')
    if (value < min(root.val)):
        return Encode(root.left, value, code + '0')


def EncodeTest(root, value, code='0'):
    '''
    Find the testing plaintext dataset's  corresponding node position in the built binary search tree and return its ciphertext.
    '''
    if (value in root.val):
        return code
    if (root.right and value > max(root.val)):
        return EncodeTest(root.right, value, code + '1')
    elif (root.left and value < min(root.val)):
        return EncodeTest(root.left, value, code + '0')
    else:
        return code


def inorderTraversal(root):
    '''
    inorder traversal the built binary search tree.
    '''
    res = []

    def inorder(root):
        if not root:
            return
        inorder(root.left)
        res.append([root.val, root.flag])
        inorder(root.right)

    inorder(root)
    return res


def getPathCode(root, values):
    '''
    Find the training plaintext dataset's  corresponding node position as its ciphertext.
    '''
    pathcodes = []
    for i in range(len(values)):
        tmpCode = Encode(root, values[i], code='0')
        if (not tmpCode): print("error")
        pathcodes.append(tmpCode)
    return pathcodes


def getTestPathCode(root, values):
    '''
    Find the testing plaintext dataset's  corresponding node position as its ciphertext.
    '''
    pathcodes = []
    for i in range(len(values)):
        tmpCode = EncodeTest(root, values[i], code='0')
        if (not tmpCode): print("error")
        pathcodes.append(tmpCode)
    return pathcodes


def getTestCode(value, root, code='0'):
    if (min(root.val) <= value <= max(root.val)):
        return code
    if (value > max(root.val)):
        if (root.right):
            return getTestCode(value, root.right, code + '1')
        else:
            return code
    if (value < min(root.val)):
        if (root.left):
            return getTestCode(value, root.left, code + '0')
        else:
            return code


def GetValue(X, dict_list):
    if (X <= dict_list[0][0]):
        return str(dict_list[0][1])
    if (X >= dict_list[len(dict_list) - 1][0]):
        return str(dict_list[len(dict_list) - 1][1])

    for i in range(1, len(dict_list)):
        if (X == dict_list[i][0]):
            return str(dict_list[i][1])
        if (dict_list[i - 1][0] < X < dict_list[i][0]):
            if (X - dict_list[i - 1][0] > dict_list[i][0] - X):
                return str(dict_list[i][1])
            else:
                return str(dict_list[i - 1][1])


if __name__ == "__main__":

    files = os.listdir("./data/")
    for file in files:
        dataname = file
        print(dataname)
        start = time.time()
        data_path = './data/' + dataname
        trainingData = pd.read_csv(data_path)
        trainingData = trainingData.sample(frac=1.0)
        X = trainingData.iloc[:, :-1].values.tolist()
        Y = trainingData.iloc[:, -1].values.tolist()

        X = list(np.round(np.array(X), 3))
        X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                            Y,
                                                            test_size=0.1,
                                                            shuffle=False,
                                                            random_state=100)

        #Get training data's cipher
        train_root = []
        pathcode_X = []
        for col in range(len(X_train[0])):
            op = OperationTree()
            root = None
            for row in range(len(X_train)):
                root = op.insert(root, X_train[row][col], Y_train[row])
            pathcodes = getPathCode(root, [item[col] for item in X_train])
            train_root.append(root)
            pathcode_X.append(pathcodes)
        pathcode_X = pd.DataFrame(pathcode_X)
        pathcode_X = pd.DataFrame(pathcode_X.values.T)  #转置
        pathcode_X.insert(pathcode_X.shape[1], 'label', Y_train)  #加label
        # print(pathcode_X)
        pathcode_X = pd.DataFrame(pathcode_X)
        pathcode_X.to_csv('./data/OPEstr/train_' + dataname, index=0)

        #Get testing data's cipher
        dict_list = []
        for col in range(len(X_train[0])):
            dict_col = {}
            for row in range(len(X_train)):
                if (X_train[row][col] not in dict_col):
                    dict_col[X_train[row][col]] = pathcode_X.iloc[row, col]
            dict_col = [(k, dict_col[k]) for k in sorted(dict_col.keys())]
            dict_list.append(dict_col)

        pathcode_X = []
        for col in range(len(X_test[0])):
            tmpPath = []
            for row in range(len(X_test)):
                tmpPath.append(GetValue(X_test[row][col], dict_list[col]))
            pathcode_X.append(tmpPath)
        pathcode_X = pd.DataFrame(pathcode_X)
        pathcode_X = pd.DataFrame(pathcode_X.values.T)
        pathcode_X.insert(pathcode_X.shape[1], 'label', Y_test)

        pathcode_X = pd.DataFrame(pathcode_X)
        pathcode_X.to_csv('./data/OPEstr/test_' + dataname, index=0)

        end = time.time()
        print("time is ", end - start)
