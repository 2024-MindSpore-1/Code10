'''
Descripttion: The Gini-impurity preserving encryption method(get each node's cipher t.cipher by the corresponding ciphertext domain [t_min, t_max]: t.cipher = (t_min+t_max)/2)
version: v2
Author: anonymous
'''

import mindspore.numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import os


class TreeNode:
    '''Definition of Binary Search Tree Node'''
    def __init__(self,
                 val,
                 label,
                 flag=True,
                 encryptMin=0,
                 encryptMax=1000000):
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
        encryptMin : float
            The ciphertext domain of the current node.
        encryptMax : float
            The ciphertext domain of the current node.
        cipher : float
            The ciphertext of the current node.
        """
        self.val = val
        self.left = None
        self.right = None
        self.label = label
        self.min = float("inf")
        self.max = float("-inf")
        self.flag = flag
        self.encryptMin = encryptMin
        self.encryptMax = encryptMax
        self.cipher = (encryptMin + encryptMax) / 2


class OperationTree:
    '''Insert plaintext to recursively build encrypted binary search tree'''
    def insert(self,
               root,
               val,
               label,
               encryptMin=0,
               encryptMax=1000000):  
        """
        Parameters
        ----------
        root : TreeNode
            The node of the binary search tree.
        val : float
            The value of the insert plaintext.
        label : int 
            The label of the insert plaintext.
        encryptMin : float
            The ciphertext domain of the current node.
        encryptMax : float 
            The ciphertext domain of the current node.
        """
        if root == None:
            root = TreeNode([val], label)
            root.min = val
            root.max = val
            root.cipher = (encryptMin + encryptMax) / 2

        else:
            # Based on the plaintext's value and label to determine its position node in the binary search tree.
            if (root.flag == True):
                if (label == root.label):
                    if (val <= max(root.val) and val >= min(
                            root.val)):  #Insert this plaintext in this node
                        root.val.append(val)
                    elif (val < min(root.val)):
                        if val < root.min:
                            root.min = val
                        if (root.left and val <= root.left.max
                            ):  #recursively find its left child
                            root.left = self.insert(root.left,
                                                    val,
                                                    label,
                                                    encryptMax=root.cipher,
                                                    encryptMin=encryptMin)
                        else:  #Insert this plaintext in this node
                            root.val.append(val)

                    elif (val > max(root.val)):  #Similarly for the right child
                        if val > root.max:
                            root.max = val
                        if (root.right and val >= root.right.min):
                            root.right = self.insert(root.right,
                                                     val,
                                                     label,
                                                     encryptMin=root.cipher,
                                                     encryptMax=encryptMax)
                        else:
                            root.val.append(val)

                else:  #label != root.label
                    if (val < min(root.val)):
                        if val < root.min:
                            root.min = val
                        root.left = self.insert(root.left,
                                                val,
                                                label,
                                                encryptMax=root.cipher,
                                                encryptMin=encryptMin)
                    elif (val > max(root.val)):
                        if val > root.max:
                            root.max = val
                        root.right = self.insert(root.right,
                                                 val,
                                                 label,
                                                 encryptMin=root.cipher,
                                                 encryptMax=encryptMax)
                    else:  #Split the current node
                        rootLeft = root.left
                        rootright = root.right
                        leftVal = [item for item in root.val if item < val]
                        rightVal = [item for item in root.val if item > val]
                        rootmin = root.min
                        rootmax = root.max
                        if (leftVal):  #The split left child
                            if (rootLeft):
                                tmp = rootLeft
                                while (tmp.right):
                                    tmp = tmp.right
                                tmpEncryptMin = tmp.cipher
                                left = TreeNode(leftVal,
                                                root.label,
                                                True,
                                                encryptMax=root.cipher,
                                                encryptMin=tmpEncryptMin)

                            else:
                                left = TreeNode(leftVal,
                                                root.label,
                                                True,
                                                encryptMax=root.cipher,
                                                encryptMin=encryptMin)
                            left.min = root.min
                            left.max = max(leftVal)
                        if (rightVal):  #The split right child
                            if (rootright):
                                tmp = rootright
                                while (tmp.left):
                                    tmp = tmp.left
                                tmpEncryptMax = tmp.cipher
                                right = TreeNode(rightVal,
                                                 root.label,
                                                 True,
                                                 encryptMin=root.cipher,
                                                 encryptMax=tmpEncryptMax)
                            else:
                                right = TreeNode(rightVal,
                                                 root.label,
                                                 True,
                                                 encryptMin=root.cipher,
                                                 encryptMax=encryptMax)
                            right.min = min(rightVal)
                            right.max = root.max
                        if (val in root.val
                            ):  # Update the current node's value
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
                        if (leftVal):  # Update the current node's child
                            root.left = left
                            root.left.left = rootLeft
                        else:
                            root.left = rootLeft
                        if (rightVal):
                            root.right = right
                            root.right.right = rootright
                        else:
                            root.right = rootright

            else:  #root.flag == False
                if (val == root.val[0]):
                    root.val.append(val)

                elif (val < root.val[0]):
                    if val < root.min:
                        root.min = val
                    root.left = self.insert(root.left,
                                            val,
                                            label,
                                            encryptMax=root.cipher,
                                            encryptMin=encryptMin)
                else:
                    if val > root.max:
                        root.max = val
                    root.right = self.insert(root.right,
                                             val,
                                             label,
                                             encryptMin=root.cipher,
                                             encryptMax=encryptMax)
        return root


def inorderTraversal(root):
    '''
    Inorder traversal the built binary search tree.
    '''
    res = []

    def inorder(root):
        if not root:
            return
        res.append([root.val, root.flag])
        inorder(root.left)
        inorder(root.right)

    inorder(root)
    # print('val', res)
    return res


def Encode(root, value):
    '''
    Find the plaintext's corresponding node position in the built binary search tree and return its ciphertext.
    '''
    if (value in root.val):
        return root.cipher
    if (value > max(root.val)):
        return Encode(root.right, value)
    if (value < min(root.val)):
        return Encode(root.left, value)


def getCipher(root, values):
    '''
    Find the plaintext's corresponding ciphertext.
    '''
    Cipher = []
    for i in range(len(values)):
        tmpCode = Encode(root, values[i])
        if (not tmpCode): print("error")
        Cipher.append(tmpCode)
    return Cipher


if __name__ == "__main__":
    #Batch encrypt the data
    files = os.listdir("./data/")
    for file in files:
        dataname = file
        print(dataname)
        start = time.time()
        data_path = './data/' + dataname
        trainingData = pd.read_csv(data_path)
        X = trainingData.iloc[:, :-1].values.tolist()
        Y = trainingData.iloc[:, -1].values.tolist()
        X = list(np.round(np.array(X), 3))
        X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                            Y,
                                                            test_size=0.1,
                                                            shuffle=False,
                                                            random_state=100)
        # Build a binary tree for each attribute
        train_root = []
        Cipher = []
        for col in range(len(X_train[0])):
            Colcipher = []
            op = OperationTree()
            root = TreeNode([X_train[0][col]], Y_train[0])
            for row in range(1, len(X_train)):
                op.insert(root, X_train[row][col], Y_train[row])
            # Get ciphertexts
            Colcipher = getCipher(root, [item[col] for item in X_train])
            Cipher.append(Colcipher)
        Cipher = pd.DataFrame(Cipher)
        Cipher = pd.DataFrame(Cipher.values.T)
        Cipher.insert(Cipher.shape[1], 'label', Y_train)
        # print(Cipher)
        Cipher.to_csv('./OPEminmax_cipher/' + dataname, index=0)
        X_train = pd.concat([pd.DataFrame(X_train),
                             pd.DataFrame(Y_train)],
                            axis=1)
        pd.DataFrame(X_train).to_csv('./OPEminmax_cipher/origin_' + dataname,
                                     index=0)
        end = time.time()
        print("time is ", end - start)
