#include <fstream>
#include <typeinfo>
#include <vector>
#include <unordered_map>
#include <climits>
#include <string>
#include <set>
#include <cstddef>
#include <iomanip>
#include <chrono>
#include <random>
#include <thread>
#include <mutex>
#include <memory>
#include <limits>
#include <algorithm>
#include <numeric>
#include <time.h>
#include <math.h>

#include <float.h>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>
#include <sstream>
#include <iostream>
#include <queue>
#include <omp.h>
#include <helib/helib.h>
using namespace std;
using namespace helib;

/*
The encryption decision tree , the feature space is encrypted by our Gini-impurity preserving encryption while the label space is encrypted
by the Homomorphic encryption CKKS through Helib library.
 */

// Definition of Decision Tree Node
struct Node
{
    vector<int> left;
    vector<int> right;
    int col;
    double value;
};

using matrix = vector<vector<double>>;

// Read the training and testing data.
matrix ReadFile(string url, int row, int col)
{
    matrix data(row, vector<double>(col));
    ifstream infile;
    infile.open(url);
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            infile >>
                data[i]
                    [j];
        }
    }
    infile.close();
    return data;
}

/*Definition of Decision Tree :
    Parameters
    ----------
    results : store the samples in the leaf node of decision tree
    index : index to the left and righe child node
    data : store the samples in the current node
*/
class Tree
{
public:
    Tree() {}

    Tree(vector<int> index)
    {
        this->index = index;
    }

    Tree(vector<double> results, matrix data)
    {
        this->results = results;
        this->data = data;
    }

    vector<double> results;
    vector<int> index;
    matrix data;
};

// Calculate the number of each label of samples in the current node
vector<double> calculateDiffCount(matrix &data,
                                  vector<int> &index,
                                  int categorynum,
                                  vector<Ctxt> &c,
                                  Ctxt &ans)
{
    int row = index.size();
    int col = data[0].size();

    vector<double> labelCount(categorynum);
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < categorynum; j++)
        {
            labelCount[j] += data[index[i]][col - categorynum + j];
        }
    }

    ans = c[index[0]];
    for (int i = 1; i < row; i++)
    {
        ans += c[index[i]];
    }
    return labelCount;
}

// Calculate the number of each label of samples in the current leaf node
vector<double> getLeafLabel(matrix &data,
                            vector<int> &indexL, vector<int> &indexR,
                            int categorynum)
{
    int rowL = indexL.size();
    int rowR = indexR.size();
    int col = data[0].size();

    vector<double> labelCount(categorynum);
    for (int i = 0; i < rowL; i++)
    {
        for (int j = 0; j < categorynum; j++)
        {
            labelCount[j] += data[indexL[i]][col - categorynum + j];
        }
    }
    for (int i = 0; i < rowR; i++)
    {
        for (int j = 0; j < categorynum; j++)
        {
            labelCount[j] += data[indexR[i]][col - categorynum + j];
        }
    }

    return labelCount;
}

// Calculate the gini impurity of the current node
double gini(matrix &data,
            vector<int> &index,
            int categorynum,
            vector<Ctxt> &c,
            Ctxt &ans,
            const SecKey &secretKey,
            Context &context)
{
    double length = index.size();
    vector<double> labelCount =
        calculateDiffCount(data, index, categorynum, c, ans);
    ans *= ans; // ciphertext p^2
    double imp = 0.0;
    return 1 - imp;
}

// Split the current node according to the splitting points
set<double> getSplitSet(matrix &data, vector<int> &index, int col)
{
    vector<double> tmp;
    for (int i = 0; i < index.size(); i++)
    {
        tmp.push_back(data[index[i]][col]);
    }
    set<double> col_value_set(tmp.begin(), tmp.end());
    return col_value_set;
    /*To speed up:
    tmp.assign(col_value_set.begin(), col_value_set.end());
    vector<double> rantmp;
    int Len = tmp.size();
    if (Len > 50) {
      for (int i = 0; i < 50; i++) {
        rantmp.push_back(tmp[rand() % Len]);
      }
    }
    set<double> col_value_set(rantmp.begin(), rantmp.end());
    return col_value_set;*/
}

// Select the optimal splitting point from all splitting points
vector<vector<int>> splitDatas(matrix &data,
                               vector<int> &index,
                               double value,
                               int column)
{
    vector<int> list1;
    vector<int> list2;
    for (int i = 0; i < index.size(); i++)
    {
        if (data[index[i]][column] <= value)
        {
            list1.push_back(index[i]);
        }
        else
        {
            list2.push_back(index[i]);
        }
    }
    return {list1, list2};
}

// Construct the decision tree by Breadth-first traversal and Multi-thread
// static omp_lock_t locklock;
Node buildLeaf(matrix &data,
               vector<int> &index,
               int category,
               vector<Ctxt> &c,
               const PubKey &publicKey,
               const SecKey &secretKey,
               Context &context)
{

    auto column_length = data[0].size();
    double rows_length = index.size();
    // double best_gain = INT_MAX;
    double best_gainCipher = -1000.0;
    // Record the best split node values
    int best_colCipher = -1;
    // double best_value = -1;
    double best_valueCipher = -1;
    vector<int> listLeft;
    vector<int> listRight;

    const int N = column_length - category;

//  omp_init_lock(&locklock); //Initialize a mutex
#pragma omp parallel for num_threads(64)
    for (int col = 1; col < N; col++)
    {
        // Possible values of each attribute
        set<double> col_value_set = getSplitSet(data, index, col);
        vector<double> col_value_vetor;
        col_value_vetor.assign(col_value_set.begin(), col_value_set.end());
        Ctxt tmpCipherL(publicKey);
        Ctxt tmpCipherR(publicKey);
        // Traverse all possible splitting point
        for (int valueIndex = 0; valueIndex < col_value_vetor.size(); valueIndex++)
        {

            double value = col_value_vetor[valueIndex];
            // cout<<"value"<<endl;
            vector<int> list1;
            vector<int> list2;

            auto tmp = splitDatas(data, index, value, col);
            list1 = tmp[0];
            list2 = tmp[1];

            double p1 = list1.size() / rows_length;
            double p2 = list2.size() / rows_length;
            double leftgini;
            double rightgini;
            if (list1.size() == 0 | list2.size() == 0)
            {
                continue;
            }
// clock_t start=clock();
#pragma omp parallel sections num_threads(2)
            {
#pragma omp section
                {
                    leftgini = gini(data, list1, category, c, tmpCipherL, secretKey, context);
                }
#pragma omp section
                {
                    rightgini = gini(data, list2, category, c, tmpCipherR, secretKey, context);
                }
            }
            tmpCipherL *= (1.0 / list1.size());
            tmpCipherR *= (1.0 / list2.size());
            tmpCipherL += tmpCipherR;
            totalSums(tmpCipherL);
            PtxtArray tmpGain(context);
            tmpGain.decrypt(tmpCipherL, secretKey);

            vector<double> PlainGain;
            tmpGain.store(PlainGain);

            // Obtain the optimal split point according to the minimum gini impurity
            if (PlainGain[0] > best_gainCipher)
            {
                // omp_set_lock(&locklock);
                best_gainCipher = PlainGain[0];
                best_colCipher = col;
                best_valueCipher = value;
                listLeft = list1;
                listRight = list2;
                // omp_unset_lock(&locklock);
            }
        }
        // omp_unset_lock(&locklock);
    }

    Node Leaves;
    Leaves.col = best_colCipher;
    Leaves.value = best_valueCipher;
    Leaves.left = listLeft;
    Leaves.right = listRight;
    return Leaves;
}

// Build the decision tree
vector<pair<Node, int>> buildDecisionTree(matrix &data,
                                          vector<int> &index,
                                          int maxleaves,
                                          int minNumInleaf,
                                          int category,
                                          vector<Ctxt> &c,
                                          const PubKey &publicKey,
                                          const SecKey &secretKey,
                                          Context &context)
{
    vector<pair<Node, int>> Tree;
    queue<vector<int>> DataIndex;
    DataIndex.push(index);
    int leavesNum = 0;
    while (DataIndex.size() > 0 && leavesNum <= 2 * maxleaves - 1)
    {
        cout << leavesNum << "leavesNum" << endl;
        // Represents whether the node is a leaf node, and  1 represents is
        int flag = 0;
        Node tmp;

        if (DataIndex.front().size() >= minNumInleaf)
        {
            tmp = buildLeaf(data, DataIndex.front(), category, c, publicKey, secretKey, context);
        }
        else
        {
            tmp.left = DataIndex.front();
            tmp.right = DataIndex.front();
            flag = 1;
        }
        DataIndex.pop();
        DataIndex.push(tmp.left);
        DataIndex.push(tmp.right);
        pair<Node, int> p1(tmp, flag);
        Tree.push_back(p1);
        leavesNum += 1;
    }
    return Tree;
}

// Classify each sample of the testing dataset
vector<double> classify(vector<double> &data, matrix &trainData, vector<pair<Node, int>> &tree, int index, int categorynum)
{
    if (tree[index].second == 1 || 2 * index + 2 >= tree.size())
    {
        Node result = tree[index].first;
        vector<double> labelCount = getLeafLabel(trainData, result.left, result.right, categorynum);
        return labelCount;
    }
    else
    {
        double v = data[tree[index].first.col];
        if (v <= tree[index].first.value)
        {
            return classify(data, trainData, tree, 2 * index + 1, categorynum);
        }
        else
        {
            return classify(data, trainData, tree, 2 * index + 2, categorynum);
        }
    }
}

// Classify  testing dataset
double predict(matrix &data, matrix &trainData, vector<pair<Node, int>> &tree, int category)
{
    double acc = 0.0;
    for (int i = 0; i < data.size(); i++)
    {
        // cout << "testdata" << " " << data[i][data[0].size() - 1] << endl;
        vector<double> result = classify(data[i], trainData, tree, 0, category); // labelCount
        double num = 0.0;
        double ans = -1; // label
        for (int i = 0; i < result.size(); i++)
        {
            if (result[i] > num)
            {
                ans = i;
                num = result[i];
            }
        }
        if (data[i][data[0].size() - category + ans])
        {
            acc += 1;
        }
    }
    return acc / data.size();
}

int main(int argc, char *argv[])
{
    vector<string> name = {"adult", "dataset_32_pendigits", "digits", "house", "bank", "amazon", "runwalk", "minist", "covtype", "phpB0xrNj", "cifar10"};
    vector<int> trainLen = {43956, 9890, 5057, 20504, 40688, 29491, 79728, 62999, 522909, 7015, 53999};
    vector<int> testLen = {4884, 1099, 561, 2278, 4521, 3276, 8858, 6999, 58101, 779, 5999};
    vector<int> attributeLen = {111, 27, 75, 19, 54, 12, 9, 795, 57, 644, 1035};
    vector<int> labelLen = {2, 10, 10, 2, 2, 2, 2, 10, 2, 26, 10};
    vector<string> name = {"cifar10"};
    vector<int> trainLen = {53999};
    vector<int> testLen = {5999};
    vector<int> attributeLen = {1035};
    vector<int> labelLen = {10};

    for (int itration = 0; itration < name.size(); itration++)
    {
        cout << name[itration] << endl;
        printf("Decision tree building........ \n");
        // training dataset
        int Data_row = trainLen[itration];
        int Data_col = attributeLen[itration];
        int category = labelLen[itration]; // one-hot label
        string train_url = "./data/train_" + name[itration] + ".csv";

        matrix train_data = ReadFile(train_url, Data_row, Data_col);

        // testing dataset
        Data_row = testLen[itration];
        string test_url = "./data/test_" + name[itration] + ".csv";
        matrix test_data = ReadFile(train_url, Data_row, Data_col);

        cout << train_data[0][0] << ',' << train_data[1][0] << endl;
        cout << train_data[2][0] << ',' << train_data[3][0] << endl;

        cout << test_data[0][0] << ',' << test_data[1][0] << endl;
        cout << test_data[2][0] << ',' << test_data[3][0] << endl;

        // Set the CKKS and encrypt the labels with CKKS
        Context context =
            ContextBuilder<CKKS>().m(16384).bits(119).precision(20).c(2).build();

        cout << "securityLevel=" << context.securityLevel() << "\n";

        long n = context.getNSlots();
        cout << "long n=" << context.securityLevel() << "\n";
        SecKey secretKey(context);
        secretKey.GenSecKey();
        addSome1DMatrices(secretKey);
        const PubKey &publicKey = secretKey;
        vector<int> index;

        // one-hot label ckks encrypt
        vector<vector<double>> v0(train_data.size());
        for (int i = 0; i < train_data.size(); i++)
        {
            index.push_back(train_data[i][0]);
            vector<double> v0itm(category);
            for (long j = 0; j < category; j++)
            {
                v0itm[j] =
                    train_data[i][Data_col - category + j];
            }
            v0[i] = v0itm;
        }

        vector<Ctxt> c;
        vector<PtxtArray> p;

        for (int i = 0; i < train_data.size(); i++)
        {
            p.emplace_back(context);
            p[i] = (context, v0[i]);
            c.emplace_back(publicKey);
            p[i].encrypt(c[i]);
        }

        // Decision tree training
        cout << "train decision tree" << endl;
        clock_t start, end;
        start = clock();
        int maxleaves = 64;
        int minNumInleaf = 20;
        double startOPm = omp_get_wtime();
        vector<pair<Node, int>> tree = buildDecisionTree(train_data,
                                                         index, maxleaves,
                                                         minNumInleaf,
                                                         category,
                                                         c,
                                                         publicKey,
                                                         secretKey,
                                                         context);
        end = clock();
        cout << "time = " << double(end - start) / CLOCKS_PER_SEC << "s" << endl;
        double endOPm = omp_get_wtime();
        cout << "Thread time = " << double(endOPm - startOPm) << "s" << endl;
        double result = predict(test_data, train_data, tree, category);
        cout << "test acc:" << result << endl;
    }
    return 0;
}