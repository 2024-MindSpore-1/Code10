## Code description

### Our Gini-impurity preserving encryption
- Version 1:
   (1) Run the ***GIPE_string.py*** and get the string cipher
   (2) Then run the ***GIPE_string2int.py*** to process the string (above) to int
- Version 2:
   (1) Directly run the ***GIPE_minmax.py*** and get the encrypted dataset (type: int)

(Here we give two methods to realize our Gini-impurity preserving encryption for the features, while the Version 2 is consistent with the pseudo code in our paper)

### The encrypted DT and RF
(1) ***EncryptRandomForest.py*** can train the random forest by the sklearn after we encrypt the feature space by our Gini-impurity preserving encryption
(2) ***EncryptDecisionTree_CKKS.py*** can train each tree in the random forest while the feature space is encrypted by our  Gini-impurity preserving encryption and label space is encrypted by the CKKS (HElib library)

 