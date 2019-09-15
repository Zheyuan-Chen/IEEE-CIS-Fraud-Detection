# Competition Description
IEEE-CIS works across a variety of AI and machine learning areas, including deep 
neural networks, fuzzy systems, evolutionary computation, and swarm intelligence. 
Today they’re partnering with the world’s leading payment service company, 
Vesta Corporation, seeking the best solutions for fraud prevention industry, 
and now you are invited to join the challenge.

In this competition, you’ll benchmark machine learning models on a challenging 
large-scale dataset. The data comes from Vesta's real-world e-commerce transactions 
and contains a wide range of features from device type to product features. 
You also have the opportunity to create new features to improve your results.

If successful, you’ll improve the efficacy of fraudulent transaction alerts for millions 
of people around the world, helping hundreds of thousands of businesses reduce their 
fraud loss and increase their revenue. And of course, you will save party people just 
like you the hassle of false positives.

# Data Description
In this [competition](https://www.kaggle.com/c/ieee-fraud-detection/overview/description)
you are predicting the probability that an online transaction is 
fraudulent, as denoted by the binary target isFraud.

The data is broken into two files identity and transaction, which are joined 
by TransactionID. Not all transactions have corresponding identity information.

 ## Transaction Table 
* TransactionDT: timedelta from a given reference datetime (not an actual timestamp)
* TransactionAMT: transaction payment amount in USD
* ProductCD: product code, the product for each transaction
* card1 - card6: payment card information, such as card type, card category, issue bank, 
country, etc.
* addr: address
* dist: distance
* P_ and (R__) emaildomain: purchaser and recipient email domain
* C1-C14: counting, such as how many addresses are found to be associated with the payment card, etc. The actual meaning is masked.
* D1-D15: timedelta, such as days between previous transaction, etc.
* M1-M9: match, such as names on card and address, etc.
* Vxxx: Vesta engineered rich features, including ranking, counting, and other entity relations.

### Categorical Features:
* ProductCD
* card1 - card6
* addr1, addr2
* Pemaildomain Remaildomain
* M1 - M9

## Identity Table 
Variables in this table are identity information – network connection information (IP, ISP, 
Proxy, etc) and digital signature (UA/browser/os/version, etc) associated with transactions.
They're collected by Vesta’s fraud protection system and digital security partners.
(The field names are masked and pairwise dictionary will not be provided for privacy 
protection and contract agreement)

### Categorical Features:
* DeviceType
* DeviceInfo
* id12 - id38

# Files
train_{transaction, identity}.csv - the training set
test_{transaction, identity}.csv - the test set (you must predict the isFraud value 
for these observations)
sample_submission.csv - a sample submission file in the correct format.

# Submission
Submissions are evaluated on area under the ROC curve between the predicted probability 
and the observed target.