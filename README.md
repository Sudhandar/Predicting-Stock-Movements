# Predicting-Stock-Movements

The transformer-based models have become a cornerstone technology in processing the sequential textual data and can learn the underlying relation between tokens in a large text, e.g., financial, legal datasets. One of the largest sources of financial text is Twitter where the market trends of various companies can be identified from the tweets. 
An efficient prediction mechanism is necessary to realize the overall relationship between the financial texts and the market. This project investigates the tweets related to the stock market to predict the relevant stock movements. The transformer-based models such as BERT and GPT-2 are used to perform sequence classification by understand the textual input and these models are utilized to predict the stock movements.

## Dataset

The stock data can be categorized into the following 9 groups, basic materials, consumer goods, healthcare, services, utilities, conglomerates, financial, industrial goods, and technology. This project uses the StockNet dataset presented in the paper [1]. This project focuses on the binary classification of the stock movement, (i.e., high or low), for a given
stock on a particular day. To generate the target variable, the movement percentage for each day is derived. To address the issue of stocks with extremely minor movement percentages, the stocks with movement percentages ≤ −0.5% are classified as 0 (low) and movement percentages > 0.5% are classified as 1 (high). Using the above setting, 26623 
targets are identified with 13368 targets with positive (i.e.,1) labels and 13255 targets with negative (i.e.,0) labels respectively. The tweets from the StockNet dataset have been linked to their target stocks based on their timestamp. To predict the stock movement for stock S on a particular target day d,  the tweets posted between the closure of the stock market on d − t days to the current opening on target day d, are considered. The reason for this is to prevent future information from entering the prediction on a target day d. After experimentation, a lag of 3 days has been considered.

## Approach

To predict the stock movement, two methods are. First, the tweets related to stock S are fed into BERT/GPT-2 to train the models to predict the stock movements. The models can directly predict the stock movement based on the information learned from the textual corpora,
(i.e., 0/1 ). In the second method, the system utilizes a larger twitter dataset with approximately 1.6 million samples. The BERT model is trained on the 1.6 million tweet samples to predict the sentiments. This trained model is used to predict the sentiments on the StockNet Twitter dataset.  The average of the sentiments is calculated for a target stock S, on a target trading day d and compared with the stock movements for the target stock S, on a target trading day d. 
