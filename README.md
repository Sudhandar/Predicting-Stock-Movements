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

### Approach 1

![alt text](https://github.com/Sudhandar/Predicting-Stock-Movements/blob/main/images/approach_1.png)

### Approach 2

![alt text](https://github.com/Sudhandar/Predicting-Stock-Movements/blob/main/images/approach_2.png)

## Fine tuning BERT

Instead of using a single learning rate, several variations of layer-wise learning rate decays have been used to improve the performance of BERT/GPT-2. The final results show that the technique significantly enhances the performance. Furthermore,another type of layer-wiselearning rate decay where 12 layers of BERT are grouped into sets of 4 layers and different learning rates are applied to each set known as grouped layer-wise Learning Rate Decay is incorporated. The method resulted in even better performance of the BERT model. The following are the final learning rates used for the model,

- Set 1: Embeddings + Layer 0, 1, 2, 3, (learning rate: 1e−6)
- Set 2: Embeddings + Layer 4, 5, 6, 7, (learning rate: 1.75e−6)
- Set 3: Embeddings + Layer 8, 9, 10, 11, (learning rate: 3.5e−6)

### Layer-wise learning rate decay ([Image Source](https://towardsdatascience.com/advanced-techniques-for-fine-tuning-transformers-82e4e61e16e))

![alt text](https://github.com/Sudhandar/Predicting-Stock-Movements/blob/main/images/layer_rate_decay.png)


### Grouped layer-wise learning rate decay ([Image Source](https://towardsdatascience.com/advanced-techniques-for-fine-tuning-transformers-82e4e61e16e))

![alt text](https://github.com/Sudhandar/Predicting-Stock-Movements/blob/main/images/grouped_rate_decay.png)


### Re-initializing pre-trained layers of BERT

BERT has 12 layers, and each layer of the BERT captures various kinds of information. The lower layers contain low-level representations and store generic information. The task-related information is stored on the top layers of the BERT closer to the output. (Zhang et al. [2]) in their paper,suggested that re-initializing these top layers will increase the performance of BERT on several downstream tasks. Based on their work,  the top 3 layers of the BERT have been reinitialzed for the direct stock movement prediction model.

## Results

### BERT Results

| Models                                                                                                   | Accuracy | MCC    |
|----------------------------------------------------------------------------------------------------------|---------|--------|
| Base BERT model                                                                                          | 46.7    | -0.03  |
| BERT + Layer wise learning rate decay                                                                    | 47.8    | 0.005  |
| BERT + Grouped Layer wise learning rate decay                                                            | 51      | 0.007  |
| BERT + Grouped Layer wise learning rate decay +<br>Reinitializing top 3 layers of BERT + 50 warmup steps | 53      | 0.0344 |

### BERT Approach 1 vs Approach 2 results

| Models                                       |Accuracy | MCC    |
|----------------------------------------------|---------|--------|
| BERT direct prediction                       | 53      | 0.0344 |
| BERT prediction based on sentiments analysis | 52.3    | 0.025  |

### GPT-2 Results

| Models                  |Accuracy | MCC   |
|-------------------------|---------|-------|
| GPT-2 Base              | 51      | 0.011 |
| GPT-2 after fine-tuning | 54      | 0.024 |

### Comparison with baseline models

The performance of the proposed models is compared with the baseline models used in [1]. The following are the baseline models considered,
- RAND: a naive predictor making random guesses up or down.
- ARIMA: Autoregressive Integrated Moving Average, an advanced technical analysis method using only price signals
- RANDFOREST: a discriminative Random Forest classifier using Word2vec text representations
- TSLDA: a generative topic model jointly learning topics and sentiments
- HAN: a state-of-the-art discriminative deep neural network with hierarchical attention

| Models     |Accuracy | MCC       | Models                                           |Accuracy | MCC    |
|------------|---------|-----------|--------------------------------------------------|---------|--------|
| RAND       | 50.89   | −0.002266 | BERT direct prediction                           | 53      | 0.0344 |
| ARIMA      | 51.31   | −0.020588 | BERT prediction based on <br>sentiments analysis | 52.3    | 0.025  |
| RANDFOREST | 50.08   | 0.012929  | GPT-2                                            | 54      | 0.024  |
| TSLDA      | 54.07   | 0.065382  |                                                  |         |        |
| HAN        | 57.64   | 0.0518    |                                                  |         |        |


Furthermore, the performance of the proposed models is also compared with the models introduced in [1].The following are presented as the StockNet variations introduced in [1],

- TECHNICALANALYST: the generative StockNet using only historical prices.
- FUNDAMENTALANALYST: the generative StockNet using only tweet information.
- INDEPENDENTANALYST: the generative StockNet without temporal auxiliary targets.
- DISCRIMINATIVEANALYST: the discriminative StockNet directly optimizing the likeli-
hood objective.

| StockNet variations    |Accuracy | MCC      | Models                                       |Accuracy | MCC    |
|------------------------|---------|----------|----------------------------------------------|---------|--------|
| TECHNICAL ANALYST      | 54.96   | 0.016456 | BERT direct prediction                       | 53      | 0.0344 |
| FUNDAMENTAL ANALYST    | 58.23   | 0.071704 | BERT prediction based on sentiments analysis | 52.3    | 0.025  |
| INDEPENDENT ANALYST    | 57.54   | 0.03661  | GPT-2                                        | 54      | 0.024  |
| DISCRIMINATIVE ANALYST | 56.15   | 0.056493 |                                              |         |        |
| HEDGEFUND ANALYST      | 58.23   | 0.080796 |                                              |         |        |

### Results Evaluation

Even though BERT and GPT-2 are trained on large text corpora and surpass state-of-the-art results in many language tasks it is still not able to achieve higher accuracy and MCC than certain StockNet Variations like Fundamental Analyst and Hedgefund analyst. On further analyzing the reason behind this and found out that the main reason is the use of the temporal auxiliary attention mechanism in StockNet. It acts as a denoising regularizer which helps the model to filter out noises like a temporary rise in a positive movement when the market has an upward trend and helps the model to focus on the main target and generalize well by denoising. This task-specific attention mechanism is not found in models like BERT and GPT-2 even though they learn context by masked self-attention.

## Reference

- [1] Xu, Y., &amp; Cohen, S. B. (2018). Stock movement prediction from tweets and historical prices. Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). https://doi.org/10.18653/v1/p18-1183 
- [2] Zhang, T., Wu, F., Katiyar, A., Weinberger, K. Q., & Artzi, Y. (2020). Revisiting few-sample BERT fine-tuning. arXiv preprint arXiv:2006.05987.
- [3] https://towardsdatascience.com/advanced-techniques-for-fine-tuning-transformers-82e4e61e16e