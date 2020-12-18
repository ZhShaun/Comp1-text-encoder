# Pretrained encoders
參考report的[github](https://github.com/samirsen/image-generator)。

大多數pretrained encoder都是end to end架構，接收字串等input後就會輸出固定shape的feature vector。正確安裝後，api相對簡單。但因為那些encoder安裝過程大多需要下載較大的model file，實際安裝還是需要自己本地裝。
pretrained encoder輸出的是n_word\*features的shape。不用LSTM等架構的情況下就是直接average各單字獲得size為features的feature vector。

本專案提供的是安裝上需要注意的細節(包跨版本相容等原始專案不容易找到解決方法的問題)，以及需要替換以解決問題的檔案。安裝完畢後可以直接使用encoder_lazyApi方便使用。

## lazyApi Encoder
```python
from encoder_lazyApi import Encoder
# path = $model_path (only needed if using word2vec)
encoder = Encoder(*model_name*, *path*)
features = encoder(*image caption*)
```
path指的是你下載的word2vec pretrained path，只接受.bin檔。使用skip thought的話不用輸入。
input直接輸入image caption就可以了，不需要前處理。
output是各model的features大小(平均每個單字)。skip: 4800; word2vec: 300。

## Skip-thought 

**相比word2vec或GloVe優點是skip比較能捕捉語意，所以都是直接套的話就算效果都會比較好。缺點是字彙量比較小(800k)，不夠精確在我們的domain。適合當base line。**

**根據參考的那篇report，如果不額外想辦法domain adaption的話效果上限就在那邊。**

主要照[github project](https://github.com/ryankiros/skip-thoughts)安裝即可，但因為有版本問題需要額外注意一些細節

### requirement

和官網不一樣的部分就是我自己試要改才能正確跑的部分。

* Python3都可
    * py檔大多只有python2和3的語法衝突，skipthought.py換掉即可
    * 其他py檔雖然語法也有問題，但目前我們用不到
* 最新Theano
    * 官網的版本內建的Cython和Python3.7有問題，需要更新到最新
* A recent version of NumPy and SciPy
* scikit-learn
* NLTK 3
* Keras (for Semantic-Relatedness experiments only)
* gensim (for vocabulary expansion when training new models)
    * 網路上使用word2vec的library大多是用這個，當然用keras版本的word2vec api也是可以

### installation

1. git clone https://github.com/ryankiros/skip-thoughts.git
2. 用本專案的版本替換掉原來的 skipthoughts.py
3. 需要下載他的字典和pretrained model，**檔案總共5G左右比較大需要下載比較久，想要可以先下載**

```shell=
mkdir word_embeddings
cd word_embeddings
wget http://www.cs.toronto.edu/~rkiros/models/dictionary.txt
wget http://www.cs.toronto.edu/~rkiros/models/utable.npy
wget http://www.cs.toronto.edu/~rkiros/models/btable.npy
wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz
wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl
wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz
wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl
```
4. skipthoughts.py 裡面的path_to_models和path_to_tables要改成你本機word_embeddings資料夾的位置
### Usage

```python=
import skipthoughts
model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)
vectors = encoder.encode(X)
```
vectors[1]就是我們要的feature vectors。vectors[0]是input size。

Input: 原本的image caption即可。不須經過前處理。
Output: n_words\*4800 。

### Word expansion

skip可以透過訓練linear mapping來從其他大的word vector，像是word2vec轉換成skip自己的字彙量。這個功能需要稍微實作一下。

但是我有稍微從image caption上的單字去找，發現大多數情況skip還是有我們的單字。所以我覺得效果不夠好問題比較是在於太過general，不夠貼近我們的task。

## Word2Vec

**優點是字彙量很大，缺點是捕捉語意效果沒那麼好，也不夠domain specific。**

[model zoo](https://github.com/3Top/word2vec-api#where-to-get-a-pretrained-models)在這。各model差別只在訓練的data不一樣。我用的是規模最大的google news 300。

安裝相對簡單，下載好pretrained model後用gensim或keras word2vec api就可以直接用。需要做基本的text preprocessing。

### Usage (gensim)

需要做基本的text preprocessing。
1) Convert to lower 
2) Remove punctuations/symbols/numbers (but it is your choice) 
3) Normalize the words (lemmatize and stem the words)


[官網參考](https://radimrehurek.com/gensim/models/word2vec.html)

```python=
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.test.utils import datapath
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format(datapath("/home/anomaly_detection_project_2020/weight_lifting_trainer/dl/comp3/GoogleNews-vectors-negative300.bin"), binary=True)
vectors = model[X]
```
其中X是前處理過的word array。

input:前處理過的image caption。
output: n_words\*400。

## Domain adaption

原因是假如說單字同樣為red，我們的task中就會希望他預測的下一個字是flower的比重要比較高。但是word vector本身訓練成本高，且字彙量很大的情況下額外訓練效果不彰。

參考report中的想法是freeze住encoder，在encoder後面接上並訓練LSTM和fc layer。訓練來源是在GAN訓練過程中generator的gradient。也就是說不是直接透過平均各單字，而是使用RNN來得到更準確的representation。