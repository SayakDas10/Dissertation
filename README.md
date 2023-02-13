# Table of contents
1. [First Meeting: Topic Selection (24 Jan, 2023)](#TopicSelection)
    * [NLP Task on Legal Documents](#NLP-Task-on-Legal-Documents)
    * [Bangla NLP](#Bangla-NLP)
    * [Summarization](#Summarization)

## Topic Selection (24 Jan, 2023)<a name='#TopicSelection'></a>

In the first meeting we discuss what topic to choose for my dissertation.

Possible topics in NLP include, 

* NLP task on legal documents.
* Bangla NLP
* Summarization

<font color='green'>Task before next meeting:</font> Explore the mentioned topics and select one for dissertation. 

### NLP Task on Legal Documents
* <a href='https://sites.google.com/view/legaleval/home?pli=1'>LegalEval: Understanding Legal Texts</a>

Legal texts are different from natural text. This calls for domain specific NLP models. They have proposed three subtasks as building blocks for developing large legal AI applications.

1. Rhetorical Roles Prediction: Structuring unstructured legal documents into semantically coherent units. We have to do the following task: Given a long unstructured legal document, automatically segment it into semantically coherent text segments and assign a label to each of them such as preamble, fact, ratio, arguments etc. These are Rhetorical Roles. More info about rhetorical roles, dataset and baseline can be found in this <a href='https://github.com/Legal-NLP-EkStep/rhetorical-role-baseline'> GitHub </a> link. They are using a weighted F1 score based on hidden dataset for evaluation. <a href='https://colab.research.google.com/drive/1FRxgadvvMem8Z_Wtq_-CChk97X2DVdt2#scrollTo=EZRDvJeYNS-7'> Colab baseline </a>  
2. Legal Named Entity Recognition (L-NER): Identifying relevant entities in a legal document. Our task is the following: Standard NERs (StanfordNLP NER <a href='https://stanfordnlp.github.io/CoreNLP/ner.html'>link</a>) struggle to recognize legal document entities like names of petitioner, respondent, court, statute, provision, precedents, etc. A list of legal named entities and datasets are  given <a href='https://github.com/Legal-NLP-EkStep/legal_NER'> here </a>. Datasets for preamble and judgement are provided separately. They are using standard F1 score for evaluation. 
3. Court Judgement Prediction with Explanation (CJPE): Predicting the outcome of a case along with an explanation. Our task is to predict the outcome (binary: accepted or denied) of the case along with an explanation. Naturally, this task can be broken into two parts. Firstly, we have to predict the outcome and then we have to come up with an explanation of the prediction. They have used F1 score for the outcome and BLEU, METEOR and ROUGE scores for machine explanation for evaluation of the two subtasks. Dataset and the explanation can be found in this <a href='https://github.com/Exploration-Lab/CJPE'> GitHub </a> link. 

* <a href='https://arxiv.org/pdf/2201.13125.pdf'>Corpus for Automatic Structuring of Legal Documents </a>
* <a href='https://arxiv.org/pdf/2112.01836.pdf'>Semantic Segmentation of Legal Documents via Rhetorical Roles</a>

* <a href='https://arxiv.org/pdf/2211.03442.pdf'> Named Entity Recognition in Indian court judgements </a>

* <a href='https://arxiv.org/pdf/2105.13562.pdf'>ILDC for CJPE: Indian Legal Documents Corpus for Court Judgement Prediction and Explanation </a>

### Bangla NLP
* <a href='https://github.com/sagorbrur/bnlp'>Bangle NLP GitHub</a>

### Summarization
* <a href='https://github.com/sagorbrur/bnlp'>ACL Website</a>

## Next Meeting (31 Jan, 2023 [16:00])

* Read The ILDC for CJPE paper throughly
    * Understand the baseline model architectures
* Wait for dataset response
* Read Attention is all you need paper
* Implement Attention block using cross-entropy loss.  


## Meeting 6 Feb
 * Dataset summary (number of sentences, words etc) verify with the paper
 * implement one of the baselines they have used and try to reproduce their results
 * prepare slides and explain sir their models

## Next meeting Friday 4pm
