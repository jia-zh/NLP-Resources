# NLP-Resources
[![](https://img.shields.io/badge/update-everyday-success.svg)](https://github.com/jia-zh/NLP-Resources)
  
A useful list of NLP(Natural Language Processing) resources 
  
自然语言处理的相关资源列表，持续更新
  
  
## Contents
- [NLP Toolkit 自然语言工具包](#nlp-toolkit-自然语言工具包)
- [NLP Corpus 自然语言处理语料库](#nlp-corpus-自然语言处理语料库)
- [ML and DL 机器学习和深度学习](#ml-and-dl-机器学习和深度学习)
- [NLP Technology 自然语言处理相关技术](#nlp-technology-自然语言处理相关技术)
  - [Bert Model](#bert-model)
  - [Text Modeling](#text-modeling)
  - [Text Similarity](#text-similarity)
  - [Information Extraction](#information-extraction)
  - [Knowledge Graph](#knowledge-graph)
  - [Reading Comprehension](#reading-comprehension)
- [NLP Blogs and Courses 博客和课程](#nlp-blogs-and-courses-博客和课程)
  - [NLP Blogs 博客](#nlp-blogs)
  - [NLP Courses 博客](#nlp-courses)
- [NLP Organizations 学术组织](#nlp-organizations-学术组织)
  - [中国大陆地区高校/研究所](#china-school)
  - [中国大陆地区企业](#china-company)
  - [中国香港/澳门/台湾地区](#china-hmt)
  - [新加坡/日本/以色列/澳大利亚](#east-asia)
  - [北美地区](#north-america)
  - [欧洲地区](#europe)
  
  
### NLP-Toolkit 自然语言工具包

- [CoreNLP](https://github.com/stanfordnlp/CoreNLP)： a set of natural language analysis tools written in **Java**，by Stanford

- [NLTK](http://www.nltk.org/)：a **Python** Natural Language Toolkit includes corpora, lexical resources and text processing libraries

- [gensim](https://radimrehurek.com/gensim/)：[Github](https://github.com/RaRe-Technologies/gensim)，a **Python** library for topic modelling, document indexing and similarity retrieval with large corpora

- [LTP](https://github.com/HIT-SCIR/ltp)：[语言技术平台](http://ltp.ai/)，中文NLP工具，支持**Java & Python**，by 哈工大

- [jieba](https://github.com/fxsjy/jieba)：结巴中文分词，做最好的 **Python** 中文分词组件，现已覆盖几乎所有的语言和系统

- [NLPIR](https://github.com/NLPIR-team/NLPIR)：**Java** 分词组件，by 中科院/北理工

- [HanLP](https://github.com/hankcs/HanLP)：中文NLP模型与算法工具包，支持**Java & Python**，by 上海林原信息科技有限公司

- [THULAC](http://thulac.thunlp.org/)：高效的中文词法分析工具包，支持**C++ & Java & Python**，by 清华

- [pkuseg](https://github.com/lancopku/pkuseg-python)：多领域中文分词工具包，支持细分领域分词，支持**Python**，by 北大

- [FudanNLP](https://github.com/FudanNLP/fnlp)：中文NLP工具包、机器学习算法和数据集，支持**Java**，by 复旦

  
### NLP Corpus 自然语言处理语料库

- [中文 Wikipedia Dump](https://dumps.wikimedia.org/zhwiki/)
  
- [人民日报199801标注语料](https://pan.baidu.com/s/10_CQck5mKsKyfpA08slyFA)
  
- [Sogou Labs](http://www.sogou.com/labs/resource/list_pingce.php) 互联网词库、中文词语搭配库、全网新闻数据（2012）、搜狐新闻数据（2012）、互联网语料库、链接关系库等
       
- [中文聊天语料](https://github.com/codemayq/chaotbot_corpus_Chinese) chatterbot、豆瓣多轮、PTT八卦语料、青云语料、电视剧对白语料、贴吧论坛回帖语料、微博语料、小黄鸡语料
         
- [领域中文词库](https://github.com/thunlp/THUOCL) IT、财经、成语、地名、历史名人、诗词、医学、饮食、法律、汽车、动物
         
- [汉语词库](http://www.hankcs.com/nlp/corpus/tens-of-millions-of-giant-chinese-word-library-share.html) 各种类型词库如人名库、金融专业相关词、政府机关团体机构大全等

- [中文依存语料库](http://www.hankcs.com/nlp/corpus/chinese-treebank.html) 第二届自然语言处理与中文计算会议（NLP&CC 2013）的技术评测中文树库语料
       
- [微信公众号语料库](https://github.com/nonamestreet/weixin_public_corpus) 网络抓取的微信公众号的文章，包括微信公众号名字、微信公众号ID、题目和正文
       
- [中文谣言微博数据](https://github.com/thunlp/Chinese_Rumor_Dataset) 从新浪微博不实信息举报平台抓取的中文谣言数据
  
- [Tencent AI Lab Embedding Corpus](https://ai.tencent.com/ailab/nlp/embedding.html) A corpus on continuous distributed representations of Chinese words and phrases
  
- [Word2vec Slim](https://github.com/eyaler/word2vec-slim) word2vec Google News model slimmed down to 300k English words
    
- [Chinese Word2vec Model](https://github.com/to-shimo/chinese-word2vec)
     
- [Chinese Word Vectors](https://github.com/Embedding/Chinese-Word-Vectors)
  
- [NLP Chinese Corpus](https://github.com/brightmart/nlp_chinese_corpus) 维基百科中文词条、新闻语料、百科问答、社区问答、翻译语料
       
- [中文诗歌古典文集数据库](https://github.com/chinese-poetry/chinese-poetry)

- [Chinese RC Dataset](https://github.com/ymcui/Chinese-RC-Dataset) A Chinese Reading Comprehension Dataset
  
- [Chinese Word Ordering Errors Detection and Correction Corpus](http://nlg.csie.ntu.edu.tw/nlpresource/woe_corpus/)
  
- [中文文本分类数据集THUCNews](http://thuctc.thunlp.org/#%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E6%95%B0%E6%8D%AE%E9%9B%86THUCNews) 根据新浪新闻RSS订阅频道2005~2011年间的历史数据筛选过滤生成，包含74万篇新闻文档
   
- [公司名语料库](https://github.com/wainshine/Company-Names-Corpus) 公司名语料库、机构名语料库、公司简称、品牌词等
  
- [中文人名语料库](https://github.com/wainshine/Chinese-Names-Corpus) 中文常见人名、中文古代人名、日文人名、翻译人名、中文姓氏、中文称呼、成语词典
     
- [中文简称词库](https://github.com/zhangyics/Chinese-abbreviation-dataset)
         
- [对联数据集](https://github.com/wb14123/couplet-dataset)
  
- [无忧无虑中学语文网](http://jyc.5156edu.com/) 常见中文词语工具，包括近义词、反义词、汉字拼音转换、简繁转换等
  
- [EmotionLexicon](https://github.com/songkaisong/EmotionLexicon) 细粒度情感词典、网络词汇、否定词典、停用词典
  
- [Chinese_Dictionary](https://github.com/guotong1988/chinese_dictionary) 同义词表、反义词表、否定词表
  
- [Synonyms](https://github.com/huyingxi/Synonyms) 中文近义词工具包
      
- [NLP太难了系列](https://github.com/fighting41love/hardNLU)
  
### ML and DL 机器学习和深度学习

- [Machine Learning](https://github.com/wepe/MachineLearning) 一些常见的机器学习算法的实现代码
  
- [Tensorflow Cookbook](https://github.com/taki0112/Tensorflow-Cookbook) Simple Tensorflow Cookbook for easy-to-use

- [Awesome pytorch List](https://github.com/taki0112/Tensorflow-Cookbook) A comprehensive list of pytorch related content such as different models, implementations. helper libraries, tutorials etc.
  
- [PyTorch实战指南](https://zhuanlan.zhihu.com/p/29024978) PyTorch代码的组织和结构设计

### NLP Technology 自然语言处理相关技术

<a name="bert-model"></a>

- Bert Model
  
  - [BERT](https://github.com/google-research/bert) TensorFlow code and pre-trained models for [BERT](https://arxiv.org/abs/1810.04805)
  
  - [Bert as Service](https://github.com/hanxiao/bert-as-service) Using BERT model as a sentence encoding service, i.e. mapping a variable-length sentence to a fixed-length vector
  
  - [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT) A PyTorch implementation of Google AI's BERT model provided with Google's pre-trained models, examples and utilities.
  
  - [BERT Classification Tutorial](https://github.com/Socialbird-AILab/BERT-Classification-Tutorial)

  - [BERT Utils](https://github.com/terrifyzhao/bert-utils) BERT生成句向量，BERT做文本分类、文本相似度计算

  - [BERT BiLSTM CRF NER](https://github.com/macanv/BERT-BiLSTM-CRF-NER) Tensorflow solution of NER task Using BiLSTM-CRF model with Google BERT Fine-tuning

  - [BERT Chinese NER](https://github.com/ProHiryu/bert-chinese-ner) 使用预训练语言模型BERT做中文NER

<a name="text-modeling"></a>

- Text Modeling
  
  - [Self Attention Mechanism](https://github.com/roomylee/self-attentive-emb-tf) Simple Tensorflow Implementation of "[A Structured Self-attentive Sentence Embedding](https://arxiv.org/abs/1703.03130)" (ICLR 2017) 
  
  - [Encoder Decoder](https://github.com/jacoxu/encoder_decoder) Four styles of encoder decoder model by Python, Theano, Keras and Seq2Seq
  
  - [Seq2seq](https://github.com/farizrahman4u/seq2seq) Sequence to Sequence Learning with Keras
  
  - [Keras Language Modeling](https://github.com/codekansas/keras-language-modeling) Some code for doing language modeling with Keras, in particular for question-answering tasks
  
  - [CNN for Sentence Classification in Keras](https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras) Simple Keras Implementation of "[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)" (EMNLP 2014) 
  
  - [CNN for Classification](https://github.com/Shijihao/CNN_for_classification) Simple Pytorch Implementation of "[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)" (EMNLP 2014) 


<a name="text-similarity"></a>

- Text Similarity
  
  - [Cilin and Hownet](https://github.com/yaleimeng/Final_word_Similarity) 综合了同义词词林扩展版与知网（Hownet）的词语相似度计算方法

<a name="information-extraction"></a>

- Information Extraction
    
  - [Open IE Papers](https://github.com/NPCai/Open-IE-Papers) This list containts OpenIE and ORE papers and resources
  
  - [关系抽取(分类)总结](http://shomy.top/2018/02/28/relation-extraction/) 对近几年(到2017)一些关系抽取/分类(Relation Extraction)的部分文献的一个简单总结

  - [LM-LSTM-CRF](https://github.com/jia-zh/NLP-Resourcesss/blob/master/Information%20Extraction.md) PyTorch Implementation of "[Empower Sequence Labeling with Task-Aware Neural Language Model](http://arxiv.org/abs/1709.04109)" (AAAI 2018)
  
  - [Named Entity Realtion Extraction](https://github.com/twjiang/NamedEntity_realtion_extraction) 基于句法分析的命名实体关系抽取程序
  
  - [Pytorch Relation Extraction](https://github.com/ShomyLiu/pytorch-relation-extraction) Pytorch 复现 [PCNN + MIL (Zeng 2015)](http://aclweb.org/anthology/D/D15/D15-1203.pdf) 与 [PCNN + ATT (Lin 2016)](http://nlp.csai.tsinghua.edu.cn/~lyk/publications/acl2016_nre.pdf)
  
  - [Zh NER TF](https://github.com/fighting41love/zh-NER-TF) A very simple BiLSTM-CRF model for Chinese Named Entity Recognition 中文命名实体识别 (TensorFlow)
  
  - [BERT BiLSTM CRF NER](https://github.com/macanv/BERT-BiLSTM-CRF-NER) Tensorflow solution of NER task Using BiLSTM-CRF model with Google BERT Fine-tuning
  
  - [Text Info Exp](https://github.com/Roshanson/TextInfoExp) TF-IDF、文本分类、聚类、词向量、情感识别、关系抽取等

<a name="knowledge-graph"></a>

- Knowledge Graph
  
  - [Agriculture KnowledgeGraph](https://github.com/cjm1044642385/Agriculture_KnowledgeGraph) 农业领域的信息检索，命名实体识别，关系抽取，分类树构建，数据挖掘

<a name="reading-comprehension"></a>

- Reading Comprehension
  
  - [CMRC 2017](https://github.com/ymcui/cmrc2017) The First Evaluation Workshop on Chinese Machine Reading Comprehension (CMRC 2017)
  
  - [CMRC 2018](https://github.com/ymcui/cmrc2018) The Second Evaluation Workshop on Chinese Machine Reading Comprehension (CMRC2018)
  
### NLP Blogs and Courses 博客和课程
  
<a name="nlp-blogs"></a>
  
- NLP Blogs 博客

  - [52nlp 我爱自然语言处理](http://www.52nlp.cn/)
  
  - [hankcs 码农场](http://www.hankcs.com/)

  - [剑指汉语自然语言处理](https://blog.csdn.net/FontThrone/column/info/16265)

  - [natural language processing blog](https://nlpers.blogspot.com/)

  - [Google AI Blog](https://ai.googleblog.com/)
  
  - [Language Log](http://languagelog.ldc.upenn.edu/nll/)

  - [Jay Alammar](http://jalammar.github.io/)
  
<a name="nlp-courses"></a>
  
- NLP Courses 课程

  - [斯坦福CS224d](http://cs224d.stanford.edu/syllabus.html)
  
  - [Oxford-CS-Deepnlp-2017](https://github.com/oxford-cs-deepnlp-2017)
  
  - [Gt NLP Class CS 4650 and 7650](https://github.com/jacobeisenstein/gt-nlp-class)
  
### NLP Organizations 学术组织
***排名不分先后，收集不全，欢迎完善***

- [ACL Anthology](https://aclanthology.info/)

- [NLP Conference Calender](http://cs.rochester.edu/~omidb/nlpcalendar/)
  
<a name="china-school"></a>
  
- 中国大陆地区高校/研究所

  - [清华大学自然语言处理与人文计算实验室](http://nlp.csai.tsinghua.edu.cn/site2/index.php/zh)[Github](https://github.com/thunlp)

  - [智能技术与系统国家重点实验室信息检索课题组](http://www.thuir.cn/)

  - [北京大学计算语言学教育部重点实验室](http://klcl.pku.edu.cn/)

  - [北京大学计算语言所](http://icl.pku.edu.cn/)

  - [北京大学计算机科学技术研究所语言计算与互联网挖掘研究室](http://59.108.48.12/lcwm/index.php?title=%E9%A6%96%E9%A1%B5)

  - [中科院计算所自然语言处理研究组](http://nlp.ict.ac.cn/index_zh.php)

  - [中科院自动化所自然语言处理研究组](http://nlp.ict.ac.cn/index_zh.php)

  - [中科院软件所中文信息处理实验室](http://www.icip.org.cn/zh/homepage/)

  - [哈工大社会计算与信息检索研究中心](http://ir.hit.edu.cn/)

  - [哈工大智能技术与自然语言处理实验室](http://insun.hit.edu.cn/main.htm)

  - [哈工大机器智能与翻译研究室](http://mitlab.hit.edu.cn/)

  - [复旦大学自然语言处理研究组](http://nlp.fudan.edu.cn/)

  - [苏州大学自然语言处理组](http://nlp.suda.edu.cn/)

  - [苏州大学人类语言技术研究所](http://hlt.suda.edu.cn/)

  - [南京大学自然语言处理研究组](http://nlp.nju.edu.cn/homepage/)

  - [东北大学自然语言处理实验室](http://www.nlplab.com/)

  - [厦门大学自然语言处理实验室](http://nlp.xmu.edu.cn/)

  - [郑州大学自然语言处理实验室](http://nlp.zzu.edu.cn/)
  
<a name="china-company"></a>
  
- 中国大陆地区企业

  - [微软亚洲研究院自然语言计算组](https://www.microsoft.com/en-us/research/group/natural-language-computing/)

  - [华为诺亚方舟实验室](http://www.noahlab.com.hk/)
  
  - [Tencent AI Lab NLP Center](https://ai.tencent.com/ailab/nlp/)
  
  - [百度自然语言处理部](https://nlp.baidu.com/)
  
  - [头条人工智能实验室（Toutiao AI Lab）](http://lab.toutiao.com/)
  
<a name="china-hmt"></a>
  
- 中国香港/澳门/台湾地区
  
  - [CUHK Text Mining Group](http://www1.se.cuhk.edu.hk/~textmine/)（香港中文大学文本挖掘组）

  - [PolyU Social Media Mining Group](http://www4.comp.polyu.edu.hk/~cswjli/Group.html)（香港理工大学社交媒体挖掘组）

  - [HKUST Human Language Technology Center](http://www.cse.ust.hk/~hltc/)（香港科技大学人类语言技术中心）
  
  - [NLP<sup>2</sup>CT @ University of Macau](http://nlp2ct.cis.umac.mo/index.html)（澳门大学自然语言处理与中葡机器翻译实验室）

  - [National Taiwan University NLP Lab](http://nlg.csie.ntu.edu.tw/)（台湾大学自然语言处理实验室）

<a name="east-asia"></a>
  
- 新加坡/日本/以色列/澳大利亚

  - [NUS Natural Language Processing Group](http://www.comp.nus.edu.sg/~nlp/index.html)（新加坡国立大学自然语言处理组）

  - [NLP and Big Data Research Group in the ISTD pillar at the Singapore University of Technology and Design](http://www.statnlp.org/) （新加坡科技设计大学自然语言处理和大数据研究组） 
  
  - [NLP Research Group at the Nanyang Technological University](https://ntunlpsg.github.io/)（南洋理工大学自然语言处理组）
  
  - [Advanced Translation Technology Laboratory at National Institute of Information and Communications Technology](http://att-astrec.nict.go.jp/en/)（日本情报通讯研究所高级翻译技术实验室）
  
  - [Nakayama Laboratory at University of Tokyo](http://www.nlab.ci.i.u-tokyo.ac.jp/index-e.html) （东京大学中山实验室） 
  
  - [Natural Language Processing Lab at Bar-Ilan University](http://u.cs.biu.ac.il/~nlp/)  （以色列巴伊兰大学自然语言处理实验室）

  - [The University of Melbourne NLP Group](http://hum.csse.unimelb.edu.au/nlp-group/)（澳大利亚墨尔本大学自然语言处理组）

<a name="north-america"></a>
  
- 北美地区

  - [Natural Language Processing - Research at Google](https://research.google.com/pubs/NaturalLanguageProcessing.html) （Google自然语言处理组）

  - [The Redmond-based Natural Language Processing group](http://research.microsoft.com/en-us/groups/nlp/) （微软自然语言处理组）

  - [Facebook AI Research (FAIR)](https://research.fb.com) （Facebook AI 研究部）

  - [IBM Thomas J. Watson Research Center](http://researchweb.watson.ibm.com/labs/watson/index.shtml)（IBM Thomas J. Watson研究中心）

  - [The Stanford Natural Language Processing Group](http://nlp.stanford.edu/) （斯坦福大学自然语言处理组）
  
  - [The Berkeley Natural Language Processing Group](http://nlp.cs.berkeley.edu/index.shtml)（伯克利加州大学自然语言处理组）
  
  - [Natural Language Processing research at Columbia University](http://www1.cs.columbia.edu/nlp/index.cgi)（哥伦比亚大学自然语言处理组）
  
  - [Graham Neubig's lab at the Language Technologies Instititute of Carnegie Mellon University](http://www.cs.cmu.edu/~neulab/) （卡内基梅隆大学语言技术研究所Graham Neubig实验室）
  
  - [RPI Blender Lab](http://nlp.cs.rpi.edu/)（伦斯勒理工学院Blender Lab）
  
  - [UC Santa Barbara Natural Language Processing Group](http://nlp.cs.ucsb.edu/)（加州大学圣巴巴拉分校自然语言处理组）
  
  - [The Natural Language Group at the USC Information Sciences Institute](http://nlg.isi.edu/) （南加利福尼亚大学信息科学研究所自然语言处理组）
  
  - [Natural Language Processing @USC](https://cl.usc.edu/) （南加利福尼亚大学自然语言处理组）
  
  - [Natural Language Processing Group at University of Notre Dame](http://nlp.nd.edu/) （圣母大学自然语言处理组）
  
  - [Artificial Intelligence Research Group at Harvard](http://www.eecs.harvard.edu/ai/) （哈佛大学人工智能研究组）
  
  - [The Harvard natural-language processing group](http://nlp.seas.harvard.edu/) （哈佛大学自然语言处理组）
  
  - [Computational Linguistics and Information Processing at Maryland](https://wiki.umiacs.umd.edu/clip/index.php/Main_Page) （马里兰大学计算语言学和信息处理实验室）
  
  - [Language and Speech Processing at Johns Hopkins University](http://www.clsp.jhu.edu/about-clsp/)（约翰斯·霍普金斯大学语言语音处理实验室）
  
  - [Human Language Technology Center of Excellence at Johns Hopkins University](http://hltcoe.jhu.edu/)（约翰斯·霍普金斯大学人类语言技术卓越中心）
  
  - [Machine Translation Group at The Johns Hopkins University](http://www.statmt.org/jhu/)（约翰斯·霍普金斯大学机器翻译组）
  
  - [Machine Translation Research at Rochester](https://www.cs.rochester.edu/~gildea/mt/)（罗切斯特大学机器翻译组）
  
  - [NLP @ University of Illinois at Urbana-Champaign](http://nlp.cs.illinois.edu/)（伊利诺伊大学厄巴纳-香槟分校自然语言处理组）
  
  - [UIC Natural Language Processing Laboratory](http://nlp.cs.uic.edu/)（伊利诺伊大学芝加哥分校自然语言处理组）
  
  - [Human Language Technology Research Institute at The University of Texas at Dallas](http://www.hlt.utdallas.edu/)（德克萨斯大学达拉斯分校人类语言技术研究所
  
  - [Natural Language Processing Group at MIT CSAIL](http://nlp.csail.mit.edu/)（麻省理工学院自然语言处理组）
  
  - [Natural Language Processing Group at Texas A&M University](http://nlp.cs.tamu.edu/)（德克萨斯A&M大学自然语言处理组）
  
  - [The Natural Language Processing Group at Northeastern University](https://nlp.ccis.northeastern.edu/)（东北大学自然语言处理组）
  
  - [Cornell NLP group](https://confluence.cornell.edu/display/NLP/Home/)（康奈尔大学自然语言处理组）
  
  - [Natural Language Processing group at University Of Washington](https://www.cs.washington.edu/research/nlp)（华盛顿大学自然语言处理组）
  
  - [Natural Language Processing Research Group at University of Utah](https://www.cs.utah.edu/nlp/)（犹他大学自然语言处理组）
  
  - [Natural Language Processing and Information Retrieval group at University of Pittsburgh](http://www.isp.pitt.edu/research/nlp-info-retrieval-group)（匹兹堡大学自然语言处理和信息检索小组）
  
  - [Brown Laboratory for Linguistic Information Processing (BLLIP)](http://bllip.cs.brown.edu/)（布朗大学布朗语言信息处理实验室）
  
  - [Natural Language Processing (NLP) group at University of British Columbia](https://www.cs.ubc.ca/cs-research/lci/research-groups/natural-language-processing/)（不列颠哥伦比亚大学自然语言处理组）
  
<a name="europe"></a>
  
- 欧洲地区
  
  - [Natural Language and Information Processing Research Group at University of Cambridge](http://www.cl.cam.ac.uk/research/nl/)（英国剑桥大学自然语言和信息处理组）
  
  - [The Computational Linguistics Group at Oxford University](http://www.clg.ox.ac.uk/)（英国牛津大学计算语言学组）
  
  - [Human Language Technology and Pattern Recognition Group at the RWTH Aachen](https://www-i6.informatik.rwth-aachen.de/)（德国亚琛工业大学人类语言技术与模式识别组）
  
  - [The Natural Language Processing Group at the University of Edinburgh (EdinburghNLP)](http://edinburghnlp.inf.ed.ac.uk/)（英国爱丁堡大学自然语言处理研究组）
  
  - [Statistical Machine Translation Group at the University of Edinburgh](http://www.statmt.org/ued/?n=Public.HomePage)（英国爱丁堡大学统计机器翻译组）
  
  - [Natural Language Processing Research Group at The University of Sheffield](http://nlp.shef.ac.uk/)（英国谢菲尔德大学自然语言处理研究组）
  
  - [Speech Research Group at University of Cambridge](http://mi.eng.cam.ac.uk/Main/Speech/)（英国剑桥大学语音研究组）
  
  - [Statistical Machine Translation Group at the University of Cambridge](http://divf.eng.cam.ac.uk/smt)（英国剑桥大学统计机器翻译组）
  
  - [Computational Linguistics group at Uppsala University](http://www.lingfil.uu.se/forskning/datorlingvistik/?languageId=1)（瑞典乌普萨拉大学计算语言学组）

  - [The Center for Information and Language Processing at University of Munich](http://www.cis.uni-muenchen.de/ueber_uns/)（德国慕尼黑大学信息与语言处理中心）

  - [National Centre for Language Technology at Dublin City University](http://www.nclt.dcu.ie/)（爱尔兰都柏林城市大学国家语言技术中心）

  - [The National Centre for Text Mining (NaCTeM) at University of Manchester](http://nactem.ac.uk/)（英国曼彻斯特大学国家文本挖掘中心）

  - [The Information and Language Processing Systems group at the University of Amsterdam](http://ilps.science.uva.nl/)（荷兰阿姆斯特丹大学信息与语言处理系统组）

  - [Institute of Formal and Applied Linguistics at Charles University](http://ufal.mff.cuni.cz/)（捷克查理大学语言学应用与规范研究所）

  - [DFKI Language Technology Lab](https://www.dfki.de/lt/)（德国人工智能研究中心自然语言处理组）

  - [IXA in University of the Basque Country](http://ixa.eus/)（西班牙巴斯克大学自然语言处理组）

  - [Statistical Natural Language Processing Group at the Institute for Computational Linguistics at Heidelberg University](http://www.cl.uni-heidelberg.de/statnlpgroup/)（德国海德堡大学计算语言学研究所统计自然语言处理组）

  - [NLP Research at the University of Helsinki](https://blogs.helsinki.fi/language-technology/)（芬兰赫尔辛基大学自然语言处理组）
    
    
    
### Reference
- [Awesome Chinese NLP](https://github.com/crownpku/Awesome-Chinese-NLP)  
  
- [FunNLP](https://github.com/fighting41love/funNLP)  
  
- [国内外自然语言处理(NLP)研究组](https://blog.csdn.net/wangxinginnlp/article/details/44890553) 
