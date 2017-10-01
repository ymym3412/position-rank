# PositionRank
PositionRank is a keyphrase extraction method described in the ACL 2017 paper [PositionRank: An Unsupervised Approach to Keyphrase Extraction from Scholarly Documents](http://aclweb.org/anthology/P/P17/P17-1102.pdf).  
This method search keyphrase by graph-based algorithm, which is biased PageRank by co-occurence word's position information.  
You can use this method not only English scholarly documents, but also any other language's document if you create your tokenizer for other language.  

```py
>>>from position_rank import position_rank
>>>from tokenizer import StanfordCoreNlpTokenizer

>>>title = "PositionRank: An Unsupervised Approach to Keyphrase Extraction from Scholarly Documents."
>>>abstract = """The large and growing amounts of online
... scholarly data present both challenges and
... opportunities to enhance knowledge discovery.
... One such challenge is to automatically
... extract a small set of keyphrases
... from a document that can accurately describe
... the document’s content and can facilitate
... fast information processing. In
... this paper, we propose PositionRank, an
... unsupervised model for keyphrase extraction
... from scholarly documents that incorporates
... information from all positions of a
... word’s occurrences into a biased PageRank.
... Our model obtains remarkable improvements
... in performance over PageRank
... models that do not take into account
... word positions as well as over strong baselines
... for this task. Specifically, on several
... datasets of research papers, PositionRank
... achieves improvements as high as 29.09%."""

>>>tokenizer = StanfordCoreNlpTokenizer("http://localhost")
>>>position_rank(title + abstract, tokenizer)
['PositionRank', 'online_scholarly_data', 'fast_information_processing', 'account_word_positions', 'Keyphrase', 'Extraction', 'Approach', 'Unsupervised', 'document', 'Scholarly']
```

## SETUP
### Prerequirement
#### For English
Java 1.8+ (for Stanford CoreNLP) ([Download](http://www.oracle.com/technetwork/cn/java/javase/downloads/jdk8-downloads-2133151-zhs.html))  
Stanford CoreNLP 3.7.0 ([Download](https://stanfordnlp.github.io/CoreNLP/download.html))

#### For Japanese
Mecab ([Installation](http://taku910.github.io/mecab/#install))

### Install Python libraries
```shell
$ pip install -r requirements.txt
```

## USAGE (English document)
### Start up Stanford CoreNLP Server
First, you start up Stanford CoreNLP server.
```shell
$cd /path/to/stanford_corenlp/
$ java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
[main] INFO CoreNLP - --- StanfordCoreNLPServer#main() called ---
[main] INFO CoreNLP - setting default constituency parser
[main] INFO CoreNLP - warning: cannot find edu/stanford/nlp/models/srparser/englishSR.ser.gz
[main] INFO CoreNLP - using: edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz instead
[main] INFO CoreNLP - to use shift reduce parser download English models jar from:
[main] INFO CoreNLP - http://stanfordnlp.github.io/CoreNLP/download.html
[main] INFO CoreNLP -     Threads: 4
[main] INFO CoreNLP - Starting server...
[main] INFO CoreNLP - StanfordCoreNLPServer listening at /0:0:0:0:0:0:0:0:9000

```

### Simple example
```py
from position_rank import position_rank
from tokenizer import StanfordCoreNlpTokenizer

title = "PositionRank: An Unsupervised Approach to Keyphrase Extraction from Scholarly Documents."
abstract = """The large and growing amounts of online
scholarly data present both challenges and
opportunities to enhance knowledge discovery.
..."""

tokenizer = StanfordCoreNlpTokenizer("http://localhost")
position_rank(title + abstract, tokenizer)
["keyphrase1", "keyphrase2", ..., "keyphrase10"]
```

Edit number of output keyphrase.  
```py
position_rank(title + abstract, tokenizer, num_keyphrase=5)
["keyphrase1", "keyphrase2", "keyphrase3", "keyphrase4", "keyphrase5"]
```

Switch other algorith parameters.  
```py
position_rank(title + abstract, tokenizer, alpha=0.6, window_size=4, num_keyphrase=10, lang="en")
["keyphrase1", "keyphrase2", ..., "keyphrase10"]
```

## USAGE (Japanese document)
### Simple example
```py
from position_rank import position_rank
from tokenizer import MecabTokenizer

title = "{日本語論文のタイトル}"
abstract = "{日本語論文の概要}"

tokenizer = MecabTokenizer()
position_rank(title + abstract, tokenizer, lang="ja")
["keyphrase1", "keyphrase2", ..., "keyphrase10"]
```

Use dictionary for Mecab. Add Mecab's option string to MecabTokenizer.  
```py
from position_rank import position_rank
from tokenizer import MecabTokenizer

title = "{日本語論文のタイトル}"
abstract = "{日本語論文の概要}"

tokenizer = MecabTokenizer("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")
position_rank(title + abstract, tokenizer, lang="ja")
["keyphrase1", "keyphrase2", ..., "keyphrase10"]
```
Switch other algorith parameters. 
```py
position_rank(title + abstract, tokenizer, alpha=0.6, window_size=4, num_keyphrase=10, lang="ja")
["keyphrase1", "keyphrase2", ..., "keyphrase10"]
```

## CUSTOMIZE
You can use PositionRank for other language if you create your tokenizer.  
Customize tokenizer must have `tokenize()` method. `tokenize()` returns two list, `token list` and `phrase list`.  
`Phrase` means continuous tokens which have specific POS(Part-of-Speech) pattern `(adjective)*(noun)+` and length are more than 3.  
This is sample customize tokenizer.  
```py
class CustomizeTokenizer(object):

    def __init__(self):
        # Initialize your tokenizer.

    def tokenize(self, sentence):
        # tokenize sentence and create phrase list, then return them.
        # Tokens must be filterd only adjective and noun POS in your language.
        return token_list, phrase_list

title = "{other language's title}"
abstract = "{other language's abstract}"

tokenizer = CustomTokenizer()
position_rank(title + abstract, tokenizer, alpha=0.85, window_size=6, num_keyphrase=10, lang="custom")

```
