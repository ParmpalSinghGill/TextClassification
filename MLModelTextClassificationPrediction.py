# ## Import Libraries
# Loading all libraries to be used
import pickle as pk
import nltk
from newspaper import Article
nltk.download('stopwords')

MODELS="MODELS/"

with open(MODELS+"vectorizer.pk","rb") as f:
    vectorizer=pk.load(f)

with open(MODELS+"lencoder.pk","rb") as f:
    encoder=pk.load(f)

with open(MODELS+"VThreshhold.pk","rb") as f:
    selection=pk.load(f)

with open(MODELS+"MultinomialNB.pk","rb") as f:
    nb=pk.load(f)

def predictClass(text):
    x=vectorizer.transform([text])
    x =selection.transform(x)
    y=nb.predict(x)[0]
    # print(y,encoder.classes_[y])
    return encoder.classes_[y]


def ClassifyTEXTUrlArticle(url,arg="title"):
    try:
        article = Article(url)
        article.download()
        article.parse()
        content=article.__getattribute__(arg)
        return {"topic":predictClass(content),arg:content}
    except Exception as e:
        print("Exception ",e)
        return {'code': 500}, 500

with open("../Files/Ulrs.txt") as f:
    urls = [d[:-1] for d in f.readlines()]
for url in urls:
    print(ClassifyTEXTUrlArticle(url,arg="title"))
    print(ClassifyTEXTUrlArticle(url,arg="meta_description"))