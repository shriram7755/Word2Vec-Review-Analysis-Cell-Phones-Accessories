# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 09:44:30 2023

@author: acer
"""

# pip install gensim
# pip install python-Levenshtein
import pandas as pd
import gensim
df=pd.read_json("C:/Users/acer/Downloads/reviews_Cell_Phones_and_Accessories_5 (1).json.gz",compression='infer',lines=True)
df
df.shape
# (194439, 9)


# simple preprocessing and tokenizatation
review_text=df.reviewText.apply(gensim.utils.simple_preprocess)
review_text
# lets us check first word of each review
review_text.loc[0]
#lets check first row of dataframe
df.reviewText.loc[0]
# Training the Word2Vec model
model=gensim.models.Word2Vec(
    window=10,
    min_count=2,
    workers=4
    )

'''
where window is having how many words you are going to 
consider as sliding window you can choose any count
min_count there must min 2 words in each sentence
woekers:no.of threads
'''

# build Vocabulary
model.build_vocab(review_text,progress_per=1000)
# progress_per:after 1000 words it shows progress 
# train the Word2Vec model
# it will take time,have patience
model.train(review_text,total_examples=model.corpus_count,epochs=model.epochs)
# save the model
model.save("C:/Users/acer/Downloads/reviews_Cell_Phones_and_Accessories_5.model")
# fimndings sumilar words and similarity
model.wv.most_similar("bad")
model.wv.similarity(w1='cheap', w2='inexpensive')
model.wv.similarity(w1='great', w2='good')

































