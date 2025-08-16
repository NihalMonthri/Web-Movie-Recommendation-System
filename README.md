
# Movie Recommendation System using Machine Learning



## 🎯Objective
The main goal of the project is to create a movie recommender system that recommends movies based on their similarity to the movie entered by the user.
## 🛠️Techniques/Algorithms/Methods used in Project
𝟏.𝐑𝐞𝐜𝐨𝐦𝐦𝐞𝐧𝐝𝐞𝐫 𝐒𝐲𝐬𝐭𝐞𝐦: A recommendation system (or recommender system) is a class of machine learning that uses data to help predict, narrow down, and find what people are looking for among an exponentially growing number of options.  
It is an artificial intelligence or AI algorithm, usually associated with machine learning, that uses Big Data to suggest or recommend additional products to consumers. There are three types of recommender systems-
Content, Collaborative and Hybrid. Content Recommender system is used in this project.

![App Screenshot](https://www.nvidia.com/content/dam/en-zz/Solutions/glossary/data-science/recommendation-system/img-3.png)

Content recommender system uses the attributes or features of an item  to recommend other items similar to the user’s preferences.

𝟐.𝐃𝐚𝐭𝐚 𝐏𝐫𝐞𝐩𝐫𝐨𝐜𝐞𝐬𝐬𝐢𝐧𝐠: Data preprocessing is the process of detecting and correcting (or removing) corrupt or inaccurate records from a dataset, or and refers to identifying incorrect, incomplete, irrelevant parts of the data and then modifying, replacing, or deleting the dirty or coarse data.

𝟑.𝐂𝐨𝐮𝐧𝐭 𝐕𝐞𝐜𝐭𝐨𝐫𝐢𝐳𝐞𝐫: CountVectorizer is a text  preprocessing technique commonly used in natural language processing (NLP) tasks for converting a collection of text documents into a numerical representation. It is part of the scikit-learn library, a popular machine learning library in Python.   
It operates by tokenizing the text data and counting the occurrences of each token. It then creates a matrix where the rows represent the documents, and the columns represent the tokens. The cell values indicate the frequency of each token in each document. This matrix is known as the “document-term matrix.”

![App Screenshot](https://www.educative.io/api/edpresso/shot/5197621598617600/image/6596233398321152)

To use CountVectorizer, you have to import the module as shown below:
```bash
  from sklearn.feature_extraction.text import CountVectorizer
  cv=CountVectorizer(max_features= 5000, stop_words='english')
```

𝟒.𝐏𝐨𝐫𝐭𝐞𝐫 𝐒𝐭𝐞𝐦𝐦𝐢𝐧𝐠 𝐀𝐥𝐠𝐨𝐫𝐢𝐭𝐡𝐦: The Porter Stemming algorithm (or Porter Stemmer) is used to remove the suffixes from an English word and obtain its stem which becomes very useful in the field of Information Retrieval (IR).  
This process reduces the number of terms kept by an IR system which will be advantageous both in terms of space and time complexity. The process of reducing such inflected (or sometimes derived) words to their word stem is known as Stemming. For example, CONNECTED, CONNECTION and CONNECTING can be reduced to the stem CONNECT.

![App Screenshot](https://raw.githubusercontent.com/markfullmer/porter2/master/demo/stemmer-demo.png)

To use PorterStemmer, you have to import the module as shown below:
```bash
  from nltk.stem.porter import PorterStemmer
  ps=PorterStemmer()
```

𝟓.𝐂𝐨𝐬𝐢𝐧𝐞 𝐒𝐢𝐦𝐢𝐥𝐚𝐫𝐢𝐭𝐲: Similarity measure refers to distance with dimensions representing features of the data object, in a dataset. If this distance is less, there will be a high degree of similarity, but when the distance is large, there will be a low degree of similarity.   
Cosine similarity measures the similarity between two vectors of an inner product space. It is measured by the cosine of the angle between two vectors and determines whether two vectors are pointing in roughly the same direction. It is often used to measure document similarity in text analysis. A document can be represented by thousands of attributes, each recording the frequency of a particular word (such as a keyword) or phrase in the document.
![App Screenshot](https://cdn.botpenguin.com/assets/website/Cosine_Similarity_f1e08fbad8.webp)
![App Screenshot](https://www.oreilly.com/api/v2/epubs/9781785283451/files/assets/d258ae34-f4f8-4143-b3c2-0cb10f2b82de.png)

To use Cosine Similarity, you have to import the module as shown below:
```bash
  from sklearn.metrics.pairwise import cosine_similarity
  similarity=cosine_similarity(vectors)
```
𝟔.𝐀𝐛𝐬𝐭𝐫𝐚𝐜𝐭 𝐒𝐲𝐧𝐭𝐚𝐱 𝐓𝐫𝐞𝐞: AST (Abstract Syntax Tree) is a module present in the python standard library. Before transforming python code to “byte code”(.pyc files), it is converted to an AST. The most important function of the AST module is to generate this AST.

![App Screenshot](https://miro.medium.com/v2/resize:fit:522/format:webp/0*ykaApIklGcJ7Qzhw)

To use Abstract Syntax Tree, you have to import the module as shown below:
```bash
  import ast
```










## 📋DataSets Used
In this project 4 DataSets are used. They are:

𝟏.𝐜𝐫𝐞𝐝𝐢𝐭𝐬.𝐜𝐬𝐯: This is a dataset that contains the major details of movies. It has four attributes- movie_id, title, cast and crew. The cast and crew information is the major part of this dataset. It consists of 4803 tuples, each with four attributes(some may have null values). This dataset is scraped from IMDB website, which contains a large database of movies and their data. This dataset is available on kaggle.

𝟐.𝐦𝐨𝐯𝐢𝐞𝐬.𝐜𝐬𝐯: This is a dataset that contains all other details of movies. It has 20 attributes- budget, genres, homepage, id, keywords, original_language, original_title, overview, popularity, production_companies, production_countries, release_date, revenue, runtime, spoken_languages, status, tagline, title, vote_average and vote_count. It consists of 4803 tuples, each with four attributes(some may have null values). This dataset is scraped from IMDB website, which contains a large database of movies and their data. This dataset is available on kaggle. A lot of the information in this dataset is redundant and unnecessary and will be omitted.

𝟑.𝐦𝐨𝐯𝐢𝐞𝐬_𝐝𝐟(merged): This dataset is the combination of movies and credit datasets. All of the redundant information from both datasets is removed and null values are erased. It consists of 7 attributes- movie_id, title, overview, genres, keywords, cast and crew. it has 4805 tuples. All the future preprocessing takes place on this dataset.

𝟒.𝐧𝐞𝐰_𝐝𝐟:This dataset contains tuples from movies_df with even more redundant information removed and also preprocessed.
it contains 3 attributes and 4805 tuples- movie_id, title and tags.
tags is an attribute that contains all the data from the other tuples of the movies_df dataset in the form of singular words. It is used to check similarity between the movies. This dataset is used for all the main functions.




## ⬆️Results
Below are two cases where different movies are provided as input-
Case1:
```bash
    recommend("Batman Begins")
    #output
    The Dark Knight
    The Dark Knight Rises
    Amidst the Devil's Wings
    Batman
    Batman & Robin
```
Case2:
```bash
    recommend("The Avengers")
    #output
    Avengers: Age of Ultron
    Iron Man 3
    Iron Man
    Iron Man 2
    Captain America: Civil War
```
As you can see from the above outputs, The movie recommender system is clearly functioning as intended, providing movies that are very or somewhat similar to the entered movie. This shows that the code has no bugs and is ready for use. You can give any other movie as input and it will provide required similar movies. You can check for accuracy of the program by indivdually comparing the cosine similarities of the movies.

This is also implemented into steamlit via link https://web-movie-recommendation-system-tg5yvtovtsgauq2cbmvgjc.streamlit.app/
## 🏁Conclusion
Hence, We can say that the movie recommender system has been created successfully using machine learning. This algorithm can be used for other types of recommender systems as well such as shopping, music etc. This movie recommender system can be used to build a website that gives you movie recommendations based on your tastes. This was a very satisfying project to work on and also helped me learn a lot of different things in the fields of AI, Githib, Anaconda and so on. Such projects are very much worthwile.
## Authors
Nihal Monthri - 160122748042   
Thirupati Rao - 160122748049   
Parthiv       - 160122748039   
Sai Teja      - 160122748056    

