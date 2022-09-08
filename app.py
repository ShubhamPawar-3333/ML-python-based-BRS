from pydoc import render_doc
from flask import Flask,render_template,request
import pickle
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from numpy  import *

top_rated_books_list = pickle.load(open('Top Rated Books.pkl','rb'))
Books_BRS = pickle.load(open('Books_BRS.pkl','rb'))

app = Flask(__name__, template_folder='template')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/top_rated_books')
def top_rated_books():
    return render_template('Top rated Books.html',
                            cover_image = list(top_rated_books_list['coverImg'].values),
                            book_name = list(top_rated_books_list['title'].values),
                            author = list(top_rated_books_list['author'].values),
                            genre = list(top_rated_books_list['genres'].values)
                            )

@app.route('/recommendation_by_book_title')
def recommendation_by_book_title():
    return render_template('recommendation_by_book_title.html')

@app.route('/recommendation_by_book_title_list', methods=['POST'])
def recommendation_by_book_title_list():
    selected_features = ['series', 'author','language','genres']
    for feature in selected_features:
        Books_BRS[feature] = Books_BRS[feature].fillna('')
    combined_features = Books_BRS['series']+' '+Books_BRS['author']+' '+Books_BRS['language']+' '+Books_BRS['genres']
    vectorizer = TfidfVectorizer()
    feature_vector = vectorizer.fit_transform(combined_features)
    similarity = cosine_similarity(feature_vector)

    book_name = request.form.get('user_input')
    
    Books_data = Books_BRS['title'].tolist()
    
    find_close_match = difflib.get_close_matches(book_name, Books_data)
    
    close_match = find_close_match[0]
    
    index_of_the_book = Books_BRS[Books_BRS.title == close_match].index[0]

    similarity_score = list(enumerate(similarity[index_of_the_book]))

    sorted_similar_books = sorted(similarity_score, key = lambda x:x[1], reverse = True)

    data=[]
    j=1
    for i in sorted_similar_books:
        index = i[0]
        item=[]
        temp_df=Books_BRS[Books_BRS.index == index]
        item.extend(list(temp_df['title'].values))
        item.extend(list(temp_df['author'].values))
        item.extend(list(temp_df['description'].values))
        item.extend(list(temp_df['coverImg'].values))
        if (j<=10):
            data.append(item)
            j+=1
    return render_template('recommendation_by_book_title.html',data=data)

@app.route('/recommendation_by_author_name')
def recommendation_by_author_name():
    return render_template('recommendation_by_author_name.html')

@app.route('/recommendation_by_author_name_list', methods=['POST'])
def recommendation_by_author_name_list():
    
    selected_features = ['series', 'author','language','genres']
    for feature in selected_features:
        Books_BRS[feature] = Books_BRS[feature].fillna('')
    combined_features = Books_BRS['series']+' '+Books_BRS['author']+' '+Books_BRS['language']+' '+Books_BRS['genres']
    vectorizer = TfidfVectorizer()
    feature_vector = vectorizer.fit_transform(combined_features)
    similarity = cosine_similarity(feature_vector)
    
    Author = request.form.get('author_name')

    Books_author = Books_BRS['author'].tolist()

    find_close_match = difflib.get_close_matches(Author, Books_author)

    close_match = find_close_match[0]

    index_of_the_book = Books_BRS[Books_BRS.author == close_match].index[0]

    similarity_score = list(enumerate(similarity[index_of_the_book]))

    sorted_similar_books = sorted(similarity_score, key = lambda x:x[1], reverse = True)

    data_a=[]
    j=1
    for i in sorted_similar_books:
        index = i[0]
        item=[]
        temp_df=Books_BRS[Books_BRS.index == index]
        item.extend(list(temp_df['title'].values))
        item.extend(list(temp_df['author'].values))
        item.extend(list(temp_df['description'].values))
        item.extend(list(temp_df['coverImg'].values))
        if (j<=10):
            data_a.append(item)
            j+=1
            
    return render_template('recommendation_by_author_name.html',data_a=data_a)

@app.route('/recommendation_based_on_genre')
def recommendation_based_on_genre():
    return render_template('recommendation_based_on_genre.html')

@app.route('/recommendation_based_on_genre_list', methods=['POST'])
def recommendation_based_on_genre_list():
    word = request.form.get('genre')
    E=[]
    for ind in range(Books_BRS.shape[0]):
        try:
            arr = Books_BRS.loc[Books_BRS.index == ind]['genres'].values[0]
        except IndexError:
            continue
        B=[]
        try:
            n=len(arr)
        except TypeError:
            continue
        for i in range(n):
            for j in range(len(arr[i])):
                B.append(arr[i][j])
        C=[0,len(B)]
        for k in range(len(B)):
            if B[k]==',':
                C.append(k)
                C.append(k+2)
                C.sort()
        P=[]
        Q=[]
        for l in range(len(C)):
            if (l % 2 == 0):
                P.append(l)
            else:
                Q.append(l)
        D=[]
        for f, b in zip(P, Q):
            d=''.join(B[C[f]:C[b]])
            D.append(d)
        
        word_found = False
        for element in D:
            if word == element:
                word_found = True
                break
        if word_found:
            E.append(ind)
        else:
            continue    
    df = Books_BRS.loc[E]
    df_sorted = df[df['numRatings']>100000].sort_values('rating',ascending=False).head(10).reset_index(drop=True)
    df_sorted.drop(df_sorted.columns[[1,5,6,7,8,10,11,12,13,14,15]], axis = 1,inplace=True)
    data_b=[]
    for i in range(10):
        item=[]
        temp_df=df_sorted[df_sorted.index == i]
        item.extend(df_sorted.loc[i].tolist())
        
        data_b.append(item)
        
    return render_template('recommendation_based_on_genre.html',data_b=data_b)

if __name__=='__main__':
    app.run(debug=True)