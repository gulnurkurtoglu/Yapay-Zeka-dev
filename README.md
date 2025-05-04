# Yapay-Zeka-dev
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re
import pandas as pd
df = pd.read_csv("C:/Users/Gülnur/Desktop/kadin_giyim_urunleri.csv", encoding='utf-8')
df.head(100 )
renk = df['Renk'].str.lower().str.split()
print("Sütunlar:", df.columns.tolist())
!pip help install
renk_listesi = [
    'kırmızı', 'mavi', 'yeşil', 'sarı', 'siyah', 'beyaz', 'gri', 'mor',
    'turuncu', 'pembe', 'bej', 'kahverengi', 'lacivert', 'bordo', 'altın', 'gümüş'
]

color_palette = {
    'kırmızı': (255, 0, 0),
    'mavi': (0, 0, 255),
    'yeşil': (0, 128, 0),
    'sarı': (255, 255, 0),
    'siyah': (0, 0, 0),
    'beyaz': (255, 255, 255),
    'gri': (128, 128, 128),
    'mor': (128, 0, 128),
    'turuncu': (255, 165, 0),
    'pembe': (255, 192, 203),
    'bej': (245, 245, 220),
    'kahverengi': (165, 42, 42),
    'lacivert': (0, 0, 139),
    'bordo': (128, 0, 32),
    'altın': (255, 215, 0),
    'gümüş': (192, 192, 192)
}
def tokenize(text):
    return re.findall(r'\b\w+\b', str(text).lower())

def renk_ayikla(tokens):
    return [tok for tok in tokens if tok in renk_listesi]
    df['description'] = df['Ürün Adı'] + ' ' + df['Renk'] + ' ' + df['Stil Önerisi']
    def tokenize(text):
    return text.lower().split()
    def renk_ayikla(tokens):
    renkler_listesi = [
        'siyah', 'beyaz', 'kırmızı', 'mavi', 'yeşil', 'sarı',
        'kahverengi', 'bej', 'lacivert', 'gri', 'turuncu', 'mor',
        'pembe', 'zeytin', 'bordo', 'ekru'
    ]
    return [token for token in tokens if token in renkler_listesi]
    df['tokens'] = df['description'].apply(tokenize)
df['renkler'] = df['tokens'].apply(renk_ayikla)
print(df[['description', 'tokens', 'renkler']].head())
genis_renk_eslestirme = {
    'açık mavi': 'mavi',
    'koyu mavi': 'mavi',
    'açık yeşil': 'yeşil',
    'koyu yeşil': 'yeşil',
    'açık pembe': 'pembe',
    'koyu pembe': 'pembe',
    'lacivert': 'mavi',
    'bej': 'bej',
    'krem': 'beyaz',
    'siyah': 'siyah',
    'beyaz': 'beyaz',
    'gri': 'gri',
    'kahverengi': 'kahverengi',
    'zeytin yeşili': 'yeşil',
    'bordo': 'kırmızı',
    'ekru': 'beyaz',
    'mavi': 'mavi',
    'yeşil': 'yeşil',
    'kırmızı': 'kırmızı',
    'turuncu': 'turuncu',
    'mor': 'mor',
    'pembe': 'pembe',
    'sarı': 'sarı'
}
def renk_ayikla_genis_ve_kisa(description):
    description = description.lower()
    matched_colors = []

    for renk_ifadesi, temel_renk in genis_renk_eslestirme.items():
        if renk_ifadesi in description:
            matched_colors.append(temel_renk)

    return list(set(matched_colors))
    if 'description' not in df.columns:
    df['description'] = df['Ürün Adı'] + ' ' + df['Renk'] + ' ' + df['Stil Önerisi']
    df['renkler'] = df['description'].apply(renk_ayikla_genis_ve_kisa)
    print(df[['description', 'renkler']].head())
    pip install pandas gensim scikit-learn
    from gensim.models import Word2Vec
    descriptions = df['description'].apply(tokenize)
    model = Word2Vec(sentences=descriptions, vector_size=100, window=5, min_count=1, workers=4)
    word_vector = model.wv['pantolon']
    print(word_vector)
    from gensim.models import Word2Vec
    descriptions = df['description'].apply(tokenize)
    model = Word2Vec(sentences=descriptions, vector_size=100, window=5, min_count=1, workers=4)
word_vector = model.wv['elbise']
print(word_vector)
color_palette = {
    'kırmızı': [255, 0, 0],
    'yeşil': [0, 255, 0],
    'mavi': [0, 0, 255],
    'sarı': [255, 255, 0],
    'beyaz': [255, 255, 255],
    'siyah': [0, 0, 0],
    'gri': [128, 128, 128],
    'bej': [245, 245, 220],
    'krem': [255, 253, 208],
    'lacivert': [0, 0, 128],
    'kahverengi': [139, 69, 19],
    'zeytin yeşili': [128, 128, 0],
    'bordo': [128, 0, 0],
    'ekru': [255, 255, 230],
    'açık mavi': [173, 216, 230],
    'koyu mavi': [0, 0, 139],
    'açık yeşil': [144, 238, 144],
    'koyu yeşil': [0, 100, 0],
    'açık pembe': [255, 182, 193],
    'koyu pembe': [255, 20, 147],
    'turuncu': [255, 165, 0],
    'mor': [128, 0, 128],
    'pembe': [255, 192, 203],
    'sarı': [255, 255, 0]
}
rgb_vectors = {
    renk: np.pad(np.array(rgb), (0, 100 - len(rgb)), 'constant')
    for renk, rgb in color_palette.items()
}
renk_rgb_eslesme = {}
for renk, vec in rgb_vectors.items():
    similarities = {
        rgb_name: cosine_similarity([vec], [rgb_vec])[0][0]
        for rgb_name, rgb_vec in rgb_vectors.items()
    }
    en_yakin_rgb = max(similarities, key=similarities.get)
    renk_rgb_eslesme[renk] = {
        'en_yakin_rgb': en_yakin_rgb,
        'rgb_degeri': color_palette[en_yakin_rgb],
        'benzerlik': similarities[en_yakin_rgb]
    }
    for renk, bilgiler in renk_rgb_eslesme.items():
    print(f"{renk}: En yakın RGB = {bilgiler['en_yakin_rgb']}, "
          f"RGB değeri = {bilgiler['rgb_degeri']}, "
          f"Benzerlik = {bilgiler['benzerlik']:.4f}")
          sentences = [
    ['kırmızı', 'yeşil', 'mavi'],
    ['siyah', 'beyaz', 'gri'],
    ['bej', 'krem', 'ekru'],
    ['lacivert', 'koyu mavi'],
    ['açık pembe', 'pembe'],
    ['bordo', 'kahverengi'],
    ['turuncu', 'sarı'],
    ['zeytin yeşili', 'koyu yeşil']
]
model = Word2Vec(sentences=sentences, vector_size=100, window=2, min_count=1, workers=4)
vector_kirmizi = model.wv['kırmızı']
vector_yesil = model.wv['yeşil']
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity([vector_kirmizi], [vector_yesil])[0][0]
print(f"Kırmızı ve Yeşil benzerliği: {similarity:.4f}")
metinler = df['Ürün Adı'].astype(str).str.lower().str.cat(sep=' ')
kelimeler = metinler.split()
from collections import Counter
frekanslar = Counter(kelimeler)
sorted_freqs = sorted(frekanslar.values(), reverse=True)
ranks = np.arange(1, len(sorted_freqs) + 1)
plt.figure(figsize=(10, 6))
plt.plot(np.log(ranks), np.log(sorted_freqs), marker='o', linestyle='-')
plt.title("Zipf Yasası - Ürün Adları")
plt.xlabel("log(Sıra)")
plt.ylabel("log(Frekans)")
plt.grid(True)
plt.tight_layout()
plt.show()
urun_adlari = df['Ürün Adı'].dropna().astype(str)
vectorizer = TfidfVectorizer(lowercase=True)
tfidf_matrix = vectorizer.fit_transform(urun_adlari)
kelimeler = vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=kelimeler)
print(tfidf_df.head())
print("\nVeri hazırlığı yapılıyor...")
df['description'] = df['Ürün Adı'].astype(str) + ' ' + df['Renk'].astype(str) + ' ' + df['Stil Önerisi'].astype(str)
turkish_stop_words = set([
    'acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri', 'birkaç', 
    'birşey', 'biz', 'bu', 'çok', 'çünkü', 'da', 'daha', 'de', 'defa', 
    'diye', 'eğer', 'en', 'gibi', 'hem', 'hep', 'hepsi', 'her', 'hiç', 
    'için', 'ile', 'ise', 'kez', 'ki', 'kim', 'mı', 'mu', 'mü', 'nasıl', 
    'ne', 'neden', 'nerde', 'nerede', 'nereye', 'niçin', 'niye', 'o', 
    'sanki', 'şey', 'siz', 'şu', 'tüm', 've', 'veya', 'ya', 'yani'
])
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
turkish_stop_words = set(stopwords.words('turkish'))
def preprocess_text(text, process_type='lemmatize'):
    try:
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in turkish_stop_words]
        tokens = [token for token in tokens if not token.isdigit()]
        tokens = [token for token in tokens if len(token) > 1]
        if process_type == 'lemmatize':
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
        elif process_type == 'stem':
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(token) for token in tokens]
        else:
            print("Geçersiz işlem türü, 'lemmatize' veya 'stem' seçiniz.")
        return tokens
    except Exception as e:
        print("Ön işleme hatası:", e)
        return []
        print("\nMetin ön işleme uygulanıyor...")
df['lemmatized'] = df['description'].apply(lambda x: preprocess_text(x, 'lemmatize'))
df['stemmed'] = df['description'].apply(lambda x: preprocess_text(x, 'stem'))
print("\nÖn işleme sonrası örnek çıktı:")
print(df[['description', 'lemmatized', 'stemmed']].head())
def plot_zipf(data, title):
    try:
        all_words = [word for sublist in data for word in sublist]
        word_counts = Counter(all_words)
        sorted_counts = sorted(word_counts.values(), reverse=True)
        ranks = np.arange(1, len(sorted_counts)+1)
        
        plt.figure(figsize=(10,6))
        plt.loglog(ranks, sorted_counts, marker=".")
        plt.title(f"Zipf Yasası - {title}")
        plt.xlabel("Kelime Sırası (log)")
        plt.ylabel("Frekans (log)")
        plt.grid(True)
        plt.show()
        print(f"{title} Zipf grafiği oluşturuldu.")
    except Exception as e:
        print(f"Zipf grafiği oluşturulamadı: {e}")
        print("\nZipf analizi yapılıyor...")
plot_zipf(df['description'].apply(lambda x: x.split()), "Ham Veri")
plot_zipf(df['lemmatized'], "Lemmatized Veri")
plot_zipf(df['stemmed'], "Stemmed Veri")
print("\nTF-IDF vektörleştirme yapılıyor...")
def create_tfidf(data, name):
    try:
        corpus = [" ".join(tokens) for tokens in data]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        df_tfidf = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
        df_tfidf.to_csv(f"tfidf_{name}.csv", index=False)
        print(f"tfidf_{name}.csv başarıyla kaydedildi. Boyut:", df_tfidf.shape)
        return df_tfidf
    except Exception as e:
        print(f"TF-IDF oluşturulamadı ({name}):", e)
        return None
tfidf_lemmatized = create_tfidf(df['lemmatized'], "lemmatized")
tfidf_stemmed = create_tfidf(df['stemmed'], "stemmed")
from gensim.models import Word2Vec

print("\nWord2Vec modelleri eğitiliyor...")

def train_word2vec(sentences, name, model_type='cbow', window=5, vector_size=100):
    try:
        model = Word2Vec(
            sentences=sentences,
            sg=1 if model_type == 'skipgram' else 0,
            window=window,
            vector_size=vector_size,
            min_count=1,
            workers=4
        )
        model.save(f"word2vec_{name}_{model_type}_w{window}_d{vector_size}.model")
        print(f"Model {name}_{model_type} kaydedildi.")
    except Exception as e:
        print(f"Model eğitilirken hata oluştu: {e}")
def train_word2vec(sentences, name, model_type='cbow', window=5, vector_size=100):
    try:
        model = Word2Vec(
            sentences=sentences,
            sg=1 if model_type == 'skipgram' else 0,
            window=window,
            vector_size=vector_size,
            min_count=1,
            workers=4
        )
        try:
            similar = model.wv.most_similar('elbise', topn=5)
            print(f"'elbise' için benzer kelimeler: {similar}")
        except:
            print("'elbise' kelimesi modelde bulunamadı")
        model.save(f"word2vec_{name}_{model_type}_w{window}_d{vector_size}.model")
        print(f"Model {name}_{model_type} kaydedildi.")

        return model
    except Exception as e:
        print(f"Word2Vec modeli eğitilemedi ({name}):", e)
        return None
        params = [
    {'model_type': 'cbow', 'window': 2, 'vector_size': 100},
    {'model_type': 'skipgram', 'window': 2, 'vector_size': 100},
    {'model_type': 'cbow', 'window': 5, 'vector_size': 100},
    {'model_type': 'skipgram', 'window': 5, 'vector_size': 100},
    {'model_type': 'cbow', 'window': 2, 'vector_size': 300},
    {'model_type': 'skipgram', 'window': 2, 'vector_size': 300},
    {'model_type': 'cbow', 'window': 5, 'vector_size': 300},
    {'model_type': 'skipgram', 'window': 5, 'vector_size': 300}
]
for param in params:
    print(f"\n{param['model_type']} modeli eğitiliyor (window={param['window']}, dim={param['vector_size']})...")
    train_word2vec(
        sentences=df['lemmatized'],
        name='lemmatized',
        model_type=param['model_type'],
        window=param['window'],
        vector_size=param['vector_size']
    )
     train_word2vec(
        sentences=df['stemmed'],
        name='stemmed',
        model_type=param['model_type'],
        window=param['window'],
        vector_size=param['vector_size']
    )
print("\nTüm işlemler başarıyla tamamlandı!")
