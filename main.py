from sklearn import manifold, datasets
from ParseDataset import ParseDataset
from CompareDimReduction import CompareDimReduction
import matplotlib.pyplot as plt
from functools import partial
import matplotlib.image as mpimg
from math import ceil
from sklearn.feature_extraction.text import TfidfVectorizer

MAX = 15
def callbackImages(images, labels, path):
    fig1 = plt.figure()
    width = 4
    height = ceil(len(images)/4)
    for idx, im in enumerate(images):
        if idx >= MAX:
            break
        print(path+labels[im])
        img = mpimg.imread(path+labels[im])
        ax = fig1.add_subplot(width, height, idx+1)
        ax.imshow(img)
    plt.show()
    return

# START

# Reuters
# news_data = ParseDataset('/home/evarildo/UnB/TG1/Dataset/noticias.data')
# news = CompareDimReduction(news_data.data, news_data.data_class, 'News')
# news.start()

# # Imagens Corel
corel_data = ParseDataset('/home/evarildo/UnB/TG1/Dataset/ImagensCorel.data')
print(corel_data.entry_labels)
corel = CompareDimReduction(corel_data.data, corel_data.data_class, 'Corel100', partial(callbackImages, labels=corel_data.entry_labels, path = '/home/evarildo/UnB/TG1/Dataset/imagensCorel/'))
corel.start()

# # Circles
# circle_data, circle_classes = datasets.make_circles(n_samples=500, factor=.5, noise=.05)
# circle = CompareDimReduction(circle_data, circle_classes, 'Circles')
# circle.start()

# # S Curve
n_points = 1000
curve_data, color = datasets.samples_generator.make_s_curve(n_points, random_state=0)
curve = CompareDimReduction(curve_data, color, 'S Curve', partial(callbackImages, labels=curve_data, path = '/home/evarildo/UnB/TG1/Dataset/imagensCorel/'))
curve.start()

# Iris
iris = datasets.load_iris()
iris_data = iris.data[:, :]  # we only take the first two features.
iris_classes = iris.target
iris = CompareDimReduction(iris_data, iris_classes, 'Iris Dataset')
iris.start()

# 20 News Groups
# x = 1000
# categories = ['alt.atheism', 'talk.religion.misc']
# newsgroups = datasets.fetch_20newsgroups(subset='train', categories=categories)
# vectorizer = TfidfVectorizer()
# newsgroups_data = vectorizer.fit_transform(newsgroups.data)
# print(len(newsgroups.data), newsgroups_data.shape)
# newsgroups_classes = newsgroups.target
# print(newsgroups_classes)
# newsgroups_out = CompareDimReduction(newsgroups_data.toarray(), newsgroups_classes, '20 News Groups')
# newsgroups_out.start()