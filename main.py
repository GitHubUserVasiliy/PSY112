# imports

import vk_api
import pandas as pd
from datetime import date, timedelta
import nltk
from nltk import TweetTokenizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pymorphy2
from nltk.probability import FreqDist
import pickle
import os.path


# definitions

def get_vk_session(vk_token, vk_app_id):
    return vk_api.VkApi(token=vk_token, app_id=vk_app_id).get_api()


def get_posts(vk_session, domain, start_date=date.today(), end_date=date.today() - timedelta(7)):
    posts = []
    offset = 0
    post_date = start_date
    while True:
        wall = vk_session.wall.get(domain=domain, count=100, offset=offset)
        for post in wall["items"]:
            post_date = date.fromtimestamp(post["date"])
            if (post_date >= start_date) & (post_date <= end_date):
                posts.append(post)
        offset = offset + 100
        if post_date < end_date:
            break
    return posts


def get_comments(vk_session, owner_id, post_id):
    comments = vk_session.wall.getComments(owner_id=owner_id, post_id=post_id, count=100)
    texts = []
    for comment in comments["items"]:
        texts.append(comment["text"])
    return texts


def get_comments_after(posts_after):
    comm_after = []
    for post in posts_after:
        comm_after = comm_after + get_comments(vk, post["owner_id"], post["id"])
    return comm_after


def get_comments_tokens(comments_after):
    morph = pymorphy2.MorphAnalyzer()
    t = TweetTokenizer()

    stop_words = ["быть", "этот", "который"]

    comment_tokens = []
    for text in comments_after:
        tokens = t.tokenize(text)
        filtered_tokens = []
        for token in tokens:
            word = morph.parse(token)[0]
            if word.tag.POS in ["VERB", "NOUN", "ADJF", "ADJS", "INFN", "ADVB"]:
                if word.normal_form not in stop_words:
                    filtered_tokens.append(word.normal_form)
        comment_tokens.append(filtered_tokens)
    return comment_tokens


def format_tokens(comment_tokens):
    texts = []
    for tokens in comment_tokens:
        if tokens:
            texts.append(" ".join(tokens))
    return texts


def show_in_graph(texts):
    plt.imshow(WordCloud().generate(" ".join(texts)), interpolation='bilinear')
    plt.axis("off")
    plt.show()


def to_file(name, text):
    with open(name + '.txt', 'wb') as filehandle:
        pickle.dump(text, filehandle)


def from_file(name):
    with open(name + '.txt', 'rb') as filehandle:
        db = pickle.load(filehandle)
    return db


def load_data(group_name):
    if os.path.exists(group_name + ".txt"):
        group_opinion = from_file(group_name)
    else:
        group_opinion = format_tokens(get_comments_tokens(
            get_comments_after(get_posts(vk, group_name, date(2022, 5, 1), date(2022, 5, 25)))))
        to_file(group_name, group_opinion)
    return group_opinion


# main
# суть проекта - выявление устойчивых речевых выражений в среде киберспортсменов
# анализ устойчивых слов и выражений на предмет агрессивности, построение первичного патерна
# через наложение на формат комментариев новостей.
# развитие проекта - создание и обучение классов нейронной сети на автопаттерны.

# установление подключения
vk = get_vk_session("371d7b52371d7b52371d7b524c3761bd313371d371d7b52559245f2cba45d9946a2a458", 8177251)

# загрузка комментариев из групп вконтакте

group_esportsffru = load_data("esportsffru")
group_cyber_goths_cybers = load_data("cyber_goths_cybers")
group_ria = load_data("ria")

# графический анализ и построение паттерна
FreqDist(group_esportsffru + group_cyber_goths_cybers).plot(30, cumulative=False)


# сопоставление изучаемой с новостной будет предметом анализа по сигмоидному распределению (нейросеть)
FreqDist(list(set(group_esportsffru + group_cyber_goths_cybers) & set(group_ria))).plot(30, cumulative=False)

show_in_graph(group_esportsffru + group_cyber_goths_cybers + group_ria)

# на данном этапе исследования выявлено, что киберспортивная среда снижает агрессивность
# через новые термины из цифровой сферы.


