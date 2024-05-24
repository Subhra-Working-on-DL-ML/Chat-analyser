import pandas as pd
from urlextract import URLExtract
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from collections import Counter
import emoji

nltk.download('stopwords')


def fetch_stats(selected_user, df):
    extractor = URLExtract()
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]
    num_messages = df.shape[0]
    words = []
    links = []
    for message in df['message']:
        words.extend(message.split())
        urls = extractor.find_urls(message)
        links.extend(urls)

    num_media_messages = df[df['message'] == "<Media omitted>\n"].shape[0]

    return num_messages, len(words), num_media_messages, len(links)


def most_busy_users(df):
    user_count = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})

    return user_count, df


def create_wordcloud(selected_user, df):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]
    filtered_df = df[df['user'] != "group_notification"]
    filtered_df = filtered_df[filtered_df['message'] != '<Media omitted>\n']

    def remove_stop_word(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color="white")
    filtered_df['message'].apply(remove_stop_word)
    df_wc = wc.generate(filtered_df['message'].str.cat(sep=" "))
    return df_wc


def most_common_words(selected_user, df):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]
    filtered_df = df[df['user'] != "group_notification"]
    filtered_df = filtered_df[filtered_df['message'] != '<Media omitted>\n']
    words = []
    for message in filtered_df['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df


def emoji_helper(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]
    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    return emoji_df


def monthly_timeline(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + " - " + str(timeline['year'][i]))
    timeline['time'] = time
    return timeline


def daily_timeline(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]
    daily_timeline = df.groupby('date_daily').count()['message'].reset_index()

    return daily_timeline


def week_activity_map(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()


def month_activity_map(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]
    return df['month'].value_counts()


def activity_heatmap(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]
    user_activity_heatmap = df.pivot_table(index='day_name', columns='period', values='message',
                                           aggfunc='count').fillna(0)
    return user_activity_heatmap
