import streamlit as st

import helper
import preprocessor
import matplotlib.pyplot as plt
import seaborn as sns

st.sidebar.title("WhatsApp Chat Analyser")

uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    # fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")
    selected_user = st.sidebar.selectbox("Show analysis w.r.t ", user_list)

    if st.sidebar.button("Show analysis"):
        num_messages, words, num_media_messages, links = helper.fetch_stats(selected_user, df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links shared")
            st.title(links)

        if selected_user == "Overall":
            st.title("Most Busy Users")
            user_count, df_with_percent = helper.most_busy_users(df)
            fig, ax = plt.subplots()
            col1, col2 = st.columns(2)

            with col1:
                ax.bar(user_count.index, user_count.values)
                plt.xticks(rotation="vertical")
                st.pyplot(fig)

            with col2:
                st.dataframe(df_with_percent)

        # WordCloud
        st.title("Word Cloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        most_common_df = helper.most_common_words(selected_user, df)
        st.title("Most Common Words")
        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation="vertical")
        st.pyplot(fig)

        emoji_df = helper.emoji_helper(selected_user, df)
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig, ax = plt.subplots()
            ax.pie(emoji_df[1], labels=emoji_df[0])
            st.pyplot(fig)

        st.title("Monthly timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color="green")
        plt.xticks(rotation="vertical")
        st.pyplot(fig)

        st.title("Daily timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['date_daily'], daily_timeline['message'], color="green")
        plt.xticks(rotation="vertical")
        st.pyplot(fig)

        st.title("Activity Map")
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most Busy Day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values)
            plt.xticks(rotation="vertical")
            st.pyplot(fig)

        with col2:
            st.header("Most Busy Month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color="orange")
            plt.xticks(rotation="vertical")
            st.pyplot(fig)

        st.title("Weekly activity map")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

