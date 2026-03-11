import os
import numpy as np
import pandas as pd
from os.path import exists
import pickle
import torch
from tganood.utils import preprocess_sentence, preprocess_french_sentence, SBERT_embed


# ============================================================
# Closed-set helpers
# ============================================================

def get_event2012_closed_set_test_df(df):
    save_path = './data/Event2012/closed_set/'
    if not exists(save_path):
        os.makedirs(save_path)
    test_set_df_np_path = save_path + 'test_set.npy'
    if not exists(test_set_df_np_path):
        test_mask = torch.load('./raw_data/Event2012/masks/test_mask.pt').cpu().detach().numpy()
        test_mask = list(np.where(test_mask == True)[0])
        test_df = df.iloc[test_mask]
        test_df_np = test_df.to_numpy()
        np.save(test_set_df_np_path, test_df_np)
    return


def get_event2018_closed_set_test_df(df):
    save_path = './data/Event2018/closed_set/'
    if not exists(save_path):
        os.makedirs(save_path)
    test_set_df_np_path = save_path + 'test_set.npy'
    if not exists(test_set_df_np_path):
        with open('./raw_data/Event2018/data_splits/test_indices.pkl', 'rb') as f:
            test_indices = pickle.load(f)
        test_df = df.iloc[test_indices]
        test_df_np = test_df.to_numpy()
        np.save(test_set_df_np_path, test_df_np)
    return


# ============================================================
# Closed-set SBERT embeddings
# ============================================================

def get_event2012_closed_set_messages_embeddings():
    save_path = './data/Event2012/closed_set/'
    SBERT_embedding_path = f'{save_path}/SBERT_embeddings.pkl'
    if not exists(SBERT_embedding_path):
        test_set_df_np_path = save_path + 'test_set.npy'
        test_df_np = np.load(test_set_df_np_path, allow_pickle=True)
        test_df = pd.DataFrame(data=test_df_np, columns=[
            "event_id", "tweet_id", "text", "user_id", "created_at", "user_loc",
            "place_type", "place_full_name", "place_country_code", "hashtags",
            "user_mentions", "image_urls", "entities", "words", "filtered_words", "sampled_words"
        ])
        print("Dataframe loaded.")
        processed_text = [preprocess_sentence(s) for s in test_df['text'].values]
        print('Message text contents preprocessed.')
        embeddings = SBERT_embed(processed_text, language='English')
        with open(SBERT_embedding_path, 'wb') as fp:
            pickle.dump(embeddings, fp)
        print('SBERT embeddings stored.')
    return


def get_event2018_closed_set_messages_embeddings():
    save_path = './data/Event2018/closed_set/'
    SBERT_embedding_path = f'{save_path}/SBERT_embeddings.pkl'
    if not exists(SBERT_embedding_path):
        test_set_df_np_path = save_path + 'test_set.npy'
        test_df_np = np.load(test_set_df_np_path, allow_pickle=True)
        test_df = pd.DataFrame(data=test_df_np, columns=[
            "tweet_id", "user_name", "text", "time", "event_id", "user_mentions",
            "hashtags", "urls", "words", "created_at", "filtered_words", "entities", "sampled_words"
        ])
        print("Dataframe loaded.")
        processed_text = [preprocess_french_sentence(s) for s in test_df['text'].values]
        print('Message text contents preprocessed.')
        embeddings = SBERT_embed(processed_text, language='French')
        with open(SBERT_embedding_path, 'wb') as fp:
            pickle.dump(embeddings, fp)
        print('SBERT embeddings stored.')
    return


# ============================================================
# Open-set SBERT embeddings
# ============================================================

def get_event2012_open_set_messages_embeddings():
    """Get the SBERT embeddings for messages in blocks 1-21."""
    save_path = './data/Event2012/open_set/'
    for i in range(21):
        block = i + 1
        print('\n\n====================================================')
        print('block: ', block)
        SBERT_embedding_path = f'{save_path}{block}/SBERT_embeddings.pkl'
        if not exists(SBERT_embedding_path):
            df_np = np.load(f'{save_path}{block}/{block}.npy', allow_pickle=True)
            df = pd.DataFrame(data=df_np, columns=[
                "original_index", "event_id", "tweet_id", "text", "user_id", "created_at", "user_loc",
                "place_type", "place_full_name", "place_country_code", "hashtags", "user_mentions",
                "image_urls", "entities", "words", "filtered_words", "sampled_words", "date"
            ])
            print("Dataframe loaded.")
            df['processed_text'] = [preprocess_sentence(s) for s in df['text']]
            print('Message text contents preprocessed.')
            embeddings = SBERT_embed(df['processed_text'].tolist(), language='English')
            with open(SBERT_embedding_path, 'wb') as fp:
                pickle.dump(embeddings, fp)
            print('SBERT embeddings stored.')
    return


def get_event2018_open_set_messages_embeddings():
    save_path = './data/Event2018/open_set/'
    for i in range(16):
        block = i + 1
        print('\n\n====================================================')
        print('block: ', block)
        SBERT_embedding_path = f'{save_path}{block}/SBERT_embeddings.pkl'
        if not exists(SBERT_embedding_path):
            df_np = np.load(f'{save_path}{block}/{block}.npy', allow_pickle=True)
            df = pd.DataFrame(data=df_np, columns=[
                "original_index", "tweet_id", "user_name", "text", "time", "event_id",
                "user_mentions", "hashtags", "urls", "words", "created_at",
                "filtered_words", "entities", "sampled_words", "date"
            ])
            print("Dataframe loaded.")
            df['processed_text'] = [preprocess_french_sentence(s) for s in df['text']]
            print('Message text contents preprocessed.')
            embeddings = SBERT_embed(df['processed_text'].tolist(), language='French')
            with open(SBERT_embedding_path, 'wb') as fp:
                pickle.dump(embeddings, fp)
            print('SBERT embeddings stored.')
    return


# ============================================================
# Open-set block splitting
# ============================================================

def split_open_set(df, root_path, dataset='2012'):
    if not exists(root_path):
        os.makedirs(root_path)
    df = df.sort_values(by='created_at').reset_index()
    df['date'] = [d.date() for d in df['created_at']]
    distinct_dates = df.date.unique()

    # First week -> block 0
    folder = root_path + '0/'
    if not exists(folder):
        os.mkdir(folder)
    df_np_path = folder + '0.npy'
    if not exists(df_np_path):
        ini_df = df.loc[df['date'].isin(distinct_dates[:7])]
        np.save(df_np_path, ini_df.to_numpy())

    # Following dates -> block 1, 2, ...
    if dataset == '2012':
        end = len(distinct_dates) - 1
    else:
        end = len(distinct_dates)
    for i in range(7, end):
        folder = root_path + str(i - 6) + '/'
        if not exists(folder):
            os.mkdir(folder)
        df_np_path = folder + str(i - 6) + '.npy'
        if not exists(df_np_path):
            incr_df = df.loc[df['date'] == distinct_dates[i]]
            np.save(df_np_path, incr_df.to_numpy())
    return


# ============================================================
# Top-level preprocessing entry points
# ============================================================

def preprocess_event2012():
    p_part1 = './raw_data/Event2012/68841_tweets_multiclasses_filtered_0722_part1.npy'
    p_part2 = './raw_data/Event2012/68841_tweets_multiclasses_filtered_0722_part2.npy'
    df_np = np.concatenate(
        (np.load(p_part1, allow_pickle=True), np.load(p_part2, allow_pickle=True)), axis=0
    )
    print("Loaded data.")
    df = pd.DataFrame(data=df_np, columns=[
        "event_id", "tweet_id", "text", "user_id", "created_at", "user_loc",
        "place_type", "place_full_name", "place_country_code", "hashtags",
        "user_mentions", "image_urls", "entities", "words", "filtered_words", "sampled_words"
    ])
    print("Data converted to dataframe.")
    df = df.sample(frac=0.1, random_state=42)
    print("Sampled 10% of the data.")

    # Open-set
    root_path = './data/Event2012/open_set/'
    split_open_set(df, root_path, dataset='2012')
    get_event2012_open_set_messages_embeddings()

    # Closed-set
    get_event2012_closed_set_test_df(df)
    get_event2012_closed_set_messages_embeddings()
    return


def preprocess_event2018():
    columns = [
        "tweet_id", "user_name", "text", "time", "event_id", "user_mentions",
        "hashtags", "urls", "words", "created_at", "filtered_words", "entities", "sampled_words"
    ]
    df_np = np.load('./raw_data/Event2018/french_tweets.npy', allow_pickle=True)
    df = pd.DataFrame(data=df_np, columns=columns)
    print("Data converted to dataframe.")
    df = df.sample(frac=0.1, random_state=42)
    print("Sampled 10% of the data.")

    # Open-set
    root_path = './data/Event2018/open_set/'
    split_open_set(df, root_path, dataset='2018')
    get_event2018_open_set_messages_embeddings()

    # Closed-set
    get_event2018_closed_set_test_df(df)
    get_event2018_closed_set_messages_embeddings()
    return


if __name__ == "__main__":
    preprocess_event2012()
    preprocess_event2018()