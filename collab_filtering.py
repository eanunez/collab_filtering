import os
import time
import pandas as pd
from datetime import datetime
import numpy as np
import json
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


class RecommenderNet(keras.Model):

    def __init__(self, num_users, num_items, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.item_embedding = layers.Embedding(
            num_items,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.item_bias = layers.Embedding(num_items, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        item_vector = self.item_embedding(inputs[:, 1])
        item_bias = self.item_bias(inputs[:, 1])
        dot_user_item = tf.tensordot(user_vector, item_vector, 2)
        # Add all the components (including bias)
        x = dot_user_item + user_bias + item_bias
        # The sigmoid activation forces the rating to between 0 and 1
        return tf.nn.sigmoid(x)


class NeuralCollabFiltering:

    base_dir = os.path.abspath(os.path.dirname(__file__))
    col_map = {}
    min_rating = 0
    max_rating = 0
    today = datetime.today().date()
    x_train, y_train, x_val, y_val = [], [], [], []
    df, model, user_encodings, item_encodings, item_encoded2item = None, None, None, None, None

    def __init__(self, domain=None, cols=None):
        """
        :param domain: Site domain to train
        :param cols: Columns to use for training. If None, defaults
        to ['userId', 'itemId', 'rating']"""
        
        self.domain = domain
        if self.domain:
            self.site = domain.split('.')[1].lower()
        else:
            self.site = ''
            
        def_cols = ['userId', 'itemId', 'rating']
        if not cols:
            self.col_map = dict(zip(def_cols, def_cols))
        else:
            self.col_map = dict(zip(def_cols, cols))

    def pre_processing(self, csv_path, encoding_path=None, training_csv=None):
        """Prepares data for training
        
        :param csv_path: Path to csv file containing data.
        :param encoding_path: Output file to save the encoded data.
        :param training_csv: Csv path used for training.
        """
        t0 = time.time()
        df = pd.read_csv(csv_path)
        
        # Reduce and Clean
        users = df[self.col_map['userId']].value_counts()
        items = df[self.col_map['itemId']].value_counts()
        users = users[users >= 2]
        items = items[items >= 2]

        df = df.merge(pd.DataFrame({self.col_map['userId']: users.index})).merge(pd.DataFrame({self.col_map['itemId']: items.index}))
        
        user_ids = df[self.col_map['userId']].unique().tolist()
        user_encodings = {x: i for i, x in enumerate(user_ids)}
        # userencoded2user = {i: x for i, x in enumerate(user_ids)}
        item_ids = df[self.col_map['itemId']].unique().tolist()
        item_encodings = {x: i for i, x in enumerate(item_ids)}
        item_encoded2item = {i: x for i, x in enumerate(item_ids)}
        df['user'] = df[self.col_map["userId"]].map(user_encodings)
        df['item'] = df[self.col_map['itemId']].map(item_encodings)

        df[self.col_map['rating']] = df[self.col_map['rating']].values.astype(np.float32)
        # min and max ratings will be used to normalize the ratings later
        self.min_rating = min(df[self.col_map["rating"]])
        self.max_rating = max(df[self.col_map["rating"]])

        print(
            f"Number of {self.col_map['userId']}: {len(user_encodings)}, "
            f"Number of {self.col_map['itemId']}: {len(item_encodings)}, "
            f"Min {self.col_map['rating']}: {self.min_rating}, "
            f"Max {self.col_map['rating']}: {self.max_rating}"
            )
        if not encoding_path:
            encoding_path = os.path.join(self.base_dir, f"model/{self.site}_encoders_{self.today.strftime('%Y%m%d')}.json")
        encodings = {'user_encodings': user_encodings, 'item_encodings': item_encodings}
        with open(encoding_path, 'w+', encoding='utf-8', newline='') as f:
            json.dump(encodings, f, indent=4, sort_keys=True)
            print(f"Encodings data saved at {encoding_path}")
            
        if not training_csv:
            csv_file = f"{os.path.basename(csv_path).split('.csv')[0]}_reduced.csv"
            training_csv = os.path.join(self.base_dir, f"dataset/{csv_file}")
        with open(training_csv, 'w+', encoding='utf-8', newline='') as f:
            df.to_csv(f, index=False)
            print(f"New training data saved at: {training_csv}")
            
        # set config
        json_path = os.path.join(self.base_dir, 'config.json')
        data = {}
        with open(json_path, 'r') as jf:
            data = json.load(jf)
            if not data:
                data[self.site] = {'encoding': encoding_path,
                                   'data': training_csv}
            else:
                data[self.site]['encoding'] = encoding_path
                data[self.site]['data'] =  training_csv
                
        with open(json_path, 'w+') as jf:
            json.dump(data, jf)
        print(f"Pre-processing time {time.time() - t0} secs")

        return df

    def train_test_split(self, df, test_size=.1, seed=42):
        """Splits data to training and testing."""
        df = df.sample(frac=1, random_state=seed)
        x = df[["user", "item"]].values
        # Normalize the targets between 0 and 1. Makes it easy to train.
        y = df[self.col_map["rating"]].apply(
            lambda x: (x - self.min_rating) / (self.max_rating - self.min_rating)).values
        # Assuming training on 90% of the data and validating on 10%.
        train_indices = int((1 - test_size) * df.shape[0])
        self.x_train, self.x_val, self.y_train, self.y_val = (
            x[:train_indices],
            x[train_indices:],
            y[:train_indices],
            y[train_indices:],
        )

    @staticmethod
    def build_model(num_users, num_items, embedding_size=64):
        """"""
        model = RecommenderNet(num_users, num_items, embedding_size)
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(lr=0.001)
        )
        return model

    def train(self, num_users, num_items, embedding_size=64, x_train=None, y_train=None,
              validation_data=(None, None), batch_size=200, epochs=10, verbose=1, plot=True, save_model=True):

        t0 = time.time()
        
        if not all([x_train, y_train]) and not all(validation_data):
            x_train = self.x_train
            y_train = self.y_train
            validation_data = (self.x_val, self.y_val)
        model = self.build_model(num_users, num_items, embedding_size)
        history = model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_data=validation_data,
        )
        if plot:
            plt.plot(history.history["loss"])
            plt.plot(history.history["val_loss"])
            plt.title("model loss")
            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.legend(["train", "test"], loc="upper left")
            plt.show()
        if save_model:
            model_path = os.path.join(self.base_dir, f"model/{self.site}_ncf_model_{self.today.strftime('%Y%m%d')}")
            model.save(model_path, save_format='tf')
            
        # set config
        json_path = os.path.join(self.base_dir, 'config.json')
        with open(json_path, 'r') as jf:
            data = json.load(jf)
            data[self.site]['model'] = model_path
        with open(json_path, 'w+') as jf:
            json.dump(data, jf)
        
        print(f'Training time: {(time.time() - t0)/3600} hours.')
        
        return model
    
    def load_model(self, csv_path, model_path, encodings_path):
        """Loads pre-trained model to memory.
        
        :param csv_path: Path containing the raw data as reference
        :param model_path: path of the model
        :param encodings_path: path ot user and item encodings."""
        self.df = pd.read_csv(csv_path)
        assert encodings_path.endswith('.json')
        with open(encodings_path, 'r', encoding='utf-8') as f:
            encoded = json.load(f)
            self.user_encodings = encoded['user_encodings']
            self.item_encodings = encoded['item_encodings']
        self.item_encoded2item = {v: k for k, v in self.item_encodings.items()}
        self.model = tf.keras.models.load_model(model_path, custom_objects=None, compile=True, options=None)
        
    def recommend(self, user_id, verbose=True):
        # if not user_id:
        #    user_id = self.df[self.df.duplicated(subset=[self.col_map['userId']])][self.col_map['userId']].sample(1).iloc[0]
            
        items_watched_by_user = self.df[self.df[self.col_map['userId']] == user_id]
        if items_watched_by_user.empty:
            return []
        items_not_watched = self.df[~self.df[self.col_map['itemId']].isin(
            items_watched_by_user[self.col_map['itemId']].values)][self.col_map['itemId']]
        items_not_watched = list(
            set(items_not_watched).intersection(set(self.item_encodings.keys()))
        )
        items_not_watched = [[self.item_encodings.get(x)] for x in items_not_watched]
        user_encoder = self.user_encodings.get(user_id)
        user_item_array = np.hstack(
            ([[user_encoder]] * len(items_not_watched), items_not_watched)
        )
        ratings = self.model.predict(user_item_array).flatten()
        top_ratings_indices = ratings.argsort()[-10:][::-1]
        recommended_item_ids = [
            self.item_encoded2item.get(items_not_watched[x][0]) for x in top_ratings_indices
        ]
        recommended_items = self.df[
                self.df[self.col_map["itemId"]].isin(recommended_item_ids)].drop_duplicates(subset=[self.col_map['itemId']])
        if verbose:
            print("Showing recommendations for user: {}".format(user_id))
            print("====" * 9)
            print("Articles with high time on page from user")
            print("----" * 8)
            top_items_user = (
                items_watched_by_user.sort_values(by=self.col_map['itemId'], ascending=False).drop_duplicates()
                    .head(5)
                    .slug.values
            )
            item_df_rows = self.df[self.df[self.col_map['itemId']].isin(top_items_user)].drop_duplicates(subset=[self.col_map['itemId']])
            for item in item_df_rows[self.col_map['itemId']]:
                print(item)

            print("----" * 8)
            print("Top 10 article recommendations")
            print("----" * 8)
            
            for item in recommended_items[self.col_map['itemId']]:
                print(item)
        return recommended_items[self.col_map['itemId']].values.tolist()
        

    def recommend_(self, csv_path, model, encodings, user_id=None, verbose=True):
        """Recommends items to users
        :param csv_path: Path containing the raw data as reference
        :param model: Model
        :param user_id: User to predict
        :param encodings: Path to user and item encodings."""
        # Let us get a user and see the top recommendations.
        # user_id = df.summit_client_id.sample(1).iloc[0]

        df = pd.read_csv(csv_path)
        assert encodings.endswith('.json')
        with open(encodings, 'r', encoding='utf-8') as f:
            encoded = json.load(f)
            user_encodings = encoded['user_encodings']
            item_encodings = encoded['item_encodings']
        item_encoded2item = {v: k for k, v in item_encodings.items()}

        if not user_id:
            user_id = df[df.duplicated(subset=[self.col_map['userId']])][self.col_map['userId']].sample(1).iloc[0]
            
        items_watched_by_user = df[df[self.col_map['userId']] == user_id]
        if items_watched_by_user.empty:
            return []
        items_not_watched = df[~df[self.col_map['itemId']].isin(
            items_watched_by_user[self.col_map['itemId']].values)][self.col_map['itemId']]
        items_not_watched = list(
            set(items_not_watched).intersection(set(item_encodings.keys()))
        )
        items_not_watched = [[item_encodings.get(x)] for x in items_not_watched]
        user_encoder = user_encodings.get(user_id)
        user_item_array = np.hstack(
            ([[user_encoder]] * len(items_not_watched), items_not_watched)
        )
        ratings = model.predict(user_item_array).flatten()
        top_ratings_indices = ratings.argsort()[-10:][::-1]
        recommended_item_ids = [
            item_encoded2item.get(items_not_watched[x][0]) for x in top_ratings_indices
        ]
        recommended_items = df[
                df[self.col_map["itemId"]].isin(recommended_item_ids)].drop_duplicates(subset=[self.col_map['itemId']])
        if verbose:
            print("Showing recommendations for user: {}".format(user_id))
            print("====" * 9)
            print("Articles with high time on page from user")
            print("----" * 8)
            top_items_user = (
                items_watched_by_user.sort_values(by=self.col_map['itemId'], ascending=False).drop_duplicates()
                    .head(5)
                    .slug.values
            )
            item_df_rows = df[df[self.col_map['itemId']].isin(top_items_user)].drop_duplicates(subset=[self.col_map['itemId']])
            for item in item_df_rows[self.col_map['itemId']]:
                print(item)

            print("----" * 8)
            print("Top 10 article recommendations")
            print("----" * 8)
            
            for item in recommended_items[self.col_map['itemId']]:
                print(item)
        return recommended_items[self.col_map['itemId']].values.tolist()


if __name__ == '__main__':
    # PARSING ARGUMENTS
    parser = argparse.ArgumentParser(prog='python3 collab_filtering.py')
    parser.add_argument("domain", type=str, help="Input the site domain to train.")
    parser.add_argument("infile", type=str, help="Complete path to training file (.csv)")
    
    cols = ['summit_client_id', 'slug', 'time_on_page']
    ncf = NeuralCollabFiltering(args.domain, cols)
    df = ncf.pre_processing(args.infile)
    n_users = df.summit_client_id.unique().shape[0]
    n_items = df.slug.unique().shape[0]
    ncf.train_test_split(df)
    ncf.train(n_users, n_items)