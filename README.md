# Article Recommender - Neural Collaborative Filtering

This is an article recommender which is based on collaborative filtering. 
The idea is borrowed from Keras' [demo](https://keras.io/examples/structured_data/collaborative_filtering_movielens/) on collaborative filtering 
and modified to fit for article recommender system, replacing rating with time on page. 
The methods are generic to accept users and items of different header columns.

## To install:
1. `$ ./install.sh`
2. Change to home directory:
`$ cd collab_filtering`
3. Activate virtual environment
`$ source venv/bin/activate`

## Training:
`cols = [<client_id>, <article_id>, <time_on_page>]`
`ncf = NeuralCollabFiltering(cols)'
`ncf.pre_process(csv_path, 'dataset/encodings.json')`
