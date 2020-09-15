import os
# from neural_recommender.collab_filtering import NeuralCollabFiltering
from collab_filtering import NeuralCollabFiltering

# Init
base_dir = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(base_dir, 'model/spot_ncf_model_20200915')
encoder_path = os.path.join(base_dir, 'model/encodings_20200915.json')
ref_path = os.path.join(base_dir, 'dataset/spot_scid_slug_20200301-20200831_reduced.csv')

# Define column headers
cols = ['summit_client_id', 'slug', 'time_on_page']

ncf = NeuralCollabFiltering(cols)
ncf.load_model(ref_path, model_path, encoder_path)

def reco(summit_id, **kwargs):
    return ncf.recommend(summit_id, **kwargs)