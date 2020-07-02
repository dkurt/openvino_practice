import os
import cv2 as cv
import numpy as np

from addict import Dict
from compression.graph import load_model, save_model
from compression.data_loaders.data_loader import DataLoader
from compression.engines.ie_engine import IEEngine
from compression.custom.metric import Metric
from compression.pipeline.initializer import create_pipeline

class DatasetsDataLoader(DataLoader):

    def __init__(self, config):
        assert config['images']
        self.dataset = config['images']
        self.images = os.listdir(config['images'])

    @property
    def size(self):
        return len(self.images)

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        print('.')
        img = cv.imread(os.path.join(self.dataset, self.images[item]))

        img = cv.resize(img, (320, 240))
        img = img[:,:,[2,1,0]]
        inp = img.transpose(2, 0, 1).reshape(1, 3, 240, 320).astype(np.float32)

        return (item, None), inp


model_config = Dict({
    'model_name': 'candy',
    "model": os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'candy.xml'),
    "weights": os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'candy.bin')
})
engine_config = {
    'device': 'CPU',
    'stat_requests_number': 4,
    'eval_requests_number': 4
}
dataset_config = {
    'images': None,
}
algorithms = [
    {
        'name': 'DefaultQuantization',
        'params': {
            'target_device': 'CPU',
            'preset': 'performance',
            'stat_subset_size': 300,
        }
    }
]

model = load_model(model_config)

data_loader = DatasetsDataLoader(dataset_config)

engine = IEEngine(engine_config, data_loader, metric=None, loss=None)
pipeline = create_pipeline(algorithms, engine)

compressed_model = pipeline.run(model)
save_model(compressed_model, 'optimized', model_name='candy_int8')
