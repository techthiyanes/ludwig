import os
import tempfile

from ludwig.api import LudwigModel
from tests.integration_tests.utils import category_feature, generate_data, numerical_feature


def test_pretraining(tmpdir):
    with tempfile.TemporaryDirectory() as outdir:
        input_features = [
            category_feature(vocab_size=3, reduce_input="sum"),
            numerical_feature(),
        ]
        output_features = [category_feature(vocab_size=2, reduce_input="sum")]

        csv_filename = os.path.join(tmpdir, "training.csv")
        data_csv = generate_data(input_features, output_features, csv_filename)

        config = {
            "input_features": input_features,
            "output_features": output_features,
            "pretraining": {"epochs": 3},
            "training": {"epochs": 2},
        }

        model = LudwigModel(config)
        _, _, output_directory = model.train(dataset=data_csv, output_directory=outdir)
