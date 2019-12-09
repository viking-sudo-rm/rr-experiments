# Model hyperparameters.
local RNN = std.extVar("RNN");
local LAYERS = std.extVar("LAYERS");

# Task parameters.
local NUM_TRAIN = 100000;
local NUM_VALID = 10000;
local LENGTH = 64;


{

  "dataset_reader": {
    "type": "prefix",
    "seed": 2,
  },

  "train_data_path": NUM_TRAIN + ":" + LENGTH,
  "validation_data_path": NUM_VALID + ":" + LENGTH,

  "model": {
    "type": "simple_tagger",

    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": 4
        }
      }
    },

    "encoder": {
      "type": RNN,
      "input_size": 4,
      "hidden_size": 10,
      "num_layers": LAYERS,
    }

  },

  "iterator": {
      "type": "bucket",
      "sorting_keys": [["tokens", "num_tokens"]],
      "batch_size": 16,
  },
  "trainer": {
    "optimizer": "adam",
    "validation_metric": "-loss",
    "num_epochs": 100,
    "patience": 10,
    "cuda_device": 0,
    "num_serialized_models_to_keep": 1
  }
}