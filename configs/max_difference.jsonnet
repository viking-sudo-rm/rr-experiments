# Model hyperparameters.
local RNN = std.extVar("RNN");
local HIDDEN_DIM = std.extVar("H");

# Task parameters.
local LENGTH = 64;


{

  "dataset_reader": {
    "type": "max_difference",
    "seed": 2,
  },

  "train_data_path": "10000:" + LENGTH,
  "validation_data_path": "1000:" + LENGTH,

  "model": {
    "type": "tagger2",
    "hidden_dim": HIDDEN_DIM,

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
      "num_layers": 1
    }

  },

  "iterator": {
      "type": "bucket",
      "sorting_keys": [["tokens", "num_tokens"]],
      "batch_size": 16,
  },
  "trainer": {
    "optimizer": "adam",
    "validation_metric": "+accuracy",
    "num_epochs": 100,
    "patience": 10,
    "cuda_device": 0,
    "num_serialized_models_to_keep": 1
  }
}