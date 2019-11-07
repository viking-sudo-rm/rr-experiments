{

  "dataset_reader": {
    "type": "anbk",
    "prefix": true,
    "abs_value": true
  },

  "train_data_path": "2:300",
  "validation_data_path": "310:320",

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
      "type": "lstm",
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
    "num_epochs": 100,
    "patience": 10,
    "cuda_device": -1
  }
}