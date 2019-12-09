local TASK = std.extVar("TASK");


{

  "dataset_reader": {
    "type": TASK,
    "seed": 2,
  },

  # Since "training" doesn't matter, just use 10:10.
  "train_data_path": "10:10",
  "validation_data_path": "10:10",

  "model": {
    "type": "mock_false_tagger",
    "value": true,
  },

  "iterator": {
      "type": "bucket",
      "sorting_keys": [["tokens", "num_tokens"]],
      "batch_size": 16,
  },
  "trainer": {
    "optimizer": "adam",
    "num_epochs": 1,
  }
}