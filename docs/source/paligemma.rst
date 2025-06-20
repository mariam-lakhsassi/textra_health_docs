Fine-tuning PaLI-Gemma for Handwritten Prescription Recognition
==============================================================

This documentation explains step-by-step how to fine-tune the PaLI-Gemma Vision-Language Model (VLM) on a custom dataset of handwritten medical prescriptions. It covers data preparation, environment setup, model loading, training, evaluation, and checkpoint saving.

Dataset Structure
-----------------

Your dataset should be organized as follows:

.. code-block:: text

   Doctor’s Handwritten Prescription BD dataset/
   ├── Training/
   │   ├── images/
   │   │   ├── 0.png
   │   │   ├── 1.png
   │   │   └── ...
   │   └── training_labels.json
   ├── Validation/
   │   ├── images/
   │   │   ├── 0.png
   │   │   └── ...
   │   └── validation_labels.json
   └── Testing/
       ├── images/
       │   ├── 0.png
       │   └── ...
       └── testing_labels.json

Each JSON file contains one object per line:

.. code-block:: json

   {"prefix": "", "suffix": "Aceta", "image": "0.png"}
   {"prefix": "", "suffix": "Ibuprofen", "image": "1.png"}

- `prefix`: always an empty string (can be used for prompts if needed)
- `suffix`: the label (medicine name) for the image
- `image`: filename of the image in the corresponding `images/` folder

Data Preparation
----------------

If your labels are in CSV format, you can convert them to JSONL as follows:

.. code-block:: python

   import os
   import pandas as pd
   import json

   base_dir = '/content/drive/MyDrive/S8/Textra_health/data (1)/Doctor’s Handwritten Prescription BD dataset'
   subdirs = ['Testing', 'Training', 'Validation']

   for subdir in subdirs:
       csv_path = os.path.join(base_dir, subdir, f"{subdir.lower()}_labels.csv")
       json_path = os.path.join(base_dir, subdir, f"{subdir.lower()}_labels.json")
       if os.path.exists(csv_path):
           df = pd.read_csv(csv_path)
           with open(json_path, 'w', encoding='utf-8') as f:
               for _, row in df.iterrows():
                   json_item = {
                       "prefix": "",
                       "suffix": row["MEDICINE_NAME"],
                       "image": row["IMAGE"]
                   }
                   f.write(json.dumps(json_item, ensure_ascii=False) + '\n')
           print(f"Created: {json_path}")

Environment Setup
-----------------

1. **Clone big_vision and install dependencies**

   .. code-block:: python

      !git clone --quiet --branch=main --depth=1 https://github.com/google-research/big_vision big_vision_repo
      !pip3 install -q "overrides" "ml_collections" "einops~=0.7" "sentencepiece" tensorflow

   Add the repo to your Python path:

   .. code-block:: python

      import sys
      if "big_vision_repo" not in sys.path:
          sys.path.append("big_vision_repo")

2. **Configure Kaggle API (for model download)**

   - In Colab, add your Kaggle username and API key as secrets (`KAGGLE_USERNAME`, `KAGGLE_KEY`).
   - Accept the model terms on [Kaggle](https://www.kaggle.com/models/google/paligemma/).

   .. code-block:: python

      import os
      from google.colab import userdata
      os.environ["KAGGLE_USERNAME"] = userdata.get('KAGGLE_USERNAME')
      os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')

3. **Mount Google Drive (if using Colab)**

   .. code-block:: python

      from google.colab import drive
      drive.mount('/content/drive')

Download Model and Tokenizer
----------------------------

.. code-block:: python

   import kagglehub

   LLM_VARIANT = "gemma2_2b"
   MODEL_PATH = "./paligemma2-3b-pt-224.b16.npz"
   KAGGLE_HANDLE = "google/paligemma-2/jax/paligemma2-3b-pt-224"

   if not os.path.exists(MODEL_PATH):
       print("Downloading the checkpoint from Kaggle...")
       MODEL_PATH = kagglehub.model_download(KAGGLE_HANDLE, MODEL_PATH)

   TOKENIZER_PATH = "./paligemma_tokenizer.model"
   if not os.path.exists(TOKENIZER_PATH):
       print("Downloading the model tokenizer...")
       !gsutil cp gs://big_vision/paligemma_tokenizer.model {TOKENIZER_PATH}

Check your data directory:

.. code-block:: python

   DATA_DIR = "/content/drive/MyDrive/S8/Textra_health/data (1)/Doctor’s Handwritten Prescription BD dataset"
   print(os.path.exists(DATA_DIR))

Model Initialization
--------------------

.. code-block:: python

   import ml_collections
   import sentencepiece
   from big_vision.models.proj.paligemma import paligemma
   from big_vision.trainers.proj.paligemma import predict_fns

   model_config = ml_collections.FrozenConfigDict({
       "llm": {"vocab_size": 257_152, "variant": LLM_VARIANT, "final_logits_softcap": 0.0},
       "img": {"variant": "So400m/14", "pool_type": "none", "scan": True, "dtype_mm": "float16"}
   })
   model = paligemma.Model(**model_config)
   tokenizer = sentencepiece.SentencePieceProcessor(TOKENIZER_PATH)

   params = paligemma.load(None, MODEL_PATH, model_config)
   decode_fn = predict_fns.get_all(model)['decode']
   decode = functools.partial(decode_fn, devices=jax.devices(), eos_token=tokenizer.eos_id())

Parameter Management and Sharding
---------------------------------

To fit the model on a T4 GPU, only the attention layers are fine-tuned and others are frozen (float16). Parameters are sharded if multiple devices are available.

.. code-block:: python

   def is_trainable_param(name, param):
       if name.startswith("llm/layers/attn/"):  return True
       if name.startswith("llm/"):              return False
       if name.startswith("img/"):              return False
       raise ValueError(f"Unexpected param name {name}")

   trainable_mask = big_vision.utils.tree_map_with_names(is_trainable_param, params)
   mesh = jax.sharding.Mesh(jax.devices(), ("data"))
   data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
   params_sharding = big_vision.sharding.infer_sharding(params, strategy=[('.*', 'fsdp(axis="data")')], mesh=mesh)

   @functools.partial(jax.jit, donate_argnums=(0,), static_argnums=(1,))
   def maybe_cast_to_f32(params, trainable):
       return jax.tree.map(lambda p, m: p.astype(jnp.float32) if m else p.astype(jnp.float16), params, trainable)

   # Load params param by param to avoid RAM issues
   params, treedef = jax.tree.flatten(params)
   sharding_leaves = jax.tree.leaves(params_sharding)
   trainable_leaves = jax.tree.leaves(trainable_mask)
   for idx, (sharding, trainable) in enumerate(zip(sharding_leaves, trainable_leaves)):
       params[idx] = big_vision.utils.reshard(params[idx], sharding)
       params[idx] = maybe_cast_to_f32(params[idx], trainable)
       params[idx].block_until_ready()
   params = jax.tree.unflatten(treedef, params)

Preprocessing Functions
-----------------------

.. code-block:: python

   def preprocess_image(image, size=224):
       image = np.asarray(image)
       if image.ndim == 2:
           image = np.stack((image,)*3, axis=-1)
       image = image[..., :3]
       image = tf.constant(image)
       image = tf.image.resize(image, (size, size), method='bilinear', antialias=True)
       return image.numpy() / 127.5 - 1.0

   def preprocess_tokens(prefix, suffix=None, seqlen=None):
       separator = "\n"
       tokens = tokenizer.encode(prefix, add_bos=True) + tokenizer.encode(separator)
       mask_ar = [0] * len(tokens)
       mask_loss = [0] * len(tokens)
       if suffix:
           suffix = tokenizer.encode(suffix, add_eos=True)
           tokens += suffix
           mask_ar += [1] * len(suffix)
           mask_loss += [1] * len(suffix)
       mask_input = [1] * len(tokens)
       if seqlen:
           padding = [0] * max(0, seqlen - len(tokens))
           tokens = tokens[:seqlen] + padding
           mask_ar = mask_ar[:seqlen] + padding
           mask_loss = mask_loss[:seqlen] + padding
           mask_input = mask_input[:seqlen] + padding
       return jax.tree.map(np.array, (tokens, mask_ar, mask_loss, mask_input))

   def postprocess_tokens(tokens):
       tokens = tokens.tolist()
       try:
           eos_pos = tokens.index(tokenizer.eos_id())
           tokens = tokens[:eos_pos]
       except ValueError:
           pass
       return tokenizer.decode(tokens)

Data Iterators
--------------

.. code-block:: python

   train_DIR = os.path.join(DATA_DIR, "Training")
   val_DIR = os.path.join(DATA_DIR, "Validation")
   train_dataset = big_vision.datasets.jsonl.DataSource(
       os.path.join(train_DIR, "training_labels.json"),
       fopen_keys={"image": train_DIR})
   val_dataset = big_vision.datasets.jsonl.DataSource(
       os.path.join(val_DIR, "validation_labels.json"),
       fopen_keys={"image": val_DIR})

   def train_data_iterator():
       dataset = train_dataset.get_tfdata().shuffle(1_000).repeat()
       for example in dataset.as_numpy_iterator():
           image = Image.open(io.BytesIO(example["image"]))
           image = preprocess_image(image)
           prefix = "caption en"
           suffix = example["suffix"].decode().lower()
           tokens, mask_ar, mask_loss, _ = preprocess_tokens(prefix, suffix, SEQLEN)
           yield {
               "image": np.asarray(image),
               "text": np.asarray(tokens),
               "mask_ar": np.asarray(mask_ar),
               "mask_loss": np.asarray(mask_loss),
           }

   def validation_data_iterator():
       for example in val_dataset.get_tfdata(ordered=True).as_numpy_iterator():
           image = Image.open(io.BytesIO(example["image"]))
           image = preprocess_image(image)
           prefix = "caption en"
           tokens, mask_ar, _, mask_input = preprocess_tokens(prefix, seqlen=SEQLEN)
           yield {
               "image": np.asarray(image),
               "text": np.asarray(tokens),
               "mask_ar": np.asarray(mask_ar),
               "mask_input": np.asarray(mask_input),
           }

Training Loop
-------------

.. code-block:: python

   @functools.partial(jax.jit, donate_argnums=(0,))
   def update_fn(params, batch, learning_rate):
       imgs, txts, mask_ar = batch["image"], batch["text"], batch["mask_ar"]
       def loss_fn(params):
           text_logits, _ = model.apply({"params": params}, imgs, txts[:, :-1], mask_ar[:, :-1], train=True)
           logp = jax.nn.log_softmax(text_logits, axis=-1)
           mask_loss = batch["mask_loss"][:, 1:]
           targets = jax.nn.one_hot(txts[:, 1:], text_logits.shape[-1])
           token_pplx = jnp.sum(logp * targets, axis=-1)
           example_loss = -jnp.sum(token_pplx * mask_loss, axis=-1)
           example_loss /= jnp.clip(jnp.sum(mask_loss, -1), 1)
           return jnp.mean(example_loss)
       loss, grads = jax.value_and_grad(loss_fn)(params)
       def apply_grad(param, gradient, trainable):
           if not trainable: return param
           return param - learning_rate * gradient
       params = jax.tree_util.tree_map(apply_grad, params, grads, trainable_mask)
       return params, loss

   # Training loop
   BATCH_SIZE = 8
   TRAIN_STEPS = 7000
   LEARNING_RATE = 0.01
   sched_fn = big_vision.utils.create_learning_rate_schedule(
       total_steps=TRAIN_STEPS+1, base=LEARNING_RATE,
       decay_type="cosine", warmup_percent=0.05)
   train_data_it = train_data_iterator()
   for step in range(1, TRAIN_STEPS+1):
       examples = [next(train_data_it) for _ in range(BATCH_SIZE)]
       batch = jax.tree.map(lambda *x: np.stack(x), *examples)
       batch = big_vision.utils.reshard(batch, data_sharding)
       learning_rate = sched_fn(step)
       params, loss = update_fn(params, batch, learning_rate)
       if step == 1 or step % 500 == 0:
           loss = jax.device_get(loss)
           print(f"step: {step:6d}/{TRAIN_STEPS:6d}   lr: {learning_rate:.5f}   loss: {loss:.4f}")

Evaluation
----------

.. code-block:: python

   def make_predictions(data_iterator, *, num_examples=None, batch_size=8, seqlen=SEQLEN, sampler="greedy"):
       outputs = []
       while True:
           examples = []
           try:
               for _ in range(batch_size):
                   examples.append(next(data_iterator))
                   examples[-1]["_mask"] = np.array(True)
           except StopIteration:
               if len(examples) == 0:
                   return outputs
           while len(examples) % batch_size:
               examples.append(dict(examples[-1]))
               examples[-1]["_mask"] = np.array(False)
           batch = jax.tree.map(lambda *x: np.stack(x), *examples)
           batch = big_vision.utils.reshard(batch, data_sharding)
           tokens = decode({"params": params}, batch=batch, max_decode_len=seqlen, sampler=sampler)
           tokens, mask = jax.device_get((tokens, batch["_mask"]))
           tokens = tokens[mask]
           responses = [postprocess_tokens(t) for t in tokens]
           for example, response in zip(examples, responses):
               outputs.append((example["image"], response))
               if num_examples and len(outputs) >= num_examples:
                   return outputs

   # Evaluate on validation set
   for image, caption in make_predictions(validation_data_iterator(), batch_size=8):
       # Display or save results as needed
       pass

Saving the Fine-tuned Model
---------------------------

.. code-block:: python

   def npsave(pytree, path):
       names_and_vals, _ = big_vision.utils.tree_flatten_with_names(pytree)
       with open(path, "wb") as f:
           np.savez(f, **{k:v for k, v in names_and_vals})

   npsave(params, '/content/drive/MyDrive/S8/textra_health/my-custom-paligemma-ckpt.npz')

   # Check if saved
   import os
   print(os.path.exists('/content/drive/MyDrive/S8/textra_health/my-custom-paligemma-ckpt.npz'))

How to Use the Fine-tuned Model
-------------------------------

- Upload your fine-tuned checkpoint (`my-custom-paligemma-ckpt.npz`) and tokenizer to your workspace or Google Drive.
- When running inference, load the model and parameters as shown above, but set `MODEL_PATH` to your fine-tuned checkpoint.

.. code-block:: python

   params = paligemma.load(None, '/content/drive/MyDrive/S8/textra_health/my-custom-paligemma-ckpt.npz', model_config)

- Use the same preprocessing and decoding functions for inference.

Summary
-------

This procedure allows you to fine-tune PaLI-Gemma on a custom dataset of prescription images, using a simple JSONL format for labels and images. The notebook demonstrates all steps from data preparation to training, evaluation, and saving the final model for later use.