.. _sparseml:

SparseML
========

`SparseML <https://github.com/neuralmagic/sparseml>`__ provides GPU-class performance on CPUs through sparsification, pruning, and quantization.
For more details, see `SparseML docs <https://docs.neuralmagic.com/sparseml/>`__.

With multiple machines, the command has to be run on all machines either manually, or using an orchestration system such as SLURM or TorchElastic. More information can be seen in the Pytorch Lightning `Computing Cluster <https://pytorch-lightning.readthedocs.io/en/latest/advanced/cluster.html#computing-cluster>`_.

GPU Config
----------

We provide out of the box configs to use SparseML. Below is an example of how you can swap to the default trainer config for SparseML when using the text classification task.

.. code-block:: bash

    python train.py task=nlp/text_classification                 \
        dataset=nlp/text_classification/glue                     \
        dataset.cfg.dataset_config_name=sst2                     \
        backbone.pretrained_model_name_or_path=bert-base-uncased \
        trainer=sparseml                                         \
        ++trainer.callbacks.output_dir=/content/MODELS           \
        ++trainer.callbacks.recipe_path=/content/recipe.yaml     \
        log=True                                                 \

These commands are only useful when a recipe has already been created. Example recipes can be found `here <https://github.com/neuralmagic/sparseml/tree/main/integrations/huggingface-transformers/recipes>`__.

After training, this will leave two ONNX models in the trainer.callbacks.output_dir folder: small_model.onnx and model.onnx. small_model.onnx is excellent for demos. For reliable inference, it is recommended to optimize model.onnx with your compression algorithm.

CPU Config
----------

If you want to use SparseML only as a callback, or if you would like to train/test on CPU, use a variation of the following config:

.. code-block:: bash

    python train.py task=nlp/text_classification                 \
        dataset=nlp/text_classification/glue                     \
        dataset.cfg.dataset_config_name=sst2                     \
        backbone.pretrained_model_name_or_path=bert-base-uncased \
        trainer.gpus=0                                           \
        +trainer/callbacks=sparseml                              \
        +trainer/logger=sparsewandb                              \
        ++trainer.callbacks.output_dir=/content/MODELS           \
        ++trainer.callbacks.recipe_path=/content/recipe.yaml     \
        log=True                                                 \
