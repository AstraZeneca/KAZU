.. _scaling_kazu:

Scaling with Ray
=================


Usually, we want to run Kazu over large number of documents, so we need a framework to handle the distributed processing.

`Ray <https://www.ray.io//>`_ is a simple to use Actor style framework that works extremely well for this. In this example,

we demonstrate how Ray can be used to scale Kazu over multiple cores.

.. note::
    Ray can also be used in a multi node environment, for extreme scaling. Please refer to the Ray docs for this.



Overview
-----------

We'll use the Kazu :class:`.LLMNERStep` with some clean up actions to build a Kazu pipeline. We'll then create multiple
Ray actors to instantiate this pipeline, then feed those actors Kazu :class:`.Document`\s through :class:`ray.util.queue.Queue`\.
The actors will process the documents, and write the results to another :class:`ray.util.queue.Queue`\. The main process will then
read from this second queue and write the results to disk.

The code for this orchestration is in ```scripts/examples/annotate_with_llm.py``` and the configuration is in
```scripts/examples/conf/annotate_with_llm/default.yaml```

The script can be executed with:

.. code-block:: console

   $ python scripts/examples/annotate_with_llm.py --config-path /<fully qualified>/kazu/scripts/examples/conf hydra.job.chdir=True


.. note::
    You will need to add values for the configuration keys marked ???, such as your input directory, vertex config etc.
