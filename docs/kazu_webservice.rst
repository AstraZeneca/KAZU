Kazu as a Web Server (REST API) with FastAPI
============================================

The Kazu framework can be run in REST API mode with a few additional steps. This page provides a simple tutorial on how to host a REST API server using FastAPI.

Installation
------------

First, clone the repository:

.. code-block:: bash

    git clone https://github.com/AstraZeneca/KAZU.git
    cd KAZU

Install the library with the `webserver` option:

.. code-block:: bash

    export VERSION=2.1.1
    pip install kazu[webserver]==${VERSION}

    wget https://github.com/AstraZeneca/KAZU/releases/download/v${VERSION}/kazu_model_pack_public-v${VERSION}.zip
    unzip kazu_model_pack_public-v${VERSION}.zip

    export KAZU_MODEL_PACK=${PWD}/kazu_model_pack_public-v${VERSION}

Run and Test the Server
-----------------------

Run the web server:

.. code-block:: bash

    mkdir run_dir
    python -m kazu.web.server --config-path "${KAZU_MODEL_PACK}/conf" hydra.run.dir="${PWD}/run_dir"

It will take a few minutes to be fully deployed. Once fully deployed, you will see a completion message. 

Following is an example of such a message:

.. code-block:: bash

    2024-07-11 01:44:28,366 INFO api.py:609 -- Deployed app 'default' successfully.
    2024-07-11 01:44:28,379 INFO server.py:648 -- ServeStatus(proxies={'0d2e08c45946473': <ProxyStatus.HEALTHY: 'HEALTHY'>}, 
    applications={'default': ApplicationStatusOverview(status=<ApplicationStatus.RUNNING: 'RUNNING'>, message='', last_deployed_time_s=172067695.0692, 
    deployments={'KazuWebAPI': DeploymentStatusOverview(status=<DeploymentStatus.HEALTHY: 'HEALTHY'>, status_trigger=<DeploymentStatusTrigger.CONFIG_UPDATE_COMPLETED: 
    'CONFIG_UPDATE_COMPLETED'>, replica_states={'RUNNING': 1}, message='')})}, target_capacity=None)

To test, execute the following example from any machine on the same network as the server:

.. code-block:: bash

    # Set IP address according to the server.
    # If the web server is running on the same server, use:
    export IP_ADDR=127.0.0.1
    export PORT=8080

    curl --header "Content-Type: application/json" --request POST \
     --data '{"text": "EGFR is an important gene in breast cancer study."}' \
     http://${IP_ADDR}:${PORT}/api/kazu

FastAPI
-------

You can see the FastAPI documentation at :

http://<your-ip-address>:<port>/docs

One successful example is http://kazu.korea.ac.kr/docs.
