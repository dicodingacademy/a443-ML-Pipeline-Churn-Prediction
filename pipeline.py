import os
import sys
from typing import Text

from absl import logging
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

PIPELINE_NAME = "customer-churn-pipeline"

# pipeline inputs
root = os.getcwd()
data_dir = os.path.join(root, "data")
training_module = os.path.join(root, "module", "customer_churn_trainer.py")
transform_module = os.path.join(root, "module", "customer_churn_transform.py")
requirement_file = os.path.join(root, "requirements.txt")

# pipeline outputs
output_base = os.path.join(root, "output", PIPELINE_NAME)
serving_model_dir = os.path.join(output_base, PIPELINE_NAME)
pipeline_root = os.path.join(output_base, "pipeline_root")
metadata_path = os.path.join(pipeline_root, "metadata.sqlite")


def init_beam_pipeline(
    components, pipeline_root: Text, direct_num_workers: int
) -> pipeline.Pipeline:

    logging.info(f"Pipeline root set to: {pipeline_root}")
    beam_arg = (
        f"--direct_num_workers={direct_num_workers}",
        f"--requirements_file={requirement_file}",  # optional
        "--direct_running_mode=multi_processing",
    )

    p = pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=False,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        beam_pipeline_args=beam_arg,
    )
    return p

if __name__ == "__main__":

    logging.set_verbosity(logging.INFO)

    module_path = os.getcwd()
    if module_path not in sys.path:
        print(module_path)
        sys.path.append(module_path)

    from module.components import init_components

    components = init_components(
        data_dir,
        training_module=training_module,
        transform_module=transform_module,
        training_steps=5000,
        eval_steps=100,
        serving_model_dir=serving_model_dir,
    )
    direct_num_workers = int(os.cpu_count() / 2)
    direct_num_workers = 1 if direct_num_workers < 1 else direct_num_workers
    pipeline = init_beam_pipeline(components, pipeline_root, direct_num_workers)
    BeamDagRunner().run(pipeline)