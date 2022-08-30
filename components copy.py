import os

import tensorflow_model_analysis as tfma
import tfx
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy
)
from tfx.types import Channel 
from tfx.proto import example_gen_pb2, pusher_pb2, trainer_pb2


def init_components(
    data_dir,
    transform_module,
    training_module,
    training_steps,
    eval_steps,
    serving_model_dir=None,
):
    """initiate tfx pipeline components

    Args:
        data_dir (_type_): _description_
        transform_module (_type_): _description_
        training_module (_type_): _description_
        training_steps (_type_): _description_
        eval_steps (_type_): _description_
        serving_model_dir (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    
    output = example_gen_pb2.Output(
        split_config = example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
            example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2)
        ])
    )

    example_gen = tfx.components.CsvExampleGen(
        input_base=os.path.join(os.getcwd(), data_dir), 
        output_config=output
    )
    
    statistics_gen = tfx.components.StatisticsGen(
        examples=example_gen.outputs["examples"]
    )
    
    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs["statistics"],
        infer_feature_shape=False,
    )
    
    example_validator = tfx.components.ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )

    transform = tfx.components.Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=transform_module,
    )
    
    trainer = tfx.components.Trainer(
        module_file= training_module,
        examples= transform.outputs["transformed_examples"],
        schema= schema_gen.outputs["schema"],
        transform_graph= transform.outputs["transform_graph"],
        train_args= trainer_pb2.TrainArgs(num_steps=training_steps),
        eval_args= trainer_pb2.EvalArgs(num_steps=eval_steps),
    )
    
    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=tfx.types.standard_artifacts.Model),
        model_blessing=Channel(
            type=tfx.types.standard_artifacts.ModelBlessing
        )
    )
    
    slicing_specs=[
        tfma.SlicingSpec(), 
        tfma.SlicingSpec(feature_keys=[
            "gender",
            "Partner"
        ])
    ]

    metrics_specs = [
        tfma.MetricsSpec(metrics=[
                tfma.MetricConfig(class_name='AUC'),
                tfma.MetricConfig(class_name="Precision"),
                tfma.MetricConfig(class_name="Recall"),
                tfma.MetricConfig(class_name="ExampleCount"),
                tfma.MetricConfig(class_name='BinaryAccuracy',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value':0.5}),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value':0.0001})
                        )
                )
            ])
    ]

    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key='Churn')],
        slicing_specs=slicing_specs,
        metrics_specs=metrics_specs
    )
    
    evaluator = tfx.components.Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config
    )
    
    pusher = tfx.components.Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        ),
    )
    
    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        model_resolver,
        evaluator,
        pusher,
    ]
    
    return components