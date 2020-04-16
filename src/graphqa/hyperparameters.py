import time
from typing import Optional, Dict, Any

from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from torch.utils.tensorboard._convert_np import make_np
from tensorboard.compat.proto.summary_pb2 import Summary, SummaryMetadata
from tensorboard.plugins.hparams.plugin_data_pb2 import (
    HParamsPluginData,
    SessionEndInfo,
    SessionStartInfo,
)
from tensorboard.plugins.hparams.api_pb2 import (
    Experiment,
    HParamInfo,
    MetricInfo,
    MetricName,
    Status,
    DatasetType,
    DataType,
)

PLUGIN_NAME = "hparams"
PLUGIN_DATA_VERSION = 0

EXPERIMENT_TAG = "_hparams_/experiment"
SESSION_START_INFO_TAG = "_hparams_/session_start_info"
SESSION_END_INFO_TAG = "_hparams_/session_end_info"


def _make_experiment_summary(hparam_infos, metric_infos, experiment):
    """Define hyperparameters and metrics.

    Args:
        hparam_infos: information about all hyperparameters (name, description, type etc.),
            list of dicts containing 'name' (required), 'type', 'description', 'display_name',
            'domain_discrete', 'domain_interval'
        metric_infos: information about all metrics (tag, description etc.),
            list of dicts containing 'tag' (required), 'dataset_type', 'description', 'display_name'
        experiment: dict containing 'name' (required), 'description', 'time_created_secs', 'user'

    Returns:

    """

    def make_hparam_info(hparam):
        data_types = {
            None: DataType.DATA_TYPE_UNSET,
            str: DataType.DATA_TYPE_STRING,
            list: DataType.DATA_TYPE_STRING,
            tuple: DataType.DATA_TYPE_STRING,
            bool: DataType.DATA_TYPE_BOOL,
            int: DataType.DATA_TYPE_FLOAT64,
            float: DataType.DATA_TYPE_FLOAT64,
        }
        return HParamInfo(
            name=hparam["name"],
            type=data_types[hparam.get("type")],
            description=hparam.get("description"),
            display_name=hparam.get("display_name"),
            domain_discrete=hparam.get("domain_discrete"),
            domain_interval=hparam.get("domain_interval"),
        )

    def make_metric_info(metric):
        return MetricInfo(
            name=MetricName(tag=metric["tag"]),
            dataset_type=DatasetType.Value(
                f'DATASET_{metric.get("dataset_type", "UNKNOWN").upper()}'
            ),
            description=metric.get("description"),
            display_name=metric.get("display_name"),
        )

    def make_experiment_info(experiment, metric_infos, hparam_infos):
        return Experiment(
            name=experiment["name"],
            description=experiment.get("description"),
            time_created_secs=experiment.get("time_created_secs"),
            user=experiment.get("user"),
            metric_infos=metric_infos,
            hparam_infos=hparam_infos,
        )

    metric_infos = [make_metric_info(m) for m in metric_infos]
    hparam_infos = [make_hparam_info(hp) for hp in hparam_infos]
    experiment = make_experiment_info(experiment, metric_infos, hparam_infos)

    experiment_content = HParamsPluginData(
        experiment=experiment, version=PLUGIN_DATA_VERSION
    )
    experiment_summary_metadata = SummaryMetadata(
        plugin_data=SummaryMetadata.PluginData(
            plugin_name=PLUGIN_NAME, content=experiment_content.SerializeToString()
        )
    )
    experiment_summary = Summary(
        value=[Summary.Value(tag=EXPERIMENT_TAG, metadata=experiment_summary_metadata)]
    )

    return experiment_summary


def _make_session_start_summary(
    hparam_values,
    group_name: Optional[str] = None,
    start_time_secs: Optional[int] = None,
):
    """Assign values to the hyperparameters in the context of this session.

    Args:
        hparam_values: a dict of ``hp_name`` -> ``hp_value`` mappings
        group_name: optional group name for this session
        start_time_secs: optional starting time in seconds

    Returns:

    """
    if start_time_secs is None:
        start_time_secs = int(time.time())
    session_start_info = SessionStartInfo(
        group_name=group_name, start_time_secs=start_time_secs
    )

    for hp_name, hp_value in hparam_values.items():
        # Logging a None would raise an exception when setting session_start_info.hparams[hp_name].number_value = None.
        # Logging a float.nan instead would work, but that run would not show at all in the tensorboard hparam plugin.
        # The best thing to do here is to skip that value, it will show as a blank cell in the table view of the
        # tensorboard plugin. However, that run would not be shown in the parallel coord or in the scatter plot view.
        if hp_value is None:
            logger.warning(
                f"Hyper parameter {hp_name} is `None`: the tensorboard hp plugin "
                f"will show this run in table view, but not in parallel coordinates "
                f"view or in scatter plot matrix view"
            )
            continue

        if isinstance(hp_value, (str, list, tuple)):
            session_start_info.hparams[hp_name].string_value = str(hp_value)
            continue

        if isinstance(hp_value, bool):
            session_start_info.hparams[hp_name].bool_value = hp_value
            continue

        if not isinstance(hp_value, (int, float)):
            hp_value = make_np(hp_value)[0]

        session_start_info.hparams[hp_name].number_value = hp_value

    session_start_content = HParamsPluginData(
        session_start_info=session_start_info, version=PLUGIN_DATA_VERSION
    )
    session_start_summary_metadata = SummaryMetadata(
        plugin_data=SummaryMetadata.PluginData(
            plugin_name=PLUGIN_NAME, content=session_start_content.SerializeToString()
        )
    )
    session_start_summary = Summary(
        value=[
            Summary.Value(
                tag=SESSION_START_INFO_TAG, metadata=session_start_summary_metadata
            )
        ]
    )

    return session_start_summary


def _make_session_end_summary(status: str, end_time_secs: Optional[int] = None):
    """

    Args:
        status: outcome of this run, one of of 'UNKNOWN', 'SUCCESS', 'FAILURE', 'RUNNING'
        end_time_secs: optional ending time in seconds

    Returns:

    """
    status = Status.DESCRIPTOR.values_by_name[f"STATUS_{status.upper()}"].number
    if end_time_secs is None:
        end_time_secs = int(time.time())

    session_end_summary = SessionEndInfo(status=status, end_time_secs=end_time_secs)
    session_end_content = HParamsPluginData(
        session_end_info=session_end_summary, version=PLUGIN_DATA_VERSION
    )
    session_end_summary_metadata = SummaryMetadata(
        plugin_data=SummaryMetadata.PluginData(
            plugin_name=PLUGIN_NAME, content=session_end_content.SerializeToString()
        )
    )
    session_end_summary = Summary(
        value=[
            Summary.Value(
                tag=SESSION_END_INFO_TAG, metadata=session_end_summary_metadata
            )
        ]
    )

    return session_end_summary


def add_hparam_summary(
    writer: SummaryWriter, hparam_values: Dict[str, Any], metric_infos
):
    hparam_infos = [
        {"name": key, "type": type(value)} for key, value in hparam_values.items()
    ]

    metric_infos = [{"dataset_type": "validation", **mi} for mi in metric_infos]
    experiment = {"name": "experiment", "time_created_secs": 0}

    experiment_summary = _make_experiment_summary(
        hparam_infos, metric_infos, experiment
    )
    writer.file_writer.add_summary(experiment_summary)


def add_session_start(writer: SummaryWriter, hparam_values: Dict[str, Any]):
    session_summary = _make_session_start_summary(hparam_values)
    writer.file_writer.add_summary(session_summary)


def add_session_end(writer: SummaryWriter, status: str):
    status = status.upper()
    if status not in {"UNKNOWN", "SUCCESS", "FAILURE", "RUNNING"}:
        logger.warning(f"Invalid status: {status}")

    session_end_summary = _make_session_end_summary(status)
    writer.file_writer.add_summary(session_end_summary)
