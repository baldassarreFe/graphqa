from typing import Optional

from tensorboardX.x2num import make_np
from tensorboardX.proto.summary_pb2 import Summary, SummaryMetadata

from tensorboardX.proto.plugin_hparams_pb2 import HParamsPluginData, SessionEndInfo, SessionStartInfo
from tensorboardX.proto.api_pb2 import Experiment, HParamInfo, MetricInfo, MetricName, Status, DatasetType, DataType
from six import string_types


PLUGIN_NAME = 'hparams'
PLUGIN_DATA_VERSION = 0

EXPERIMENT_TAG = '_hparams_/experiment'
SESSION_START_INFO_TAG = '_hparams_/session_start_info'
SESSION_END_INFO_TAG = '_hparams_/session_end_info'


def make_experiment_summary(hparam_infos, metric_infos, experiment):
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
        data_type = hparam.get('type')
        if hparam.get('type') is None:
            data_type = DataType.DATA_TYPE_UNSET
        elif hparam.get('type') in string_types:
            data_type = DataType.DATA_TYPE_STRING
        elif hparam.get('type') is bool:
            data_type = DataType.DATA_TYPE_BOOL
        elif hparam.get('type') in (float, int):
            data_type = DataType.DATA_TYPE_FLOAT64
        return HParamInfo(
            name=hparam['name'],
            type=data_type,
            description=hparam.get('description'),
            display_name=hparam.get('display_name'),
            domain_discrete=hparam.get('domain_discrete'),
            domain_interval=hparam.get('domain_interval'),
        )

    def make_metric_info(metric):
        return MetricInfo(
            name=MetricName(tag=metric['tag']),
            dataset_type=DatasetType.Value(f'DATASET_{metric.get("dataset_type", "UNKNOWN").upper()}'),
            description=metric.get('description'),
            display_name=metric.get('display_name')
        )

    def make_experiment_info(experiment, metric_infos, hparam_infos):
        return Experiment(
            name=experiment['name'],
            description=experiment.get('description'),
            time_created_secs=experiment.get('time_created_secs'),
            user=experiment.get('user'),
            metric_infos=metric_infos,
            hparam_infos=hparam_infos
        )

    metric_infos = [make_metric_info(m) for m in metric_infos]
    hparam_infos = [make_hparam_info(hp) for hp in hparam_infos]
    experiment = make_experiment_info(experiment, metric_infos, hparam_infos)

    experiment_content = HParamsPluginData(experiment=experiment, version=PLUGIN_DATA_VERSION)
    experiment_summary_metadata = SummaryMetadata(plugin_data=SummaryMetadata.PluginData(
        plugin_name=PLUGIN_NAME,
        content=experiment_content.SerializeToString())
    )
    experiment_summary = Summary(value=[Summary.Value(
        tag=EXPERIMENT_TAG, 
        metadata=experiment_summary_metadata
    )])
    
    return experiment_summary


def make_session_start_summary(hparam_values, group_name: Optional[str] = None, start_time_secs: Optional[int] = None):
    """Assign values to the hyperparameters in the context of this session.
    
    Args:
        hparam_values: a dict of ``hp_name`` -> ``hp_value`` mappings
        group_name: optional group name for this session
        start_time_secs: optional starting time in seconds

    Returns:

    """
    if start_time_secs is None:
        import time
        start_time_secs = int(time.time())
    session_start_info = SessionStartInfo(group_name=group_name, start_time_secs=start_time_secs)

    for hp_name, hp_value in hparam_values.items():
        # Logging a None would raise an exception when setting session_start_info.hparams[hp_name].number_value = None.
        # Logging a float.nan instead would work, but that run would not show at all in the tensorboard hparam plugin.
        # The best thing is to skip that value, it will show as blank in tensorboard.
        if hp_value is None:
            continue

        if isinstance(hp_value, string_types):
            session_start_info.hparams[hp_name].string_value = hp_value
            continue

        if isinstance(hp_value, bool):
            session_start_info.hparams[hp_name].bool_value = hp_value
            continue

        if not isinstance(hp_value, (int, float)):
            hp_value = make_np(hp_value)[0]

        session_start_info.hparams[hp_name].number_value = hp_value

    session_start_content = HParamsPluginData(session_start_info=session_start_info, version=PLUGIN_DATA_VERSION)
    session_start_summary_metadata = SummaryMetadata(plugin_data=SummaryMetadata.PluginData(
        plugin_name=PLUGIN_NAME,
        content=session_start_content.SerializeToString()
    ))
    session_start_summary = Summary(value=[Summary.Value(
        tag=SESSION_START_INFO_TAG,
        metadata=session_start_summary_metadata
    )])

    return session_start_summary


def make_session_end_summary(status: str, end_time_secs: Optional[int] = None):
    """

    Args:
        status: outcome of this run, one of of 'UNKNOWN', 'SUCCESS', 'FAILURE', 'RUNNING'
        end_time_secs: optional ending time in seconds

    Returns:

    """
    status = Status.DESCRIPTOR.values_by_name[f'STATUS_{status.upper()}'].number
    if end_time_secs is None:
        import time
        end_time_secs = int(time.time())

    session_end_summary = SessionEndInfo(status=status, end_time_secs=end_time_secs)
    session_end_content = HParamsPluginData(session_end_info=session_end_summary, version=PLUGIN_DATA_VERSION)
    session_end_summary_metadata = SummaryMetadata(plugin_data=SummaryMetadata.PluginData(
        plugin_name=PLUGIN_NAME,
        content=session_end_content.SerializeToString()
    ))
    session_end_summary = Summary(value=[Summary.Value(
        tag=SESSION_END_INFO_TAG,
        metadata=session_end_summary_metadata
    )])

    return session_end_summary
