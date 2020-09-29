import logging

logger = logging.getLogger(__name__)


def index_to_time(index, metadata):
    """
    Find time units (in terms of base units) at a given index (typically obtained from a timestamped filename)
    :param index: Index obtained from a timestamped filename (.out.press.00055.pdb etc.)
    :param metadata: A Metadata object obtained by parsing a .pfmetadata file
    :return: Time steps elapsed since start for index.
    """

    # TODO: The following logic probably suffers from one-off errors, and can also be improved in terms
    # of runtime by direct comparison of values (using log(..) etc.).
    # However, as a first-pass implementation, it's meant to be direct and easy to follow.

    base_unit = metadata['TimingInfo.BaseUnit']
    dump_interval = metadata['TimingInfo.DumpInterval']
    timestep_type = metadata['TimeStep.Type']
    start_time = metadata['TimingInfo.StartTime']
    start_count = metadata['TimingInfo.StartCount']

    if timestep_type == 'Constant':
        return start_time + ((index - start_count) * dump_interval / base_unit)

    elif timestep_type == 'Growth':
        step0 = metadata['TimeStep.InitialStep']
        factor = metadata['TimeStep.GrowthFactor']
        step_min = metadata['TimeStep.MinStep']
        step_max = metadata['TimeStep.MaxStep']

        t = start_time
        step = max(step_min, min(step_max, step0))
        i = 0

        logger.debug('index_to_time calculations:')
        while True:
            logger.debug(f'{i}: step={step}, time={t}')
            # TODO: What about base_unit here?
            if (index - start_count) * dump_interval == i:
                return t
            i += 1
            t += step
            step = max(step_min, min(step_max, step * factor))
