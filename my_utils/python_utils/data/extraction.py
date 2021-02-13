import numpy as np


# OK
def extract_data_by_class_counts(
    labels, num_classes, num_class_outputs,
    shuffle=False, seed=None,
    return_unselected=False, return_select_mask=False):

    """
    labels: A list of all labels

    num_class_outputs: A list of length 'num_classes' whose each element describes
    the number of samples for the corresponding class

    If 'num_class_outputs' is an int,
    each class will have the same number of samples equal to 'num_class_outputs'
    """

    labels = np.asarray(labels)
    assert len(labels.shape) == 1, "'labels' must be a 1D array!"

    ids = np.arange(len(labels))

    if shuffle:
        rs = np.random.RandomState(seed=seed)
        rs.shuffle(ids)

    if np.isscalar(num_class_outputs):
        num_class_outputs = [num_class_outputs for _ in range(num_classes)]
    assert len(num_class_outputs) == num_classes, "len(num_output_labels)={} " \
        "while num_classes={}".format(len(num_class_outputs), num_classes)

    remaining = np.array(num_class_outputs, copy=True)
    total_remaining = np.sum(num_class_outputs)

    selected_ids = []
    unselected_ids = []

    for n, idx in enumerate(ids):
        if total_remaining <= 0:
            break

        if remaining[labels[idx]] > 0:
            selected_ids.append(idx)
            remaining[labels[idx]] -= 1
            total_remaining -= 1
        else:
            unselected_ids.append(idx)

    assert len(selected_ids) == np.sum(num_class_outputs), \
        "len(output_ids)={} but sum(num_output_labels)={}!".format(
            len(selected_ids), np.sum(num_class_outputs))

    selected_ids = np.asarray(selected_ids, dtype=np.int32)
    assert len(selected_ids) == np.sum(num_class_outputs), "Cannot get enough samples!"
    outputs = [selected_ids]

    if return_unselected:
        unselected_ids = np.concatenate(
            [np.asarray(unselected_ids, dtype=np.int32),
             ids[n:]], axis=0)
        assert (len(selected_ids) + len(unselected_ids)) == len(labels), \
            "len(selected_ids)={}, len(unselected_ids)={}, len(labels)={}!".format(
                len(selected_ids), len(unselected_ids), len(labels))
        outputs.append(unselected_ids)

    if return_select_mask:
        select_mask = np.full(len(labels), fill_value=False, dtype=np.bool)
        select_mask[selected_ids] = True
        outputs.append(select_mask)

    if len(outputs) == 1:
        return outputs[0]
    else:
        return tuple(outputs)


# Extend the first version with "return_selected_by_class"
def extract_data_by_class_counts_v2(
    labels, num_classes, num_class_outputs,
    shuffle=False, seed=None,
    return_selected_by_class=False, return_unselected=False, return_select_mask=False):

    """
    labels: A list of all labels

    num_class_outputs: A list of length 'num_classes' whose each element describes
    the number of samples for the corresponding class

    If 'num_class_outputs' is an int,
    each class will have the same number of samples equal to 'num_class_outputs'
    """

    labels = np.asarray(labels)
    assert len(labels.shape) == 1, "'labels' must be a 1D array!"

    ids = np.arange(len(labels))

    if shuffle:
        rs = np.random.RandomState(seed=seed)
        rs.shuffle(ids)

    if np.isscalar(num_class_outputs):
        num_class_outputs = [num_class_outputs for _ in range(num_classes)]
    assert len(num_class_outputs) == num_classes, "len(num_output_labels)={} " \
        "while num_classes={}".format(len(num_class_outputs), num_classes)

    remaining = np.array(num_class_outputs, copy=True)
    total_remaining = np.sum(num_class_outputs)

    selected_ids = []
    selected_ids_by_class = [[] for _ in range(num_classes)]
    unselected_ids = []

    for n, idx in enumerate(ids):
        if total_remaining <= 0:
            break

        if remaining[labels[idx]] > 0:
            selected_ids.append(idx)
            selected_ids_by_class[labels[idx]].append(idx)
            remaining[labels[idx]] -= 1
            total_remaining -= 1
        else:
            unselected_ids.append(idx)

    assert len(selected_ids) == np.sum(num_class_outputs), \
        "len(output_ids)={} but sum(num_output_labels)={}!".format(
            len(selected_ids), np.sum(num_class_outputs))

    selected_ids = np.asarray(selected_ids, dtype=np.int32)
    assert len(selected_ids) == np.sum(num_class_outputs), "Cannot get enough samples!"
    outputs = [selected_ids]

    if return_selected_by_class:
        num_by_class = [len(l) for l in selected_ids_by_class]
        assert np.array_equal(num_by_class, num_class_outputs), \
            f"num_by_class={num_by_class} while num_class_outputs={num_class_outputs}"
        outputs.append(selected_ids_by_class)

    if return_unselected:
        unselected_ids = np.concatenate([np.asarray(unselected_ids, dtype=np.int32), ids[n:]], axis=0)
        assert (len(selected_ids) + len(unselected_ids)) == len(labels), \
            "len(selected_ids)={}, len(unselected_ids)={}, len(labels)={}!".format(
                len(selected_ids), len(unselected_ids), len(labels))
        outputs.append(unselected_ids)

    if return_select_mask:
        select_mask = np.full(len(labels), fill_value=False, dtype=np.bool)
        select_mask[selected_ids] = True
        outputs.append(select_mask)

    if len(outputs) == 1:
        return outputs[0]
    else:
        return tuple(outputs)


# OK
def extract_data_by_total_counts(
    labels, num_classes, num_outputs,
    shuffle=False, seed=None,
    return_unselected=False, return_select_mask=False):

    classes, class_counts = np.unique(labels, return_counts=True)
    num_samples = len(labels)
    assert num_samples > 0

    class_num_dict = {classes[i]: int(num_outputs * class_counts[i] * 1.0 / num_samples)
                      for i in range(len(classes))}

    remaining = num_outputs - np.sum(list(class_num_dict.values()))
    assert 0 <= remaining <= len(classes), "remaining={}".format(remaining)

    for i in range(remaining):
        class_num_dict[classes[i]] += 1

    assert np.sum(list(class_num_dict.values())) == num_outputs

    num_class_outputs = []
    for c in range(num_classes):
        if c in class_num_dict:
            num_class_outputs.append(class_num_dict[c])
        else:
            num_class_outputs.append(0)

    return extract_data_by_class_counts(labels, num_classes,
                                        num_class_outputs=num_class_outputs,
                                        shuffle=shuffle, seed=seed,
                                        return_unselected=return_unselected,
                                        return_select_mask=return_select_mask)


# OK
# Use class proportions
# Convert 'class_output_props' into 'num_class_outputs'
# Support imbalanced classes if 'class_output_props' is the same
def extract_data_by_class_proportions(
    labels, num_classes, class_output_props,
    shuffle=False, seed=None,
    return_unselected=False, return_select_mask=False):

    classes, class_counts = np.unique(labels, return_counts=True)
    class_counter = {classes[i]: class_counts[i] for i in range(len(classes))}
    if np.isscalar(class_output_props):
        class_output_props = [class_output_props for _ in range(num_classes)]

    for i in range(len(class_output_props)):
        assert 0.0 <= class_output_props[i] <= 1.0, "Proportion of class {} is not " \
            "in the range [0, 1]. Found {}!".format(i, class_output_props[i])

    num_class_outputs = []
    for c in range(num_classes):
        if c in class_counter:
            num_class_outputs.append(int(class_counter[c] * class_output_props[c]))
        else:
            num_class_outputs.append(0)

    return extract_data_by_class_counts(labels, num_classes,
                                        num_class_outputs=num_class_outputs,
                                        shuffle=shuffle, seed=seed,
                                        return_unselected=return_unselected,
                                        return_select_mask=return_select_mask)


# OK
def split_data(data_size_or_ids, proportion, shuffle=True, seed=None, sort=False):
    # Simply split data (don't care about labels)
    if isinstance(data_size_or_ids, int):
        data_size = data_size_or_ids
        ids = list(range(data_size_or_ids))
    else:
        assert hasattr(data_size_or_ids, '__len__')
        ids = data_size_or_ids.tolist() if isinstance(data_size_or_ids, np.ndarray) \
            else list(data_size_or_ids)
        data_size = len(data_size_or_ids)

    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(ids)

    proportion = np.asarray(proportion)
    s = np.sum(proportion)
    if s != 1.0:
        proportion = proportion * 1.0/s

    split_sizes = proportion * data_size
    int_split_sizes = split_sizes.astype(np.int32)

    residual_sizes = split_sizes - int_split_sizes
    sorted_ids = np.argsort(residual_sizes, axis=0)[::-1]  # Sort in decreasing order

    k = 0
    while np.sum(int_split_sizes) != data_size:
        int_split_sizes[sorted_ids[k % len(proportion)]] += 1
        k += 1

    # `int_split_sizes` contains true number of elements for each proportion
    split_points = np.cumsum(int_split_sizes, axis=0).tolist()
    assert split_points[-1] == data_size
    split_points.insert(0, 0)

    if sort:
        return tuple(sorted(ids[split_points[i-1]: split_points[i]]) for i in range(1, len(split_points)))
    else:
        return tuple(ids[split_points[i-1]: split_points[i]] for i in range(1, len(split_points)))