import zipfile
import csv
import os
import io
import numpy as np
import pickle
import logging
# dataset names are these keys:

FILES = dict(numeric='TMNIST_Data.csv.zip',
             alphanumeric='94_character_TMNIST.csv.zip')
# Stored w/keys:  names, labels, data, test_mask (for test/train split)
CACHE_FILES = dict(numeric='TMNIST_Data.cache.npz',
                   alphanumeric='94_character_TMNIST.cache.npz')

# Lower-case letters and punctuation are scaled inconsistently, just use these:

GOOD_CHAR_SET = subset = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
                          'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                          'Y', 'Z', '~', '@', '!', '#', '$', '%', '&', '{', '}', '[', ']', '(', ')', '|', '?']


def _load_and_filter_zipped_csv(filepath: str, label_filters: set = None):
    """


    Function generated from Google AI prompt (and modified): 

            Show a python script to load a zipped csv file with the following format:
            - header row: name, label, 1, 2, ..., 784
            - data row: <string>, <string>, uint8, uint8, ..., uint8

            The file is very large, so it will need to be efficient, and also the function 
            can take an optional argument, label_filters=set(string1, string2, ...), only 
            rows with the second column, "labels", whose value matches a string in 
            label_filters is returned.


    Loads a CSV file from within a .zip archive, filters rows based on the 'label' column,
    and converts numeric columns to uint8.

    Args:
        filepath (str): Path to the .zip file.
        label_filters (set, optional): A set of strings. Only rows where the 'label'
                                       column value is present in this set will be returned.
                                       If None, all rows are returned. Defaults to None.
    Returns:

    """
    if label_filters is not None:
        label_filters = set(label_filters)

    csv_filename_in_zip = os.path.splitext(filepath)[0]
    labels = []
    names = []
    data_rows = []

    with zipfile.ZipFile(filepath, 'r') as zf:
        with zf.open(csv_filename_in_zip, 'r') as csv_file_bytes:
            # Wrap the binary stream in TextIOWrapper to read it as text
            with io.TextIOWrapper(csv_file_bytes, encoding='utf-8', newline='') as text_file:
                reader = csv.reader(text_file)

                # Read the header row
                header = next(reader)

                for row_num, row in enumerate(reader, start=2):

                    if len(row) != len(header):
                        print(
                            f"Warning: Skipping malformed row {row_num} in '{csv_filename_in_zip}' with unexpected number of columns: {row}")
                        continue
                    name = row[0]
                    label = str(row[1])
                    data_strs = row[2:]

                    if label_filters is not None and label not in label_filters:
                        continue

                    names.append(name)
                    labels.append(label)
                    try:
                        data_rows.append(np.array([int(float(x)) for x in data_strs], dtype=np.uint8))
                    except:
                        print("Error in row %i: '%s'" % (row_num, row))

    return np.array(data_rows, dtype=np.uint8), np.array(labels), np.array(names)


def _load(dataset,  test_frac=0.15):
    dataset_file = FILES[dataset]
    cache_file = CACHE_FILES[dataset]

    cache_file = "(test=%.2f)-%s" % (test_frac, cache_file)

    if os.path.exists(cache_file):
        logging.info("Loading cached data from %s", cache_file)
        with np.load(cache_file) as data:
            return data['data'], data['labels'], data['names'], data['test_mask']
    else:
        logging.info("Cache not found, loading data from %s", dataset_file)
        data, labels, names = _load_and_filter_zipped_csv(dataset_file)
        mask_inds = np.random.choice(np.arange(len(data)), int(len(data) * test_frac), replace=False)
        test_mask = np.zeros(len(data), dtype=bool)
        test_mask[mask_inds] = True
        logging.info("Loaded %d rows from %s, caching...", len(data), dataset_file)
        with open(cache_file, 'wb') as f:
            np.savez(f, data=data, labels=labels, names=names, test_mask=test_mask)
        logging.info("Cached data to %s", cache_file)

    return data, labels, names, test_mask


def _split_data(data, labels, names, test_mask):
    train = (data[~test_mask], labels[~test_mask], names[~test_mask])
    test = (data[test_mask], labels[test_mask], names[test_mask])
    return train, test


def load_numeric(subset=None, test_train_split=0.15, w_names=False):
    data, labels, names, test_mask = _load('numeric', test_frac=test_train_split)
    data = (data/255.0).astype(np.float32)
    labels = np.array([int(l) for l in labels]).astype(int)
    names = np.array(names)
    if subset is not None:
        mask = np.isin(labels, subset)
        data = data[mask]
        labels = labels[mask]
        names = names[mask]
        test_mask = test_mask[mask]
    train, test = _split_data(data, labels, names, test_mask)
    if not w_names:
        return ((train[0], train[1]), (test[0], test[1]))
    return train, test


def load_alphanumeric(subset=None, test_train_split=0.15, w_names=False, numeric_labels=True):
    """
    :param subset: A list of characters to include in the dataset, or None for all.
    :param test_train_split: The fraction of the dataset to use for testing.
    :param w_names:  return (x,y,font_name) for each test/train set.
    :param numeric_labels: Whether to use numeric labels (0-9) instead of character labels (A-Z).
    :returns:  (x_train, y_train), (x_test, y_test), if w_names is false, else
       (x_train, y_train, font_name_train), (x_test, y_test, font_name_test)
    """
    data, labels, names, test_mask = _load('alphanumeric', test_frac=test_train_split)
    data = data.astype(np.float32)
    data /= 255.0

    names = np.array(names)

    if subset is not None:
        mask = np.isin(labels, subset)
        data = data[mask]
        labels = labels[mask]
        names = names[mask]
        test_mask = test_mask[mask]

    if numeric_labels:

        label_classes = sorted(list(set(labels)))
        lc_lut = {lc: i for i, lc in enumerate(label_classes)}
        n_labels = np.array([lc_lut[l] for l in labels])
        labels = n_labels

    train, test = _split_data(data, labels, names, test_mask)
    if not w_names:
        return ((train[0], train[1]), (test[0], test[1]))
    return train, test


def _tl(wo_names, w_names):
    (x_train, y_train), (x_test, y_test) = wo_names
    (x_traina, y_traina, names_train), (x_testa, y_testa, names_test) = w_names

    assert x_train.shape[1] == x_test.shape[1] == 784
    assert y_train.size == x_train.shape[0]
    assert y_test.size == x_test.shape[0]
    assert len(names_train) == len(x_train) and len(names_test) == len(x_test)


def test_load_numeric():
    wo_names = load_numeric()
    w_names = load_numeric(w_names=True)
    _tl(wo_names, w_names)


def test_load_alphanumeric():
    wo_names = load_alphanumeric()
    w_names = load_alphanumeric(w_names=True)
    _tl(wo_names, w_names)


def _test(subset, which, print_names=False):
    (data, labels, names), _ = load_alphanumeric(subset=subset, w_names=True)\
        if which == 'alphanumeric' else load_numeric(subset=subset, w_names=True)
    print("Dataset:  %s,  subset:  %s  -->  Read %i rows, %i cols" % (which, subset, data.shape[0], data.shape[1]))
    label_arr = np.array(labels)
    label_list = sorted(list(set(labels)))
    l_counts = {l: np.sum(label_arr == l) for l in label_list}

    order = np.argsort([l_counts[l] for l in label_list])
    name_set = set(names)
    print("names: %i" % (len(name_set),))
    print("label counts (%i):" % (len(label_list),))
    for ind in order:
        print("\t%s: %i" % (label_list[ind], l_counts[label_list[ind]]))
    if not print_names:
        return
    name_arr = np.array(names)
    name_list = sorted(list(name_set))
    n_counts = {n: np.sum(name_arr == n) for n in name_list}
    order = np.argsort([n_counts[n] for n in name_list])
    print("name_counts")
    for ind in order:
        print("\t%s: %i" % (name_list[ind], n_counts[name_list[ind]]))


def test_digits():
    _test(None, 'numeric')
    _test([0, 1], 'numeric')
    _test([0, 1, 2, 3, 4, 5], 'numeric')


def test_capital_letters():
    caps = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    _test(caps, 'alphanumeric')
    _test(None, 'alphanumeric')


# Example Usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_load_numeric()
    test_load_alphanumeric()
    test_digits()
    test_capital_letters()
    logging.info("All tests pass.")
