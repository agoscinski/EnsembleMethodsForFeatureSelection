import io_utils
import numpy as np
import pandas as pd
import shutil


class DataSets:
    root_dir = ".."
    data_sets = {
        'colon': (
            {
                "path": "/COLON/COLON/colon.data",
            },
            {
                "path": "/COLON/COLON/colon.labels",
                "apply_transform": np.sign
            }
        ),
        'arcene': (
            {
                "path": "/ARCENE/ARCENE/arcene.data",
                "apply_transform": np.transpose,
                "feat_labels": "/ARCENE/ARCENE/arcene_feat.labels"

            },
            {
                'path': "/ARCENE/ARCENE/arcene.labels",
            }
        ),
        'dexter': (
            {
                "feat_labels": "/DEXTER/DEXTER/dexter_feat.labels",
                "path": "/DEXTER/DEXTER/dexter.data",
                "method": "sparse_matrix",
                "args": [20000]
            },
            {
                "path": "/DEXTER/DEXTER/dexter.labels",
            }
        ),
        "dorothea": (
            {
                "feat_labels": "/DOROTHEA/DOROTHEA/dorothea_feat.labels",
                "path": "/DOROTHEA/DOROTHEA/dorothea.data",
                "method": "sparse_binary_matrix",
                "args": [100001],
                "apply_transform": lambda x: x[:, :150]
            },
            {
                "path": "/DOROTHEA/DOROTHEA/dorothea.labels",
                "apply_transform": lambda x: x[:150]
            }
        ),
        "gisette": (
            {
                "feat_labels": "/GISETTE/GISETTE/gisette_feat.labels",
                "path": "/GISETTE/GISETTE/gisette_valid.data",
                "apply_transform": lambda x: np.transpose(x)[:, :200],
            },
            {
                "path": "/GISETTE/GISETTE/gisette_valid.labels",
                "apply_transform": lambda x: x[:200]
            }
        ),
        "artificial": (
            {
                "feat_labels": "/ARTIFICIAL/ARTIFICIAL/artificial_feat.labels",
                "path": "/ARTIFICIAL/ARTIFICIAL/artificial.data.npy",
                "method": "numpy_matrix",
            },
            {
                "path": "/ARTIFICIAL/ARTIFICIAL/artificial.labels.npy",
                "method": "numpy_matrix",
            }
        )
    }

    @staticmethod
    def save_artificial(data, labels, feature_labels):
        PreComputedData.delete("artificial")

        artificial_data_dir = DataSets.root_dir + "/ARTIFICIAL/ARTIFICIAL"

        io_utils.mkdir(artificial_data_dir)

        data_file_name = artificial_data_dir + "/artificial.data"
        label_file_name = artificial_data_dir + "/artificial.labels"
        feature_label_file_name = artificial_data_dir + "/artificial_feat.labels"

        np.save(data_file_name, data)
        np.save(label_file_name, labels)
        np.savetxt(feature_label_file_name, feature_labels, fmt='%d')

    @staticmethod
    def load(data_set):
        data_info, labels_info = DataSets.data_sets[data_set]
        labels = DataSets.__load_data_set_file(labels_info)
        data = DataSets.__load_data_set_file(data_info)

        feature_labels = DataSets.load_features_labels(data_set)
        if feature_labels is not None:
            features = data[[feature_labels == 1]]
            probes = data[[feature_labels == -1]]
            data = np.vstack((features, probes))

        return data, labels

    @staticmethod
    def __load_data_set_file(info):
        data = getattr(io_utils, info.get('method', 'regular_matrix'))(
            DataSets.root_dir + info['path'],
            *info.get('args', []),
            **info.get('kwargs', {})
        )
        apply_transform = info.get('apply_transform', False)
        if apply_transform:
            return apply_transform(data)
        return data

    @staticmethod
    def load_features_labels(data_set):
        if data_set not in DataSets.data_sets:
            return None

        data_info, _ = DataSets.data_sets[data_set]
        feat_labels_filename = data_info.get('feat_labels', None)

        if feat_labels_filename is not None:
            return np.loadtxt(DataSets.root_dir + feat_labels_filename)

        return None


class PreComputedData:
    @staticmethod
    def load(data_set, cv, assessment_method, feature_selector):
        filename = PreComputedData.file_name(data_set, cv, assessment_method, feature_selector)
        try:
            return np.load(filename)
        except FileNotFoundError:
            print("File " + filename + " not found")
            raise

    @staticmethod
    def file_name(data_set, cv, assessment_method, feature_selector):
        return "{data_dir}/{feature_selector}.npy".format(
            data_dir=PreComputedData.dir_name(data_set, cv, assessment_method),
            feature_selector=feature_selector.__name__
        )

    @staticmethod
    def load_cv(data_set, cv):
        file_name = PreComputedData.cv_file_name(data_set, cv)
        try:
            return np.load(file_name)
        except FileNotFoundError:
            print("CV {} was never generated".format(type(cv).__name__))
            raise

    @staticmethod
    def delete(data_set):
        try:
            shutil.rmtree(PreComputedData.root_dir(data_set))
        except FileNotFoundError:
            pass

    @staticmethod
    def cv_file_name(data_set, cv):
        return PreComputedData.cv_dir(data_set, cv) + "/indices.npy"

    @staticmethod
    def dir_name(data_set, cv, assessment_method):
        return "{cv_dir}/{method}".format(
            cv_dir=PreComputedData.cv_dir(data_set, cv),
            method=assessment_method
        )

    @staticmethod
    def cv_dir(data_set, cv):
        return "{data_set_dir}/{cv}".format(
            data_set_dir=PreComputedData.root_dir(data_set),
            cv=type(cv).__name__
        )

    @staticmethod
    def root_dir(data_set):
        return "{root_dir}/pre_computed_data/{data_set}".format(
            root_dir=DataSets.root_dir,
            data_set=data_set
        )


class Analysis:
    @staticmethod
    def load_csv(data_set, cv, assessment_method, feature_method):
        filename = Analysis.file_name(data_set, cv, assessment_method, feature_method) + ".csv"
        try:
            stats = pd.read_csv(filename)
            return stats
        except FileNotFoundError:
            print("File " + filename + " not found")
            raise

    @staticmethod
    def file_name(data_set, cv, assessment_method, feature_method):
        return Analysis.dir_name(data_set, cv, assessment_method) + "/" + feature_method.__name__

    @staticmethod
    def dir_name(data_set, cv, method):
        return "{root_dir}/pre_computed_data/{data_set}/{cv}".format(
            root_dir=DataSets.root_dir,
            method=method,
            data_set=data_set,
            cv=type(cv).__name__
        )
