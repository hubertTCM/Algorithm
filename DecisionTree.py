from math import log2


class TreeNode:

    def __init__(self, label):
        self.__label__ = label
        self.__children__ = {}  # value: TreeNode
        self.__value__ = None
        
    def add_child(self, value, child):
        self.__children__[value] = child


class DecisionTree:

    def __init__(self, data_set, labels):
        self.__root__ = None    
        self.__create_tree__(data_set, labels)
        
    def __create_tree__(self, data_set, labels):
        if (len(data_set[0]) == 1):
            return None  # TODO: create tree node here
                
        class_list = [item[-1] for item in data_set]
        first_class = class_list[0]
        is_same_class = True
        for item in class_list:
            if (item != first_class):
                is_same_class = False
                break
        if (is_same_class):
            return None  # TODO:
        
        feature_index = self.__find_best_feature_index__(data_set)
        label = labels[feature_index]
        node = TreeNode(label)
        if (self.__root__ is None):
            self.__root__ = node            
        
        sub_labels = labels[: feature_index]
        sub_labels.extend(labels[feature_index + 1 : ])    
        unique_feature_values = set([feature[feature_index] for feature in data_set])
        for feature_value in unique_feature_values:
            sub_tree = self.__filter_data_set__(data_set, feature_index, feature_value)
            child = self.__create_tree__(sub_tree, sub_labels)
            node.add_child(feature_value, child)       
        
    def __filter_data_set__(self, data_set, feature_index, value):
        result = []
        for feature in data_set:
            if (feature[feature_index] != value):
                continue
            temp = feature[:feature_index]
            temp.extend(feature[feature_index + 1 : ])
            result.append(temp)
        return result
    
    def __find_best_feature_index__(self, data_set):
        feature_count = len(data_set[0]) - 1
        best_feature_index = 0
        best_feature_gain = -1
        for i in range(feature_count):
            gain = self.__calculate_gain__(data_set, i)
            if (gain < 0):
                continue
            if (best_feature_gain < 0 or gain < best_feature_gain):
                best_feature_gain = gain
                best_feature_index = i
            
        return best_feature_index
    
    def __calculate_entropy__(self, data_set):
        # class_list = [item[-1] for item in data_set]
        class_count = {}
        for feature in data_set:
            class_value = feature[-1]
            if class_value in class_count:
                class_count[class_value] += 1
            else:
                class_count[class_value] = 1
        
        entropy = 0
        entity_number = len(data_set) * 1.0
        for key, value in class_count:
            possibility = value * 1.0 / entity_number
            entropy += -1 * possibility * log2(possibility)
            
        return entropy
    
    def __calculate_gain__(self, data_set, feature_index):
        unique_feature_values = set([feature[feature_index] for feature in data_set])
        total_entropy = self.__calculate_entropy__(data_set)
        sub_entropy = 0
        conditional_entropy = 0
        for feature_value in unique_feature_values:
            sub_data_set = self.__filter_data_set__(data_set, feature_index, feature_value)
            temp_sub_entropy = self.__calculate_entropy__(sub_data_set)
            sub_entropy += temp_sub_entropy
            conditional_entropy += len(sub_data_set) * 1.0 / len(data_set) * temp_sub_entropy
        gain = self.__calculate_info_gain_ratio_core__(total_entropy, sub_entropy, conditional_entropy)
        return gain
    
    def __calculate_info_gain_core__(self, total_entropy, sub_entropy, conditional_entropy):
        return total_entropy - conditional_entropy
    
    def __calculate_info_gain_ratio_core__(self, total_entropy, sub_entropy, conditional_entropy):
        return self.__calculate_entropy__(total_entropy, sub_entropy, conditional_entropy) / sub_entropy


if __name__ == "__main__":
    print("done")