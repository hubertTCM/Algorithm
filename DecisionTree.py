from math import log
from collections import OrderedDict
import json


class TreeNode(object):

    def __init__(self, label=None, value=None):
        self.__label__ = label
        self.__children__ = {}  # value: TreeNode
        self.__value__ = value
        
    def add_child(self, value, child):
        self.__children__[value] = child
        
    def get_export_values(self):
        
        if (len(self.__children__) == 0):
            return {'value': self.__value__}
        
        result = {'label': self.__label__}
        child_property = []
        for value, child in self.__children__.items():
            child_property.append((value, child.get_export_values()))
        result['children'] = child_property
        return result


class DecisionTree:

    def __init__(self, data_set, labels):
        self.__root__ = None    
        self.__root__ = self.__create_tree__(data_set, labels)
        
    @property
    def root(self):
        return self.__root__
        
    def __create_tree__(self, data_set, labels):
        class_list = [item[-1] for item in data_set]
                
        sorted_class_count = OrderedDict({})
        for item in class_list:
            if (item not in sorted_class_count):
                sorted_class_count[item] = 1
            else:
                sorted_class_count[item] += 1
        
        if (len(sorted_class_count) == 1 or len(data_set[0]) == 1):
            first_class = next(reversed(sorted_class_count))
            return TreeNode(None, first_class)
        
        feature_index = self.__find_best_feature_index__(data_set)
        label = labels[feature_index]
        node = TreeNode(label, None)        
        
        sub_labels = labels[: feature_index]
        sub_labels.extend(labels[feature_index + 1 : ])    
        unique_feature_values = set([feature[feature_index] for feature in data_set])
        for feature_value in unique_feature_values:
            sub_tree = self.__filter_data_set__(data_set, feature_index, feature_value)
            child = self.__create_tree__(sub_tree, sub_labels)
            node.add_child(feature_value, child)       
        return node
        
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
            # print("gain:" + str(gain) + " i:" + str(i))
            if (gain < 0):
                continue
            if (best_feature_gain < 0 or gain > best_feature_gain):
                best_feature_gain = gain
                best_feature_index = i
            
        # print("best_feature_index:" + str(best_feature_index) + " len(data_set):" + str(len(data_set)) + " best_feature_gain:" + str(best_feature_gain))
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
        for key, value in class_count.items():
            # possibility = value * 1.0 / entity_number
            entropy += self.__calculate_entropy_item__(value, entity_number)  # -1 * possibility * log2(possibility)            
        return entropy
    
    def __calculate_float__(self, denominator, numerator):
        return float(denominator) / float(numerator)
    
    def __calculate_entropy_item__(self, denominator, numerator):
        possibility = self.__calculate_float__(denominator, numerator)
        return -1 * possibility * log(possibility, 2)
    
    def __calculate_gain__(self, data_set, feature_index):
        unique_feature_values = set([feature[feature_index] for feature in data_set])
        total_entropy = self.__calculate_entropy__(data_set)
        iv = 0
        conditional_entropy = 0
        total_items = len(data_set)
        for feature_value in unique_feature_values:
            sub_data_set = self.__filter_data_set__(data_set, feature_index, feature_value)
            temp_sub_entropy = self.__calculate_entropy__(sub_data_set)
            
            sub_items_count = len(sub_data_set)
            temp_float = self.__calculate_float__(sub_items_count, total_items)
            # print(feature_value + " temp_sub_entropy:" + str(temp_sub_entropy) + " sub_items_count:" + str(sub_items_count) + " total_items:" + str(total_items) + " temp_float:" + str(temp_float))
            conditional_entropy += temp_float * temp_sub_entropy
            iv += self.__calculate_entropy_item__(sub_items_count, total_items)
        
        #gain = self.__calculate_info_gain_ratio_core__(total_entropy, iv, conditional_entropy)
        gain = self.__calculate_info_gain_core__(total_entropy, iv, conditional_entropy)
        return gain
    
    def __calculate_info_gain_core__(self, total_entropy, sub_entropy, conditional_entropy):        
        # print("total_entropy:" + str(total_entropy) + " sub_entropy:" + str(sub_entropy) + " conditional_entropy:" + str(conditional_entropy))
        result = total_entropy - conditional_entropy 
        # print("result:" + str(result))
        return result
    
    def __calculate_info_gain_ratio_core__(self, total_entropy, iv, conditional_entropy):
        return self.__calculate_info_gain_core__(total_entropy, iv, conditional_entropy) / iv


if __name__ == "__main__":
    f = open('resource/decisiontree.txt')
    data_set = [line.strip().split('\t') for line in f.readlines()]
    labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    tree = DecisionTree(data_set, labels)
    root = tree.root
    print(json.dumps(root.get_export_values()))
    print("done")
