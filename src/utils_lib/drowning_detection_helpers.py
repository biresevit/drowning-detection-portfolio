from classes_enum import ClassesEnum


def contains_drowning_prediction(info_predictions, conf_high_thres):
    for class_name, conf, _ in info_predictions:
        if class_name == ClassesEnum.drowning and conf >= conf_high_thres:
            return True
    return False


def is_drowning_detected(
    predicted_class, confidence, conf_high_thres, info_predictions
):
    if predicted_class == ClassesEnum.drowning and confidence >= conf_high_thres:
        return True
    return False