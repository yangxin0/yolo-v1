
class YOLOTrainConfig:
    Epochs = 135
    BatchSize = 64
    WarnupEpochs = 0
    LearningRate = 1e-4
    DataPath = "./data"
    LabelPath = "./data/labels.json"


class YOLOv1ModelConfig:
    S = 7 # Divide each image into a SxS gride
    B = 2 # Number of bounding boxes to predict
    C = 20 # Number of classes in the dataset
    Epsilon = 1e-6
    ImageSize = (448, 448)
    PretrainImageSize = (224, 224)
