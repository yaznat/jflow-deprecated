package jflow.utils;

public class LayerManifest {
    public static final String[] SUPPORTED_LAYERS = {
        "BatchNorm", "Conv2D", "Dense", "Dropout", "Embedding", "Flatten",
        "GELU", "GlobalAveragePooling2D", "LayerNorm", "LeakyReLU", 
        "MaxPool2D", "Mish", "ReLU", "Reshape", "Sigmoid", "Softmax", "Swish",
        "Tanh", "Upsampling2D"
    };

    public static boolean isSupported(String layerClassName) {
        for (String layer : SUPPORTED_LAYERS) {
            if (layer.equals(layerClassName)) {
                return true;
            }
        }
        return false;
    }
}
