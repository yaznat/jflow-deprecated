package jflow.model;

import java.util.function.BiFunction;
import java.util.function.Function;

import jflow.data.JMatrix;
import jflow.layers.*;

/**
 * Import this class statically for access to JFlow Layers, Optimizers, and other components.
 */
public class Builder {

    /**
     * A flat input shape for a Sequential layer.
     * @param flattenedSize         The 1D input shape.
     */
    public static InputShape InputShape(int flattenedSize) {
        return new InputShape(flattenedSize);
    }
    /**
     * A 3D input shape for a Sequential layer.
     * @param channels              The channel dimension.
     * @param height                The input height.
     * @param width                 The input width.
     */
    public static InputShape InputShape(int channels, int height, int width) {
        return new InputShape(channels, height, width);
    }

    /**
     * A Dense layer with bias.
     * @param size                  The output size of the Dense layer.
     * @param input                 The 1D input shape to the Dense layer.
     */
    public static Dense Dense(int size, InputShape input) {
        return new Dense(size, input.getShape(), true);
    }

    /**
     * A Dense layer.
     * @param size                  The output size of the Dense layer.
     * @param input                 The 1D input shape to the Dense layer.
     * @param useBias               Whether or not to use biases.
     */
    public static Dense Dense(int size, InputShape input, boolean useBias) {
        return new Dense(size, input.getShape(), useBias);
    }
    /**
     * A Dense layer with bias.
     * @param size                  The output size of the Dense layer.
     */
    public static Dense Dense(int size) {
        return new Dense(size, null, true);
    }

    /**
     * A Dense layer.
     * @param size                  The output size of the Dense layer.
     * @param useBias               Whether or not to use biases.
     */
    public static Dense Dense(int size, boolean useBias) {
        return new Dense(size, null, useBias);
    }
    /**
     * A Conv2D layer.
     * @param numFilters            The number of filters in the Conv2D layer.
     * @param filterSize            The square size of each filter. Currently only square filters are supported.
     * @param padding               The type of padding to use. Current options: <p> 
     *                                  - same_padding - valid_padding
     * @param stride                The stride to apply in convolution.
     */
    public static Conv2D Conv2D(int numFilters, int filterSize, int stride, String padding) {
        return new Conv2D(numFilters, filterSize, stride, padding);
    }
        /**
     * A Conv2D layer.
     * @param numFilters            The number of filters in the Conv2D layer.
     * @param filterSize            The square size of each filter. Currently only square filters are supported.
     * @param padding               The type of padding to use. Current options: <p> 
     *                                  - same_padding - valid_padding
     * @param stride                The stride to apply in convolution.
     * @param InputShape            The 3D input shape to the Conv2D layer.
     */
    public static Conv2D Conv2D(int numFilters, int filterSize, int stride, String padding, InputShape input) {
        return new Conv2D(numFilters, filterSize, stride, padding, input.getShape());
    }
    /**
     * A MaxPool2D layer.
     * @param poolSize              The pool size to use in max pooling.
     * @param stride                The stride to use in max pooling.
     */
    public static MaxPool2D MaxPool2D(int poolSize, int stride) {
        return new MaxPool2D(poolSize, stride);
    }
    /**
     * Averages spatial dimensions along every channel.
     */
    public static Layer GlobalAveragePooling2D() {
        return new GlobalAveragePooling2D();
    }

    /**
     * A flatten layer. Flattens 4D data.
     */
    public static Flatten Flatten() {
        return new Flatten();
    }
    /**
     * The ReLU activation.
     */
    public static ReLU ReLU() {
        return new ReLU();
    }
    /**
     * The LeakyRelu activation.
     * @param alpha             The slope to use in the ReLU function.
     */
    public static Layer LeakyReLU(double alpha) {
        return new LeakyReLU(alpha);
    }

    /**
     * The Softmax activation.
     */
    public static Softmax Softmax() {
        return new Softmax();
    }

    /**
     * The Swish activation.
     */
    public static Swish Swish() {
        return new Swish();
    }

    /**
     * The Mish activation.
     */
    public static Mish Mish() {
        return new Mish();
    }

    /**
     * The GELU activation.
     */
    public static GELU GELU() {
        return new GELU();
    }

    /**
     * The Dropout function. <p>
     * @param alpha             The percent of nuerons to drop.
     * 
     */
    public static Dropout Dropout(double alpha) {
        return new Dropout(alpha);
    }

    /**
     * The Sigmoid activation.
     */
    public static Sigmoid Sigmoid() {
        return new Sigmoid();
    }

    /**
     * Reshapes data in forward() and return it to its original shape in backward().
     * @param channels              The channel dimension to reshape to.
     * @param height                The height dimension to reshape to.
     * @param width                 The width dimension to reshape to.
     */
    public static Reshape Reshape(int channels, int height, int width) {
        return new Reshape(channels, height, width);
    }

    /**
     * Reshapes data in forward() and return it to its original shape in backward().
     * @param length                The length dimension to reshape to.
     * @param channels              The channel dimension to reshape to.
     * @param height                The height dimension to reshape to.
     * @param width                 The width dimension to reshape to.
     */
    public static Reshape Reshape(int length, int channels, int height, int width) {
        return new Reshape(length, channels, height, width);
    }

    /**
     * The Upsampling2D Layer.
     * @param scaleFactor            The factor to upsample by.
     */
    public static Upsampling2D Upsampling2D(int scaleFactor) {
        return new Upsampling2D(scaleFactor);
    }

    /**
     * The BatchNorm layer.
     */
    public static BatchNorm BatchNorm() {
        return new BatchNorm();
    }

    /**
     * The LayerNorm layer.
     */
    public static LayerNorm LayerNorm() {
        return new LayerNorm();
    }

    /**
     * The Embedding layer.
     * @param vocabSize             The number of tokens.
     * @param embedDim              The size of embedding vectors.
     * @return
     */
    public static Embedding Embedding(int vocabSize, int embedDim) {
        return new Embedding(vocabSize, embedDim);
    }

    /**
     * The Embedding layer.
     * @param vocabSize             The number of tokens.
     * @param embedDim              The size of embedding vectors.
     * @param inputShape            The seqLength of the Embedding layer.
     * @return
     */
    public static Embedding Embedding(int vocabSize, int embedDim, InputShape inputShape) {
        return new Embedding(vocabSize, embedDim, inputShape.getShape());
    }
    /**
     * Add a custom activation function to the model.
     * @param activation                A function that applies the activation in forward propagation.
     * @param dActivation               A function that applies the derivative of the activation 
     *                                  in backward propagation, given: <p>
     *                                      - The post activation output (Z). <p>
     *                                      - The gradient passed back to the layer (dX_prev).
     * @param name                      The name of the activation.
     */
    public static Layer customActivation(Function<JMatrix, JMatrix> activation, 
        BiFunction<JMatrix, JMatrix, JMatrix> dActivation, String name) {
        return new CustomActivation(activation, dActivation, name);
    }

    /**
     * The Tanh activation.
     */
    public static Tanh Tanh() {
        return new Tanh();
    }


    /**
     * The Adam Optimizer.
     * @param learningRate              The learning rate for parameter updates.
     * @param beta1                     The momentum coefficient of the first moment.
     * @param beta2                     The momentum coefficient of the second moment.
     */
    public static Optimizer Adam(double learningRate, double beta1, double beta2) {
        return new Adam(learningRate, beta1, beta2);
    }

    /**
     * The Adam Optimizer.
     * @param learningRate              The learning rate for parameter updates.
     */
    public static Optimizer Adam(double learningRate) {
        return new Adam(learningRate);
    }

    /**
     * The SGD optimizer without momentum.
     * 
     * @param learningRate The learning rate for parameter updates.
     */
    public static Optimizer SGD(double learningRate) {
        return new SGD(learningRate);
    }

    /**
     * The SGD optimizer with momentum.
     * 
     * @param learningRate The learning rate for parameter updates.
     * @param momentum The momentum coefficient.
     * @param useNesterov Whether to use Nesterov accelerated gradient.
     */
    public static Optimizer SGD(double learningRate, double momentum, boolean useNesterov) {
        return new SGD(learningRate, momentum, useNesterov);
    }


    /**
     * The RMSprop optimizer with default parameters and without momentum.
     * 
     * @param learningRate The learning rate for parameter updates.
     */
    public static Optimizer RMSprop(double learningRate) {
        return new RMSprop(learningRate);
    }

    /**
     * The RMSprop optimizer with custom parameters and without momentum.
     * 
     * @param learningRate                  The learning rate for parameter updates.
     * @param decay                         The decay rate for running average of squared gradients.
     * @param epsilon                       Small constant for numerical stability.
     */
    public static Optimizer RMSprop(double learningRate, double decay, double epsilon) {
        return new RMSprop(learningRate, decay, epsilon);
    }

    /**
     * The RMSprop optimizer with custom parameters and momentum.
     * 
     * @param learningRate                  The learning rate for parameter updates.
     * @param decay                         The decay rate for running average of squared gradients.
     * @param epsilon                       Small constant for numerical stability.
     * @param momentum                      The momentum coefficient.
     */
    public static Optimizer RMSprop(double learningRate, double decay, double epsilon, double momentum) {
        return new RMSprop(learningRate, decay, epsilon, momentum);
    }

    /**
     * The Adagrad optimizer.
     * 
     * @param learningRate The base learning rate for parameter updates.
     */
    public static Optimizer AdaGrad(double learningRate) {
        return new AdaGrad(learningRate);
    }

    /**
     * The Adagrad optimizer with custom epsilon.
     * @param learningRate                  The base learning rate for parameter updates. 
     * @param epsilon                       Small constant for numerical stability. 
     */
    public static Optimizer AdaGrad(double learningRate, double epsilon) {
        return new AdaGrad(learningRate, epsilon);
    }

    /**
     * Passes data to the train function to faciliate the saving of model checkpoints.
     * @param metric                            the metric to track for improvement. Supported: 
     *                                            <ul> <li> val_accuracy <li> val_loss 
     *                                                 <li> train_accuracy <li> train_loss </ul>
     * @param savePath                          the path to save checkpoints to.
     */
    public static ModelCheckpoint ModelCheckpoint(String metric, String savePath) {
        return new ModelCheckpoint(metric, savePath);
    }
}
