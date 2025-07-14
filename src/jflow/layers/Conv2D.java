package jflow.layers;

import java.util.stream.IntStream;

import jflow.data.JMatrix;
import jflow.layers.templates.TrainableLayer;

import java.util.concurrent.ThreadLocalRandom;

public class Conv2D extends TrainableLayer {
    private JMatrix filters;
    private JMatrix dFilters;
    private JMatrix lastInput;
    private JMatrix biases;
    private JMatrix dBiases;

    private int numFilters;
    private int filterSize;
    private int stride;
    private int numChannels;
    private int inputHeight;
    private int inputWidth;
    private int numImages;

    private String padding;
    // Hyperparameters for gradient clipping
    final double epsilon = 1e-8;         // Small constant for numerical stability
    final double clipThreshold = 5.0;   // Global gradient clipping threshold

    /**
     * Represents a convolutional layer.
     * 
     * <p><b>Do not instantiate directly.</b> Use the static builder method:
     * {@code import static jflow.model.builder.*;}
     * and call {@code Conv2D(...)} instead of {@code new Conv2D(...)}.
     */
    public Conv2D(int numFilters, int filterSize, int stride, String padding) {
        super("conv_2d");
        this.numFilters = numFilters;
        this.filterSize = filterSize;
        this.stride = stride;
        this.padding = padding;
        if (!(padding.equals("same_padding") || padding.equals("valid_padding"))) {
            throw new IllegalArgumentException("Only same_padding and valid_padding allowed.");
        }
    }

    /**
     * Represents a convolutional layer.
     * 
     * <p><b>Do not instantiate directly.</b> Use the static builder method:
     * {@code import static jflow.model.builder.*;}
     * and call {@code Conv2D(...)} instead of {@code new Conv2D(...)}.
     */
    public Conv2D(int numFilters, int filterSize, int stride, String padding, int[] inputShape) {
        this(numFilters, filterSize, stride, padding);

        if (inputShape.length != 3) {
            throw new IllegalArgumentException(
                "Conv2D input shape should have 3 dimensions. Got: "
                + inputShape.length + "."
            );
        }
        setInputShape(inputShape);
    }

    @Override
    protected void build(int IDnum) {
        super.build(IDnum);
        int[] shape = getInputShape();
        if (shape == null) {
            throw new IllegalStateException(
                    "In " + this.getClass().getSimpleName() + 
                    ": Cannot build the first layer without an input shape."
                );
        } else {
            this.numChannels = shape[1];
        }
        setNumTrainableParameters(numFilters * numChannels * filterSize * filterSize + numFilters);

        // He initialization
        double stdDev = Math.sqrt(2.0 / (numChannels * filterSize * filterSize));

        double filterScale = 1.0;

        int filterSizeTotal = numFilters * numChannels * filterSize * filterSize;
        float[] filters = new float[filterSizeTotal];

        IntStream.range(0, filterSizeTotal).parallel().forEach(i -> {
            filters[i] = (float)(ThreadLocalRandom.current().nextGaussian() * stdDev * filterScale);
        });

        this.filters = JMatrix.wrap(filters, numFilters, numChannels, filterSize, filterSize).label("filters");

        // Initialize biases to 0
        biases = JMatrix.zeros(numFilters, 1, 1, 1).label("biases");

        dFilters = JMatrix.zeros(numFilters, numChannels, filterSize, filterSize).label("dFilters");
        dBiases = JMatrix.zeros(numFilters, 1, 1, 1).label("dBiases");
    }

    @Override
    public JMatrix forward(JMatrix input, boolean training) {
        if (training) {
            lastInput = input;
        }
        this.inputHeight = input.height();
        this.inputWidth = input.width();
        this.numImages = input.length();
    
        // Calculate output dimensions based on padding and stride
        int outputHeight, outputWidth;
        if (padding.equals("same_padding")) {
            outputHeight = (int)Math.ceil((double)inputHeight / stride);
            outputWidth = (int)Math.ceil((double)inputWidth / stride);
        } else { // valid padding
            outputHeight = (inputHeight - filterSize) / stride + 1;
            outputWidth = (inputWidth - filterSize) / stride + 1;
        }
    
        // Initialize output matrix with proper dimensions
        JMatrix A = JMatrix.zeros(numImages, numFilters, outputHeight, outputWidth);
        
        // Calculate forward output
        if (numImages <= Runtime.getRuntime().availableProcessors() / 2) {
            // For each image in the batch
            for (int imageIndex = 0; imageIndex < numImages; imageIndex++) {
                final int imgIdx = imageIndex;
                int startIdx = imgIdx * numChannels * inputHeight * inputWidth;
                
                // Parallelize across filters
                IntStream.range(0, numFilters).parallel().forEach(filterIndex -> {
                    int outputIdx = (imgIdx * numFilters + filterIndex) * outputHeight * outputWidth;
                    convolveWithKernel(A.unwrap(), outputIdx, input.unwrap(), startIdx,
                            filters.unwrap(), filterIndex, biases.get(filterIndex), padding);
                });
            }
        } else {
            // Parallelize across batch for larger batch sizes
            IntStream.range(0, numImages).parallel().forEach(imageIndex -> {
                for (int filterIndex = 0; filterIndex < numFilters; filterIndex++) {
                    int startIdx = imageIndex * numChannels * inputHeight * inputWidth;
                    int outputIdx = (imageIndex * numFilters + filterIndex) * outputHeight * outputWidth;
                    convolveWithKernel(A.unwrap(), outputIdx, input.unwrap(), startIdx,
                            filters.unwrap(), filterIndex, biases.get(filterIndex), padding);
                }
            });
        }
       
        return trackOutput(A, training);
    }

    @Override
    public JMatrix backward(JMatrix input) {
        // Calculate output dimensions
        int outputHeight, outputWidth;
        if (padding.equals("same_padding")) {
            outputHeight = (int)Math.ceil((double)inputHeight / stride);
            outputWidth = (int)Math.ceil((double)inputWidth / stride);
        } else { // valid padding
            outputHeight = (inputHeight - filterSize) / stride + 1;
            outputWidth = (inputWidth - filterSize) / stride + 1;
        }
        
        // Initialize dX with proper dimensions
        JMatrix dX = JMatrix.zeros(numImages, numChannels, inputHeight, inputWidth);
        
        // Calculate padding for same_padding
        int padTop; int padLeft;
        if (padding.equals("same_padding")) {
            int padTotal_h = Math.max(0, (outputHeight - 1) * stride + filterSize - inputHeight);
            int padTotal_w = Math.max(0, (outputWidth - 1) * stride + filterSize - inputWidth);
            padTop = padTotal_h / 2;
            padLeft = padTotal_w / 2;
        } else {
            padTop = 0;
            padLeft = 0;
        }
        
        // Calculate bias gradients - parallelize across filters
        IntStream.range(0, numFilters).parallel().forEach(k -> {
            float biasGrad = 0;
            for (int i = 0; i < numImages; i++) {
                for (int oh = 0; oh < outputHeight; oh++) {
                    for (int ow = 0; ow < outputWidth; ow++) {
                        int dZIdx = (i * numFilters + k) * outputHeight * outputWidth + (oh * outputWidth + ow);
                        biasGrad += input.get(dZIdx);
                    }
                }
            }
            dBiases.set(k, biasGrad);
        });
        
        // Calculate filter gradients - parallelize across filters
        IntStream.range(0, numFilters).parallel().forEach(k -> {
            for (int c = 0; c < numChannels; c++) {
                for (int fh = 0; fh < filterSize; fh++) {
                    for (int fw = 0; fw < filterSize; fw++) {
                        float filterGrad = 0;
                        
                        // Sum over all images and output positions
                        for (int i = 0; i < numImages; i++) {
                            for (int oh = 0; oh < outputHeight; oh++) {
                                for (int ow = 0; ow < outputWidth; ow++) {
                                    // Calculate corresponding input position
                                    int ih, iw;
                                    if (padding.equals("same_padding")) {
                                        ih = oh * stride + fh - padTop;
                                        iw = ow * stride + fw - padLeft;
                                    } else {
                                        ih = oh * stride + fh;
                                        iw = ow * stride + fw;
                                    }
                                    
                                    // Check if input position is valid
                                    if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                                        int inputIdx = (i * numChannels + c) * inputHeight * inputWidth + (ih * inputWidth + iw);
                                        int dZIdx = (i * numFilters + k) * outputHeight * outputWidth + (oh * outputWidth + ow);
                                        
                                        filterGrad += lastInput.get(inputIdx) * input.get(dZIdx);
                                    }
                                }
                            }
                        }
                        
                        // Use consistent indexing with forward pass
                        int filterIdx = ((k * numChannels + c) * filterSize + fh) * filterSize + fw;
                        dFilters.set(filterIdx, filterGrad);
                    }
                }
            }
        });
        
        // Calculate input gradients (dX) - parallelize with tiling for memory efficiency
        final int TILE_SIZE = 32;
        int numTilesH = (inputHeight + TILE_SIZE - 1) / TILE_SIZE;
        int numTilesW = (inputWidth + TILE_SIZE - 1) / TILE_SIZE;
        
        // Create parallel tasks for each image+channel+tile combination
        IntStream.range(0, numImages * numChannels * numTilesH * numTilesW).parallel().forEach(taskIdx -> {
            // Decode task index
            int remaining = taskIdx;
            int i = remaining / (numChannels * numTilesH * numTilesW);
            remaining %= (numChannels * numTilesH * numTilesW);
            int c = remaining / (numTilesH * numTilesW);
            remaining %= (numTilesH * numTilesW);
            int tileH = remaining / numTilesW;
            int tileW = remaining % numTilesW;
            
            // Calculate tile boundaries
            int ih_start = tileH * TILE_SIZE;
            int ih_end = Math.min(ih_start + TILE_SIZE, inputHeight);
            int iw_start = tileW * TILE_SIZE;
            int iw_end = Math.min(iw_start + TILE_SIZE, inputWidth);
            
            // Process this tile
            for (int ih = ih_start; ih < ih_end; ih++) {
                for (int iw = iw_start; iw < iw_end; iw++) {
                    float inputGrad = 0;
                    
                    // Sum contributions from all filters
                    for (int k = 0; k < numFilters; k++) {
                        // For each filter position
                        for (int fh = 0; fh < filterSize; fh++) {
                            for (int fw = 0; fw < filterSize; fw++) {
                                // Calculate which output position this input position contributes to
                                // when convolved with this filter element
                                int oh_numerator, ow_numerator;
                                if (padding.equals("same_padding")) {
                                    oh_numerator = ih + padTop - fh;
                                    ow_numerator = iw + padLeft - fw;
                                } else {
                                    oh_numerator = ih - fh;
                                    ow_numerator = iw - fw;
                                }
                                
                                // Check if this creates a valid output position with the given stride
                                if (oh_numerator >= 0 && ow_numerator >= 0 && 
                                    oh_numerator % stride == 0 && ow_numerator % stride == 0) {
                                    
                                    int oh = oh_numerator / stride;
                                    int ow = ow_numerator / stride;
                                    
                                    // Check if output position is valid
                                    if (oh < outputHeight && ow < outputWidth) {
                                        int filterIdx = ((k * numChannels + c) * filterSize + fh) * filterSize + fw;
                                        int dZIdx = (i * numFilters + k) * outputHeight * outputWidth + (oh * outputWidth + ow);
                                        
                                        inputGrad += filters.get(filterIdx) * input.get(dZIdx);
                                    }
                                }
                            }
                        }
                    }
                    
                    int dXIdx = (i * numChannels + c) * inputHeight * inputWidth + (ih * inputWidth + iw);
                    dX.set(dXIdx, inputGrad);
                }
            }
        });
        
        // Apply gradient clipping
        adaptiveScale(dFilters, dBiases, dX);
        
        return trackGradient(dX);
    }
    
    // Apply convolution to one image at a time
    private void convolveWithKernel(float[] output, int outIdx, float[] input, int inIdx, 
                               float[] kernel, int filterIdx, float bias, String padding) {
        // Calculate padding and output dimensions
        int outputHeight, outputWidth;
        int padTop = 0, padBottom = 0, padLeft = 0, padRight = 0;
        
        if (padding.equals("same_padding")) {
            int padTotal_h = Math.max(0, (inputHeight - 1) * stride + filterSize - inputHeight);
            int padTotal_w = Math.max(0, (inputWidth - 1) * stride + filterSize - inputWidth);
            
            padTop = padTotal_h / 2;
            padBottom = padTotal_h - padTop;
            padLeft = padTotal_w / 2;
            padRight = padTotal_w - padLeft;
            
            outputHeight = (int)Math.ceil((double)inputHeight / stride);
            outputWidth = (int)Math.ceil((double)inputWidth / stride);
        } else { // valid padding
            outputHeight = (inputHeight - filterSize) / stride + 1;
            outputWidth = (inputWidth - filterSize) / stride + 1;
        }
        
        // Process each position in the output feature map
        for (int oh = 0; oh < outputHeight; oh++) {
            for (int ow = 0; ow < outputWidth; ow++) {
                float sum = bias;
                
                // For each input channel
                for (int c = 0; c < numChannels; c++) {
                    int inputChannelOffset = inIdx + (c * inputHeight * inputWidth);
                    int filterChannelOffset = (filterIdx * numChannels + c) * filterSize * filterSize;
                    
                    // For each element in the filter
                    for (int fh = 0; fh < filterSize; fh++) {
                        for (int fw = 0; fw < filterSize; fw++) {
                            // Calculate input position
                            int ih, iw;
                            
                            if (padding.equals("same_padding")) {
                                ih = oh * stride + fh - padTop;
                                iw = ow * stride + fw - padLeft;
                            } else { // valid padding
                                ih = oh * stride + fh;
                                iw = ow * stride + fw;
                            }
                            
                            // Check bounds and accumulate weighted value if within bounds
                            if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                                int inputPos = inputChannelOffset + (ih * inputWidth + iw);
                                int filterPos = filterChannelOffset + (fh * filterSize + fw);
                                
                                sum += input[inputPos] * kernel[filterPos];
                            }
                        }
                    }
                }
                
                // Store result in output
                output[outIdx + (oh * outputWidth + ow)] = sum;
            }
        }
    }

    @Override
    public void updateParameters(JMatrix[] parameterUpdates) {
        filters.subtractInPlace(parameterUpdates[0]);
        biases.subtractInPlace(parameterUpdates[1]);
    }

    private void adaptiveScale(JMatrix dFilters, JMatrix dBiases, JMatrix dX) {
        final double FILTER_CLIP_THRESHOLD = 1.0;  
        final double BIAS_CLIP_THRESHOLD = 1.0;   
        final double INPUT_CLIP_THRESHOLD = 5.0;  
        
        // Simple absolute clipping
        clipGradients(dFilters, FILTER_CLIP_THRESHOLD);
        clipGradients(dBiases, BIAS_CLIP_THRESHOLD);
        clipGradients(dX, INPUT_CLIP_THRESHOLD);
    }
    
    private void clipGradients(JMatrix gradients, double threshold) {
        double norm = gradients.l2Norm();
        if (norm > threshold) {
            double scale = threshold / norm;
            gradients.multiplyInPlace(scale);
        }
    }
    
    
    @Override
    public JMatrix[] getParameters() {
        return new JMatrix[]{filters, biases};
    }

    @Override
    public JMatrix[] getParameterGradients() {
        return new JMatrix[]{dFilters, dBiases};
    }


    @Override
    public int[] outputShape() {
        int[] outputShape;
        if (getOutput() != null) {
            outputShape = getOutput().shape();
        } else {
            int[] inputShape = getInputShape();
            int[] prevShape = new int[]{1, inputShape[1], inputShape[2], inputShape[3]};
            
            if (padding.equals("same_padding")) {
                outputShape = new int[]{
                    prevShape[0],
                    numFilters,
                    (int)Math.ceil((double)prevShape[2] / stride),
                    (int)Math.ceil((double)prevShape[3] / stride)
                };
            } else { // valid padding
                outputShape = new int[]{
                    prevShape[0],
                    numFilters,
                    (prevShape[2] - filterSize) / stride + 1,
                    (prevShape[3] - filterSize) / stride + 1
                };
            }
        }
        return outputShape;
    }
   
}