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
    public void build(int IDnum) {
        super.build(IDnum);
        if (internalGetInputShape() != null) {
            this.numChannels = internalGetInputShape()[0];
        } else {
            if (getPreviousLayer() == null) {
                throw new IllegalStateException(
                    "In " + this.getClass().getSimpleName() + 
                    ": Cannot build the first layer without an input shape."
                );
            }
            this.numChannels = getPreviousLayer().outputShape()[1];
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

        this.filters = JMatrix.wrap(filters, numFilters, numChannels, filterSize, filterSize).setName("filters");

        // Initialize biases to 0
        biases = JMatrix.zeros(numFilters, 1, 1, 1).setName("biases");

        dFilters = JMatrix.zeros(numFilters, numChannels, filterSize, filterSize).setName("dFilters");
        dBiases = JMatrix.zeros(numFilters, 1, 1, 1).setName("dBiases");
        
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
                    convolveWithKernel(A.getMatrix(), outputIdx, input.getMatrix(), startIdx,
                            filters.getMatrix(), filterIndex, biases.get(filterIndex), padding);
                });
            }
        } else {
            // Parallelize across batch for larger batch sizes
            IntStream.range(0, numImages).parallel().forEach(imageIndex -> {
                for (int filterIndex = 0; filterIndex < numFilters; filterIndex++) {
                    int startIdx = imageIndex * numChannels * inputHeight * inputWidth;
                    int outputIdx = (imageIndex * numFilters + filterIndex) * outputHeight * outputWidth;
                    convolveWithKernel(A.getMatrix(), outputIdx, input.getMatrix(), startIdx,
                            filters.getMatrix(), filterIndex, biases.get(filterIndex), padding);
                }
            });
        }
       
        return trackOutput(A, training);
    }

    @Override
    public JMatrix backward(JMatrix input) {
        // Calculate output dimensions based on padding and stride
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
        
        // Pre-calculate padding for same_padding
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
        
        // Calculate gradients in batch
        
        // For each filter - Calculate bias and filter gradients
        IntStream.range(0, numFilters).parallel().forEach(k -> {
            // Calculate bias gradients
            float biasGrad = 0;
            for (int i = 0; i < numImages; i++) {
                int dZFilterOffset = (i * numFilters + k) * outputHeight * outputWidth;
                for (int oh = 0; oh < outputHeight; oh++) {
                    for (int ow = 0; ow < outputWidth; ow++) {
                        biasGrad += input.get(dZFilterOffset + oh * outputWidth + ow);
                    }
                }
            }
            dBiases.set(k, biasGrad);
            
            // Calculate filter gradients
            for (int c = 0; c < numChannels; c++) {
                int filterChannelOffset = ((k * numChannels) + c) * filterSize * filterSize;
                
                for (int fh = 0; fh < filterSize; fh++) {
                    for (int fw = 0; fw < filterSize; fw++) {
                        float filterGrad = 0;
                        
                        // Accumulate gradients from all images and all valid output positions
                        for (int i = 0; i < numImages; i++) {
                            int inputChannelOffset = (i * numChannels + c) * inputHeight * inputWidth;
                            int dZFilterOffset = (i * numFilters + k) * outputHeight * outputWidth;
                            
                            // For each output position
                            for (int oh = 0; oh < outputHeight; oh++) {
                                for (int ow = 0; ow < outputWidth; ow++) {
                                    // Calculate the input position that this output position reads from
                                    // when applying this specific filter element (fh, fw)
                                    int ih, iw;
                                    
                                    if (padding.equals("same_padding")) {
                                        ih = oh * stride - padTop + fh;
                                        iw = ow * stride - padLeft + fw;
                                    } else { // valid padding
                                        ih = oh * stride + fh;
                                        iw = ow * stride + fw;
                                    }
                                    
                                    // Check bounds - only accumulate if input position is valid
                                    if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                                        int inputIdx = inputChannelOffset + (ih * inputWidth + iw);
                                        int dZIdx = dZFilterOffset + (oh * outputWidth + ow);
                                        
                                        // Gradient: dL/dW = input * dL/dOutput
                                        filterGrad += lastInput.get(inputIdx) * input.get(dZIdx);
                                    }
                                }
                            }
                        }
                        
                        dFilters.set(filterChannelOffset + (fh * filterSize + fw), filterGrad);
                    }
                }
            }
        });
        
        // Calculate input gradients (dX)
        final int TILE_SIZE = 32;
        int numTilesH = (inputHeight + TILE_SIZE - 1) / TILE_SIZE;
        int numTilesW = (inputWidth + TILE_SIZE - 1) / TILE_SIZE;

        // Create parallel tasks for each channel+tile combination
        IntStream.range(0, numChannels * numTilesH * numTilesW).parallel().forEach(taskIdx -> {
            int c = taskIdx / (numTilesH * numTilesW);
            int tileIdx = taskIdx % (numTilesH * numTilesW);
            int tileH = tileIdx / numTilesW;
            int tileW = tileIdx % numTilesW;
            
            // Calculate tile boundaries
            int ih_start = tileH * TILE_SIZE;
            int ih_end = Math.min(ih_start + TILE_SIZE, inputHeight);
            int iw_start = tileW * TILE_SIZE;
            int iw_end = Math.min(iw_start + TILE_SIZE, inputWidth);
            
            // Process for all images in the batch
            for (int i = 0; i < numImages; i++) {
                int dXChannelOffset = (i * numChannels + c) * inputHeight * inputWidth;
                
                // Compute gradients for this tile
                for (int ih = ih_start; ih < ih_end; ih++) {
                    for (int iw = iw_start; iw < iw_end; iw++) {
                        float sum = 0;
                        
                        // For each filter
                        for (int k = 0; k < numFilters; k++) {
                            int filterChannelBaseOffset = (k * numChannels + c) * filterSize * filterSize;
                            int dZFilterOffset = (i * numFilters + k) * outputHeight * outputWidth;
                            
                            // For each filter position that could have contributed to this input gradient
                            for (int fh = 0; fh < filterSize; fh++) {
                                for (int fw = 0; fw < filterSize; fw++) {
                                    // Calculate which output position this filter element would affect
                                    // when applied to input position (ih, iw)
                                    int oh, ow;
                                    
                                    if (padding.equals("same_padding")) {
                                        // Check if this input position contributes to any output
                                        int numerator_h = ih + padTop - fh;
                                        int numerator_w = iw + padLeft - fw;
                                        
                                        if (numerator_h % stride != 0 || numerator_w % stride != 0) {
                                            continue; // No contribution to any output
                                        }
                                        
                                        oh = numerator_h / stride;
                                        ow = numerator_w / stride;
                                    } else { // valid padding
                                        int numerator_h = ih - fh;
                                        int numerator_w = iw - fw;
                                        
                                        if (numerator_h % stride != 0 || numerator_w % stride != 0 || 
                                            numerator_h < 0 || numerator_w < 0) {
                                            continue; // No contribution to any output
                                        }
                                        
                                        oh = numerator_h / stride;
                                        ow = numerator_w / stride;
                                    }
                                    
                                    // Check if output position is valid
                                    if (oh >= 0 && oh < outputHeight && ow >= 0 && ow < outputWidth) {
                                        int filterPos = filterChannelBaseOffset + (fh * filterSize + fw);
                                        int dZPos = dZFilterOffset + (oh * outputWidth + ow);
                                        
                                        sum += filters.get(filterPos) * input.get(dZPos);
                                    }
                                }
                            }
                        }
                        int dXPos = dXChannelOffset + (ih * inputWidth + iw);
                        dX.set(dXPos, sum);
                    }
                }
            }
        });
        
        // Apply adaptive gradient scaling
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
    public JMatrix[] getWeights() {
        return new JMatrix[]{filters, biases};
    }

    @Override
    public JMatrix[] getParameterGradients() {
        return new JMatrix[]{dFilters, dBiases};
    }


    @Override
    public int[] outputShape() {
        int[] outputShape = null;
        if (getOutput() != null) {
            outputShape = getOutput().shape();
        } else {
            int[] prevShape;
            if (getPreviousLayer() == null) {
                int[] inputShape = internalGetInputShape();
                prevShape = new int[]{-1, inputShape[0], inputShape[1], inputShape[2]};
            } else {
                prevShape = getPreviousLayer().outputShape().clone();
            }
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