package jflow.data;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;




public class JMatrix {
    private float[] matrix;
    private int length, channels, height, width;
    private Random rand = new Random();
    private String name = null;
    /*
     * Cut-off size for matrices passed to matmul(). 
     * Matrices with all dimensions under the cut-off
     * size will be handled with simpleMatmul()
     */ 
    private static int cutoffSize = -1;
    // Cache-blocking sizes tuned for modern CPUs
    private static final int BLOCK_SIZE_M = 128;
    private static final int BLOCK_SIZE_N = 128;
    private static final int BLOCK_SIZE_K = 512;

    


    private static final ForkJoinPool THREAD_POOL = new ForkJoinPool(
            Math.max(1, Runtime.getRuntime().availableProcessors() * 2), 
            ForkJoinPool.defaultForkJoinWorkerThreadFactory,
            null, true);
    

    /**
     * Initialize a new JMatrix with default values of zero.
     * @param length                The batch dimension.
     * @param channels              The channel dimension.
     * @param height                The height dimension.
     * @param width                 The width dimension.
     */
    public JMatrix(int length, int channels, int height, int width) {
        this(new float[length * channels * height * width], 
            length, channels, height, width);
    }

     /**
     * Initialize a new JMatrix with default values of zero.
     * @param shape                The desired shape (N, channels, height, width)
     * @throws IllegalArgumentException if the length of shape is not four.
     */
    public JMatrix(int[] shape) {
        this(shape[0], shape[1], shape[2], shape[3]);

        if (shape.length != 4) {
            throw new IllegalArgumentException("Invalid shape. Only length of 4 is permitted.");
        }
    }

    /**
     * Wrap an array in a new JMatrix.
     * @param length                The batch dimension.
     * @param channels              The channel dimension.
     * @param height                The height dimension.
     * @param width                 The width dimension.
     */
    public JMatrix(float[] matrix, int length, int channels, int height, int width) {
        // all constructors point to this one
        this.matrix = matrix;
        this.length = length;
        this.channels = channels;
        this.height = height;
        this.width = width;

        if (cutoffSize == -1) {
            // TODO: calibrate cut-off size
            cutoffSize = 1024;
        } 
    }

    /**
     * Initialize a new JMatrix with default values of zero.
     * @param length                The batch dimension.
     * @param channels              The channel dimension.
     * @param height                The height dimension.
     * @param width                 The width dimension.
     * @param name                  The name to assign to this JMatrix.
     */
    public JMatrix(int length, int channels, int height, int width, String name) {
        this(length, channels, height, width);
        this.name = name;
    }

     /**
     * Initialize a new JMatrix with default values of zero.
     * @param shape                The desired shape (N, channels, height, width)
     * @param name                  The name to assign to this JMatrix.
     * 
     * @throws IllegalArgumentException if the length of shape is not four.
     */
    public JMatrix(int[] shape, String name) {
        if (shape.length != 4) {
            throw new IllegalArgumentException("Invalid shape. Only 4 is permitted.");
        }

        this.length = shape[0];
        this.channels = shape[1];
        this.height = shape[2];
        this.width = shape[3];
        this.name = name;

        matrix = new float[length * channels * height * width];
    }

    /**
     * Wrap an array in a new JMatrix.
     * @param length                The batch dimension.
     * @param channels              The channel dimension.
     * @param height                The height dimension.
     * @param width                 The width dimension.
     * @param name                  The name to assign to this JMatrix.
     */
    public JMatrix(float[] matrix, int length, int channels, int height, int width, String name) {
        this.matrix = matrix;
        this.length = length;
        this.channels = channels;
        this.height = height;
        this.width = width;
        this.name = name;
    }

    protected float access(int index) {
        // Groundwork for new features
        return matrix[index];
    }

    /**
     * Access the wrapped array.
     */
    public float[] getMatrix() {
        return matrix;
    }

    /**
     * Name this JMatrix
     * @param name          The name to assign.
     */
    public JMatrix setName(String name) {
        this.name = name;
        return this; // For chaining
    }

    /**
     * Access the name of this JMatrix. 
     * @return the name if set. <li> otherwise null.
     */
    public String getName() {
        return name;
    }

    /**
     * The shape of the JMatrix.
     * @returns {length, channels, height, width} in an int[4].
     */
    public int[] shape() {
        return new int[]{length, channels, height, width};
    }

    /**
     * The shape of the JMatrix as a String.
     * @returns {length, channels, height, width} visually organized into a String.
     */
    public String shapeAsString() {
        return "(" + length + "," + channels + "," + height + "," + width + ")";
    }
    /**
     * Print the shape of the JMatrix in the format (length, channels, height, width).
     */
    public JMatrix printShape() {
        System.out.println("(" + length + "," + channels + 
            "," + height + "," + width + ")");
        return this;
    }


    /**
     * Set the wrapped array to a new value. Resize not allowed.
     * @param matrix                            The new array to replace the original. 
     * @exception IllegalArgumentException      if the number of items 
     * in the new array doesn't match the original.
     */
    public void setMatrix(float[] matrix) {
        if (matrix.length != size()) {
            throw new IllegalArgumentException(
                "Sizes must match. Original: " 
                + size() + " New: " + matrix.length
            );
        }
        this.matrix = matrix;
    }

    /**
     * Set the wrapped array to a new value. Resize allowed.
     * @param matrix                            The new array to replace the original. 
     * @param shape                             The four dimensional shape of the new matrix.
     * @exception IllegalArgumentException      if: <p> <ul> <li>  the length of shape is not four. <p> <li>
     *  the reported number of elements is unequal to the length of the matrix. <ul>
     */
    public void setMatrix(float[] matrix, int[] shape) {
        int newSize = length * channels * height * width;
        if (matrix.length != newSize) {
            throw new IllegalArgumentException(
                "Sizes must match. Reported: " 
                + newSize + " Actual: " + matrix.length
            );
        }
        this.matrix = matrix;
        this.length = shape[0];
        this.channels = shape[1];
        this.height = shape[2];
        this.width = shape[3];
    }
    
    /**
     * Copy a batch worth of elements into the proper location.
     * @param batchIndex the index along the batch dimension to copy into.
     * @param values a JMatrix of values to copy into this JMatrix.
     * @throws IllegalArgumentException if the size of values is unequal 
     * to the size of a channels * height * width slice of this JMatrix.
     */
    public void arrayCopyBatch(int batchIndex, JMatrix values) {
        int itemSize = channels * height * width;
        float[] internalValues = values.getMatrix();
        if (internalValues.length != itemSize) {
            throw new IllegalArgumentException("Unexpected length: " + values.length + 
                ". Expected: " + itemSize);
        }
        IntStream.range(0, itemSize)
            .parallel().forEach(i -> {
            matrix[batchIndex * itemSize + i] = internalValues[i];
        });
    }

    /**
     * Get all values up until a given index.
     * @param index the exclusive end index.
     *
     */
    public float[] to(int index) {
        float[] values = new float[index];

        IntStream.range(0, index).parallel().forEach(i -> {
            values[i] = access(i);
        });

        return values;
    }

    /**
     * Get all values starting at a given index.
     * @param index the inclusive start index.
     *
     */
    public float[] from(int index) {
        int returnSize = size() - index;
        float[] values = new float[returnSize];

        IntStream.range(0, index).parallel().forEach(i -> {
            values[i] = access(index + i);
        });

        return values;
    }

    /**
     * The total number of elements.
     */
    public int size() {
        return matrix.length;
    }

    /**
     * The specified batch dimension.
     */
    public int length() {
        return length;
    }
    /**
     * The specified channel dimension.
     */
    public int channels() {
        return channels;
    }
    /**
     * The specified height dimension.
     */
    public int height() {
        return height;
    }
    /**
     * The specified width dimension.
     */
    public int width () {
        return width;
    }

    /**
     * Get an individual element.
     * @param index               The 1D index of the item to get.
     */
    public float get(int index) {
        return access(index);
    }

    /**
     * Get an individual element.
     * @param lengthIndex               The batch index of the item to get.
     * @param channelIndex              The channel index of the item to get.
     * @param heightIndex               The height index of the item to get.
     * @param widthIndex                The width index of the item to get.
     */
    public float get(int lengthIndex, int channelIndex, int heightIndex, int widthIndex) {
        return access(lengthIndex * channels * height * 
            width + channelIndex * height * width + 
            heightIndex * width + widthIndex);
    }


    /**
     * Copies a slice of this JMatrix
     * @param startIdx                  the 1D start index, inclusive.
     * @param endIdx                    the 1D end index, exclusive.
     * @return                          a JMatrix containing the sliced values with shape (length, 1, 1, 1).
     */
    public JMatrix slice(int startIdx, int endIdx) {
        int length = endIdx - startIdx;
        float[] result = new float[length];

        IntStream.range(0, length).parallel().forEach(i -> {
            result[i] = matrix[i + startIdx];
        });

        return new JMatrix(result, length, 1, 1, 1);
    }
    /**
     * Get a channels * height * width element.
     * @param lengthIndex The index along the batch dimension.
     */
    public double[] getImage(int lengthIndex) {
        int sliceSize = channels * height * width;
        int startIdx = lengthIndex * sliceSize;  
        double[] slice = new double[sliceSize];
        System.arraycopy(matrix, startIdx, slice, 0, sliceSize);
        return slice;
    }


    /**
     * Get a channels * height * width element wrapped in a JMatrix.
     * @param lengthIndex The index along the batch dimension.
     */
     public JMatrix getWrapped(int lengthIndex) {
        int sliceSize = channels * height * width;
        int startIdx = lengthIndex * sliceSize;  
        float[] slice = new float[sliceSize];
        System.arraycopy(matrix, startIdx, slice, 0, sliceSize);
        return new JMatrix(slice, 1, channels, height, width);
    }

    /**
     * Get a height * width element.
     * @param lengthIndex The index along the batch dimension.
     * @param channelIndex The index along the channel dimension.
     */
    public float[] get(int lengthIndex, int channelIndex) {
        int sliceSize = height * width;
        int startIdx = lengthIndex * channels * sliceSize + channelIndex * sliceSize;
        float[] slice = new float[sliceSize];
        System.arraycopy(matrix, startIdx, slice, 0, sliceSize);
        return slice;
    }

    /**
     * Get a height * width element wrapped in a JMatrix.
     * @param lengthIndex The index along the batch dimension.
     * @param channelIndex The index along the channel dimension.
     */
    public JMatrix getWrapped(int lengthIndex, int channelIndex) {
        int sliceSize = height * width;
        int startIdx = lengthIndex * channels * sliceSize + channelIndex * sliceSize;
        float[] slice = new float[sliceSize];
        System.arraycopy(matrix, startIdx, slice, 0, sliceSize);
        return new JMatrix(slice, 1, 1, height, width);
    }

    /**
     * Set a height * width element.
     * @param lengthIndex The index along the batch dimension.
     * @param channelIndex The index along the channel dimension.
     * @param values The values to copy into the specified region.
     * @throws IllegalArgumentException If the number of values is
     * unequal to the height * width of the JMatrix.
     */
    public void set(int lengthIndex, int channelIndex, double[] values) {
        int sliceSize = height * width;
        if (values.length != sliceSize) {
            throw new IllegalArgumentException("Invalid slice size. Expected " + sliceSize + " values.");
        }
        int startIdx = lengthIndex * channels * sliceSize + channelIndex * sliceSize;
        System.arraycopy(values, 0, matrix, startIdx, sliceSize);
    }

    
    /**
     * Alter the dimensional information stored in the JMatrix.
     * @param newLength                 The new batch dimension.
     * @param newChannels               The new channel dimension.
     * @param newHeight                 The new height dimension.
     * @param newWidth                  The new width dimension.
     * @throws IllegalArgumentException If the total size of the reshape 
     * is unequal to that of the original.
     * @return                          A new JMatrix with the changes applied.
     */
    public JMatrix reshape(int newLength, int newChannels, int newHeight, int newWidth) {
        int numItems = size();
        int newNumItems = newLength * newChannels * newHeight * newWidth;

        if (numItems != newNumItems) {
            throw new IllegalArgumentException(
                "Invalid reshape: total elements must match. Original: " 
                + numItems + " Reshape: " + newNumItems);
        }

        return new JMatrix(matrix, newLength, newChannels, newHeight, newWidth);
    }

     /**
     * Alter the dimensional information stored in the JMatrix.
     * @param newLength                 The new shape;
     * @throws IllegalArgumentException If the total size of the reshape 
     * is unequal to that of the original, or if shape.length != 4.
     * @return                          A new JMatrix with the changes applied.
     */
    public JMatrix reshape(int[] shape) {
        if (shape.length != 4) {
            throw new IllegalArgumentException(
                "Invalid shape: length does not equal 4.");
        }
        return reshape(shape[0], shape[1], shape[2], shape[3]);
    }
    /**
     * Set an item with 1D indexing.
     * @param index             The 1D index to alter.
     * @param value             The value to set, cast to a float.            
     */
    public void set(int index, double value) {
        matrix[index] = (float)value;
    }

    /**
     * Set an item with 4D indexing.
     * @param lengthIndex               The batch index of the item to set.
     * @param channelIndex              The channel index of the item to set.
     * @param heightIndex               The height index of the item to set.
     * @param widthIndex                The width index of the item to set.
     * @param value             The value to set, cast to a float.            
     */
    public void set(int lengthIndex, int channelIndex, int widthIndex, int heightIndex, double value) {
        matrix[lengthIndex * channels * height * 
        width + channelIndex * height * width + 
        heightIndex * width + widthIndex] = (float)value;
    }

// Statistics

    /**
     * The max value in the JMatrix.             
     */
    public double max() {
        double max = Double.NEGATIVE_INFINITY;
        for (double d : matrix) {
            max = Math.max(max, d);
        }
        return max;
    }
    /**
     * The max absolute value in the JMatrix.             
     */
    public double absMax() {
        double max = Double.NEGATIVE_INFINITY;
        for (double d : matrix) {
            max = Math.max(max, Math.abs(d));
        }
        return max;
    }
    /**
     * The 1D index of the max value.
     */
    public int argmax() {
        double maxValue = Double.NEGATIVE_INFINITY;
        int maxIndex = 0;
        for (int i = 0; i < size(); i++) {
            if (access(i) > maxValue) {
                maxValue = access(i);
                maxIndex = i;
            }
        }
        return maxIndex;
    }
    /**
     * Finds the index of max values along a given axis.
     * @param axis              The specified axis in the range [0,3] inclusive.
     * @return                  An array containing the indexes of max values along the given axis
     */
    public int[] argmax(int axis) {
        if (axis < 0 || axis > 3) {
            throw new IllegalArgumentException("Axis must be between 0 and 3");
        }
    
        int[] dimensions = new int[]{length, channels, height, width};
        int[] strides = new int[]{channels * height * width, height * width, width, 1};
    
        int axisSize = dimensions[axis];
        int resultSize = matrix.length / axisSize;
        int[] result = new int[resultSize];
    
        IntStream.range(0, resultSize).parallel().forEach(opIndex -> {
            int baseOffset = indexHelper(opIndex, axis, dimensions, strides);
    
            float maxVal = Float.NEGATIVE_INFINITY;
            int maxIdx = 0;
    
            for (int i = 0; i < axisSize; i++) {
                int offset = baseOffset + i * strides[axis];
                if (matrix[offset] > maxVal) {
                    maxVal = matrix[offset];
                    maxIdx = i;
                }
            }
    
            result[opIndex] = maxIdx;
        });
    
        return result;
    }
    /**
     * The mean of all values in the JMatrix.
     */
    public double mean() {
        double mean = 0;

        int size = size();
        for (int i = 0; i < size; i++) {
            mean += access(i);
        
        }
        return mean / size;
    }
    /**
     * The sum of all values in the JMatrix.
     */
    public double sum() {
        double sum = 0;

        int size = size();
        for (int i = 0; i < size; i++) {
            sum += access(i);
        
        }
        return sum;
    }

    /**
     * Sums the elements of the matrix along a specified axis.
     * 
     * @param axis The axis along which to perform the sum (0=batch, 1=channel, 2=height, 3=width)
     * @return A new JMatrix with the summed values, with dimension 1 in the summed axis
     */
    public JMatrix sum(int axis) {
        int[] shape = new int[]{length, channels, height, width};
        
        // Create a new matrix with the summed dimension set to 1
        shape[axis] = 1;
        JMatrix result = new JMatrix(shape[0], shape[1], shape[2], shape[3]);
        
        // Perform the sum based on the specified axis
        switch (axis) {
            case 0: // Sum across batch dimension
                for (int n = 0; n < length; n++) {
                    for (int c = 0; c < channels; c++) {
                        for (int h = 0; h < height; h++) {
                            for (int w = 0; w < width; w++) {
                                result.set(0, c, h, w, result.get(0, c, h, w) + 
                                    get(n, c, h, w));
                            }
                        }
                    }
                }
                break;
                
            case 1: // Sum across channel dimension
                for (int n = 0; n < length; n++) {
                    for (int c = 0; c < channels; c++) {
                        for (int h = 0; h < height; h++) {
                            for (int w = 0; w < width; w++) {
                                result.set(n, 0, h, w, result.get(n, 0, h, w) + 
                                    get(n, c, h, w));
                            }
                        }
                    }
                }
                break;
                
            case 2: // Sum across height dimension
                for (int n = 0; n < length; n++) {
                    for (int c = 0; c < channels; c++) {
                        for (int h = 0; h < height; h++) {
                            for (int w = 0; w < width; w++) {
                                result.set(n, c, 0, w, result.get(n, c, 0, w) + 
                                    get(n, c, h, w));
                            }
                        }
                    }
                }
                break;
                
            case 3: // Sum across width dimension
                for (int n = 0; n < length; n++) {
                    for (int c = 0; c < channels; c++) {
                        for (int h = 0; h < height; h++) {
                            for (int w = 0; w < width; w++) {
                                result.set(n, c, h, 0, result.get(n, c, h, 0) + 
                                get(n, c, h, w));
                            }
                        }
                    }
                }
                break;
                
            default:
                throw new IllegalArgumentException("Axis must be between 0 and 3");
        }
        
        return result;
    }
   
    /**
     * The frobenius norm is calculated as: <p>
     * For every x -> <p>
     * - Raise x ^ 2. <p>
     * - Add it to the sum. <p>
     * Finally, square the sum.
     */
    public float frobeniusNorm() {
        float sum = 0;

        int size = size();
        for (int i = 0; i < size; i++) {
            float pixel = access(i);
            sum += pixel * pixel;
        
        }
        return (float)Math.sqrt(sum);
    }

    /**
     * The l1 norm is calculated as: <p>
     * For every x -> <p>
     * Take the abosolute value and add it to the sum.
     * 
     */
    public float l1Norm() {
        float sum = 0;
        int size = size();
        for (int i = 0; i < size; i++) {
            sum += Math.abs(access(i));
        }
        return sum;
    }

    /**
     * Count the number of items in the JMatrix of a certain value.
     * @param value             The value to count instances of.
     */
    public int count(int value) {
        int count = 0;
        int size = size();
        for (int i = 0; i < size; i++) {
            if (access(i) == value) {
                count++;
            }
        }
        return count;
    }

    /**
     * Perform softmax along a given axis. Avoids overflow.
     * @param axis the axis to apply softmax to: <p>
     * <ul> <li> 0 = across batch dimension 
     * <li> 1 = across channel dimension
     * <li> 2 = across height dimension
     * <li> 3 = across width dimension </ul>
     * @return A new JMatrix with softmax applied along the specified axis.
     */
    public JMatrix softmax(int axis) {
        if (axis < 0 || axis > 3) {
            throw new IllegalArgumentException("Axis must be between 0 and 3");
        }
        
        // Create result matrix with same dimensions
        float[] result = new float[matrix.length];
        
        try {
            THREAD_POOL.submit(() -> {
                // Determine the sizes and strides based on dimensions
                int[] dimensions = new int[]{length, channels, height, width};
                int[] strides = new int[]{channels * height * width, height * width, width, 1};
                
                // Calculate the size of the axis to apply softmax to
                int axisSize = dimensions[axis];
                
                // Calculate the number of softmax operations to perform
                int numOperations = matrix.length / axisSize;
                
                // Process each softmax operation in parallel
                IntStream.range(0, numOperations).parallel().forEach(opIndex -> {
                    // Calculate the starting offset for this operation
                    
                    int baseOffset = indexHelper(opIndex, axis, dimensions, strides);
                    
                    // Find max value for numerical stability
                    float maxVal = Float.NEGATIVE_INFINITY;
                    for (int i = 0; i < axisSize; i++) {
                        int offset = baseOffset + i * strides[axis];
                        maxVal = Math.max(maxVal, matrix[offset]);
                    }
                    
                    // Calculate sum of exponentials
                    float sumExp = 0.0f;
                    for (int i = 0; i < axisSize; i++) {
                        int offset = baseOffset + i * strides[axis];
                        sumExp += Math.exp(matrix[offset] - maxVal);
                    }
                    
                    // Calculate softmax values
                    for (int i = 0; i < axisSize; i++) {
                        int offset = baseOffset + i * strides[axis];
                        result[offset] = (float) Math.exp(matrix[offset] - maxVal) / sumExp;
                    }
                });
            }).get();
        } catch (Exception e) {
            throw new RuntimeException("Error in JMatrix.Softmax(axis = " + axis + ")", e);
        }
        
        return new JMatrix(result, length, channels, height, width);
    }
    /**
     * Perform log softmax along a given axis. Avoids overflow and underflow.
     * @param axis the axis to apply log softmax to: <p>
     * <ul> <li> 0 = across batch dimension
     * <li> 1 = across channel dimension
     * <li> 2 = across height dimension
     * <li> 3 = across width dimension </ul>
     * @return A new JMatrix with log softmax applied along the specified axis.
     */
    public JMatrix logSoftmax(int axis) {
        if (axis < 0 || axis > 3) {
            throw new IllegalArgumentException("Axis must be between 0 and 3");
        }

        float[] result = new float[size()];
        try {
            THREAD_POOL.submit(() -> {
                // Determine the sizes and strides based on dimensions
                int[] dimensions = new int[]{length, channels, height, width};
                int[] strides = new int[]{channels * height * width, height * width, width, 1};

                // Calculate the size of the axis to apply log softmax to
                int axisSize = dimensions[axis];

                // Calculate the number of log softmax operations to perform
                int numOperations = matrix.length / axisSize;

                IntStream.range(0, numOperations).parallel().forEach(opIndex -> {
                    // Calculate the starting offset
                    int baseOffset = indexHelper(opIndex, axis, dimensions, strides);
                    // Find max value for numerical stability
                    float maxVal = Float.NEGATIVE_INFINITY;
                    for (int i = 0; i < axisSize; i++) {
                        int offset = baseOffset + i * strides[axis];
                        maxVal = Math.max(maxVal, matrix[offset]);
                    }
                    // Calculate sum of exponentials
                    float sumExp = 0.0f;
                    for (int i = 0; i < axisSize; i++) {
                        int offset = baseOffset + i * strides[axis];
                        sumExp += Math.exp(matrix[offset] - maxVal);
                    }
                    // Calculate log softmax values
                    float logSumExp = (float) Math.log(sumExp) + maxVal;
                    for (int i = 0; i < axisSize; i++) {
                        int offset = baseOffset + i * strides[axis];
                        result[offset] = matrix[offset] - logSumExp;
                    }
                });
            }).get();
        } catch (Exception e) {
            throw new RuntimeException("Error in JMatrix.LogSoftmax(axis = " + axis + ")", e);
        }
        return new JMatrix(result, length, channels, height, width);
    }

    // NOTE: THIS METHOD IS ADAPTED FROM CODE GENERATED BY CLAUDE.AI
    private static int indexHelper(int opIndex, int axis, int[] dimensions, int[] strides) {
        int[] indices = new int[4];
        int remainingIndex = opIndex;
        
        for (int dim = 0; dim < 4; dim++) {
            if (dim == axis) continue;
            
            // Calculate the size of all dimensions after this one (except the softmax axis)
            int productOfLaterDims = 1;
            for (int laterDim = dim + 1; laterDim < 4; laterDim++) {
                if (laterDim != axis) {
                    productOfLaterDims *= dimensions[laterDim];
                }
            }
            
            // Calculate the index for this dimension
            indices[dim] = remainingIndex / productOfLaterDims;
            remainingIndex %= productOfLaterDims;
        }
        
        // Set the index for the softmax axis to 0
        indices[axis] = 0;
        
        // Calculate the base offset
        int baseOffset = 0;
        for (int dim = 0; dim < 4; dim++) {
            baseOffset += indices[dim] * strides[dim];
        }
        
        return baseOffset;
    }
    
    /**
     * Creates a one-hot encoded representation of token indices.
     * 
     * @param indices A JMatrix containing integer token indices
     * @param vocabSize The size of the vocabulary (number of possible token values)
     * @return A new JMatrix with one-hot encoding
     */
    public static JMatrix oneHot(JMatrix indices, int vocabSize) {
        int batchSize = indices.shape()[0];
        int seqLen = indices.shape()[1];
        
        JMatrix oneHotMatrix = new JMatrix(batchSize, seqLen, vocabSize, 1);
        
        // Fill in the one-hot encoded values
        for (int batch = 0; batch < batchSize; batch++) {
            for (int seq = 0; seq < seqLen; seq++) {
                // Get the token index at this position
                int tokenIndex = (int)indices.get(batch, seq, 0, 0);
                
                // Set the corresponding position to 1.0 if the token index is valid
                if (tokenIndex >= 0 && tokenIndex < vocabSize) {
                    oneHotMatrix.set(batch, seq, tokenIndex, 0, 1.0);
                }
            }
        }
        
        return oneHotMatrix;
    }

    /**
     * Create a JMatrix with random values in the range [0,1]
     * @param length                        the N dimension of the JMatrix.
     * @param channels                      the channel dimension of the JMatrix.
     * @param height                        the height dimension of the JMatrix.
     * @param width                         the width dimension of the JMatrix.
     * 
     * @return                              a new JMatrix with the specified dimensions 
     * and random values in the range [0,1].
     * 
     */
    public static JMatrix randn(int length, int channels, int height, int width) {
        int size = length * channels * height * width;

        float[] noise = new float[size];
        IntStream.range(0, size).parallel().forEach(i -> {
            noise[i] = (float)ThreadLocalRandom.current().nextDouble();
        });

        return new JMatrix(noise, length, channels, height, width);
    }

    /**
     * Generates a position ID matrix corresponding to the input token ID matrix.
     *
     * For each token in the input tokenIDs matrix, this function assigns a position index (0 to seqLen - 1)
     * along the sequence dimension. The output is a new JMatrix of shape [batch, seqLen, 1, 1],
     * where each entry holds the position index of the corresponding token in the sequence.
     *
     * @param tokenIDs A JMatrix of shape [batch, seqLen] representing token IDs for a batch of sequences.
     * @return A JMatrix of shape [batch, seqLen, 1, 1] where each entry contains the position index of the token.
     */
    public static JMatrix positionIDs(JMatrix tokenIDs) {
        int batch = tokenIDs.shape()[0];
        int seqLen = tokenIDs.shape()[1];
        JMatrix positions = new JMatrix(batch, seqLen, 1, 1);
    
        for (int b = 0; b < batch; b++) {
            for (int t = 0; t < seqLen; t++) {
                positions.set(b, t, 0, 0, t);
            }
        }
    
        return positions;
    }
    

    /**
     * Scale values in the JMatrix from [0, n] to [0, 1].
     * @return A new JMatrix with the changes applied.
     */
    public JMatrix scaleSigmoid() {
        double max = max();
        return multiply(1.0 / max);
    }

    /**
     * Add Gaussian noise to each item in the JMatrix.
     * @param mean              The mean of the Gaussian noise.
     * @param stdDev            The standard deviation of the Gaussian noise.
     * @return A new JMatrix with the changes applied.
     */
    public JMatrix addGaussianNoise(double mean, double stdDev) {
        JMatrix noisy = this.zerosLike();
        for (int i = 0; i < size(); i++) {
            noisy.set(i, access(i) + (float)(rand.nextGaussian() * stdDev + mean));
        }
        return noisy;
    }

    /**
     * Sum values along rows for 2D use cases.
     * @param scale Whether or not to scale results by 1 / numRows.
     */
    public float[] sum0(boolean scale) {
        int rows = length;
        int cols = channels * height * width;

        float scaleFactor = (scale) ? 1.0f / rows : 1;

        float[] sum = new float[rows];
        IntStream.range(0, rows).parallel().forEach(i -> { 
            for (int j = 0; j < cols; j++) {
                sum[i] = access(i * cols + j);
            }
            sum[i] *= scaleFactor;
        });
        return sum;
    }

    /**
     * Rotate the JMatrix 90 degrees clockwise for 2D use cases.
     * Swaps the batch dimension (N) with the item dimension (C * H * W).
     * @return A new JMatrix with the changes applied.
     */
    public JMatrix transpose2D() {
        int oldHeight = length;
        int oldWidth = channels * height * width;
        int newHeight = oldWidth;
        int newWidth = oldHeight;

        float[] rotated = new float[size()];
        
        // Use cache-friendly block size
        final int BLOCK_SIZE = 64;
        final int PARALLEL_THRESHOLD = 4096;
        
        // For small matrices, use sequential processing
        if (oldHeight * oldWidth <= PARALLEL_THRESHOLD) {
            transposeBlock2D(rotated, 0, 0, oldHeight, oldWidth, oldWidth, oldHeight);
        } else {
            // Process in blocks for better cache locality
            int numBlocksH = (oldHeight + BLOCK_SIZE - 1) / BLOCK_SIZE;
            int numBlocksW = (oldWidth + BLOCK_SIZE - 1) / BLOCK_SIZE;
            
            IntStream.range(0, numBlocksH * numBlocksW).parallel().forEach(blockIdx -> {
                int blockRow = blockIdx / numBlocksW;
                int blockCol = blockIdx % numBlocksW;
                
                int rowStart = blockRow * BLOCK_SIZE;
                int rowEnd = Math.min(rowStart + BLOCK_SIZE, oldHeight);
                int colStart = blockCol * BLOCK_SIZE;
                int colEnd = Math.min(colStart + BLOCK_SIZE, oldWidth);
                
                transposeBlock2D(rotated, rowStart, colStart, rowEnd, colEnd, oldWidth, newWidth);
            });
        }
        // float[] rotated = MetalTranspose.transpose2D(matrix, oldHeight, oldWidth, false);
        return new JMatrix(rotated, newHeight, newWidth, 1, 1);
    }

    /**
     * Process a block of the 2D transpose operation
     */
    private void transposeBlock2D(float[] rotated, int rowStart, int colStart,
                            int rowEnd, int colEnd, int oldWidth, int newWidth) {
    for (int row = rowStart; row < rowEnd; row++) {        
        for (int col = colStart; col < colEnd; col++) {    
            // Convert feature index back to position in N,C,H,W 
            int n = row;  // batch index
            int chw = col; // flattened feature index
            

            int oldIndex = n * (channels * height * width) + chw;
            
            int newIndex = col * newWidth + row;  
            rotated[newIndex] = access(oldIndex);
        }
    }
}

    /**
     * Transpose the matrix by rearranging dimension according to a particular order
     * @param axis1 The dimension to use as the first dimension (0=N, 1=C, 2=H, 3=W)
     * @param axis2 The dimension to use as the second dimension (0=N, 1=C, 2=H, 3=W)
     * @param axis3 The dimension to use as the third dimension (0=N, 1=C, 2=H, 3=W)
     * @param axis4 The dimension to use as the fourth dimension (0=N, 1=C, 2=H, 3=W)
     */
    public JMatrix transpose(int axis1, int axis2, int axis3, int axis4) {
        // Check input values
        int[] axes = {axis1, axis2, axis3, axis4};
        boolean[] used = new boolean[4];
        for (int axis : axes) {
            if (axis < 0 || axis > 3) {
                throw new IllegalArgumentException("Axis values must be between 0 and 3 inclusive");
            }
            if (used[axis]) {
                throw new IllegalArgumentException("Axis values must be a permutation of 0, 1, 2, 3");
            }
            used[axis] = true;
        }

        int[] dims = {length, channels, height, width};
        
        // Calculate new dimensions after transposition
        int newLength = dims[axis1];
        int newChannels = dims[axis2];
        int newHeight = dims[axis3];
        int newWidth = dims[axis4];
        
        // Output array
        float[] transposed = new float[size()];
        
        // Pre-calculate the permutation mapping for faster index calculation
        int[] oldDims = {length, channels, height, width};
        
        // Pre-calculate strides for both original and new array
        int[] oldStrides = new int[4];
        oldStrides[3] = 1;                           // W dimension (innermost)
        oldStrides[2] = width;                       // H dimension
        oldStrides[1] = height * width;              // C dimension
        oldStrides[0] = channels * height * width;   // N dimension (outermost)
        
        int[] newStrides = new int[4];
        newStrides[3] = 1;                                   // W dimension
        newStrides[2] = newWidth;                            // H dimension
        newStrides[1] = newHeight * newWidth;                // C dimension
        newStrides[0] = newChannels * newHeight * newWidth;  // N dimension
        
        // Calculate exact number of elements for better work distribution
        final int totalElements = length * channels * height * width;
        final int THRESHOLD = 1024;
        
        // For small matrices, use a single-threaded approach
        if (totalElements <= THRESHOLD) {
            transposeSequential(transposed, axes, oldDims, oldStrides, newStrides);
        } else {
            int numThreads = Runtime.getRuntime().availableProcessors();
            int blockSize = Math.max(1, totalElements / (numThreads * 4));
            
            IntStream.range(0, (totalElements + blockSize - 1) / blockSize)
                    .parallel()
                    .forEach(block -> {
                        int start = block * blockSize;
                        int end = Math.min(start + blockSize, totalElements);
                        transposeBlock(transposed, axes, oldDims, oldStrides, newStrides, start, end);
                    });
        }
        
        return new JMatrix(transposed, newLength, newChannels, newHeight, newWidth);
    }

    /**
     * Sequential transposition for small matrices
     */
    private void transposeSequential(float[] transposed, int[] axes, int[] oldDims, int[] oldStrides, int[] newStrides) {
        int length = oldDims[0];
        int channels = oldDims[1];
        int height = oldDims[2];
        int width = oldDims[3];
        
        for (int n = 0; n < length; n++) {
            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        // Original coordinates
                        int[] coords = {n, c, h, w};
                        
                        // Calculate original index directly without using array lookups in hot loop
                        int originalIndex = n * oldStrides[0] + c * oldStrides[1] + 
                                        h * oldStrides[2] + w * oldStrides[3];
                        
                        // Calculate new coordinates after transposition
                        int newN = coords[axes[0]];
                        int newC = coords[axes[1]];
                        int newH = coords[axes[2]];
                        int newW = coords[axes[3]];
                        
                        // Calculate new index directly
                        int newIndex = newN * newStrides[0] + newC * newStrides[1] + 
                                    newH * newStrides[2] + newW * newStrides[3];
                        
                        // Copy the value
                        transposed[newIndex] = access(originalIndex);
                    }
                }
            }
        }
    }


    private void transposeBlock(float[] transposed, int[] axes, int[] oldDims, int[] oldStrides, int[] newStrides, 
                            int startIdx, int endIdx) {
        // Pre-calculated values for the innermost loop
        int axis1 = axes[0], axis2 = axes[1], axis3 = axes[2], axis4 = axes[3];
        
        // Get the actual dimensions for bounds checking
        int maxN = oldDims[0], maxC = oldDims[1], maxH = oldDims[2], maxW = oldDims[3];
        
        // Get a linear index and convert to 4D coordinates
        for (int linearIdx = startIdx; linearIdx < endIdx; linearIdx++) {
            // Convert linear index to NCHW coordinates
            int remaining = linearIdx;
            int n = remaining / oldStrides[0];
            remaining %= oldStrides[0];
            int c = remaining / oldStrides[1];
            remaining %= oldStrides[1];  
            int h = remaining / oldStrides[2];
            int w = remaining % oldStrides[2]; // This is correct - remaining mod width
            
            // Bounds check to prevent IndexOutOfBounds
            if (n >= maxN || c >= maxC || h >= maxH || w >= maxW) {
                continue; // Skip invalid coordinates
            }
            
            // Original coordinates
            int[] oldCoords = {n, c, h, w};
            
            // Map to new coordinates based on axis permutation
            int newN = oldCoords[axis1];
            int newC = oldCoords[axis2];
            int newH = oldCoords[axis3]; 
            int newW = oldCoords[axis4];
            
            // Calculate new index
            int newIdx = newN * newStrides[0] + newC * newStrides[1] + 
                        newH * newStrides[2] + newW * newStrides[3];
            
            // Bounds check for output array
            if (newIdx >= 0 && newIdx < transposed.length) {
                // Copy the value 
                transposed[newIdx] = access(linearIdx);
            }
        }
    }

    /**
     * Rotate every height * width * channels item by 90 degrees for 3D use cases.
     * @return A new JMatrix with the changes applied.
     */
    public JMatrix transpose3D() {
        int numBatches = length;     
        int oldC = channels;         
        int oldH = height;           
        int oldW = width;           
    
        int newH = oldH * oldW; // Flatten spatial dimensions
        int newW = oldC; // Channels become features
        float[] transposed = new float[size()]; 
    
        int oldPerBatch = oldC * oldH * oldW;
        int newPerBatch = newH * newW;
    
        IntStream.range(0, numBatches).parallel().forEach(batch -> {
            for (int c = 0; c < oldC; c++) {
                for (int h = 0; h < oldH; h++) {
                    for (int w = 0; w < oldW; w++) {
                        int hwIndex = h * oldW + w;
                        int oldIndex = batch * oldPerBatch + c * oldH * oldW + h * oldW + w;
                        int newIndex = batch * newPerBatch + hwIndex * oldC + c;
                        transposed[newIndex] = access(oldIndex);
                    }
                }
            }
        });
    
        return new JMatrix(transposed, numBatches, newH, newW, 1);
    }

    /**
     * Rotate every height * width element by 90 degrees.
     * @return A new JMatrix with the changes applied.
     */
    public JMatrix transpose4D() {
        int newHeight = width;
        int newWidth = height;
        
        float[] resultData = new float[size()];
        
        IntStream.range(0, length * channels).parallel().forEach(batchChannel -> {
            int batch = batchChannel / channels;
            int channel = batchChannel % channels;
            
            int oldOffset = batch * (channels * height * width) + channel * (height * width);
            int newOffset = batch * (channels * newHeight * newWidth) + channel * (newHeight * newWidth);
            
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int oldIndex = oldOffset + h * width + w;
                    int newIndex = newOffset + w * newWidth + h; 
                    resultData[newIndex] = matrix[oldIndex];
                }
            }
        });
        
        return new JMatrix(resultData, length, channels, newHeight, newWidth);
    }

    

    /**
     * Compare the shape of two JMatrixes.
     * @param other             The JMatrix to compare this JMatrix with.
     * @return                  True if the JMatrixes have the same shape.
     */
    public boolean shapeEquals(JMatrix other) {
        return length == other.length() && channels == other.channels()
            && height == other.height() && width == other.width();
    }

    /**
     * Returns an exact copy of this JMatrix.
     */
    public JMatrix copy() {
        return new JMatrix(matrix.clone(), length, channels, height, width);
    }
    /**
     * Returns a new empty JMatrix with the same dimensions as this JMatrix.
     */
    public JMatrix zerosLike() {
        return new JMatrix(new float[size()], length, channels, height, width);
    }

    /**
     * Clips all values in the JMatrix to a desired range.
     * @param min               The minimum allowed value, cast to a float.
     * @param max               The maximum allowed value, cast to a float.
     */
    public JMatrix clip(double min, double max) {
        float fMin = (float)min;
        float fMax = (float)max;
        IntStream.range(0, matrix.length).parallel().forEach(i -> {
            matrix[i] = Math.max(fMin, Math.min(fMax, matrix[i]));
        });
        return this; // For chaining
    }

    /**
     * Fills the JMatrix with a certain value.
     * @param fillValue             The value to assign to all items of the JMatrix.
     */
    public JMatrix fill(double fillValue) {
        float valueF = (float)fillValue;
        IntStream.range(0, matrix.length).parallel().forEach(i -> {
            matrix[i] = valueF;
        });
        return this; // For chaining
    }

    /**
     * Perform matrix multiplication with another JMatrix for 2D use cases. <p>
     * This function treats data as 2D tensors where:
     * <p> - First dimension: rows (N)
     * <p> - Second dimension: columns (C * H * W)
     * @param secondMatrix The second JMatrix to perform matrix multiplication with.
     * @param scale Whether or not to scale values by 1 / rows.
     * @return A new JMatrix representing the dot product.
     */

    public JMatrix matmul(JMatrix secondMatrix, boolean scale) {
        // Treat channels * height * width as flat
        int m = length;
        int n = secondMatrix.channels() * secondMatrix.height() * secondMatrix.width();
        int k = channels * height * width;
        
        if (k != secondMatrix.length()) {
            throw new IllegalArgumentException(
                "Matrix multiplication not possible for " +
                "arrays with shape: (" + m + "," + k +
                ") and (" + secondMatrix.length() + "," +
                n + ")"
            );
        }
        
        float[] matrixA = matrix;
        float[] matrixB = secondMatrix.getMatrix();
        
        // Use simple algorithm for small matrices
        if (m < cutoffSize && n < cutoffSize && k < cutoffSize) {
            return simpleMatmul(secondMatrix, scale, m, n, k);
        }

        float[] result = OptimizedMatmul.matmul(matrixA, matrixB, m, n, k, scale, 
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, THREAD_POOL);
        
        return new JMatrix(result, m, secondMatrix.channels(), secondMatrix.height(), secondMatrix.width());
    }
    
    /**
     * Simple matrix multiplication for smaller matrices.
     */
    private JMatrix simpleMatmul(JMatrix secondMatrix, boolean scale, int m, int n, int k) {
        float scaleFactor = (float)(1.0f / Math.sqrt(k));
        float[] result = new float[m * n];
        
        float[] matrixA = matrix;
        float[] matrixB = secondMatrix.getMatrix();
        
        IntStream.range(0, m).parallel().forEach(i -> {
            for (int j = 0; j < n; j++) {
                float sum = 0;
                
                int rowStartA = i * k;
                int colJ = j;
                    
                // Unroll by 8 for small matrices
                for (int kIndex = 0; kIndex < k - 7; kIndex += 8) {
                    sum += matrixA[rowStartA + kIndex] * matrixB[kIndex * n + colJ]
                             + matrixA[rowStartA + kIndex + 1] * matrixB[(kIndex + 1) * n + colJ]
                             + matrixA[rowStartA + kIndex + 2] * matrixB[(kIndex + 2) * n + colJ]
                             + matrixA[rowStartA + kIndex + 3] * matrixB[(kIndex + 3) * n + colJ]
                             + matrixA[rowStartA + kIndex + 4] * matrixB[(kIndex + 4) * n + colJ]
                             + matrixA[rowStartA + kIndex + 5] * matrixB[(kIndex + 5) * n + colJ]
                             + matrixA[rowStartA + kIndex + 6] * matrixB[(kIndex + 6) * n + colJ]
                             + matrixA[rowStartA + kIndex + 7] * matrixB[(kIndex + 7) * n + colJ];
                }
                    
                // Handle remaining elements
                for (int kIndex = k - (k % 8); kIndex < k; kIndex++) {
                    sum += matrixA[rowStartA + kIndex] * matrixB[kIndex * n + colJ];
                }

                result[i * n + j] = scale ? sum * scaleFactor : sum;
            }
        });
        
        return new JMatrix(result, m, secondMatrix.channels(), secondMatrix.height(), secondMatrix.width());
    }

    /**
     * Perform batch matrix multiplication for 3D tensor operations.
     * This function treats data as 3D tensors where:
     * - First dimension: batch (N)
     * - Second dimension: channels (C)
     * - Third dimension: spatial size (H * W)
     * 
     * @param secondMatrix The second JMatrix to perform batch matrix multiplication with.
     * @param scale Whether or not to scale values by 1 / (height * width).
     * @return A new JMatrix representing the batch dot product.
     */
    public JMatrix batchMatmul(JMatrix secondMatrix, boolean scale) {
        int batchSize = length;
        int inputChannels = channels;
        int flatSpatialDim = height * width;
        
        int outputChannels = secondMatrix.channels();
        int outFlatSpatialDim = secondMatrix.height() * secondMatrix.width();
        
        if (flatSpatialDim != secondMatrix.channels()) {
            throw new IllegalArgumentException(
                "Batch matrix multiplication not possible for " +
                "tensors with shapes: (" + batchSize + "," + inputChannels + "," + flatSpatialDim +
                ") and (" + secondMatrix.length() + "," + outputChannels + "," + outFlatSpatialDim + ")"
            );
        }

        float[] result = OptimizedMatmul.batchMatmul(matrix, secondMatrix.getMatrix(), length, 
            inputChannels, outFlatSpatialDim, outputChannels, scale, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, THREAD_POOL);

        return new JMatrix(result, batchSize, channels, outFlatSpatialDim / secondMatrix.width(), secondMatrix.width());
    }

    /**
     * Set every item x in the JMatrix to 1 / x.
     * @return A new JMatrix with the changes applied.
     */
    public JMatrix reciprocal() {
        int size = size();
        float[] result = new float[size];

        IntStream.range(0, size).parallel().forEach(i -> {
            result[i] = (float)(1.0 / access(i));
        });

        return new JMatrix(result, length, channels, height, width);
    }

    /**
     * Set every item x in the JMatrix to x ^ 1/2.
     * @return A new JMatrix with the changes applied.
     */
    public JMatrix sqrt() {
        int size = size();
        float[] result = new float[size];

        IntStream.range(0, size).parallel().forEach(i -> {
            result[i] = (float)(Math.sqrt(access(i)));
        });

        return new JMatrix(result, length, channels, height, width);
    }
 
    /**
     * Perform broadcast subtraction with another JMatrix.
     * @param secondMatrix              The JMatrix to subtract from this JMatrix.
     * @throws IllegalArgumentException If the JMatrixes are incompatible. Full match and (1,C,1,1) match are supported.
     * @return                          A new JMatrix representing the difference.
     */
    public JMatrix subtract(JMatrix secondMatrix) {
        int size = size();

        // Full element-wise subtraction
        if (size == secondMatrix.size()) {
            float[] result = new float[size];
            IntStream.range(0, size).parallel().forEach(i -> {
                result[i] = access(i) - secondMatrix.access(i);
            });
            return new JMatrix(result, length, channels, height, width);
        }

        // Channel-wise broadcasting: (1, C, 1, 1) over (N, C, H, W)
        if (secondMatrix.length() == 1 && secondMatrix.channels() == channels &&
            secondMatrix.height() == 1 && secondMatrix.width() == 1) {

            float[] result = new float[size];
            int channelSize = height * width;

            IntStream.range(0, length).parallel().forEach(n -> {
                for (int c = 0; c < channels; c++) {
                    float subtractor = secondMatrix.access(c); // one subtractor per channel
                    int offset = n * channels * channelSize + c * channelSize;
                    for (int i = 0; i < channelSize; i++) {
                        result[offset + i] = access(offset + i) - subtractor;
                    }
                }
            });

            return new JMatrix(result, length, channels, height, width);
        }

        // If matrices are not broadcastable
        throw new IllegalArgumentException(
            "Sizes " + size + " and " + secondMatrix.size() +
            " cannot be broadcast together. Supported: full match or (1,channels,1,1)."
        );
    }

        /**
     * Perform broadcast subtraction with another JMatrix in place.
     * @param secondMatrix              The JMatrix to subtract from this JMatrix.
     * @throws IllegalArgumentException If the JMatrixes are incompatible. Full match and (1,C,1,1) match are supported.
     */
    public JMatrix subtractInPlace(JMatrix secondMatrix) {
        int size = size();

        // Full element-wise subtraction
        if (size == secondMatrix.size()) {
            IntStream.range(0, size).parallel().forEach(i -> {
                matrix[i] = access(i) - secondMatrix.access(i);
            });
            return this; // For chaining
        }

        // Channel-wise broadcasting: (1, C, 1, 1) over (N, C, H, W)
        if (secondMatrix.length() == 1 && secondMatrix.channels() == channels &&
            secondMatrix.height() == 1 && secondMatrix.width() == 1) {

            int channelSize = height * width;

            IntStream.range(0, length).parallel().forEach(n -> {
                for (int c = 0; c < channels; c++) {
                    float subtractor = secondMatrix.access(c); // one subtractor per channel
                    int offset = n * channels * channelSize + c * channelSize;
                    for (int i = 0; i < channelSize; i++) {
                        matrix[offset + i] = access(offset + i) - subtractor;
                    }
                }
            });
            return this; // For chaining
        }

        // If matrices are not broadcastable
        throw new IllegalArgumentException(
            "Sizes " + size + " and " + secondMatrix.size() +
            " cannot be broadcast together. Supported: full match or (1,channels,1,1)."
        );
    }


    /**
     * Perform broadcast addition with another JMatrix.
     * @param secondMatrix              The JMatrix to add to this JMatrix.
     * @throws IllegalArgumentException If the JMatrixes are incompatible. Full match and (1,C,1,1) match are supported.
     * @return                          A new JMatrix representing the sum.
     */
     public JMatrix add(JMatrix secondMatrix) {
        int size = size();

        // Full element-wise addition
        if (size == secondMatrix.size()) {
            float[] result = new float[size];
            IntStream.range(0, size).parallel().forEach(i -> {
                result[i] = access(i) + secondMatrix.access(i);
            });
            return new JMatrix(result, length, channels, height, width);
        }

        // Channel-wise broadcasting: (1, C, 1, 1) over (N, C, H, W)
        if (secondMatrix.length() == 1 && secondMatrix.channels() == channels &&
            secondMatrix.height() == 1 && secondMatrix.width() == 1) {

            float[] result = new float[size];
            int channelSize = height * width;

            IntStream.range(0, length).parallel().forEach(n -> {
                for (int c = 0; c < channels; c++) {
                    float adder = secondMatrix.access(c); // one adder per channel
                    int offset = n * channels * channelSize + c * channelSize;
                    for (int i = 0; i < channelSize; i++) {
                        result[offset + i] = access(offset + i) + adder;
                    }
                }
            });

            return new JMatrix(result, length, channels, height, width);
        }

        // If matrices are not broadcastable
        throw new IllegalArgumentException(
            "Sizes " + size + " and " + secondMatrix.size() +
            " cannot be broadcast together. Supported: full match or (1,channels,1,1)."
        );
    }

    /**
     * Perform broadcast addition with another JMatrix in place.
     * @param secondMatrix              The JMatrix to add to this JMatrix.
     * @throws IllegalArgumentException If the JMatrixes are incompatible. Full match and (1,C,1,1) match are supported.
     */
    public JMatrix addInPlace(JMatrix secondMatrix) {
        int size = size();

        // Full element-wise subtraction
        if (size == secondMatrix.size()) {
            IntStream.range(0, size).parallel().forEach(i -> {
                matrix[i] = access(i) + secondMatrix.access(i);
            });
            return this; // For chaining
        }

        // Channel-wise broadcasting: (1, C, 1, 1) over (N, C, H, W)
        if (secondMatrix.length() == 1 && secondMatrix.channels() == channels &&
            secondMatrix.height() == 1 && secondMatrix.width() == 1) {

            int channelSize = height * width;

            IntStream.range(0, length).parallel().forEach(n -> {
                for (int c = 0; c < channels; c++) {
                    float adder = secondMatrix.access(c); // one adder per channel
                    int offset = n * channels * channelSize + c * channelSize;
                    for (int i = 0; i < channelSize; i++) {
                        matrix[offset + i] = access(offset + i) + adder;
                    }
                }
            });
            return this; // For chaining
        }

        // If matrices are not broadcastable
        throw new IllegalArgumentException(
            "Sizes " + size + " and " + secondMatrix.size() +
            " cannot be broadcast together. Supported: full match or (1,channels,1,1)."
        );
    }
    /**
     * Perform broadcast multiplication with another JMatrix.
     * @param secondMatrix The JMatrix to multiply with this JMatrix.
     * @throws IllegalArgumentException If the JMatrixes are incompatible. Full match, (1,C,1,1) match, and (N,C,1,1) match are supported.
     * @return A new JMatrix representing the product.
     */
    public JMatrix multiply(JMatrix secondMatrix) {
        int size = size();
        // Full element-wise multiplication
        if (size == secondMatrix.size()) {
            float[] result = new float[size];
            IntStream.range(0, size).parallel().forEach(i -> {
                result[i] = access(i) * secondMatrix.access(i);
            });
            return new JMatrix(result, length, channels, height, width);
        }
        
        // Channel-wise broadcasting: (1, C, 1, 1) over (N, C, H, W)
        if (secondMatrix.length() == 1 && secondMatrix.channels() == channels &&
            secondMatrix.height() == 1 && secondMatrix.width() == 1) {
            float[] result = new float[size];
            int channelSize = height * width;
            IntStream.range(0, length).parallel().forEach(n -> {
                for (int c = 0; c < channels; c++) {
                    float multiplier = secondMatrix.access(c); // one multiplier per channel
                    int offset = n * channels * channelSize + c * channelSize;
                    for (int i = 0; i < channelSize; i++) {
                        result[offset + i] = access(offset + i) * multiplier;
                    }
                }
            });
            return new JMatrix(result, length, channels, height, width);
        }
        
        // Sample-channel-wise broadcasting: (N, C, 1, 1) over (N, C, H, W)
        // Each sample has its own set of channel multipliers
        if (secondMatrix.length() == length && secondMatrix.channels() == channels &&
            secondMatrix.height() == 1 && secondMatrix.width() == 1) {
            float[] result = new float[size];
            int channelSize = height * width;
            IntStream.range(0, length).parallel().forEach(n -> {
                for (int c = 0; c < channels; c++) {
                    // Get multiplier for this specific sample and channel
                    float multiplier = secondMatrix.access(n * channels + c);
                    int offset = n * channels * channelSize + c * channelSize;
                    for (int i = 0; i < channelSize; i++) {
                        result[offset + i] = access(offset + i) * multiplier;
                    }
                }
            });
            return new JMatrix(result, length, channels, height, width);
        }
        
        // If matrices are not broadcastable
        throw new IllegalArgumentException(
            "Sizes " + size + " and " + secondMatrix.size() +
            " cannot be broadcast together. Supported: full match, (1,C,1,1), or (N,C,1,1)."
        );
    }

    /**
     * Perform broadcast multiplication with another JMatrix in place.
     * @param secondMatrix              The JMatrix to multiply this JMatrix with.
     * @throws IllegalArgumentException If the JMatrixes are incompatible. Full match and (1,C,1,1) match are supported.
     */
    public JMatrix multiplyInPlace(JMatrix secondMatrix) {
        int size = size();

        // Full element-wise subtraction
        if (size == secondMatrix.size()) {
            IntStream.range(0, size).parallel().forEach(i -> {
                matrix[i] = access(i) * secondMatrix.access(i);
            });
            return this; // For chaining
        }

        // Channel-wise broadcasting: (1, C, 1, 1) over (N, C, H, W)
        if (secondMatrix.length() == 1 && secondMatrix.channels() == channels &&
            secondMatrix.height() == 1 && secondMatrix.width() == 1) {

            int channelSize = height * width;

            IntStream.range(0, length).parallel().forEach(n -> {
                for (int c = 0; c < channels; c++) {
                    float multiplier = secondMatrix.access(c); // one multiplier per channel
                    int offset = n * channels * channelSize + c * channelSize;
                    for (int i = 0; i < channelSize; i++) {
                        matrix[offset + i] = access(offset + i) * multiplier;
                    }
                }
            });
            return this; // For chaining
        }

        // If matrices are not broadcastable
        throw new IllegalArgumentException(
            "Sizes " + size + " and " + secondMatrix.size() +
            " cannot be broadcast together. Supported: full match or (1,channels,1,1)."
        );
    }

    /**
     * Perform broadcast division with another JMatrix.
     * @param secondMatrix              The JMatrix to divide this JMatrix by.
     * @throws IllegalArgumentException If the JMatrixes are incompatible. Full match and (1,C,1,1) match are supported.
     * @return                          A new JMatrix representing the dividend.
     */
    public JMatrix divide(JMatrix secondMatrix) {
        int size = size();

        // Full element-wise subtraction
        if (size == secondMatrix.size()) {
            float[] result = new float[size];
            IntStream.range(0, size).parallel().forEach(i -> {
                result[i] = access(i) / secondMatrix.access(i);
            });
            return new JMatrix(result, length, channels, height, width);
        }

        // Channel-wise broadcasting: (1, C, 1, 1) over (N, C, H, W)
        if (secondMatrix.length() == 1 && secondMatrix.channels() == channels &&
            secondMatrix.height() == 1 && secondMatrix.width() == 1) {

            float[] result = new float[size];
            int channelSize = height * width;

            IntStream.range(0, length).parallel().forEach(n -> {
                for (int c = 0; c < channels; c++) {
                    float divisor = secondMatrix.access(c); // one divisor per channel
                    int offset = n * channels * channelSize + c * channelSize;
                    for (int i = 0; i < channelSize; i++) {
                        result[offset + i] = access(offset + i) / divisor;
                    }
                }
            });

            return new JMatrix(result, length, channels, height, width);
        }

        // If matrices are not broadcastable
        throw new IllegalArgumentException(
            "Sizes " + size + " and " + secondMatrix.size() +
            " cannot be broadcast together. Supported: full match or (1,channels,1,1)."
        );
    }

    /**
     * Perform broadcast division with another JMatrix in place.
     * @param secondMatrix              The JMatrix to divide this JMatrix by.
     * @throws IllegalArgumentException If the JMatrixes are incompatible. Full match and (1,C,1,1) match are supported.
     */
    public JMatrix divideInPlace(JMatrix secondMatrix) {
        int size = size();

        // Full element-wise subtraction
        if (size == secondMatrix.size()) {
            IntStream.range(0, size).parallel().forEach(i -> {
                matrix[i] = access(i) / secondMatrix.access(i);
            });
            return this; // For chaining
        }

        // Channel-wise broadcasting: (1, C, 1, 1) over (N, C, H, W)
        if (secondMatrix.length() == 1 && secondMatrix.channels() == channels &&
            secondMatrix.height() == 1 && secondMatrix.width() == 1) {

            int channelSize = height * width;

            IntStream.range(0, length).parallel().forEach(n -> {
                for (int c = 0; c < channels; c++) {
                    float divisor = secondMatrix.access(c); // one divisor per channel
                    int offset = n * channels * channelSize + c * channelSize;
                    for (int i = 0; i < channelSize; i++) {
                        matrix[offset + i] = access(offset + i) / divisor;
                    }
                }
            });
            return this; // For chaining
        }

        // If matrices are not broadcastable
        throw new IllegalArgumentException(
            "Sizes " + size + " and " + secondMatrix.size() +
            " cannot be broadcast together. Supported: full match or (1,channels,1,1)."
        );
    }

    /**
     * Subtract a scalar from this JMatrix.
     * @param scalar              The scalar value to subtract from this JMatrix.
     * @return                    A new JMatrix with the changes applied.
     */
    public JMatrix subtract(double scalar) {
        int size = size();
        float[] result = new float[size];

        float fScalar = (float)scalar;

        IntStream.range(0, size).parallel().forEach(i -> {
            result[i] = access(i) - fScalar;
        });

        return new JMatrix(result, length, channels, height, width);
    }

    /**
     * Subtract a scalar from this JMatrix in place.
     * @param scalar              The scalar value to subtract from this JMatrix.
     */
    public JMatrix subtractInPlace(double scalar) {
        int size = size();

        float fScalar = (float)scalar;

        IntStream.range(0, size).parallel().forEach(i -> {
            matrix[i] = access(i) - fScalar;
        });

        return this; // For chaining
    }

    /**
     * Add a scalar to this JMatrix.
     * @param scalar              The scalar value to add to this JMatrix.
     * @return                    A new JMatrix with the changes applied.
     */
    public JMatrix add(double scalar) {
        int size = size();
        float[] result = new float[size];

        float fScalar = (float)scalar;

        IntStream.range(0, size).parallel().forEach(i -> {
            result[i] = access(i) + fScalar;
        });

        return new JMatrix(result, length, channels, height, width);
    }

    /**
     * Add a scalar to this JMatrix in place.
     * @param scalar              The scalar value to add to this JMatrix.
     */
    public JMatrix addInPlace(double scalar) {
        int size = size();

        float fScalar = (float)scalar;

        IntStream.range(0, size).parallel().forEach(i -> {
            matrix[i] = access(i) + fScalar;
        });

        return this; // For chaining
    }
    /**
     * Multiply a scalar with this JMatrix.
     * @param scalar              The scalar value to muliply with this JMatrix.
     * @return                    A new JMatrix with the changes applied.
     */
    public JMatrix multiply(double scalar) {
        int size = size();
        float[] result = new float[size];

        float fScalar = (float)scalar;

        IntStream.range(0, size).parallel().forEach(i -> {
            result[i] = access(i) * fScalar;
        });

        return new JMatrix(result, length, channels, height, width);
    }

    /**
     * Multiply a scalar with this JMatrix in place.
     * @param scalar              The scalar value to subtract from this JMatrix.
     */
    public JMatrix multiplyInPlace(double scalar) {
        int size = size();

        float fScalar = (float)scalar;

        IntStream.range(0, size).parallel().forEach(i -> {
            matrix[i] = access(i) * fScalar;
        });
        return this; // For chaining
    }

     /**
     * Divide this JMatrix by a scalar.
     * @param scalar              The scalar value to divide this JMatrix by.
     * @return                    A new JMatrix with the changes applied.
     */
    public JMatrix divide(double scalar) {
        int size = size();
        float[] result = new float[size];

        float fScalar = (float)scalar;

        IntStream.range(0, size).parallel().forEach(i -> {
            result[i] = access(i) / fScalar;
        });

        return new JMatrix(result, length, channels, height, width);
    }

    /**
     * Divide this matrix by a scalar in place.
     * @param scalar              The scalar value to divide this JMatrix by.
     */
    public JMatrix divideInPlace(double scalar) {
        int size = size();

        float fScalar = (float)scalar;

        IntStream.range(0, size).parallel().forEach(i -> {
            matrix[i] = access(i) / fScalar;
        });
        
        return this; // For chaining
    }
}
