package jflow.data;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

public class JMatrix {
    private float[] matrix;
    private int length;
    private int channels;
    private int height;
    private int width;
    private String name = null;
    private static long seed = System.nanoTime();
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
    

    private JMatrix(float[] matrix, int length, int channels, int height, int width) {
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

    private JMatrix(int length, int channels, int height, int width) {
        this(new float[length * channels * height * width], 
            length, channels, height, width);
    }

    private static void validateShape(int[] shape) {
        if (shape.length != 4) {
            throw new IllegalArgumentException("Invalid shape. Only length 4 is permitted.");
        }
    }

    /**
     * Initialize a new JMatrix with default values of zero.
     * JMatrix is in (N, C, H, W) format.
     * @param length                the batch dimension.
     * @param channels              the channel dimension.
     * @param height                the height dimension.
     * @param width                 the width dimension.
     */
    public static JMatrix zeros(int length, int channels, int height, int width) {
        return new JMatrix(length, channels, height, width);
    }

    /**
     * Initialize a new JMatrix with default values of zero.
     * JMatrix is in (N, C, H, W) format.
     * @param shape                the desired shape (N, channels, height, width).
     * @throws IllegalArgumentException if the length of <b>shape</b> is not four.
     */
    public static JMatrix zeros(int[] shape) {
        validateShape(shape);
        return new JMatrix(shape[0], shape[1], shape[2], shape[3]);
    }

    /**
     * Initialize a new JMatrix with default values of one.
     * JMatrix is in (N, C, H, W) format.
     * @param length                the batch dimension.
     * @param channels              the channel dimension.
     * @param height                the height dimension.
     * @param width                 the width dimension.
     */
    public static JMatrix ones(int length, int channels, int height, int width) {
        return new JMatrix(length, channels, height, width).fill(1.0);
    }

    /**
     * Initialize a new JMatrix with default values of one.
     * JMatrix is in (N, C, H, W) format.
     * @param shape                the desired shape (N, channels, height, width).
     * @throws IllegalArgumentException if the length of <b>shape</b> is not four.
     */
    public static JMatrix ones(int[] shape) {
        validateShape(shape);
        return new JMatrix(shape[0], shape[1], shape[2], shape[3]).fill(1.0);
    }

    /**
     * Wrap an array in a new JMatrix.
     * JMatrix is in (N, C, H, W) format.
     * @param values                the values to wrap in this JMatrix.
     * @param length                the batch dimension.
     * @param channels              the channel dimension.
     * @param height                the height dimension.
     * @param width                 the width dimension.
     * @throws IllegalArgumentException if the dimensional information 
     * doesn't correspond to the length of <b>values</b>.
     */
    public static JMatrix wrap(float[] values, int length, int channels, int height, int width) {
        int reportedLength = length * channels * height * width;
        if (values.length != reportedLength) {
            throw new IllegalArgumentException(
                String.format(
                    "Invalid dimensional information. " + 
                    "Length of values: %d. Reported length: %d",
                    values.length, reportedLength
                )
            );
        }
        return new JMatrix(values, length, channels, height, width);
    }

    /**
     * Wrap an array in a new JMatrix.
     * JMatrix is in (N, C, H, W) format.
     * @param values           the values to wrap in this JMatrix.
     * @param shape            the desired shape (N, channels, height, width).
     * @throws IllegalArgumentException if the length of <b>shape</b> is not four, 
     * or if the dimensional information doesn't correspond to the length of <b>values</b>.
     */
    public static JMatrix wrap(float[] values, int[] shape) {
        validateShape(shape);
        return wrap(values, shape[0], shape[1], shape[2], shape[3]);
    }

    /**
     * Create a JMatrix with uniform distribution in a specified range.
     * @param length          the N dimension of the JMatrix.
     * @param channels        the channel dimension of the JMatrix.
     * @param height          the height dimension of the JMatrix.
     * @param width           the width dimension of the JMatrix.
     * @param min             the minimum value, inclusive.
     * @param max             the maximum value, exclusive.
     */
    public static JMatrix uniform(
        int length,
        int channels,
        int height,
        int width,
        double min,
        double max
    ) {
        int size = length * channels * height * width;
        float[] noise = new float[size];
        double range = max - min;

        int chunkSize = 1024;

        IntStream.range(0, (size + chunkSize - 1) / chunkSize).parallel().forEach(chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, size);

            // Hash-based unique seed per chunk
            long localSeed = JMatrix.seed ^ Long.rotateLeft(chunk * 0x9E3779B97F4A7C15L, 17);
            Random rng = new Random(localSeed);

            for (int i = start; i < end; i ++) {
                // Uniform distribution: [min, max)
                noise[i] = (float)(rng.nextDouble() * range + min);
            }
        });

        return JMatrix.wrap(noise, length, channels, height, width);
    }

    /**
     * Create a JMatrix with uniform distribution in a specified range.
     * @param shape         the shape of the JMatrix.
     * @param min           the minimum value, inclusive.
     * @param max           the maximum value, exclusive.
     */
    public static JMatrix uniform(int[] shape, double min, double max) {
        validateShape(shape);
        return JMatrix.uniform(shape[0], shape[1], shape[2], shape[3], min, max);
    }

    /**
     * Create a JMatrix with normally distributed values.
     * @param length        the N dimension of the JMatrix.
     * @param channels      the channel dimension of the JMatrix.
     * @param height        the height dimension of the JMatrix.
     * @param width         the width dimension of the JMatrix.
     * @param mean          the mean of the normal distribution.
     * @param stddev        the standard deviation of the normal distribution.
     * @return              a JMatrix initialized with normal distribution.
     */
    public static JMatrix normal(
        int length,
        int channels,
        int height,
        int width,
        double mean,
        double stddev
    ) {
        int size = length * channels * height * width;
        float[] noise = new float[size];

        int chunkSize = 1024;

        IntStream.range(0, (size + chunkSize - 1) / chunkSize).parallel().forEach(chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, size);

            // Hash-based unique seed per chunk
            long localSeed = JMatrix.seed ^ Long.rotateLeft(chunk * 0x9E3779B97F4A7C15L, 17);
            Random rng = new Random(localSeed);

            for (int i = start; i < end; i++) {
                // Box-Muller transform
                double u1 = rng.nextDouble();
                double u2 = rng.nextDouble();

                double z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);

                noise[i] = (float)(z0 * stddev + mean);
            }
        });

        return JMatrix.wrap(noise, length, channels, height, width);
    }

    /**
     * Create a JMatrix with normally distributed values.
     * @param shape         the shape of the JMatrix.
     * @param mean          the mean of the normal distribution.
     * @param stddev        the standard deviation of the normal distribution.
     */
    public static JMatrix normal(int[] shape, double mean, double stddev) {
        validateShape(shape);
        return JMatrix.normal(shape[0], shape[1], shape[2], shape[3], mean, stddev);
    }

    /**
     * Set the JMatrix seed for reproducability.
     * @param seed the seed to use.
     */
    public static void setSeed(long seed) {
        JMatrix.seed = seed;
    }

    /**
     * Acquire the current JMatrix seed.
     */
    public static long currentSeed() {
        return JMatrix.seed;
    }

    /**
     * Access the wrapped array.
     */
    public float[] unwrap() {
        return matrix;
    }

    /**
     * Name this JMatrix.
     * @param name          The name to assign.
     */
    public JMatrix label(String name) {
        this.name = name;
        return this;
    }

    /**
     * Access the name of this JMatrix. 
     * @return <b> name </b> if it's set. <li> otherwise null. </li>
     */
    public String label() {
        return name;
    }

    /**
     * The shape of the JMatrix.
     * @return {length, channels, height, width} in an int[4].
     */
    public int[] shape() {
        return new int[]{length, channels, height, width};
    }

    /**
     * The size of the JMatrix along a specified axis.
     */
    public int shape(int axis) {
        if (axis < 0 || axis > 3) {
            throw new IllegalArgumentException(
                "Invalid axis: " + axis
                + ". Options: (0,1,2,3)"
            );
        }
        return shape()[axis];
    }

    /**
     * The shape of the JMatrix visually organized as a String.
     */
    public String shapeAsString() {
        return "(" + length + "," + channels + "," + height + "," + width + ")";
    }

    /**
     * The shape of the JMatrix visually organized as a String, with unused trailing dimensions excluded.
     */
    public String simpleShapeAsString() {
        if (width == 1 && height == 1 && channels == 1) {
            return "(" + length + ",)";
        }
        if (width == 1 && height == 1) {
            return "(" + length + "," + channels + ")";
        }
        if (width == 1) {
            return "(" + length + "," + channels + "," + height + ")";
        }
        return shapeAsString();
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
    public JMatrix setMatrix(float[] matrix) {
        if (matrix.length != size()) {
            throw new IllegalArgumentException(
                "Sizes must match. Original: " 
                + size() + " New: " + matrix.length
            );
        }
        this.matrix = matrix;
        return this;
    }

    /**
     * Set the wrapped array to a new value. Resize allowed.
     * @param matrix                            The new array to replace the original. 
     * @param shape                             The 4D shape of the new matrix.
     * @exception IllegalArgumentException      if: <ul> <li>  the length of shape is not four. </li> 
     * <li> the reported number of elements is unequal to the length of the matrix. </li> </ul> 
     */
    public JMatrix setMatrix(float[] matrix, int[] shape) {
        validateShape(shape);
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
        return this;
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
        float[] internalValues = values.unwrap();
        if (internalValues.length != itemSize) {
            throw new IllegalArgumentException("Unexpected length: " + internalValues.length + 
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
            values[i] = matrix[i];
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
            values[i] = matrix[index + i];
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
        return matrix[index];
    }

    /**
     * Get an individual element.
     * @param lengthIndex               The batch index of the item to get.
     * @param channelIndex              The channel index of the item to get.
     * @param heightIndex               The height index of the item to get.
     * @param widthIndex                The width index of the item to get.
     */
    public float get(int lengthIndex, int channelIndex, int heightIndex, int widthIndex) {
        return matrix[lengthIndex * channels * height * 
            width + channelIndex * height * width + 
            heightIndex * width + widthIndex];
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

        return JMatrix.wrap(result, length, 1, 1, 1);
    }

    /**
     * Get a channels * height * width element wrapped in a JMatrix.
     * @param lengthIndex The index along the batch dimension.
     */
     public JMatrix getItem(int lengthIndex) {
        int sliceSize = channels * height * width;
        int startIdx = lengthIndex * sliceSize;  
        float[] slice = new float[sliceSize];
        System.arraycopy(matrix, startIdx, slice, 0, sliceSize);
        return JMatrix.wrap(slice, 1, channels, height, width);
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
     * Set a height * width element.
     * @param lengthIndex The index along the batch dimension.
     * @param channelIndex The index along the channel dimension.
     * @param values The values to copy into the specified region.
     * @throws IllegalArgumentException If the number of values is
     * unequal to the height * width of the JMatrix.
     */
    public void setImage(int lengthIndex, int channelIndex, float[] values) {
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
     * @throws IllegalArgumentException if the provided dimensions don't match the length of the matrix.
     * @return                          A new JMatrix with the changes applied.
     */
    public JMatrix reshape(int newLength, int newChannels, int newHeight, int newWidth) {
        long numItems = size();
        long newNumItems = (long)newLength * newChannels * newHeight * newWidth;

        if (numItems != newNumItems) {
            throw new IllegalArgumentException(
                "Invalid reshape: total elements must match. Original: " 
                + numItems + " Reshape: " + newNumItems);
        }

        return JMatrix.wrap(matrix, newLength, newChannels, newHeight, newWidth);
    }

     /**
     * Alter the dimensional information stored in the JMatrix.
     * @param newLength                 The new shape
     * @return                          A new JMatrix with the changes applied.
     * @throws IllegalArgumentException If the length of shape is not 4,
     * or the provided dimensions don't match the length of the matrix.
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
     * Add to an item with 1D indexing.
     * @param index             The 1D index to alter.
     * @param value             The value to add to the existing item, cast to a float.            
     */
    public void addTo(int index, double value) {
        matrix[index] += (float)value;
    }

    /**
     * Set an item with 4D indexing.
     * @param lengthIndex               The batch index of the item to set.
     * @param channelIndex              The channel index of the item to set.
     * @param heightIndex               The height index of the item to set.
     * @param widthIndex                The width index of the item to set.
     * @param value                     The value to set, cast to a float.            
     */
    public void set(int lengthIndex, int channelIndex, int heightIndex, int widthIndex, double value) {
        matrix[lengthIndex * channels * height * 
        width + channelIndex * height * width + 
        heightIndex * width + widthIndex] = (float)value;
    }

    /**
     * Add to an item with 4D indexing.
     * @param lengthIndex               The batch index of the item to add to.
     * @param channelIndex              The channel index of the item to add to.
     * @param heightIndex               The height index of the item to add to.
     * @param widthIndex                The width index of the item to add to.
     * @param value                     The value to add to the existing item, cast to a float.            
     */
    public void addTo(int lengthIndex, int channelIndex, int heightIndex, int widthIndex, double value) {
        matrix[lengthIndex * channels * height * 
        width + channelIndex * height * width + 
        heightIndex * width + widthIndex] += (float)value;
    }

// Statistics

    /**
     * The max value in the JMatrix.             
     */
    public double max() {
        return Statistics.max(matrix);
    }
    /**
     * The max absolute value in the JMatrix.             
     */
    public double absMax() {
        return Statistics.absMax(matrix);
    }

    /**
     * The mean of all values in the JMatrix.
     */
    public double mean() {
        return Statistics.mean(matrix);

    }

    /**
     * The mean of the absolute value of all items in the JMatrix.
     */
    public double absMean() {
        return Statistics.absMean(matrix);

    }

    /**
     * Finds the index of max values along a given axis.
     * @param axis              The specified axis in the range [0,3] inclusive.
     * @return                  An array containing the indices of max values along the given axis
     */
    public int[] argmax(int axis) {
        return Statistics.argmax(matrix, axis, shape());
    }
    
    /**
     * The sum of all values in the JMatrix.
     */
    public double sum() {
        return Statistics.sum(matrix);
    }

    /**
     * Sums the elements of this JMatrix along specified axes, reducing those dimensions to 1.
     * 
     * @param axis The axis to preserve (0 = batch only, 1 = batch + channel, 2 = batch + channel + height)
     *     <ul>  <li>  axis = 0: Sum over c * h * w, result shape: (length, 1, 1, 1) </li>
     *           <li>  axis = 1: Sum over h * w, result shape: (length, channels, 1, 1) </li>
     *           <li>  axis = 2: Sum over w, result shape: (length, channels, height, 1) </li>
     *     </ul>
     * @return A new JMatrix with the summed values
     */
    public JMatrix sum(int axis) {
        // Calculate the sum
        float[] result = Statistics.sum(
            matrix,
            axis,
            shape()
        );
        // Determine the wrapper
        switch (axis) {
            case 0: // Sum over channels, height, width - preserve only batch dimension
                return JMatrix.wrap(result, length, 1, 1, 1);
                
            case 1: // Sum over height, width - preserve batch and channel dimensions
                return JMatrix.wrap(result, length, channels, 1, 1);
                
            case 2: // Sum over width - preserve batch, channel, and height dimensions  
                return JMatrix.wrap(result, length, channels, height, 1);

            default:
                throw new IllegalArgumentException("Axis must be 0, 1, or 2. Got: " + axis);
        }
    }
   
    /**
     * The l1 norm is calculated as: <p>
     * For every x -> <p>
     * Take the abosolute value and add it to the sum.
     * 
     */
    public double l1Norm() {
        return Statistics.l1Norm(matrix);
    }

    /**
     * The l2 norm is calculated as: <p>
     * For every x -> <p>
     * - Raise x ^ 2. <p>
     * - Add it to the sum. <p>
     * Finally, square the sum.
     */
    public double l2Norm() {
        return Statistics.l2Norm(matrix);
    }

    /**
     * Count the number of items in the JMatrix of a certain value.
     * @param value             The value to count instances of.
     */
    public int count(float value) {
        return Statistics.count(matrix, value);
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
     * Perform softmax along a given axis. Avoids overflow.
     * @param axis the axis to apply softmax to: <p>
     * <ul> <li> 0 = across batch dimension 
     * <li> 1 = across channel dimension
     * <li> 2 = across height dimension
     * <li> 3 = across width dimension </ul>
     * @return A new JMatrix with softmax applied along the specified axis.
     */
    public JMatrix softmax(int axis) {
        float[] result = Statistics.softmax(matrix, axis, shape(), THREAD_POOL);
        
        return JMatrix.wrap(result, shape());
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
        float[] result = Statistics.logSoftmax(matrix, axis, shape(), THREAD_POOL);
        return JMatrix.wrap(result, shape());
    }
    
    /**
     * Creates a one-hot encoded representation of token indices.
     *
     * @param indices A JMatrix containing integer token indices
     * @param numLabels The size of the vocabulary (number of possible token values)
     * @param smoothing Label smoothing factor (0.0 for no smoothing, typical values 0.1-0.3)
     * @return A new JMatrix with one-hot encoding
     */
    public static JMatrix oneHot(JMatrix indices, int numLabels, float smoothing) {
        int batchSize = indices.shape()[0];
        int seqLen = indices.shape()[1];
        
        float smoothValue = smoothing / numLabels;
        float hotValue = 1.0f - smoothing + smoothValue;
        
        JMatrix oneHotMatrix = JMatrix.zeros(batchSize, seqLen, numLabels, 1);
        oneHotMatrix.fill(smoothValue);

        for (int batch = 0; batch < batchSize; batch++) {
            for (int seq = 0; seq < seqLen; seq++) {
                int tokenIndex = (int)indices.get(batch, seq, 0, 0);

                if (tokenIndex >= 0 && tokenIndex < numLabels) {
                    oneHotMatrix.set(batch, seq, tokenIndex, 0, hotValue);
                }
            }
        }
        
        return oneHotMatrix;
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
        int batch = tokenIDs.shape(0);
        int seqLen = tokenIDs.shape(1);
        JMatrix positions = JMatrix.zeros(batch, seqLen, 1, 1);
    
        for (int b = 0; b < batch; b++) {
            for (int t = 0; t < seqLen; t++) {
                positions.set(b, t, 0, 0, t);
            }
        }
    
        return positions;
    }

    /**
     * Perform a 2D transposition, treating the matrix as rows (N)
     * and cols (C * H * W).
     * <p>
     * Equivalent to {@code permute(1,0)}
     * For higher-dimensional tensors, use {@code permute(int...)}
     * 
     * @return A new JMatrix with shape (cols, rows, 1, 1).
     */
    public JMatrix T() {
        
        int oldHeight = length;
        int oldWidth = channels * height * width;
        int newHeight = oldWidth;
        int newWidth = oldHeight;


        float[] result = new float[size()];

        MatrixOps.transpose4DMatrixByDims(
            matrix, 
            oldHeight, oldWidth, 1, 1,
            1, 0, 2, 3,
            result
        );

        // Assign all features to channels (C) for 2D use case
        return JMatrix.wrap(result, newHeight, newWidth, 1, 1);
    }

    /**
     * Perform a 4D permutation.
     * @param axis1                         The dimension to use as the batch dimension.
     * @param axis2                         The dimension to use as the channel dimension.
     * @param axis3                         The dimension to use as the height dimension.
     * @param axis4                         The dimension to use as the width dimension.
     * @throws IllegalArgumentException     If axis values are not a permuation of (0, 1, 2, 3).
     */
    public JMatrix permute(int axis1, int axis2, int axis3, int axis4) {
        float[] result = new float[size()];
        MatrixOps.transpose4DMatrixByDims(
            matrix, 
            length, channels, height, width, 
            axis1, axis2, axis3, axis4, 
            result
        );

        int[] dims = shape();
        int newLength = dims[axis1];
        int newChannels = dims[axis2];
        int newHeight = dims[axis3];
        int newWidth = dims[axis4];
        
        return JMatrix.wrap(result, newLength, newChannels, newHeight, newWidth);
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
     * An exact copy of this JMatrix.
     */
    public JMatrix copy() {
        JMatrix copy = JMatrix.wrap(matrix.clone(), shape());
        if (name != null) {
            copy.label(name);
        }
        return copy;
    }
    /**
     * Returns a new empty JMatrix with the same dimensions as this JMatrix.
     */
    public JMatrix zerosLike() {
        return JMatrix.zeros(shape());
    }

    /**
     * Clips all values in the JMatrix to a desired range.
     * @param min               The minimum allowed value, cast to a float.
     * @param max               The maximum allowed value, cast to a float.
     * @return                  This JMatrix.
     */
    public JMatrix clip(double min, double max) {
        float fMin = (float)min;
        float fMax = (float)max;
        IntStream.range(0, matrix.length).parallel().forEach(i -> {
            matrix[i] = Math.max(fMin, Math.min(fMax, matrix[i]));
        });
        return this;
    }

    /**
     * Fills the JMatrix with a certain value.
     * @param fillValue             The value to assign to all items of the JMatrix.
     * @return                      This JMatrix.
     */
    public JMatrix fill(double fillValue) {
        float valueF = (float)fillValue;
        int size = size();
        IntStream.range(0, size).parallel().forEach(i -> {
            matrix[i] = valueF;
        });
        return this;
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
                String.format(
                    "Matrix multiplication not possible for " +
                    "arrays with shape: (%d,%d) and (%d,%d)",
                    m, k, secondMatrix.length(), n
                )
            );
        }
        
        float[] matrixA = matrix;
        float[] matrixB = secondMatrix.unwrap();
        
        // Use simple algorithm for small matrices
        if (m < cutoffSize && n < cutoffSize && k < cutoffSize) {
            float[] result = OptimizedMatmul.simpleMatmul(matrixA, matrixB, scale, m, n, k);
            return JMatrix.wrap(result, m, n, 1, 1);
        }

        float[] result = OptimizedMatmul.matmul(matrixA, matrixB, m, n, k, scale, 
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, THREAD_POOL);
        
        return JMatrix.wrap(result, m, n, 1, 1);
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
                String.format(
                    "Batch matrix multiplication not possible for " +
                    "tensors with shapes: (%d,%d,%d) and (%d,%d,%d)",
                    batchSize, inputChannels, flatSpatialDim, 
                    secondMatrix.length(), outputChannels, outFlatSpatialDim
                )
            );
        }

        float[] result = OptimizedMatmul.batchMatmul(
            matrix, secondMatrix.unwrap(), 
            length, inputChannels, outFlatSpatialDim, outputChannels, 
            scale, 
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, 
            THREAD_POOL
        );

        return JMatrix.wrap(result, batchSize, channels, outFlatSpatialDim, 1);
    }

    /**
     * Take the reciprocal (1 / x) of every item in the JMatrix.
     * @return A new JMatrix with the changes applied.
     */
    public JMatrix reciprocal() {
        float[] result = new float[size()];
        Arithmetic.broadcastReciprocal(matrix, result);

        return JMatrix.wrap(result, shape());
    }

    /**
     * Take the sqrt of every item in the JMatrix.
     * @return A new JMatrix with the changes applied.
     */
    public JMatrix sqrt() {
        float[] result = new float[size()];
        Arithmetic.broadcastSqrt(matrix, result);
       
        return JMatrix.wrap(result, shape());
    }

    /**
     * Perform broadcast additino with another JMatrix.
     * Full match, (N,1,1,1) broadcast, (1,C,1,1) broadcast, 
     * and (N,C,1,1) broadcast are supported.
     * @param secondMatrix The JMatrix to add to this JMatrix.
     * @throws IllegalArgumentException If the JMatrixes are incompatible. 
     * 
     * @return A new JMatrix representing the sum.
     */
     public JMatrix add(JMatrix secondMatrix) {
        return applyBroadcastArithmetic(secondMatrix, 'a', false);
    }

    /**
     * Perform broadcast addition in-place with another JMatrix.
     * Full match, (N,1,1,1) broadcast, (1,C,1,1) broadcast, 
     * and (N,C,1,1) broadcast are supported.
     * @param secondMatrix The JMatrix to add to this JMatrix.
     * @throws IllegalArgumentException If the JMatrixes are incompatible. 
     * 
     * @return this JMatrix.
     */
    public JMatrix addInPlace(JMatrix secondMatrix) {
        return applyBroadcastArithmetic(secondMatrix, 'a', true);
    }

    /**
     * Add a scalar to this JMatrix.
     * @param scalar              The scalar value to add to this JMatrix.
     * @return                    A new JMatrix with the changes applied.
     */
    public JMatrix add(double scalar) {
        float[] result = new float[size()];
        Arithmetic.scalarAdd(matrix, (float)scalar, result);

        return JMatrix.wrap(result, shape());
    }

    /**
     * Add a scalar to this JMatrix in place.
     * @param scalar              The scalar value to add to this JMatrix.
     * @return                    this JMatrix.
     */
    public JMatrix addInPlace(double scalar) {
        // Write results onto matrix
        Arithmetic.scalarAdd(matrix, (float)scalar, matrix);
        return this;
    }


    /**
     * Perform broadcast subtraction with another JMatrix.
     * Full match, (N,1,1,1) broadcast, (1,C,1,1) broadcast, 
     * and (N,C,1,1) broadcast are supported.
     * @param secondMatrix The JMatrix to subtract from this JMatrix.
     * @throws IllegalArgumentException If the JMatrixes are incompatible. 
     * 
     * @return A new JMatrix representing the difference.
     */
    public JMatrix subtract(JMatrix secondMatrix) {
        return applyBroadcastArithmetic(secondMatrix, 's', false);

    }

    /**
     * Perform broadcast subtraction in-place with another JMatrix.
     * Full match, (N,1,1,1) broadcast, (1,C,1,1) broadcast, 
     * and (N,C,1,1) broadcast are supported.
     * @param secondMatrix The JMatrix to subtract from this JMatrix.
     * @throws IllegalArgumentException If the JMatrixes are incompatible. 
     * 
     * @return this JMatrix.
     */
    public JMatrix subtractInPlace(JMatrix secondMatrix) {
        return applyBroadcastArithmetic(secondMatrix, 's', true);
    }

    /**
     * Subtract a scalar from this JMatrix.
     * @param scalar              The scalar value to subtract from this JMatrix.
     * @return                    A new JMatrix with the changes applied.
     */
    public JMatrix subtract(double scalar) {
        float[] result = new float[size()];
        Arithmetic.scalarSubtract(matrix, (float)scalar, result);

        return JMatrix.wrap(result, shape());
    }

    /**
     * Subtract a scalar from this JMatrix in place.
     * @param scalar              The scalar value to subtract from this JMatrix.
     * @return                    this JMatrix.
     */
    public JMatrix subtractInPlace(double scalar) {
        // Write results onto matrix
        Arithmetic.scalarSubtract(matrix, (float)scalar, matrix);
        return this;
    }

    /**
     * Perform broadcast multiplication with another JMatrix.
     * Full match, (N,1,1,1) broadcast, (1,C,1,1) broadcast, 
     * and (N,C,1,1) broadcast are supported.
     * @param secondMatrix The JMatrix to multiply with this JMatrix.
     * @throws IllegalArgumentException If the JMatrixes are incompatible. 
     * 
     * @return A new JMatrix representing the product.
     */
    public JMatrix multiply(JMatrix secondMatrix) {
        return applyBroadcastArithmetic(secondMatrix, 'm', false);
    }

    /**
     * Perform broadcast multiplication in-place with another JMatrix.
     * Full match, (N,1,1,1) broadcast, (1,C,1,1) broadcast, 
     * and (N,C,1,1) broadcast are supported.
     * @param secondMatrix The JMatrix to multiply with this JMatrix.
     * @throws IllegalArgumentException If the JMatrixes are incompatible. 
     * 
     * @return this JMatrix.
     */
    public JMatrix multiplyInPlace(JMatrix secondMatrix) {
        return applyBroadcastArithmetic(secondMatrix, 'm', true);
    }

    /**
     * Multiply a scalar with this JMatrix.
     * @param scalar              The scalar value to muliply with this JMatrix.
     * @return                    A new JMatrix with the changes applied.
     */
    public JMatrix multiply(double scalar) {
        float[] result = new float[size()];
        Arithmetic.scalarMultiply(matrix, (float)scalar, result);

        return JMatrix.wrap(result, shape());
    }

    /**
     * Multiply a scalar with this JMatrix in place.
     * @param scalar              The scalar value to multiply with this JMatrix.
     * @return                    this JMatrix.
     */
    public JMatrix multiplyInPlace(double scalar) {
        // Write results onto matrix
        Arithmetic.scalarMultiply(matrix, (float)scalar, matrix);
        return this;
    }

    /**
     * Perform broadcast division with another JMatrix.
     * Full match, (N,1,1,1) broadcast, (1,C,1,1) broadcast, 
     * and (N,C,1,1) broadcast are supported.
     * @param secondMatrix The JMatrix to divide this JMatrix by.
     * @throws IllegalArgumentException If the JMatrixes are incompatible. 
     * 
     * @return A new JMatrix representing the quotient.
     */
    public JMatrix divide(JMatrix secondMatrix) {
        return applyBroadcastArithmetic(secondMatrix, 'd', false);
    }

    /**
     * Perform broadcast division in-place with another JMatrix.
     * Full match, (N,1,1,1) broadcast, (1,C,1,1) broadcast, 
     * and (N,C,1,1) broadcast are supported.
     * @param secondMatrix The JMatrix to divide this JMatrix by.
     * @throws IllegalArgumentException If the JMatrixes are incompatible. 
     * 
     * @return this JMatrix.
     */
    public JMatrix divideInPlace(JMatrix secondMatrix) {
        return applyBroadcastArithmetic(secondMatrix, 'd', true);
    }

    /**
     * Divide this JMatrix by a scalar.
     * @param scalar              The scalar value to divide this JMatrix by.
     * @return                    A new JMatrix with the changes applied.
     */
    public JMatrix divide(double scalar) {
        float[] result = new float[size()];
        Arithmetic.scalarDivide(matrix, (float)scalar, result);

        return JMatrix.wrap(result, shape());
    }

    /**
     * Divide this matrix by a scalar in place.
     * @param scalar              The scalar value to divide this JMatrix by.
     * @return                    this JMatrix.
     */
    public JMatrix divideInPlace(double scalar) {
        // Write results onto matrix
        Arithmetic.scalarDivide(matrix, (float)scalar, matrix);
        return this;
    }


    private JMatrix applyBroadcastArithmetic(JMatrix secondMatrix, char type, boolean inPlace) {
        int size = size();

        float[] result = (inPlace) ? matrix : new float[size];
        float[] broadcast = secondMatrix.unwrap();
        
        int[] broadcastDims;
        // Full element-wise broadcast
        if (size == secondMatrix.size()) {
            broadcastDims = new int[]{0, 1, 2, 3};
        }
        //Batch-wise broadcasting: (N, 1, 1, 1) over (N, C, H, W)
        else if (secondMatrix.length() == length && secondMatrix.channels() == 1 &&
            secondMatrix.height() == 1 && secondMatrix.width() == 1) {
            broadcastDims = new int[]{0};
        } 
        // Channel-wise broadcasting: (1, C, 1, 1) over (N, C, H, W)
        else if (secondMatrix.length() == 1 && secondMatrix.channels() == channels &&
            secondMatrix.height() == 1 && secondMatrix.width() == 1) {
            broadcastDims = new int[]{1};
        }
        // Item channel-wise broadcasting: (N, C, 1, 1) over (N, C, H, W)
        else if (secondMatrix.length() == length && secondMatrix.channels() == channels &&
            secondMatrix.height() == 1 && secondMatrix.width() == 1) {
            broadcastDims = new int[]{0, 1};
        } 
        // If matrices are not broadcastable
        else {
            throw new IllegalArgumentException(
            "Shapes " + shapeAsString() + 
            " and " + secondMatrix.shapeAsString() +
            " cannot be broadcast together. Supported:" + 
            "full match, (N,1,1,1), (1,C,1,1), or (N,C,1,1)."
        );
        }
        switch (type) {
            case 'a':
                Arithmetic.broadcastAdd(
                    matrix, 
                    broadcast, 
                    shape(), 
                    result, 
                    broadcastDims
                );
                break;
            case 's':
                Arithmetic.broadcastSubtract(
                    matrix, 
                    broadcast, 
                    shape(), 
                    result, 
                    broadcastDims
                );
                break;
            case 'm':
                Arithmetic.broadcastMultiply(
                    matrix, 
                    broadcast, 
                    shape(), 
                    result, 
                    broadcastDims
                );
                break;
            case 'd':
                Arithmetic.broadcastDivide(
                    matrix, 
                    broadcast, 
                    shape(), 
                    result, 
                    broadcastDims
                );
                break;
            default:
                throw new IllegalArgumentException(
                    "Unknown operator: " + type + 
                    ". Supported: 'a' (addition)," +
                    " 's' (subtraction), 'm' (multiplication) " + 
                    " 'd' (division)"
                );
        }
        if (inPlace) {
            return this;
        }
        return JMatrix.wrap(result, shape());
    }
}