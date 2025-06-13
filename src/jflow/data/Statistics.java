package jflow.data;

import java.lang.reflect.Array;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

class Statistics {

    protected static double mean(float[] array) {
        double sum = IntStream.range(0, array.length)
                    .parallel()
                    .mapToDouble(i -> array[i])
                    .sum();
        return sum / array.length;
    }

    protected static double absMean(float[] array) {
        double sum = IntStream.range(0, array.length)
                    .parallel()
                    .mapToDouble(i -> Math.abs(array[i]))
                    .sum();
        return sum / array.length;
    }

    protected static double max(float[] array) {
        double max = IntStream.range(0, array.length)
                      .parallel()
                      .mapToDouble(i -> array[i])
                      .max()
                      .orElseThrow();
        return max;
    }

    protected static double absMax(float[] array) {
        double max = IntStream.range(0, array.length)
                      .parallel()
                      .mapToDouble(i -> Math.abs(array[i]))
                      .max()
                      .orElseThrow();
        return max;
    }

    protected static double sum(float[] array) {
        double sum = IntStream.range(0, array.length)
                    .parallel()
                    .mapToDouble(i -> array[i])
                    .sum();
        return sum;
    }

    protected static double l1Norm(float[] array) {
        double sum = IntStream.range(0, array.length)
                    .parallel()
                    .mapToDouble(i -> Math.abs(array[i]))
                    .sum();
        return sum;
    }

    protected static double l2Norm(float[] array) {
        double sum = IntStream.range(0, array.length)
                    .parallel()
                    .mapToDouble(i -> Math.pow(array[i], 2))
                    .sum();
        return Math.sqrt(sum);
    }

    protected static int count(float[] array, float value) {
        int count = 0;
        int size = array.length;
        for (int i = 0; i < size; i++) {
            if (array[i] == value) {
                count++;
            }
        }
        return count;
    }

    /**
     * Sums the elements of 4D data in an array along specified axes, reducing those dimensions to 1.
     * @param array The array representing 4D data to sum.
     * @param axis The axis to preserve (0 = dim1 only, 1 = dim1 + dim2, 2 = dim1 + dim2 + dim3)
     *      <ul> <li>  axis = 0: Sum over dim2 * dim3 * dim4, result shape: (dim1, 1, 1, 1) </li>
     *           <li>  axis = 1: Sum over dim3 * dim4, result shape: (dim1, dim2, 1, 1) </li>
     *           <li>  axis = 2: Sum over dim4, result shape: (dim1, dim2, dim3, 1) </li>
     *      </ul> 
     * @param shape The 4D shape of the array.
     * @return A new array with the proper length and the summed values.
     */
    protected static float[] sum(
        float[] array,
        int axis,
        int[] shape
    ) {
        assertArrayLength(shape, "shape", 4);
        int dim1 = shape[0];
        int dim2 = shape[1];
        int dim3 = shape[2];
        int dim4 = shape[3];
        float[] result;
        // Precompute strides
        int stride1 = dim2 * dim3 * dim4;
        int stride2 = dim3 * dim4;
        int stride3 = dim4;
        switch (axis) {
            case 0: // Sum over dim2, dim3, dim4 - preserve only dim1 dimension
                result = new float[dim1];
                for (int d1 = 0; d1 < dim1; d1++) {
                    float sum = 0.0f;
                    for (int d2 = 0; d2 < dim2; d2++) {
                        for (int d3 = 0; d3 < dim3; d3++) {
                            for (int d4 = 0; d4 < dim4; d4++) {
                                sum += array[
                                    d1 * stride1 + 
                                    d2 * stride2 + 
                                    d3 * stride3 +
                                    d4
                                ];
                            }
                        }
                    }
                    result[d1] = sum;
                }
                break;
                
            case 1: // Sum over dim3, dim4 - preserve dim1 and dim2
                result = new float[dim1 * dim2];
                for (int d1 = 0; d1 < dim1; d1++) {
                    for (int d2 = 0; d2 < dim2; d2++) {
                        float sum = 0.0f;
                        for (int d3 = 0; d3 < dim3; d3++) {
                            for (int d4 = 0; d4 < dim4; d4++) {
                                sum += array[
                                    d1 * stride1 + 
                                    d2 * stride2 + 
                                    d3 * stride3 +
                                    d4
                                ];
                            }
                        }
                        result[d1 * dim2 + d2] = sum;
                    }
                }
                break;
                
            case 2: // Sum over dim4 - preserve dim1, dim2, and dim3  
                result = new float[dim1 * dim2 * dim3];
                for (int d1 = 0; d1 < dim1; d1++) {
                    for (int d2 = 0; d2 < dim2; d2++) {
                        for (int d3 = 0; d3 < dim3; d3++) {
                            float sum = 0.0f;
                            for (int d4 = 0; d4 < dim4; d4++) {
                                sum += array[
                                    d1 * stride1 + 
                                    d2 * stride2 + 
                                    d3 * stride3 +
                                    d4
                                ];
                            }
                            result[d1 * dim2 * dim3 + 
                                   d2 * dim3 + d3] = sum;
                        }
                    }
                }
                break;
            default:
                throw new IllegalArgumentException("Axis must be 0, 1, or 2. Got: " + axis);
        }
        return result;
    }

    /**
     * Finds the index of max values along a given axis.
     * @param array             The array to perform argmax on.
     * @param axis              The specified axis in the range [0,3] inclusive.
     * @param shape             
     * @return                  An array containing the indices of max values along the given axis.
     */
    protected static int[] argmax(float[] array, int axis, int[] shape) {
        assertArrayLength(shape, "shape", 4);
        if (axis < 0 || axis > 3) {
            throw new IllegalArgumentException("Axis must be between 0 and 3");
        }
        
        int[] strides = new int[]{
            shape[1] * shape[2] * shape[3], 
            shape[2] * shape[3], 
            shape[4], 
            1
        };
    
        int axisSize = shape[axis];
        int resultSize = array.length / axisSize;
        int[] result = new int[resultSize];
    
        IntStream.range(0, resultSize).parallel().forEach(opIndex -> {
            int baseOffset = indexHelper(opIndex, axis, shape, strides);
    
            float maxVal = Float.NEGATIVE_INFINITY;
            int maxIdx = 0;
    
            for (int i = 0; i < axisSize; i++) {
                int offset = baseOffset + i * strides[axis];
                if (array[offset] > maxVal) {
                    maxVal = array[offset];
                    maxIdx = i;
                }
            }
    
            result[opIndex] = maxIdx;
        });
    
        return result;
    }

    /**
     * Performs softmax along a given axis. Avoids overflow.
     * @param array             The array to perform softmax on.
     * @param axis              The specified axis in the range [0,3] inclusive.
     * @param shape             
     * @return                  An array with softmax applied along the specified axis.
     */
    protected static float[] softmax(float[] array, int axis, int[] shape, ForkJoinPool threadPool) {
        assertArrayLength(shape, "shape", 4);
        if (axis < 0 || axis > 3) {
            throw new IllegalArgumentException("Axis must be between 0 and 3");
        }
        // Create result matrix with same dimensions
        float[] result = new float[array.length];
        
        try {
            threadPool.submit(() -> {
                // Determine the sizes and strides based on dimensions

                int[] strides = new int[]{
                    shape[1] * shape[2] * shape[3], 
                    shape[2] * shape[3], 
                    shape[4], 
                    1
                };
                
                // Calculate the size of the axis to apply softmax to
                int axisSize = shape[axis];
                
                // Calculate the number of softmax operations to perform
                int numOperations = array.length / axisSize;
                
                // Process each softmax operation in parallel
                IntStream.range(0, numOperations).parallel().forEach(opIndex -> {
                    // Calculate the starting offset for this operation
                    
                    int baseOffset = indexHelper(opIndex, axis, shape, strides);
                    
                    // Find max value for numerical stability
                    float maxVal = Float.NEGATIVE_INFINITY;
                    for (int i = 0; i < axisSize; i++) {
                        int offset = baseOffset + i * strides[axis];
                        maxVal = Math.max(maxVal, array[offset]);
                    }
                    
                    // Calculate sum of exponentials
                    float sumExp = 0.0f;
                    for (int i = 0; i < axisSize; i++) {
                        int offset = baseOffset + i * strides[axis];
                        sumExp += Math.exp(array[offset] - maxVal);
                    }
                    
                    // Calculate softmax values
                    for (int i = 0; i < axisSize; i++) {
                        int offset = baseOffset + i * strides[axis];
                        result[offset] = (float) Math.exp(array[offset] - maxVal) / sumExp;
                    }
                });
            }).get();
        } catch (Exception e) {
            throw new RuntimeException("Error in softmax(axis = " + axis + ")", e);
        }
        return result;
    }

    /**
     * Performs logSoftmax along a given axis. Avoids overflow and underflow.
     * @param array             The array to perform logSoftmax on.
     * @param axis              The specified axis in the range [0,3] inclusive.
     * @param shape             
     * @return                  An array with logSoftmax applied along the specified axis.
     */
    protected static float[] logSoftmax(float[] array, int axis, int[] shape, ForkJoinPool threadPool) {
        assertArrayLength(shape, "shape", 4);
        if (axis < 0 || axis > 3) {
            throw new IllegalArgumentException("Axis must be between 0 and 3");
        }

        float[] result = new float[array.length];
        try {
            threadPool.submit(() -> {
                // Determine the sizes and strides based on dimensions
                int[] strides = new int[]{
                    shape[1] * shape[2] * shape[3], 
                    shape[2] * shape[3], 
                    shape[4], 
                    1
                };

                // Calculate the size of the axis to apply log softmax to
                int axisSize = shape[axis];

                // Calculate the number of log softmax operations to perform
                int numOperations = array.length / axisSize;

                IntStream.range(0, numOperations).parallel().forEach(opIndex -> {
                    // Calculate the starting offset
                    int baseOffset = indexHelper(opIndex, axis, shape, strides);
                    // Find max value for numerical stability
                    float maxVal = Float.NEGATIVE_INFINITY;
                    for (int i = 0; i < axisSize; i++) {
                        int offset = baseOffset + i * strides[axis];
                        maxVal = Math.max(maxVal, array[offset]);
                    }
                    // Calculate sum of exponentials
                    float sumExp = 0.0f;
                    for (int i = 0; i < axisSize; i++) {
                        int offset = baseOffset + i * strides[axis];
                        sumExp += Math.exp(array[offset] - maxVal);
                    }
                    // Calculate log softmax values
                    float logSumExp = (float) Math.log(sumExp) + maxVal;
                    for (int i = 0; i < axisSize; i++) {
                        int offset = baseOffset + i * strides[axis];
                        result[offset] = array[offset] - logSumExp;
                    }
                });
            }).get();
        } catch (Exception e) {
            throw new RuntimeException("Error in logSoftmax(axis = " + axis + ")", e);
        }
        return result;
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

    private static void assertArrayLength(Object array, String name, int expectedLength) {
        int actualLength = Array.getLength(array);
        if (actualLength != expectedLength) {
            throw new IllegalArgumentException(
                String.format(
                    "Invalid %d length. Expected: %d. Got: %d.",
                    name, expectedLength, actualLength
                )
                );
        }
    }
}
