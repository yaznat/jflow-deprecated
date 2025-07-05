package jflow.data;

import java.util.stream.IntStream;

public class MatrixOps {
    
    public static void transpose2DMatrixByDims(
        float[] input,
        int oldHeight,
        int oldWidth,
        int axis1, 
        int axis2,
        float[] result
    ) {
        int[] axes = {axis1, axis2};
        boolean[] used = new boolean[2];
        for (int axis : axes) {
            if (axis < 0 || axis > 1) {
                throw new IllegalArgumentException("Axis values must be between 0 and 1 inclusive");
            }
            if (used[axis]) {
                throw new IllegalArgumentException("Axis values must be a permutation of 0, 1");
            }
            used[axis] = true;
        }

        // Fast path for identity
        if (axis1 == 0 && axis2 == 1) {
            System.arraycopy(input, 0, result, 0, input.length);
            return;
        }
        
        final int totalElements = oldHeight * oldWidth;
        
        if (totalElements <= 2048) {
            transpose2DSequential(input, oldHeight, oldWidth, result);
        } else {
            int numThreads = Runtime.getRuntime().availableProcessors();
            int blockSize = Math.max(256, totalElements / (numThreads * 2));
            
            IntStream.range(0, (totalElements + blockSize - 1) / blockSize)
                .parallel()
                .forEach(block -> {
                    int start = block * blockSize;
                    int end = Math.min(start + blockSize, totalElements);
                    transpose2DBlock(input, oldHeight, oldWidth, start, end, result);
                });
        }
    }

    private static void transpose2DSequential(float[] input, int oldHeight, int oldWidth, float[] result) {
        for (int h = 0; h < oldHeight; h++) {
            int srcBase = h * oldWidth;
            for (int w = 0; w < oldWidth; w++) {
                result[w * oldHeight + h] = input[srcBase + w];
            }
        }
    }

    private static void transpose2DBlock(float[] input, int oldHeight, int oldWidth, 
                                        int startIdx, int endIdx, float[] result) {
        for (int idx = startIdx; idx < endIdx; idx++) {
            int h = idx / oldWidth;
            int w = idx % oldWidth;
            result[w * oldHeight + h] = input[idx];
        }
    }

    public static void transpose3DMatrixByDims(
        float[] input,
        int oldBatch,
        int oldFeature1,
        int oldFeature2,
        int axis1, 
        int axis2, 
        int axis3,
        float[] result
    ) {
        int[] axes = {axis1, axis2, axis3};
        boolean[] used = new boolean[3];
        for (int axis : axes) {
            if (axis < 0 || axis > 2) {
                throw new IllegalArgumentException("Axis values must be between 0 and 2 inclusive");
            }
            if (used[axis]) {
                throw new IllegalArgumentException("Axis values must be a permutation of 0, 1, 2");
            }
            used[axis] = true;
        }

        // Fast path for identity
        if (axis1 == 0 && axis2 == 1 && axis3 == 2) {
            System.arraycopy(input, 0, result, 0, input.length);
            return;
        }

        int[] dims = {oldBatch, oldFeature1, oldFeature2};
        int newDim1 = dims[axis1];
        int newDim2 = dims[axis2];
        int newDim3 = dims[axis3];
        
        final int totalElements = oldBatch * oldFeature1 * oldFeature2;
        
        if (totalElements <= 4096) {
            transpose3DSequential(input, axis1, axis2, axis3, oldBatch, oldFeature1, oldFeature2, 
                                 newDim1, newDim2, newDim3, result);
        } else {
            int numThreads = Runtime.getRuntime().availableProcessors();
            int blockSize = Math.max(512, totalElements / (numThreads * 2));
            
            IntStream.range(0, (totalElements + blockSize - 1) / blockSize)
                .parallel()
                .forEach(block -> {
                    int start = block * blockSize;
                    int end = Math.min(start + blockSize, totalElements);
                    transpose3DBlock(input, axis1, axis2, axis3, oldBatch, oldFeature1, oldFeature2,
                                   newDim1, newDim2, newDim3, start, end, result);
                });
        }
    }

    private static void transpose3DSequential(float[] input, int axis1, int axis2, int axis3,
                                            int oldBatch, int oldFeature1, int oldFeature2,
                                            int newDim1, int newDim2, int newDim3, float[] result) {
        int oldStride0 = oldFeature1 * oldFeature2;
        int oldStride1 = oldFeature2;
        int newStride0 = newDim2 * newDim3;
        int newStride1 = newDim3;
        
        for (int c = 0; c < oldBatch; c++) {
            int srcBase = c * oldStride0;
            for (int h = 0; h < oldFeature1; h++) {
                int srcOffset = srcBase + h * oldStride1;
                for (int w = 0; w < oldFeature2; w++) {
                    int coord1 = (axis1 == 0) ? c : (axis1 == 1) ? h : w;
                    int coord2 = (axis2 == 0) ? c : (axis2 == 1) ? h : w;
                    int coord3 = (axis3 == 0) ? c : (axis3 == 1) ? h : w;
                    int newIdx = coord1 * newStride0 + coord2 * newStride1 + coord3;
                    result[newIdx] = input[srcOffset + w];
                }
            }
        }
    }

    private static void transpose3DBlock(float[] input, int axis1, int axis2, int axis3,
                                       int oldBatch, int oldFeature1, int oldFeature2,
                                       int newDim1, int newDim2, int newDim3,
                                       int startIdx, int endIdx, float[] result) {
        int oldStride0 = oldFeature1 * oldFeature2;
        int oldStride1 = oldFeature2;
        int newStride0 = newDim2 * newDim3;
        int newStride1 = newDim3;
        
        for (int idx = startIdx; idx < endIdx; idx++) {
            int c = idx / oldStride0;
            int remaining = idx % oldStride0;
            int h = remaining / oldStride1;
            int w = remaining % oldStride1;
            
            int coord1 = (axis1 == 0) ? c : (axis1 == 1) ? h : w;
            int coord2 = (axis2 == 0) ? c : (axis2 == 1) ? h : w;
            int coord3 = (axis3 == 0) ? c : (axis3 == 1) ? h : w;
            int newIdx = coord1 * newStride0 + coord2 * newStride1 + coord3;
            result[newIdx] = input[idx];
        }
    }

    public static void transpose4DMatrixByDims(
        float[] input,
        int oldLength,
        int oldChannels,
        int oldHeight,
        int oldWidth,
        int axis1, 
        int axis2, 
        int axis3, 
        int axis4,
        float[] result
    ) {
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

        // Fast path for identity
        if (axis1 == 0 && axis2 == 1 && axis3 == 2 && axis4 == 3) {
            System.arraycopy(input, 0, result, 0, input.length);
            return;
        }

        int[] dims = {oldLength, oldChannels, oldHeight, oldWidth};
        int newDim1 = dims[axis1];
        int newDim2 = dims[axis2];
        int newDim3 = dims[axis3];
        int newDim4 = dims[axis4];
        
        final int totalElements = oldLength * oldChannels * oldHeight * oldWidth;
        
        if (totalElements <= 8192) {
            transpose4DSequential(input, axis1, axis2, axis3, axis4, oldLength, oldChannels, oldHeight, oldWidth,
                                 newDim1, newDim2, newDim3, newDim4, result);
        } else {
            int numThreads = Runtime.getRuntime().availableProcessors();
            int blockSize = Math.max(1024, totalElements / (numThreads * 2));
            
            IntStream.range(0, (totalElements + blockSize - 1) / blockSize)
                .parallel()
                .forEach(block -> {
                    int start = block * blockSize;
                    int end = Math.min(start + blockSize, totalElements);
                    transpose4DBlock(input, axis1, axis2, axis3, axis4, oldLength, oldChannels, oldHeight, oldWidth,
                                   newDim1, newDim2, newDim3, newDim4, start, end, result);
                });
        }
    }

    private static void transpose4DSequential(float[] input, int axis1, int axis2, int axis3, int axis4,
                                            int oldLength, int oldChannels, int oldHeight, int oldWidth,
                                            int newDim1, int newDim2, int newDim3, int newDim4, float[] result) {
        int oldStride0 = oldChannels * oldHeight * oldWidth;
        int oldStride1 = oldHeight * oldWidth;
        int oldStride2 = oldWidth;
        int newStride0 = newDim2 * newDim3 * newDim4;
        int newStride1 = newDim3 * newDim4;
        int newStride2 = newDim4;
        
        for (int n = 0; n < oldLength; n++) {
            int srcBase0 = n * oldStride0;
            for (int c = 0; c < oldChannels; c++) {
                int srcBase1 = srcBase0 + c * oldStride1;
                for (int h = 0; h < oldHeight; h++) {
                    int srcOffset = srcBase1 + h * oldStride2;
                    for (int w = 0; w < oldWidth; w++) {
                        int coord1 = (axis1 == 0) ? n : (axis1 == 1) ? c : (axis1 == 2) ? h : w;
                        int coord2 = (axis2 == 0) ? n : (axis2 == 1) ? c : (axis2 == 2) ? h : w;
                        int coord3 = (axis3 == 0) ? n : (axis3 == 1) ? c : (axis3 == 2) ? h : w;
                        int coord4 = (axis4 == 0) ? n : (axis4 == 1) ? c : (axis4 == 2) ? h : w;
                        int newIdx = coord1 * newStride0 + coord2 * newStride1 + coord3 * newStride2 + coord4;
                        result[newIdx] = input[srcOffset + w];
                    }
                }
            }
        }
    }

    private static void transpose4DBlock(float[] input, int axis1, int axis2, int axis3, int axis4,
                                       int oldLength, int oldChannels, int oldHeight, int oldWidth,
                                       int newDim1, int newDim2, int newDim3, int newDim4,
                                       int startIdx, int endIdx, float[] result) {
        int oldStride0 = oldChannels * oldHeight * oldWidth;
        int oldStride1 = oldHeight * oldWidth;
        int oldStride2 = oldWidth;
        int newStride0 = newDim2 * newDim3 * newDim4;
        int newStride1 = newDim3 * newDim4;
        int newStride2 = newDim4;
        
        for (int idx = startIdx; idx < endIdx; idx++) {
            int n = idx / oldStride0;
            int remaining = idx % oldStride0;
            int c = remaining / oldStride1;
            remaining %= oldStride1;
            int h = remaining / oldStride2;
            int w = remaining % oldStride2;
            
            int coord1 = (axis1 == 0) ? n : (axis1 == 1) ? c : (axis1 == 2) ? h : w;
            int coord2 = (axis2 == 0) ? n : (axis2 == 1) ? c : (axis2 == 2) ? h : w;
            int coord3 = (axis3 == 0) ? n : (axis3 == 1) ? c : (axis3 == 2) ? h : w;
            int coord4 = (axis4 == 0) ? n : (axis4 == 1) ? c : (axis4 == 2) ? h : w;
            int newIdx = coord1 * newStride0 + coord2 * newStride1 + coord3 * newStride2 + coord4;
            result[newIdx] = input[idx];
        }
    }
}