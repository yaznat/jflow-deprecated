package jflow.layers;

import java.util.stream.IntStream;

import jflow.data.JMatrix;
import jflow.layers.templates.ShapePreservingLayer;

public class Softmax extends ShapePreservingLayer{
    public Softmax() {
        super("softmax");
    }

    @Override
    public JMatrix forward(JMatrix A, boolean training) {
        int batchSize = A.shape(0);
        int numClasses = A.shape(1);

        JMatrix Z = A.zerosLike();

        IntStream.range(0, batchSize).parallel().forEach(i -> {
            float max = Float.NEGATIVE_INFINITY;

            // Find max in row i
            for (int j = 0; j < numClasses; j++) {
                max = Math.max(A.get(i * numClasses + j), max);
            }

            float sum = 0;
            for (int j = 0; j < numClasses; j++) {
                sum += Math.exp(A.get(i * numClasses + j) - max);
            }

            for (int j = 0; j < numClasses; j++) {
                float exp = (float) Math.exp(A.get(i * numClasses + j) - max);
                Z.set(i * numClasses + j, exp / sum);
            }
        });

        return trackOutput(Z, training);
    }


    @Override
    public JMatrix backward(JMatrix gradient) {
        return trackGradient(getOutput().subtract(gradient));
    }
}
