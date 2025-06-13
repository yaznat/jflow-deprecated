package jflow.layers;

import java.util.stream.IntStream;

import jflow.data.JMatrix;
import jflow.layers.templates.TrainableLayer;

public class Embedding extends TrainableLayer {
    private int vocabSize;
    private int embedDim;

    private JMatrix embeddings;         // shape: [vocabSize, embedDim]
    private JMatrix gradEmbeddings;     // shape: same

    private JMatrix lastInput;          // save token IDs for backward

    public Embedding(int vocabSize, int embedDim) {
        super("embedding");
        this.vocabSize = vocabSize;
        this.embedDim = embedDim;

        this.embeddings = JMatrix
            .randn(vocabSize, embedDim, 1, 1)
            .multiply(0.02) // standard scale factor
            .setName("embedding"); 
        this.gradEmbeddings = JMatrix.zeros(vocabSize, embedDim, 1, 1).setName("dEmbedding");
    }

    public Embedding(int vocabSize, int embedDim, int[] inputShape) {
        this(vocabSize, embedDim);

        setInputShape(inputShape);
    }

    @Override
    public void build(int IDnum) {
        super.build(IDnum);

        setNumTrainableParameters(vocabSize * embedDim);
    }

    @Override
    public JMatrix forward(JMatrix tokenIDs, boolean training) {
        this.lastInput = tokenIDs;

        int batch = tokenIDs.shape()[0];
        int seqLen = tokenIDs.shape()[1];

        JMatrix output = JMatrix.zeros(batch, seqLen, embedDim, 1);

        IntStream.range(0, batch).parallel().forEach(b -> {
            for (int t = 0; t < seqLen; t++) {
                int id = (int) tokenIDs.get(b, t, 0, 0);
                for (int e = 0; e < embedDim; e++) {
                    output.set(b, t, e, 0, embeddings.get(id, e, 0, 0));
                }
            }
        });

        return trackOutput(output, training);
    }

    @Override
    public JMatrix backward(JMatrix dOutput) {

        int batch = dOutput.shape()[0];
        int seqLen = dOutput.shape()[1];

        for (int b = 0; b < batch; b++) {
            for (int t = 0; t < seqLen; t++) {
                int id = (int) lastInput.get(b, t, 0, 0);
                for (int e = 0; e < embedDim; e++) {
                    double grad = dOutput.get(b, t, e, 0);
                    gradEmbeddings.set(id, e, 0, 0, gradEmbeddings.get(id, e, 0, 0) + grad);
                }
            }
        }

        // Return nothing since Embedding has no input gradient
        return null;
    }

    @Override
    public JMatrix[] getWeights() {
        return new JMatrix[] {embeddings};
    }

    @Override
    public JMatrix[] getParameterGradients() {
        return new JMatrix[] {gradEmbeddings};
    }

    @Override
    public void updateParameters(JMatrix[] updates) {
        this.embeddings.add(updates[0]);
    }

    @Override
    public int[] outputShape() {
        return new int[] { 1, 1, embedDim, 1}; // variable batch/seq, fixed embed
    }

 
}
