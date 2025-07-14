package jflow.layers;

import java.util.stream.IntStream;

import jflow.data.JMatrix;
import jflow.layers.templates.TrainableLayer;

public class Embedding extends TrainableLayer {
    private int vocabSize;
    private int embedDim;
    private float[] tiedWeights = null;
    private boolean customInit = false;
    private double mean;
    private double stddev;

    private JMatrix embeddings;         // shape: (vocabSize, embedDim)
    private JMatrix gradEmbeddings;     // shape: (vocabSize, embedDim)

    private JMatrix lastInput;          // save token IDs for backward

    public Embedding(int vocabSize, int embedDim) {
        super("embedding");
        this.vocabSize = vocabSize;
        this.embedDim = embedDim;
    }

    public Embedding(int vocabSize, int embedDim, int[] inputShape) {
        this(vocabSize, embedDim);
        setInputShape(inputShape);
    }

    /**
     * Set the weight matrix for weight tying.
     * @param weights The weight matrix to use for this Embedding layer.
     */
    public Embedding weightTie(float[] weights, boolean removeFromParamCount) {
        this.tiedWeights = weights;
        if (removeFromParamCount) {
            // Remove the matrix from numTrainableParameters
            setNumTrainableParameters(0);
        } else {
            setNumTrainableParameters(weights.length);
        }
        return this;
    }

    public Embedding initNormal(double mean, double stddev) {
        this.customInit = true;
        this.mean = mean;
        this.stddev = stddev;
        return this;
    }

    @Override
    public void build(int IDnum) {
        super.build(IDnum);
        double mean; double stddev;
        if (customInit) {
            mean = this.mean;
            stddev = this.stddev;
        } else{
            // Standard embedding distribution
            mean = 0;
            stddev = 0.02;
        }

        embeddings = JMatrix
            .normal(vocabSize, embedDim, 1, 1, mean, stddev) 
            .label("embedding"); 
        gradEmbeddings = JMatrix.zeros(vocabSize, embedDim, 1, 1).label("dEmbedding");

        if (tiedWeights == null) {
            setNumTrainableParameters(vocabSize * embedDim);
        } else {
            embeddings.setMatrix(tiedWeights);
            tiedWeights = null;
        }
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
    public JMatrix[] getParameters() {
        return new JMatrix[] {embeddings};
    }

    @Override
    public JMatrix[] getParameterGradients() {
        return new JMatrix[] {gradEmbeddings};
    }

    @Override
    public void updateParameters(JMatrix[] updates) {
        embeddings.subtractInPlace(updates[0]);
    }

    @Override
    public int[] outputShape() {
        return new int[] {1, 1, embedDim}; // Variable batch/seq_len, fixed embed
    }

 
}
