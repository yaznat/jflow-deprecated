package jflow.layers;

import java.util.stream.IntStream;

import jflow.data.JMatrix;
import jflow.layers.templates.ParametricLayer;

/**
 * Standard embedding layer for transformer encoder-decoders. <p>
 * Input: (batch_size, seq_len, 1, 1) <p>
 * Output: (batch_size, seq_len, embed_dim, 1)
 */
public class Embedding extends ParametricLayer<Embedding> {
    private int vocabSize;
    private int embedDim;

    private JMatrix embeddings;         
    private JMatrix gradEmbeddings;     

    private float[] tiedWeights = null;
    
    /**
     * The Embedding layer.
     * 
     * <p><b>Do not instantiate directly.</b> Use the static builder method:
     * {@code import static jflow.model.builder.*;}
     * and call {@code Embedding(...)} instead of {@code new Embedding(...)}.
     */
    public Embedding(int vocabSize, int embedDim) {
        super("embedding");
        this.vocabSize = vocabSize;
        this.embedDim = embedDim;
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
    
    @Override
    protected void build(int[] inputShape) {
        if (useCustomInit()) {
            embeddings = initCustomWeight(vocabSize, embedDim, 1, 1);
        } else{
            // Standard embedding distribution
            double mean = 0.0;
            double stddev = 0.02;
            embeddings = JMatrix.normal(vocabSize, embedDim, 1, 1, mean, stddev);
        }

        embeddings.label("weights"); 
        gradEmbeddings = JMatrix
            .zeros(vocabSize, embedDim, 1, 1)
            .label("dWeights");

        if (tiedWeights == null) {
            setNumTrainableParameters(vocabSize * embedDim);
        } else {
            try {
                embeddings.setMatrix(tiedWeights);
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException(
                    "Invalid matrix size for weight tying. Expected: "
                    + embeddings.size() + " Got: " + tiedWeights.length
                );
            }
            tiedWeights = null;
        }
    }

    @Override
    public JMatrix trainableForwardPass(JMatrix tokenIDs, boolean training) {
        // Cache tokenIDs for backpropagation
        cacheInput(tokenIDs, training);

        int batch = tokenIDs.shape(0);
        int seqLen = tokenIDs.shape(1);

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
    public JMatrix trainableBackwardPass(JMatrix dOutput) {
        int batch = dOutput.shape()[0];
        int seqLen = dOutput.shape()[1];

        JMatrix lastInput = getLastInput();

        for (int b = 0; b < batch; b++) {
            for (int t = 0; t < seqLen; t++) {
                int id = (int) lastInput.get(b, t, 0, 0);
                for (int e = 0; e < embedDim; e++) {
                    double grad = dOutput.get(b, t, e, 0);
                    gradEmbeddings.addTo(id, e, 0, 0, grad);
                }
            }
        }

        // Embedding layers do not pass gradients to token IDs
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
}
