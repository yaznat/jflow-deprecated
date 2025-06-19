package jflow.layers;

import java.util.stream.IntStream;

import jflow.data.JMatrix;
import jflow.layers.templates.TrainableLayer;

/**
 * Layer Normalization implementation matching GPT-2's approach.
 * Normalizes along the embedding dimension (H in NCHW format) for each sequence position.
 * Based on the paper "Layer Normalization" (Jimmy Lei Ba et al., 2016)
 */
public class LayerNorm extends TrainableLayer {
    private JMatrix gamma;
    private JMatrix dGamma;
    private JMatrix beta;
    private JMatrix dBeta;

    // Cached for backpropagation
    private JMatrix normalizedCache;
    private float[] varianceCache;

    private final static float EPSILON = 1e-5f;

    private int embedDim;  // Size of the embedding dimension (H in NCHW format)

    public LayerNorm() {
        super("layer_norm");
    }
    
    @Override
    public void build(int IDnum) {
        super.build(IDnum);

        int[] prevShape = getPreviousLayer().outputShape();
        // In NCHW format for GPT-2: 
        // N = batch size
        // C = sequence length
        // H = embedding dimension (normalize across this)
        // W = 1
        embedDim = prevShape[2];  // H dimension
        
        setNumTrainableParameters(2 * embedDim);

        // Initialize gamma to ones and beta to zeros
        gamma = JMatrix.ones(embedDim, 1, 1, 1).setName("gamma");
        beta = JMatrix.zeros(embedDim, 1, 1, 1).setName("beta");

        // Initialize gradients as zeros
        dGamma = gamma.zerosLike().setName("dGamma");
        dBeta = beta.zerosLike().setName("dBeta");
    }



    @Override
    public JMatrix forward(JMatrix input, boolean training) {
        int batchSize = input.length();      // N
        int seqLength = input.channels();    // C
        int numPositions = batchSize * seqLength;

        JMatrix output = input.zerosLike();

        // Cache for backpropagation
        if (training) {
            this.normalizedCache = input.zerosLike();
            this.varianceCache = new float[numPositions];
        }

        // Process each position (batch × seqLen) 
        IntStream.range(0, numPositions).parallel().forEach(posIdx -> {
            int batchIdx = posIdx / seqLength;
            int seqIdx = posIdx % seqLength;
            
            // Calculate mean for this position
            float mean = 0;
            for (int e = 0; e < embedDim; e++) {
                mean += input.get(batchIdx, seqIdx, e, 0);
            }
            mean /= embedDim;
            
            // Calculate variance for this position
            float variance = 0;
            for (int e = 0; e < embedDim; e++) {
                float diff = input.get(batchIdx, seqIdx, e, 0) - mean;
                variance += diff * diff;
            }
            variance /= embedDim;
            
            // Cache statistics for backpropagation
            if (training) {
                varianceCache[posIdx] = variance;
            }
            
            float invStdDev = 1.0f / (float)Math.sqrt(variance + EPSILON);
            
            // Normalize, scale, and shift
            for (int e = 0; e < embedDim; e++) {
                float normalized = (input.get(batchIdx, seqIdx, e, 0) - mean) * invStdDev;
                
                if (training) {
                    normalizedCache.set(batchIdx, seqIdx, e, 0, normalized);
                }
                
                // Scale and shift
                float result = normalized * gamma.get(e) + beta.get(e);
                output.set(batchIdx, seqIdx, e, 0, result);
            }
        });
        return trackOutput(output, training);
    }

    @Override
    public JMatrix backward(JMatrix dOutput) {
        int batchSize = dOutput.length();
        int seqLength = dOutput.channels();
        int numPositions = batchSize * seqLength;

        JMatrix dInput = dOutput.zerosLike();

        // Calculate gradients for each position
        IntStream.range(0, numPositions).parallel().forEach(posIdx -> {
            int batchIdx = posIdx / seqLength;
            int seqIdx = posIdx % seqLength;
            
            float variance = varianceCache[posIdx];
            float invStdDev = 1.0f / (float)Math.sqrt(variance + EPSILON);
            
            // Calculate sum terms for the gradient formula
            float sumDy = 0;
            float sumDyxHat = 0;
            
            for (int e = 0; e < embedDim; e++) {
                float dy = dOutput.get(batchIdx, seqIdx, e, 0);
                float xHat = normalizedCache.get(batchIdx, seqIdx, e, 0);
                
                // Accumulate parameter gradients
                synchronized (dGamma) {
                    dGamma.set(e, dGamma.get(e) + dy * xHat);
                    dBeta.set(e, dBeta.get(e) + dy);
                }
                
                sumDy += dy;
                sumDyxHat += dy * xHat;
            }
            
            // Calculate input gradients
            for (int e = 0; e < embedDim; e++) {
                float dy = dOutput.get(batchIdx, seqIdx, e, 0);
                float xHat = normalizedCache.get(batchIdx, seqIdx, e, 0);
                
                // Following the chain rule for layer normalization
                float term1 = dy * gamma.get(e);
                float term2 = sumDy / embedDim;
                float term3 = xHat * sumDyxHat / embedDim;
                
                float dx = invStdDev * (term1 - (term2 + term3));
                dInput.set(batchIdx, seqIdx, e, 0, dx);
            }
        });

        return trackGradient(dInput);
    }

    @Override
    public JMatrix[] getParameterGradients() {
        return new JMatrix[]{dGamma, dBeta};
    }

    @Override
    public void updateParameters(JMatrix[] parameterUpdates) {
        gamma.subtractInPlace(parameterUpdates[0]);
        beta.subtractInPlace(parameterUpdates[1]);
    }

    @Override
    public JMatrix[] getParameters() {
        return new JMatrix[]{gamma, beta};
    }

    @Override
    public int[] outputShape() {
        return getPreviousLayer().outputShape();
    }
}