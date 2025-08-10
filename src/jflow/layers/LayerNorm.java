package jflow.layers;

import java.util.stream.IntStream;

import jflow.data.JMatrix;
import jflow.layers.templates.NormalizationLayer;

public class LayerNorm extends NormalizationLayer<LayerNorm> {
    // Cached for backpropagation
    private JMatrix normalizedCache;
    private JMatrix varianceCache;
    private JMatrix meanCache;

    private int embedDim;

    /**
     * The LayerNorm layer.
     * 
     * <p><b>Do not instantiate directly.</b> Use the static builder method:
     * {@code import static jflow.model.builder.*;}
     * and call {@code LayerNorm()} instead of {@code new LayerNorm()}.
     */
    public LayerNorm() {
        super("layer_norm");
    }
    
    @Override
    protected int[] parameterShape(int[] inputShape) {
        // Normalize along H in NCHW - this is the embedding dimension in transformers
        this.embedDim = inputShape[2];

        return new int[]{embedDim, 1, 1, 1};
    }


    @Override
    public JMatrix trainableForwardPass(JMatrix input, boolean training) {
        JMatrix gamma = getGamma();
        JMatrix beta = getBeta();
        final double EPSILON = getEpsilon();

        int batchSize = input.shape(0);    
        int seqLength = input.shape(1); 
        int numPositions = batchSize * seqLength;

        JMatrix output = input.zerosLike();

        // Cache for backpropagation
        if (training) {
            this.normalizedCache = input.zerosLike();
            this.varianceCache = JMatrix.zeros(numPositions, 1, 1, 1);
            this.meanCache = JMatrix.zeros(numPositions, 1, 1, 1);
        }

        IntStream.range(0, numPositions).parallel().forEach(posIdx -> {
            int batchIdx = posIdx / seqLength;
            int seqIdx = posIdx % seqLength;
            
            // Calculate mean
            float mean = 0;
            for (int e = 0; e < embedDim; e++) {
                mean += input.get(batchIdx, seqIdx, e, 0);
            }
            mean /= embedDim;
            
            // Calculate variance
            float variance = 0;
            for (int e = 0; e < embedDim; e++) {
                float diff = input.get(batchIdx, seqIdx, e, 0) - mean;
                variance += diff * diff;
            }
            variance /= embedDim;
            
            // Cache statistics for backpropagation
            if (training) {
                varianceCache.set(posIdx, variance);
                meanCache.set(posIdx, mean);
            }
                        
            // Normalize, scale, and shift
            float stdDev = (float)Math.sqrt(variance + EPSILON);
            float invStdDev = 1.0f / stdDev;

            for (int e = 0; e < embedDim; e++) {
                // Normalize to mean ≈ 0 and variance ≈ 1
                float normalized = (input.get(batchIdx, seqIdx, e, 0) - mean) * invStdDev;

                // Cache for backpropagation
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
    public JMatrix trainableBackwardPass(JMatrix dOutput) {
        JMatrix gamma = getGamma();
        JMatrix dGamma = getDGamma();
        JMatrix dBeta = getDBeta();
        final double EPSILON = getEpsilon();

        int batchSize = dOutput.shape(0);
        int seqLength = dOutput.shape(1);
        int numPositions = batchSize * seqLength;
        JMatrix dInput = dOutput.zerosLike();
        
        // Process each position for input gradients
        IntStream.range(0, numPositions).parallel().forEach(posIdx -> {
            int batchIdx = posIdx / seqLength;
            int seqIdx = posIdx % seqLength;
            float variance = varianceCache.get(posIdx);
            float invStdDev = 1.0f / (float)Math.sqrt(variance + EPSILON);
            
            // Calculate gradient terms for this position
            float sumDyGamma = 0;
            float sumDyGammaXhat = 0;
            for (int e = 0; e < embedDim; e++) {
                float dy = dOutput.get(batchIdx, seqIdx, e, 0);
                float xHat = normalizedCache.get(batchIdx, seqIdx, e, 0);
                float dyGamma = dy * gamma.get(e);
                sumDyGamma += dyGamma;
                sumDyGammaXhat += dyGamma * xHat;
            }
            
            // Calculate input gradients
            for (int e = 0; e < embedDim; e++) {
                float dy = dOutput.get(batchIdx, seqIdx, e, 0);
                float xHat = normalizedCache.get(batchIdx, seqIdx, e, 0);
                
                // Layer norm gradient formula
                float dyGamma = dy * gamma.get(e);
                float dx = invStdDev * (dyGamma - (sumDyGamma / embedDim) - 
                                      (xHat * sumDyGammaXhat / embedDim));
                dInput.set(batchIdx, seqIdx, e, 0, dx);
            }
        });
        
        // Single-threaded gradient accumulation - faster than synchronization
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLength; s++) {
                for (int e = 0; e < embedDim; e++) {
                    float dy = dOutput.get(b, s, e, 0);
                    float xHat = normalizedCache.get(b, s, e, 0);
                    
                    dGamma.addTo(e, dy * xHat);
                    dBeta.addTo(e, dy);
                }
            }
        }
        
        return trackGradient(dInput);
    }

    @Override
    protected JMatrix[] getRunningStats() {
        return null; // LayerNorm has no running stats.
    }
}