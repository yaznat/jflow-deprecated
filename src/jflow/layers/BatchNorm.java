package jflow.layers;

import java.util.stream.IntStream;

import jflow.data.JMatrix;
import jflow.layers.templates.TrainableLayer;

public class BatchNorm extends TrainableLayer {
    private int featureSize;
    private double epsilon = 1e-5;
    private double momentum = 0.99;
    private JMatrix gamma;
    private JMatrix beta;
    private JMatrix runningMean;
    private JMatrix runningVar;
    private JMatrix batchMean;
    private JMatrix batchVar;
    private JMatrix xHat;
    private JMatrix input;

    private JMatrix output;
    
    // Pre-allocated matrices for backward pass
    private JMatrix dGamma;
    private JMatrix dBeta;
    private JMatrix dx;
    private JMatrix dxHat;
    private JMatrix dxHatSum;
    private JMatrix dxHatXhatSum;


    public BatchNorm() {
        super("batch_norm");

    }

    public void build(int IDnum) {
        super.build(IDnum);
        this.featureSize = getPreviousLayer().outputShape()[1];

        setNumTrainableParameters(featureSize * 2);

        // Initialize gamma values as 1.0
        this.gamma = JMatrix.ones(1, featureSize, 1, 1).setName("gamma");
        
        // Initialize beta values as 0.0
        this.beta = JMatrix.zeros(1, featureSize, 1, 1).setName("beta");
        
        // Initialize running mean as 0.0
        this.runningMean = JMatrix.zeros(1, featureSize, 1, 1).setName("runningMean");

        // Initialize running var as 1.0
        this.runningVar = JMatrix.ones(1, featureSize, 1, 1).setName("runningVar");

        // Initialize derivatives as 0.0
        this.dGamma = JMatrix.zeros(1, featureSize, 1, 1).setName("dGamma");
        this.dBeta = JMatrix.zeros(1, featureSize, 1, 1).setName("dBeta");
        this.dxHatSum = JMatrix.zeros(1, featureSize, 1, 1);
        this.dxHatXhatSum = JMatrix.zeros(1, featureSize, 1, 1);

    }

    public JMatrix forward(JMatrix input, boolean training) {
        if (training) {
            this.input = input;
        }
        
        if (getPreviousShapeInfluencer() instanceof Dense) {
            input = input.transpose2D();
        }
        // Ensure dx and dxHat have the right dimensions
        if (dx == null || dx.length() != input.length() || dx.channels() != input.channels()) {
            dx = input.zerosLike();
            dxHat = input.zerosLike();
        }
         
        if (input.channels() != featureSize) {
            System.out.println("Warning: BatchNorm feature size doesn't match input channels");
        }
        
        if (training) {
            // Calculate batch statistics
            batchMean = calcMean(input);
            batchVar = calcVariance(input, batchMean);
            
            // Update running averages
            runningMean.multiplyInPlace(momentum).addInPlace(batchMean.multiply(1 - momentum));
            runningVar.multiplyInPlace(momentum).addInPlace(batchVar.multiply(1 - momentum));
            
            // Normalize
            xHat = normalize(input, batchMean, batchVar);
        } else {
            // Normalize using running averages
            xHat = normalize(input, runningMean, runningVar);
        }
        
        // Scale and shift
        output = scaleAndShift(xHat, gamma, beta);
        
        return trackOutput(output, training);
    }
    
    /*
     * Calculate the mean across the batch 
     * and spatial dimensions for each channel.
     */ 
    private JMatrix calcMean(JMatrix input) {
        JMatrix mean = JMatrix.zeros(1, featureSize, 1, 1);
        
        int batchSize = input.length();
        int channels = input.channels();
        int height = input.height();
        int width = input.width();
        int spatialSize = height * width;
        
        IntStream.range(0, channels).parallel().forEach(c -> {
            float sum = 0;
            for (int n = 0; n < batchSize; n++) {
                int batchOffset = n * channels * height * width;
                int channelOffset = c * height * width;
                    
                for (int h = 0; h < height; h++) {
                    int rowOffset = h * width;
                        
                    for (int w = 0; w < width; w++) {
                        int idx = batchOffset + channelOffset + rowOffset + w;
                        sum += input.get(idx);
                    }
                }
            }
            mean.set(c, sum / (batchSize * spatialSize));
        });
        
        return mean;
    }

    /*
     * Calculate the mean across the batch 
     * and spatial dimensions for each channel.
     * Reuse the mean calculation.
     */ 
    private JMatrix calcVariance(JMatrix input, JMatrix mean) {
        JMatrix var = JMatrix.zeros(1, featureSize, 1, 1);
        
        int batchSize = input.length();
        int channels = input.channels();
        int height = input.height();
        int width = input.width();
        int spatialSize = height * width;
        
        IntStream.range(0, channels).parallel().forEach(c -> {
            float sum = 0;
            float meanVal = mean.get(c);
                
            for (int n = 0; n < batchSize; n++) {
                int batchOffset = n * channels * height * width;
                int channelOffset = c * height * width;
                    
                for (int h = 0; h < height; h++) {
                    int rowOffset = h * width;
                        
                    for (int w = 0; w < width; w++) {
                        int idx = batchOffset + channelOffset + rowOffset + w;
                        double diff = input.get(idx) - meanVal;
                        sum += diff * diff;
                    }
                }
            }
            var.set(c, sum / (batchSize * spatialSize));
        });
        
        return var;
    }
    
    private JMatrix normalize(JMatrix input, JMatrix mean, JMatrix variance) {
        JMatrix normalized = input.zerosLike();
        
        int batchSize = input.length();
        int channels = input.channels();
        int height = input.height();
        int width = input.width();
        
        // Pre-compute standard deviation inverses for efficiency
        float[] stdInvs = new float[channels];
        for (int c = 0; c < channels; c++) {
            stdInvs[c] = (float)(1.0f / Math.sqrt(variance.get(c) + epsilon));
        }
        
        IntStream.range(0, batchSize * channels).parallel().forEach(nc -> {
            int n = nc / channels;
            int c = nc % channels;
                
            float meanVal = mean.get(c);
            float stdInv = stdInvs[c];
            int batchOffset = n * channels * height * width;
            int channelOffset = c * height * width;
                
            for (int h = 0; h < height; h++) {
                int rowOffset = h * width;
                    
                for (int w = 0; w < width; w++) {
                    int idx = batchOffset + channelOffset + rowOffset + w;
                    normalized.set(idx, (input.get(idx) - meanVal) * stdInv);
                }
            }
        });
        
        return normalized;
    }
    
    private JMatrix scaleAndShift(JMatrix normalized, JMatrix gamma, JMatrix beta) {
        JMatrix output = normalized.zerosLike();

        int batchSize = normalized.length();
        int channels = normalized.channels();
        int height = normalized.height();
        int width = normalized.width();
        
        // Iterate across batches and channels
        IntStream.range(0, batchSize * channels).parallel().forEach(nc -> {
            int n = nc / channels;
            int c = nc % channels;
                
            float gammaVal = gamma.get(c);
            float betaVal = beta.get(c);
            int batchOffset = n * channels * height * width;
            int channelOffset = c * height * width;
                
            for (int h = 0; h < height; h++) {
                int rowOffset = h * width;
                    
                for (int w = 0; w < width; w++) {
                    int idx = batchOffset + channelOffset + rowOffset + w;

                    output.set(idx, normalized.get(idx) * gammaVal + betaVal);
                }
            }
        });

       
        
        return output;

    }

    public JMatrix backward(JMatrix dOut) {
        int batchSize = input.length();
        int channels = input.channels();
        int height = input.height();
        int width = input.width();
        int spatialSize = height * width;
        int elements = batchSize * spatialSize;
        
        // Calculate dGamma and dBeta
        IntStream.range(0, channels).parallel().forEach(c -> {
                float dGammaSum = 0;
                float dBetaSum = 0;
                
                for (int n = 0; n < batchSize; n++) {
                    int batchOffset = n * channels * height * width;
                    int channelOffset = c * height * width;
                    
                    for (int h = 0; h < height; h++) {
                        int rowOffset = h * width;
                        
                        for (int w = 0; w < width; w++) {
                            int idx = batchOffset + channelOffset + rowOffset + w;
                            dGammaSum += dOut.get(idx) * xHat.get(idx);
                            dBetaSum += dOut.get(idx);
                        }
                    }
                }
                
                dGamma.set(c, dGammaSum);
                dBeta.set(c, dBetaSum);
            });
        
        // Calculate dxHat = dout * gamma
        IntStream.range(0, batchSize * channels).parallel().forEach(nc -> {
                int n = nc / channels;
                int c = nc % channels;
                float gammaVal = gamma.get(c);
                
                int batchOffset = n * channels * height * width;
                int channelOffset = c * height * width;
                
                for (int h = 0; h < height; h++) {
                    int rowOffset = h * width;
                    
                    for (int w = 0; w < width; w++) {
                        int idx = batchOffset + channelOffset + rowOffset + w;
                        dxHat.set(idx, dOut.get(idx) * gammaVal);
                    }
                }
            });
        
        // Calculate intermediate sums for dx calculation
        IntStream.range(0, channels).parallel().forEach(c -> {
                float dxHatSumVal = 0;
                float dxHatXhatSumVal = 0;
                
                for (int n = 0; n < batchSize; n++) {
                    int batchOffset = n * channels * height * width;
                    int channelOffset = c * height * width;
                    
                    for (int h = 0; h < height; h++) {
                        int rowOffset = h * width;
                        
                        for (int w = 0; w < width; w++) {
                            int idx = batchOffset + channelOffset + rowOffset + w;
                            dxHatSumVal += dxHat.get(idx);
                            dxHatXhatSumVal += dxHat.get(idx) * xHat.get(idx);
                        }
                    }
                }
                
                dxHatSum.set(c, dxHatSumVal);
                dxHatXhatSum.set(c, dxHatXhatSumVal);
            });
        
        // Calculate dx
        float[] stdInvs = new float[channels];
        for (int c = 0; c < channels; c++) {
            stdInvs[c] = (float)(1.0f / Math.sqrt(batchVar.get(c) + epsilon));
        }
        
        IntStream.range(0, batchSize * channels).parallel().forEach(nc -> {
            int n = nc / channels;
            int c = nc % channels;
            float stdInv = stdInvs[c];
            float dxHatSumVal = dxHatSum.get(c) / elements;
            float dxHatXhatSumVal = dxHatXhatSum.get(c) / elements;
                
            int batchOffset = n * channels * height * width;
            int channelOffset = c * height * width;
                
            for (int h = 0; h < height; h++) {
                int rowOffset = h * width;
                    
                for (int w = 0; w < width; w++) {
                    int idx = batchOffset + channelOffset + rowOffset + w;
                    dx.set(idx, stdInv * (
                        dxHat.get(idx) - 
                        dxHatSumVal - 
                        xHat.get(idx) * dxHatXhatSumVal
                    ));
                }
            }
        });
        dx.clip(-1.0f, 1.0f);
        
        return dx;
    }

    @Override
    public JMatrix[] getWeights() {
        return new JMatrix[]{gamma, beta, runningMean, runningVar};
    }

    @Override
    public int[] outputShape() {
        return getPreviousLayer().outputShape();
    }
    @Override
    public void updateParameters(JMatrix[] parameterUpdates) {
        // Scale gamma gradients to prevent large updates
        JMatrix scaledGammaUpdate = parameterUpdates[0].multiply(0.1);
        
        gamma.subtractInPlace(scaledGammaUpdate);
        beta.subtractInPlace(parameterUpdates[1]);
        
        // Ensure gamma doesn't get too small to avoid vanishing gradients
        for (int c = 0; c < featureSize; c++) {
            if (Math.abs(gamma.get(c)) < 0.01f) {
                gamma.set(c, 0.01f * Math.signum(gamma.get(c)));
            }
        }
    }

    

    @Override
    public JMatrix getOutput() {
        return output;
    }
    @Override
    public JMatrix getGradient() {
        return dx;
    }
    @Override
    public JMatrix[] getParameterGradients() {
        return new JMatrix[]{dGamma, dBeta};
    }
}