package jflow.layers;

import java.util.stream.IntStream;

import jflow.data.JMatrix;
import jflow.layers.templates.TrainableLayer;

public class Dense extends TrainableLayer {
    private JMatrix weights;
    private JMatrix dWeights;
    private JMatrix lastInput;
    private JMatrix biases;
    private JMatrix dBiases;

    private int outputSize;
    private boolean useBias = true;

    private boolean scaleDuringMatmul = true;
    private float[] tiedWeights = null;

    // For custom weight initialization
    private boolean useUniform = false;
    private boolean useNormal = false;
    private double[] uniformRange;
    private double[] normalDist;

    /**
     * Represents a fully connected layer.
     * 
     * <p><b>Do not instantiate directly.</b> Use the static builder method:
     * {@code import static jflow.model.builder.*;}
     * and call {@code Dense(...)} instead of {@code new Dense(...)}.
     */
    public Dense(int size, int[] inputShape, boolean useBias) {
        super("dense");
        this.outputSize = size;
        this.useBias = useBias;
        setInputShape(inputShape);
    }

    /**
     * Initialize the weights of this Dense layer in a uniform range.
     * @param min the minimum value.
     * @param max the maximum value.
     */
    public Dense initUniform(double min, double max) {
        useUniform = true;
        uniformRange = new double[]{min, max};
        return this;
    }

    /**
     * Initialize the weights of this Dense layer in a normal distribution.
     * @param mean the mean of the distribution.
     * @param stddev the standard deviation of the distribution.
     */
    public Dense initNormal(double mean, double stddev) {
        useNormal = true;
        normalDist = new double[]{mean, stddev};
        return this;
    }

    /**
     * Indicate if scaling by 1/k should be applied during matrix multiplication.
     * Enabled by default. <p>
     * Recommended for image data.
     * Not recommended for transformers.
     * @param enabled Enable or disable scaling during matrix multiplication.
     */
    public Dense scaleDuringMatmul(boolean enabled) {
        scaleDuringMatmul = enabled;
        return this;
    }

    /**
     * Set the weight matrix for weight tying.
     * @param weights The weight matrix to use for this Dense layer.
     * @param removeFromParamCount Indicates if the weight matrix of this Dense 
     * layer should be removed from the parameter count.
     */
    public Dense weightTie(float[] weights, boolean removeFromParamCount) {
        this.tiedWeights = weights;
        if (removeFromParamCount) {
            // Remove the matrix from numTrainableParameters
            setNumTrainableParameters((useBias) ? outputSize : 0);
        } else {
            setNumTrainableParameters((useBias) ? outputSize + weights.length : weights.length);
        }
        return this;
    }  

    @Override
    protected void build(int IDnum) {
        super.build(IDnum);
        int inputSize;
        int[] inputShape = getInputShape();

        if (inputShape == null) {
            throw new IllegalStateException(
                    "In " + this.getClass().getSimpleName() + 
                    ": Cannot build the first layer without an input shape."
                );
        } else {
            inputSize = inputShape[1]; // Feature dimension
        }

        initParams(inputSize);

        if (tiedWeights == null) {
            int numTrainableParameters = 0;
            numTrainableParameters += inputSize * outputSize;
            if (useBias) {
                numTrainableParameters += outputSize;
            }
            setNumTrainableParameters(numTrainableParameters);
        } else {
            weights.setMatrix(tiedWeights);
            tiedWeights = null;
        }
    }

    private void initParams(int inputSize) {
        // Check for custom initialization
        if (useNormal) {
            weights = JMatrix.normal(
                outputSize, inputSize, 1, 1, 
                normalDist[0], normalDist[1]
            );
            if (useBias) {
                biases = JMatrix.normal(
                    outputSize, 1, 1, 1, 
                    normalDist[0], normalDist[1]
                );
            }
        } else if (useUniform) {
            weights = JMatrix.uniform(
                outputSize, inputSize, 1, 1, 
                uniformRange[0], uniformRange[1]
            );
            if (useBias) {
                biases = JMatrix.uniform(
                    outputSize, 1, 1, 1, 
                    uniformRange[0], uniformRange[1]
                );
            }
        } else {
            // Xavier initialization
            double scale = Math.sqrt(2.0 / (inputSize + outputSize)); 
            double min = -1.0 * scale;
            double max = scale;

            weights = JMatrix.uniform(
                outputSize, inputSize, 1, 1, 
                min, max
            );
            if (useBias) {
                biases = JMatrix.uniform(
                    outputSize, 1, 1, 1, 
                    min, max
                );
            }
        }
        weights.setName("weights");

        dWeights = JMatrix.zeros(
            outputSize, inputSize, 1, 1
        )
        .setName("dWeights");

        if (useBias) {
            biases.setName("biases");
            dBiases = JMatrix.zeros(
                outputSize, 1, 1, 1
            )
            .setName("dBiases");
        }   
    }

    @Override
    public JMatrix forward(JMatrix input, boolean training) {
        // Transpose if necessary
        if (input.length() != 
                weights.shape(1) * weights.shape(2) * weights.shape(3)) {
            input = input.T();
        }
        // Store lastInput for backpropagation
        if (training) {
            lastInput = input;
        }

        // Calculate forward output
        JMatrix output = weights.matmul(input, scaleDuringMatmul); 

        if (useBias) {
            addBiasPerOutputFeature(output, biases); 
        }
    
        return trackOutput(output, training);
    }

    @Override
    public JMatrix backward(JMatrix gradient) {
        // Transpose if necessary
        if (gradient.shape(1) * gradient.shape(2) * gradient.shape(3) != 
                lastInput.shape(1) * lastInput.shape(2) * lastInput.shape(3)) {
            gradient = gradient.T();
        }

        // Calculate dWeights and dBiases
        JMatrix weightGrad = gradient.matmul(lastInput.T(), scaleDuringMatmul);
        dWeights.addInPlace(weightGrad); // Accumulate updates

        if (useBias) {
            JMatrix biasGrad = gradient.sum(0).multiplyInPlace(1.0 / gradient.length()); // Scaled sum
            dBiases.addInPlace(biasGrad); // Accumulate updates
        }

        // Free memory
        lastInput = null;
       
        // Normalize dWeights and dBiases
        adaptiveGradientClip(weights, biases, dWeights, dBiases, 1e-2);

        // Calculate loss w.r.t previous layer
        JMatrix dX = weights.T().matmul(gradient, scaleDuringMatmul);

        return trackGradient(dX);
    }

    

    @Override
    public void updateParameters(JMatrix[] parameterUpdates) {
        weights.subtractInPlace(parameterUpdates[0]);
        if (useBias) {
            biases.subtractInPlace(parameterUpdates[1]);
        }
    }

    @Override
    public JMatrix[] getParameters() {
        if (useBias) {
            return new JMatrix[]{weights, biases};
        } else {
            return new JMatrix[]{weights};
        }
        
    }

    @Override
    public JMatrix[] getParameterGradients() {
        if (useBias) {
            return new JMatrix[]{dWeights, dBiases};
        } else {
            return new JMatrix[]{dWeights};
        }
    }

    @Override
    public int[] outputShape() {
        int[] outputShape = new int[] {1, outputSize};
        return outputShape;
    }

    private void addBiasPerOutputFeature(JMatrix output, JMatrix bias) {
        // Transposed
        int batchSize = output.shape(1) * output.shape(2) * output.shape(3);
        int outputDim = output.shape(0);
    
        IntStream.range(0, batchSize).parallel().forEach(i -> {
            for (int j = 0; j < outputDim; j++) {
                int idx = i * outputDim + j;
                output.set(idx, output.get(idx) + bias.get(j));
            }
        });
    }
    
    // Adaptively clip with L2 norm
    private void adaptiveGradientClip(
        JMatrix weights, 
        JMatrix biases, 
        JMatrix dWeights, 
        JMatrix dBiases, 
        double epsilon
    ) {
        // Clip weights
        double weightNorm = weights.l2Norm();
        double gradWeightNorm = dWeights.l2Norm();

        double maxWeightNorm = Math.max(gradWeightNorm, epsilon * weightNorm);
        
        if (gradWeightNorm > maxWeightNorm) {
            double weightScale = maxWeightNorm / gradWeightNorm;
            dWeights.multiplyInPlace(weightScale);
        }
        
        // Clip biases
        if (useBias) {
            double biasNorm = biases.l2Norm();
            double gradBiasNorm = dBiases.l2Norm();

            double maxBiasNorm = Math.max(gradBiasNorm, epsilon * biasNorm);
            
            if (gradBiasNorm > maxBiasNorm) {
                double biasScale = maxBiasNorm / gradBiasNorm;
                dBiases.multiplyInPlace(biasScale);
            }
        }
    }
}



